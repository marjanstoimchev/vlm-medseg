"""Regenerate the Kaggle notebook from source: python scripts/make_notebook.py"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

C = []


def md(s):
    C.append(new_markdown_cell(s))


def co(s):
    C.append(new_code_cell(s))

md("""# Grounding VLMs + SAM for instance segmentation on PanNuke (box prompts)

Exploratory driver. All logic lives in the [`vlm-medseg`](https://github.com/marjanstoimchev/vlm-medseg) package; this notebook configures a run, calls into it, and plots. The pipeline and metrics are dataset-agnostic (PanNuke is the reference `DatasetSpec`).

**Conditions compared**

| Condition | Boxes / masks from | Isolates |
|---|---|---|
| `la_box` | LocateAnything-3B → SAM2 | the fine-tuned grounding VLM |
| `qwen_box` | stock Qwen2.5-VL-3B → SAM2 | the *un*-finetuned backbone (what LA's training buys) |
| `oracle_box` | ground-truth boxes → SAM2 | SAM2's ceiling given perfect detection |
| `sam3_text` | SAM3 text concept, end to end | a detector-free text-promptable segmenter |

`la_box` ≪ `oracle_box` ⇒ the bottleneck is VLM detection; `la_box` vs `qwen_box` ⇒ the value of LA's grounding training; `sam3_text` ⇒ whether one text-promptable model can skip the detector entirely.

> Models load **sequentially** and free their memory, so this fits a 16 GB GPU. NVIDIA's LocateAnything license is academic / non-profit research only.
""")

md("## 1 — Bootstrap\nKaggle ships a CUDA build of torch; we add only the model stacks (`transformers==4.57.1` serves LocateAnything, Qwen2.5-VL, SAM2 and SAM3) and the package from GitHub. **Enable GPU + Internet** (Settings → Accelerator: GPU, Internet: On). If a previous attempt half-installed the package, **restart the kernel first** (Run → Restart & clear outputs) so the stale copy is cleared before this cell runs.")
co('''import os, sys, subprocess
IN_KAGGLE = os.path.exists("/kaggle")
REPO = "git+https://github.com/marjanstoimchev/vlm-medseg.git@main"

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

if IN_KAGGLE:
    # Kaggle ships a CUDA build of torch -- add only the model stacks on top.
    pip("transformers==4.57.1", "decord==0.6.0", "lmdb==1.7.5", "peft", "accelerate", "einops", "timm")
    # Clean rebuild of our package only: a stale/cached copy from an earlier run can be
    # missing subpackages; --force-reinstall --no-cache-dir guarantees a fresh build,
    # --no-deps leaves Kaggle's preinstalled torch/transformers untouched.
    pip("--no-cache-dir", "--force-reinstall", "--no-deps", REPO)
else:
    pip("-e", "..")

from vlm_medseg.data import get_dataset  # smoke test: fail here, not 5 cells later
print("bootstrap done")''')

md("## 2 — Configuration")
co("""import gc, json, time, numpy as np, torch
from pathlib import Path

DATASET     = "pannuke"
FOLD        = "fold1"
N_PATCHES   = 150
SEED        = 0
CLASS_AWARE = True
IOU_THRESH  = 0.5
INPUT_SHORT = 1024            # upscale 256 -> 1024 so tiny nuclei are visible to the VLM
SAM2_MODEL  = "facebook/sam2-hiera-large"
LA_MODEL    = "nvidia/LocateAnything-3B"
QWEN_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
SAM3_MODEL  = "facebook/sam3"
SAM3_THRESH = 0.3            # SAM3 barely fires on H&E zero-shot; lower to probe recall
RUN_QWEN    = True
RUN_SAM3    = True
RUN_DIR     = Path("/kaggle/working/run" if IN_KAGGLE else "runs/kaggle_local")
RUN_DIR.mkdir(parents=True, exist_ok=True)

def free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print("run dir:", RUN_DIR)""")

md("## 3 — Load and sample the dataset")
co("""from vlm_medseg.data import get_dataset
import matplotlib.pyplot as plt
from collections import Counter
from vlm_medseg.viz import overlay_gt

ds = get_dataset(DATASET, fold=FOLD); spec = ds.spec
samples = ds.sample(N_PATCHES, seed=SEED)
print(f"{spec.name}: {len(samples)} patches | mean nuclei/patch = "
      f"{np.mean([s.num_instances for s in samples]):.1f}")
print("groups:", dict(Counter(spec.group_name(s.group) for s in samples).most_common()))

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for ax, s in zip(axes.flat, samples[:8]):
    ax.imshow(overlay_gt(s, spec=spec)); ax.axis("off")
    ax.set_title(f"{spec.group_name(s.group)} · n={s.num_instances}", fontsize=9)
plt.tight_layout(); plt.show()""")

md("## 4 — VLM detection (cached)\nRun each grounding VLM once, cache its boxes to JSONL, then free its memory. SAM2 + metrics re-run anywhere from the cache via `vlm-medseg eval-cache`.")
co("""from vlm_medseg.detect.locate_anything import LocateAnythingDetector
from vlm_medseg.pipeline.cache import save_detections

def detect_all(detector, tag):
    recs, pre = [], {}
    t0 = time.time()
    for k, s in enumerate(samples, 1):
        td = time.time(); dets = detector.detect(s)
        pre[s.sample_id] = dets
        recs.append({"sample_id": s.sample_id, "condition": tag,
                     "detector_seconds": time.time() - td, "detections": dets})
        if k % 25 == 0 or k == len(samples):
            print(f"  {tag} [{k:>3}/{len(samples)}] {(time.time()-t0)/k:.1f}s/patch · last={len(dets)}")
    save_detections(RUN_DIR / f"{tag}.jsonl", recs)
    return pre

la = LocateAnythingDetector(LA_MODEL, device="cuda", mode="detection",
                            class_aware=CLASS_AWARE, spec=spec, input_short_size=INPUT_SHORT)
la_pre = detect_all(la, "la_box")
del la; free()""")

co("""qwen_pre = None
if RUN_QWEN:
    from vlm_medseg.detect.qwen_vl import QwenVLDetector
    qwen = QwenVLDetector(QWEN_MODEL, device="cuda", class_aware=CLASS_AWARE, spec=spec)
    qwen_pre = detect_all(qwen, "qwen_box")
    del qwen; free()""")

md("## 5 — SAM2 mask + evaluate the box conditions\nOne SAM2 load scores every cached box set plus the ground-truth oracle.")
co("""from vlm_medseg.segment.sam2 import Sam2Masker
from vlm_medseg.detect.oracle import OracleBoxDetector
from vlm_medseg.pipeline.run import run_condition

masker = Sam2Masker(SAM2_MODEL, device="cuda" if IN_KAGGLE else "auto")
runs = {}
runs["la_box"] = run_condition(None, masker, samples, spec=spec, precomputed=la_pre, iou_thresh=IOU_THRESH)
if qwen_pre is not None:
    runs["qwen_box"] = run_condition(None, masker, samples, spec=spec, precomputed=qwen_pre, iou_thresh=IOU_THRESH)
runs["oracle_box"] = run_condition(OracleBoxDetector(class_aware=CLASS_AWARE, spec=spec),
                                   masker, samples, spec=spec, iou_thresh=IOU_THRESH)
del masker; free()
print("scored:", list(runs))""")

md("## 6 — SAM3 text concept segmentation (end to end, no detector)")
co("""if RUN_SAM3:
    from vlm_medseg.segment.sam3 import Sam3TextSegmenter
    from vlm_medseg.pipeline.run import run_segmenter_condition
    sam3 = Sam3TextSegmenter(SAM3_MODEL, device="cuda" if IN_KAGGLE else "auto",
                             spec=spec, class_aware=CLASS_AWARE, threshold=SAM3_THRESH)
    runs["sam3_text"] = run_segmenter_condition(sam3, samples, spec=spec, iou_thresh=IOU_THRESH)
    del sam3; free()
    print("scored sam3_text")""")

md("## 7 — Comparison table")
co("""import pandas as pd
def row(name, out):
    s = out["summary"]; d = s["detection_pooled"]
    r = {"condition": name, "PQ": s["binary_pq"], "AJI": s["aji"], "Dice": s["dice"],
         "matchedIoU": s["matched_iou"], "det_R": d["recall"], "det_F1": d["f1"],
         "pred/gt": s["counting"]["total_pred"] / max(1, s["counting"]["total_gt"])}
    if "mpq" in s: r["mPQ"] = s["mpq"]
    return r
pd.DataFrame([row(k, v) for k, v in runs.items()]).set_index("condition").round(3)""")

md("## 8 — Plots")
co("""from vlm_medseg.viz import plot_metric_bars, plot_per_group_pq, plot_confusion
summaries = {k: v["summary"] for k, v in runs.items()}
plot_metric_bars(summaries).show()
plot_per_group_pq(summaries["la_box"], spec=spec).show()
if CLASS_AWARE:
    fig = plot_confusion(summaries["la_box"], spec=spec)
    if fig: fig.show()""")

co("""from vlm_medseg.viz import gallery
pred_by_cond = {k: [r.instances for r in v["results"]] for k, v in runs.items()}
gallery(samples, pred_by_cond, spec=spec, max_rows=6).show()""")

md("## 9 — Save artifacts")
co("""summary_out = {k: v["summary"] for k, v in runs.items()}
(RUN_DIR / "summaries.json").write_text(json.dumps(summary_out, indent=2, default=str))
plot_metric_bars(summaries).savefig(RUN_DIR / "metrics.png", dpi=150, bbox_inches="tight")
print("saved to", RUN_DIR)""")

md("""## 10 — Reading the result

- **`oracle_box` PQ** is SAM2-large's ceiling given perfect boxes; what `la_box` loses below it is the **VLM detection gap**.
- **`la_box` vs `qwen_box`** measures what LocateAnything's grounding fine-tuning buys over the stock backbone.
- **`det_R` / `pred/gt`** expose undercounting — the usual failure of grounding VLMs on dense tiny nuclei.
- **`sam3_text`** tests a detector-free path; SAM3's concepts transfer weakly to H&E zero-shot, so expect low recall unless `SAM3_THRESH` is dropped.

Scale up via `N_PATCHES`, or re-score cached boxes offline:
`vlm-medseg eval-cache --detections run/la_box.jsonl --fold fold1 --out runs/la_box`.
""")

nb = new_notebook(cells=C)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "notebooks" / "locate_anything_pannuke_kaggle.ipynb"
nbf.write(nb, str(out))
print("wrote", out, "cells:", len(C))
