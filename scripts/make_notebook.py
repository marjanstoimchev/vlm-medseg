"""Regenerate the grounding-VLM comparison notebook: python scripts/make_notebook.py"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

C = []


def md(s):
    C.append(new_markdown_cell(s))


def co(s):
    C.append(new_code_cell(s))


md("""# Grounding VLMs for nuclei instance segmentation on PanNuke

Does grounding fine-tuning teach a vision-language model to localise nuclei? This
notebook isolates that question by comparing two VLMs that emit boxes for SAM2 to turn
into masks — a **stock** Qwen2.5-VL and a **grounding-tuned** LocateAnything-3B — against
a ground-truth-prompted **oracle** that fixes the achievable segmentation quality.

| method | how it localises nuclei | isolates |
|---|---|---|
| **oracle — GT boxes → SAM2** | ground-truth boxes | upper bound (segmentation ceiling) |
| **Qwen2.5-VL → SAM2** | stock grounding VLM emits boxes | the un-tuned backbone |
| **LocateAnything-3B → SAM2** | grounding-tuned VLM emits boxes | what grounding training buys |

Both VLMs share one **SAM2** backbone for masks, so the comparison isolates how each model
*points at* nuclei rather than how it draws them. The distance from either VLM to the oracle
is a **detection** gap; the distance between the two VLMs is the value of grounding
fine-tuning. An optional **UNI2-h + MLP** classifier relabels each predicted nucleus from
its pixels, separating mask quality from the difficulty of subtyping.

> **`N_IMAGES`** and **`SEED`** (in the configuration) set how many random patches are shown
> and which draw. LocateAnything is CUDA-only and its weights are released for academic /
> non-profit research only.
""")

md("## 1 — Bootstrap\nKaggle ships a CUDA build of torch; we add only the model stacks (`transformers==4.57.1`, the version LocateAnything pins and which also serves Qwen2.5-VL and SAM2) and the package from GitHub. **Enable GPU + Internet** (Settings → Accelerator: GPU, Internet: On). If a previous attempt half-installed the package, **restart the kernel first** (Run → Restart & clear outputs).\n\n- The classifier's encoder (**UNI2-h**) and the **LocateAnything** weights are gated — add a Kaggle Secret **`HF_TOKEN`** (Add-ons → Secrets) so they can download.")
co('''import os, sys, subprocess
IN_KAGGLE = os.path.exists("/kaggle")
REPO = "git+https://github.com/marjanstoimchev/vlm-medseg.git@main"

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

if IN_KAGGLE:
    # transformers 4.57.1 is LocateAnything's pinned version (also serves Qwen + SAM2).
    pip("transformers==4.57.1", "decord==0.6.0", "lmdb==1.7.5", "peft", "accelerate", "einops", "timm")
    # Clean rebuild of our package only: a stale/cached copy can be missing subpackages;
    # --no-deps leaves Kaggle's preinstalled torch/transformers untouched.
    pip("--no-cache-dir", "--force-reinstall", "--no-deps", REPO)
else:
    pip("-e", "..")

# HF auth for the gated UNI2-h (classifier encoder) and LocateAnything weights.
hf_token = os.environ.get("HF_TOKEN")
if IN_KAGGLE and not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("HF authenticated -- gated models enabled")
else:
    print("no HF_TOKEN -- gated downloads (UNI2-h classifier, LocateAnything) may fail")

from vlm_medseg.data import get_dataset  # smoke test: fail here, not 5 cells later
print("bootstrap done")''')

md("""## 2 — Configuration
Everything per-method lives here:
- **`N_IMAGES` / `SEED`** — how many random patches and which draw.
- **`INPUT_SHORT`** — short side the patch is upscaled to before each VLM (the same for both, for a fair comparison).
- **Classifier** — `USE_CLASSIFIER` + `CLASSIFIER_DIR` (the attached UNI2-h Model). When on, it overrides each predicted nucleus class from its pixels for both VLMs **except the oracle**.

To fit a 16 GB GPU the run is **phased** — VLM detection, then SAM2 masking, then classification — so only one heavy model is resident at a time.""")
co("""import gc, os, glob, numpy as np, torch
import matplotlib.pyplot as plt
from vlm_medseg.viz import set_paper_style
set_paper_style()                      # clean, consistent figure styling for every plot below
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation

# -- dataset / sampling --
DATASET     = "pannuke"
FOLD        = "fold1"
N_IMAGES    = 6            # random patches to compare (raise for a bigger gallery)
SEED        = 0            # change to draw a different random set
CLASS_AWARE = True         # per-class prompts -> class-coloured overlays
IOU_THRESH  = 0.5

# -- per-method knobs --
INPUT_SHORT = 1024         # upscale short side so both VLMs can see tiny nuclei

# -- which methods to run --
RUN_QWEN, RUN_LA = True, True

# -- post-hoc nucleus classifier: overrides predicted class from pixels (UNI2-h + MLP) --
# Attach the Kaggle Model and point CLASSIFIER_DIR at it; the .joblib is auto-found.
USE_CLASSIFIER = True
CLASSIFIER_DIR = "/kaggle/input/models/marjan1111/uni2-h-mlp-classifier/scikitlearn/default/1"

def free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
DEVICE = "cuda" if IN_KAGGLE else "auto"
print("comparing", N_IMAGES, "patches from", f"{DATASET}/{FOLD}", "· seed", SEED,
      "· classifier:", USE_CLASSIFIER)""")

md("## 3 — Sample patches\nA uniform-random draw (not stratified), so the panel reflects a typical field of view rather than a curated one — change `SEED` to resample. Each patch shows its ground-truth nuclei: the **fill** encodes class and the **yellow outline** marks each boundary; the title gives tissue type and nucleus count.")
co("""import random
from vlm_medseg.data import get_dataset
from vlm_medseg.viz import overlay_gt, class_legend_handles

ds = get_dataset(DATASET, fold=FOLD); spec = ds.spec
idx = random.Random(SEED).sample(range(len(ds)), N_IMAGES)
samples = [ds.decode(i) for i in idx]
print("nuclei per patch:", [s.num_instances for s in samples])

cols = min(N_IMAGES, 6)
rows = (N_IMAGES + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.9 * rows), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, s in zip(axes.flat, samples):
    ax.imshow(overlay_gt(s, spec=spec))
    ax.set_title(f"{spec.group_name(s.group)} · {s.num_instances} nuclei", fontsize=9)
fig.legend(handles=class_legend_handles(spec), loc="lower center", ncol=spec.num_classes,
           bbox_to_anchor=(0.5, 0.0), fontsize=9, title_fontsize=9,
           title="nucleus class — fill colour   ·   yellow outline = boundary")
fig.suptitle("Ground-truth nuclei — the targets", y=1.0)
fig.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()""")

md("""## 4 — Run each method (memory-aware, phased)
A 16 GB GPU cannot hold a VLM, SAM2 and the classifier's encoder at once, so the work is split into phases that each free their model before the next:

1. **Detect** — load each VLM, emit boxes for every patch, free it.
2. **Segment** — load SAM2 once, turn every box set (and the ground-truth oracle) into masks, free it.
3. **Classify** — load the UNI2-h classifier, relabel both VLMs' nuclei from their pixels (the oracle keeps its ground-truth classes), free it.

> LocateAnything generates boxes autoregressively, so the detect phase is the slow one — expect it to take noticeably longer per patch than Qwen.""")
co("""from vlm_medseg.segment.sam2 import Sam2Masker
from vlm_medseg.detect.oracle import OracleBoxDetector
from vlm_medseg.pipeline.run import run_condition
from vlm_medseg.eval.report import evaluate_patch, summarize
from tqdm.auto import tqdm

# -- phase 1: detection (one VLM resident at a time; SAM2 not loaded yet) --
boxes = {}   # method -> {sample_id: [Detection]}
if RUN_QWEN:
    try:
        from vlm_medseg.detect.qwen_vl import QwenVLDetector
        qwen = QwenVLDetector(device=DEVICE, class_aware=CLASS_AWARE, spec=spec, input_short_size=INPUT_SHORT)
        boxes["qwen"] = {s.sample_id: qwen.detect(s) for s in tqdm(samples, desc="qwen detect")}
        del qwen; free()
    except Exception as e:
        print("qwen skipped:", type(e).__name__, e)
if RUN_LA:
    try:
        from vlm_medseg.detect.locate_anything import LocateAnythingDetector
        la = LocateAnythingDetector(device="cuda", class_aware=CLASS_AWARE, spec=spec, input_short_size=INPUT_SHORT)
        boxes["la"] = {s.sample_id: la.detect(s) for s in tqdm(samples, desc="la detect")}
        del la; free()
    except Exception as e:
        print("locate-anything skipped:", type(e).__name__, e)
        print("  LocateAnything needs a CUDA GPU and transformers==4.57.1 (the bootstrap pins it).")

# -- phase 2: segmentation with one shared SAM2 (no VLM resident) --
masker = Sam2Masker(device=DEVICE)
runs = {}
runs["oracle"] = run_condition(OracleBoxDetector(class_aware=CLASS_AWARE, spec=spec),
                               masker, samples, spec=spec, iou_thresh=IOU_THRESH)
for name, pre in boxes.items():
    runs[name] = run_condition(None, masker, samples, spec=spec, precomputed=pre, iou_thresh=IOU_THRESH)
del masker; free()

# -- phase 3: post-hoc classification (UNI2-h resident; nothing else heavy) --
# Relabel each non-oracle method from pixels, then recompute its class-aware metrics.
if USE_CLASSIFIER:
    from vlm_medseg.classify.linear_probe import NucleusClassifier
    hits = sorted(glob.glob(os.path.join(CLASSIFIER_DIR, "*.joblib")))
    if hits:
        clf = NucleusClassifier.load(hits[0], device=DEVICE)
        print("classifier:", os.path.basename(hits[0]), "->", clf.class_names)
        for name, out in runs.items():
            if name == "oracle":          # the oracle keeps its ground-truth classes
                continue
            pm = []
            for s, res in zip(samples, out["results"]):
                clf.classify_instances(s.image, res.instances)
                pm.append(evaluate_patch(s, res.instances, num_classes=spec.num_classes, iou_thresh=IOU_THRESH))
            out["patch_metrics"] = pm
            out["summary"] = summarize(pm, num_classes=spec.num_classes,
                                       class_names=spec.class_names, group_names=spec.group_names)
        del clf; free()
    else:
        print("USE_CLASSIFIER set but no .joblib under", CLASSIFIER_DIR, "-- continuing without it")

print("ran:", list(runs))""")

md("## 5 — Qualitative comparison\nOne row per patch: **H&E · ground truth · each method**. The fill encodes the predicted class and the **yellow outline** marks each boundary, so the figure reads like a contact sheet — missed nuclei, whole-image or coarse boxes, merged neighbours, and class confusion are all apparent against the ground-truth column.")
co("""from vlm_medseg.viz import gallery
pred_by_cond = {name: [r.instances for r in out["results"]] for name, out in runs.items()}
fig = gallery(samples, pred_by_cond, spec=spec, max_rows=N_IMAGES)
fig.suptitle("H&E | GT | " + " | ".join(runs), y=1.005, fontsize=12); plt.show()""")

md("## 6 — Quantitative comparison\nThe panoptic metrics are read as a **decomposition**, not a single figure of merit. **PQ** is overall panoptic quality; **AJI** and **Dice** summarise pixel agreement; **matched IoU** is mask quality on the nuclei a method actually recovers; **detection recall / F1** and the **predicted-to-true count ratio** capture how completely it finds them. For a grounding VLM the count ratio is especially telling: a model that emits a few coarse boxes under-counts badly even when its matched IoU looks healthy.")
co("""import pandas as pd
def row(name, out):
    s = out["summary"]; d = s["detection_pooled"]
    r = {"method": name, "PQ": s["binary_pq"], "AJI": s["aji"], "Dice": s["dice"],
         "matchedIoU": s["matched_iou"], "det_R": d["recall"], "det_F1": d["f1"],
         "pred/gt": s["counting"]["total_pred"] / max(1, s["counting"]["total_gt"])}
    if "mpq" in s: r["mPQ"] = s["mpq"]
    return r
table = pd.DataFrame([row(k, v) for k, v in runs.items()]).set_index("method")
display(table.round(3))

from vlm_medseg.viz import plot_metric_bars
plot_metric_bars({k: v["summary"] for k, v in runs.items()}).show()""")

md("## 7 — A single field, up close\nThe most densely populated patch in the draw, each method beside the ground truth, to inspect boundary adherence and how completely each VLM localises individual nuclei.")
co("""from vlm_medseg.viz import comparison_panel
busiest = int(np.argmax([s.num_instances for s in samples]))
s = samples[busiest]
panel = {name: out["results"][busiest].instances for name, out in runs.items()}
fig = comparison_panel(s, panel, spec=spec, figsize_scale=3.4)
fig.suptitle(f"{spec.group_name(s.group)} · {s.num_instances} nuclei", y=1.03); plt.show()""")

md("""## 8 — Reading the comparison

- **The oracle is the segmentation ceiling.** Given perfect boxes, SAM2 delineates nuclei well, so both VLMs are best read as a *fraction* of the oracle; the distance to it is the detection gap each VLM leaves on the table.
- **Stock versus grounding-tuned.** The gap between Qwen and LocateAnything is the value of grounding fine-tuning: if LocateAnything localises more nuclei — higher recall, count ratio nearer one — while matched IoU is comparable, the training is buying *detection*, not segmentation.
- **Coarse boxes under-count.** Grounding VLMs tend to emit a handful of large boxes on a dense field; matched IoU can look healthy while recall and the count ratio reveal how many nuclei were never proposed — the gap a denser proposal strategy would have to close.
- **Naming as a separate axis.** With the classifier enabled, the fill colours reflect a pixel-level subtype prediction rather than the prompt label; toggling it isolates how much of the class confusion is a *naming* problem solvable after segmentation versus a *segmentation* one.
- **Stability of the reading.** Raising **`N_IMAGES`** widens the sample and changing **`SEED`** resamples it; the qualitative ordering of the two VLMs should persist across draws.
""")

nb = new_notebook(cells=C)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "notebooks" / "locate_anything_pannuke_kaggle.ipynb"
nbf.write(nb, str(out))
print("wrote", out, "cells:", len(C))
