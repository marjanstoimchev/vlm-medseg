#!/usr/bin/env python
"""Run one instance-segmentation method on PanNuke patches and save inspectable
results under ``results/<method>/`` — predictions, per-patch metrics, an
aggregated summary, and GT-vs-prediction overlays.

Methods (box detectors feed SAM2; SAM3 is end to end):

    oracle_box   ground-truth boxes -> SAM2        ceiling; local, fast
    gdino_box    Grounding-DINO     -> SAM2        local (MPS/CPU/CUDA)
    qwen_box     Qwen2.5-VL         -> SAM2        local MPS or Kaggle
    la_box       LocateAnything-3B  -> SAM2        CUDA / Kaggle only
    sam3_text    SAM3 text concept                 end to end; local MPS

Examples:
    python scripts/run_method.py oracle_box --n 20
    python scripts/run_method.py qwen_box  --n 8  --device mps
    python scripts/run_method.py sam3_text --n 8  --threshold 0.3
    python scripts/run_method.py gdino_box --n 8  --no-class-aware
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

from vlm_medseg.log import configure_logging, get_logger

# Route ops MPS doesn't implement (e.g. cummax in Grounding-DINO) to CPU. Must be
# set before torch initializes its MPS backend, so do it at import time.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO / "results"
log = get_logger("run_method")

BOX_METHODS = {"oracle_box", "gdino_box", "owlv2_box", "qwen_box", "la_box"}
SEG_METHODS = {"sam3_text"}

DEFAULT_MODELS = {
    "oracle_box": None,
    "gdino_box": "IDEA-Research/grounding-dino-base",
    "owlv2_box": "google/owlv2-base-patch16",
    "qwen_box": "Qwen/Qwen2.5-VL-3B-Instruct",
    "la_box": "nvidia/LocateAnything-3B",
    "sam3_text": "facebook/sam3",
}
ORACLE_METHODS = {"oracle_box"}  # use GT labels; a classifier would only degrade them


def build_detector(method: str, args, spec):
    if method == "oracle_box":
        from vlm_medseg.detect.oracle import OracleBoxDetector
        return OracleBoxDetector(class_aware=args.class_aware, spec=spec)  # no SAHI for GT

    if method == "gdino_box":
        from vlm_medseg.detect.grounding_dino import GroundingDinoDetector
        kw = {"device": args.device, "class_aware": args.class_aware, "spec": spec}
        det = GroundingDinoDetector(args.model, **kw) if args.model else GroundingDinoDetector(**kw)
    elif method == "owlv2_box":
        from vlm_medseg.detect.owlv2 import Owlv2Detector
        kw = {"device": args.device, "class_aware": args.class_aware, "spec": spec}
        det = Owlv2Detector(args.model, **kw) if args.model else Owlv2Detector(**kw)
    elif method == "qwen_box":
        from vlm_medseg.detect.qwen_vl import QwenVLDetector
        kw = {"device": args.device, "class_aware": args.class_aware, "spec": spec}
        if args.input_short_size:
            kw["input_short_size"] = args.input_short_size
        det = QwenVLDetector(args.model, **kw) if args.model else QwenVLDetector(**kw)
    elif method == "la_box":
        from vlm_medseg.detect.locate_anything import LocateAnythingDetector
        dev = args.device if args.device != "auto" else "cuda"
        kw = {"device": dev, "class_aware": args.class_aware, "spec": spec}
        if args.input_short_size:
            kw["input_short_size"] = args.input_short_size
        det = LocateAnythingDetector(args.model, **kw) if args.model else LocateAnythingDetector(**kw)
    else:
        raise ValueError(method)

    if getattr(args, "sahi", False):
        from vlm_medseg.detect.sahi import SahiDetector
        det = SahiDetector(det, tile=args.tile, overlap=args.overlap)
    return det


def build_segmenter(method: str, args, spec):
    if method == "sam3_text":
        from vlm_medseg.segment.sam3 import Sam3TextSegmenter
        kw = {"device": args.device, "class_aware": args.class_aware, "spec": spec,
              "threshold": args.threshold}
        return Sam3TextSegmenter(args.model, **kw) if args.model else Sam3TextSegmenter(**kw)
    raise ValueError(method)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_predictions(path: Path, results, spec):
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps({
                "sample_id": r.sample_id,
                "n_instances": len(r.instances),
                "detector_seconds": round(r.detector_seconds, 3),
                "instances": [
                    {"class_id": ins.class_id,
                     "class": spec.class_name(ins.class_id) if ins.class_id is not None else None,
                     "score": round(float(ins.score), 4),
                     "box": [round(float(v), 1) for v in ins.box] if ins.box else None,
                     "area_px": int(ins.mask.sum())}
                    for ins in r.instances
                ],
            }) + "\n")


def save_patch_overlay(out_dir: Path, sample, panel, spec):
    """Write one patch's overlay (``H&E | GT | <cols...>``) immediately."""
    from vlm_medseg.viz import comparison_panel

    ov = out_dir / "overlays"
    ov.mkdir(exist_ok=True)
    fig = comparison_panel(sample, panel, spec=spec)
    fig.savefig(ov / f"{sample.sample_id}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_gallery(out_dir: Path, samples, columns, spec, rows):
    """Grid overlay across patches; ``columns = label -> {sample_id: instances}``."""
    from vlm_medseg.viz import gallery

    labels = list(columns)
    fig = gallery(samples, {lab: [columns[lab][s.sample_id] for s in samples] for lab in labels},
                  spec=spec, max_rows=rows)
    fig.savefig(out_dir / "gallery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=sorted(BOX_METHODS | SEG_METHODS))
    p.add_argument("--dataset", default="pannuke")
    p.add_argument("--fold", default="fold1")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default=None, help="override the default model id")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-large")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.3, help="SAM3 detection threshold")
    p.add_argument("--input-short-size", type=int, default=None,
                   help="upscale the patch short side before the VLM (qwen_box/la_box); "
                        "default uses the detector's own (qwen 896, la 1024)")
    p.add_argument("--sahi", action="store_true",
                   help="tile the patch (SAHI) before the box detector and NMS-merge -> "
                        "helps detectors that return whole-image boxes on dense tiny objects")
    p.add_argument("--tile", type=int, default=128, help="SAHI tile size (px)")
    p.add_argument("--overlap", type=float, default=0.25, help="SAHI tile overlap fraction")
    p.add_argument("--class-aware", dest="class_aware", action="store_true", default=True)
    p.add_argument("--no-class-aware", dest="class_aware", action="store_false")
    p.add_argument("--max-overlays", type=int, default=12)
    p.add_argument("--gallery-rows", type=int, default=6)
    p.add_argument("--out", default=None, help="output dir (default results/<method>)")
    p.add_argument("--classifier", default=None,
                   help="trained probe: a .joblib path OR a bare name resolved to "
                        "models/classifiers/<name>.joblib (e.g. uni2h_mlp); keeps the method's "
                        "prompts and re-labels each mask -> before/after overlays + metrics")
    args = p.parse_args()
    configure_logging()

    from vlm_medseg.data import get_dataset
    from vlm_medseg.pipeline.run import run_condition, run_segmenter_condition

    ds = get_dataset(args.dataset, fold=args.fold)
    spec = ds.spec
    samples = ds.sample(args.n, seed=args.seed)
    log.info(f"[{args.method}] {args.dataset}/{args.fold}: {len(samples)} patches "
          f"(mean nuclei/patch={np.mean([s.num_instances for s in samples]):.1f})")

    out_dir = Path(args.out) if args.out else RESULTS_ROOT / (("sahi_" if args.sahi else "") + args.method)
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = None
    if args.classifier and args.method in ORACLE_METHODS:
        log.info(f"  note: {args.method} uses ground-truth labels -> classifier ignored.")
    elif args.classifier:
        from vlm_medseg.classify.linear_probe import NucleusClassifier
        clf_path = Path(args.classifier)
        if not clf_path.exists():  # bare name -> models/classifiers/<name>.joblib
            nm = args.classifier if args.classifier.endswith(".joblib") else f"{args.classifier}.joblib"
            clf_path = REPO / "models" / "classifiers" / nm
        classifier = NucleusClassifier.load(clf_path, device=args.device)
        log.info(f"  classifier: {clf_path} (keeps prompts; probe re-labels each mask)")

    model_id = args.model or DEFAULT_MODELS.get(args.method)
    if args.method in BOX_METHODS:
        detector = build_detector(args.method, args, spec)
        from vlm_medseg.segment.sam2 import Sam2Masker
        masker = Sam2Masker(args.sam2_model, device=args.device)
        cond_name = getattr(detector, "name", args.method)
    else:
        segmenter = build_segmenter(args.method, args, spec)
        cond_name = getattr(segmenter, "name", args.method)

    # Per-patch callback: (classify, then) write this patch's overlay immediately
    # so you can watch results appear during the run rather than only at the end.
    from vlm_medseg.eval.report import evaluate_patch
    from vlm_medseg.types import InstancePrediction
    after_by_id: dict = {}
    after_metrics: list = []
    drawn = [0]

    def on_result(sample, res):
        if classifier is not None:
            ci = [InstancePrediction(mask=i.mask, class_id=i.class_id, score=i.score, box=i.box, point=i.point)
                  for i in res.instances]                 # copy (shares mask); probe overrides class
            classifier.classify_instances(sample.image, ci)
            after_by_id[sample.sample_id] = ci
            after_metrics.append(evaluate_patch(sample, ci, num_classes=spec.num_classes, iou_thresh=args.iou))
            panel = {"before (prompts)": res.instances, "after (classifier)": ci}
        else:
            panel = {cond_name: res.instances}
        if drawn[0] < args.max_overlays:
            try:
                save_patch_overlay(out_dir, sample, panel, spec)
            except Exception as e:
                log.info("  (overlay skip:", e, ")")
            drawn[0] += 1

    t0 = time.time()
    if args.method in BOX_METHODS:
        out = run_condition(detector, masker, samples, spec=spec, iou_thresh=args.iou, on_result=on_result)
    else:
        out = run_segmenter_condition(segmenter, samples, spec=spec, iou_thresh=args.iou, on_result=on_result)
    elapsed = time.time() - t0

    results = out["results"]                       # model labels = "before"
    by_id = {r.sample_id: r for r in results}
    summary_before = out["summary"]
    columns = {cond_name: {r.sample_id: r.instances for r in results}}
    summary, final_metrics, final_results, summary_after = (
        summary_before, out["patch_metrics"], results, None
    )

    if classifier is not None:
        from vlm_medseg.eval.report import summarize
        from vlm_medseg.types import PatchResult
        summary_after = summarize(after_metrics, num_classes=spec.num_classes,
                                  class_names=spec.class_names, group_names=spec.group_names)
        columns = {"before (prompts)": {sid: by_id[sid].instances for sid in by_id},
                   "after (classifier)": after_by_id}
        summary, final_metrics = summary_after, after_metrics
        final_results = [PatchResult(sample_id=s.sample_id, condition=cond_name,
                                     instances=after_by_id[s.sample_id],
                                     detector_seconds=by_id[s.sample_id].detector_seconds)
                         for s in samples]

    # ── persist ──────────────────────────────────────────────────────────
    (out_dir / "config.json").write_text(json.dumps({
        "method": args.method, "dataset": args.dataset, "fold": args.fold,
        "n_patches": len(samples), "seed": args.seed, "device": args.device,
        "model": str(model_id), "sam2_model": args.sam2_model if args.method in BOX_METHODS else None,
        "class_aware": args.class_aware, "iou_thresh": args.iou,
        "classifier": args.classifier if classifier is not None else None,
        "threshold": args.threshold if args.method in SEG_METHODS else None,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    if summary_after is not None:
        (out_dir / "summary_before.json").write_text(json.dumps(_jsonable(summary_before), indent=2))
    with (out_dir / "patch_metrics.jsonl").open("w") as fh:
        for m in final_metrics:
            fh.write(json.dumps(_jsonable(m)) + "\n")
    save_predictions(out_dir / "predictions.jsonl", final_results, spec)
    try:
        save_gallery(out_dir, samples, columns, spec, args.gallery_rows)
    except Exception as e:  # plotting is non-essential; never lose the metrics
        log.info("  (gallery rendering skipped:", e, ")")

    # ── report ───────────────────────────────────────────────────────────
    d = summary["detection_pooled"]
    log.info(f"\n  wrote -> {out_dir}  ({elapsed:.1f}s, {elapsed/len(samples):.2f}s/patch)")
    log.info(f"  binary PQ={summary['binary_pq']:.3f}  SQ={summary['sq']:.3f}  DQ={summary['dq']:.3f}  "
          f"AJI={summary['aji']:.3f}  Dice={summary['dice']:.3f}  matchedIoU={summary['matched_iou']:.3f}")
    log.info(f"  detection  P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}  "
          f"(pred/gt={summary['counting']['total_pred']}/{summary['counting']['total_gt']})")
    if summary["counting"]["total_pred"] == 0:
        log.info("  !! 0 instances predicted -> overlays are empty and nothing was classified. "
              "For sam3_text lower --threshold (e.g. 0.05); for detectors lower the score threshold.")
    if "mpq" in summary:
        log.info(f"  mPQ={summary['mpq']:.3f}  per-class PQ={ {k: round(v,3) for k,v in summary['per_class_pq'].items()} }")
        log.info(f"  class accuracy (matched)={summary['classification']['accuracy']:.3f}")
    if summary_after is not None:
        b, a = summary_before, summary_after
        bc = b.get("classification", {}).get("accuracy", 0.0)
        ac = a.get("classification", {}).get("accuracy", 0.0)
        log.info(f"  [classifier correction]  mPQ {b.get('mpq',0):.3f} -> {a.get('mpq',0):.3f}   "
              f"class-acc {bc:.3f} -> {ac:.3f}   (masks identical; only labels change)")


if __name__ == "__main__":
    main()
