"""Command-line entry point (dataset-agnostic; box-prompt workflows).

LocateAnything runs in the Kaggle notebook (needs a CUDA GPU); the CLI covers
everything else, locally:

  vlm-medseg sample     --dataset pannuke --fold fold1 --n 150 --out runs/ids.json
  vlm-medseg oracle     --dataset pannuke --fold fold1 --n 30  --out runs/oracle_box
  vlm-medseg eval-cache --detections la_box.jsonl --fold fold1 --out runs/la_box
  vlm-medseg report     runs/la_box/summary.json

``eval-cache`` is the offline bridge: take the LocateAnything boxes cached on
Kaggle, run SAM2 + the full metric suite locally (MPS/CPU), no GPU needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .log import configure_logging, get_logger

log = get_logger(__name__)


def _load_samples(dataset: str, fold: str, ids: list[str] | None, n: int | None, seed: int):
    from .data import get_dataset

    ds = get_dataset(dataset, fold=fold)
    if ids is not None:
        idxs = [int(s.split("-")[-1]) for s in ids]
    elif n is not None:
        idxs = ds.stratified_indices(n, seed=seed)
    else:
        idxs = list(range(len(ds)))
    return [ds.decode(i) for i in idxs], ds.spec


def _write_summary(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"wrote {out_dir / 'summary.json'}")


def cmd_sample(args) -> None:
    samples, _ = _load_samples(args.dataset, args.fold, None, args.n, args.seed)
    ids = [s.sample_id for s in samples]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": args.dataset, "fold": args.fold, "n": len(ids), "seed": args.seed,
         "sample_ids": ids}, indent=2))
    log.info(f"selected {len(ids)} patches -> {out}")


def cmd_oracle(args) -> None:
    from .detect.oracle import OracleBoxDetector
    from .pipeline.run import run_condition
    from .segment.sam2 import Sam2Masker

    samples, spec = _load_samples(args.dataset, args.fold, None, args.n, args.seed)
    masker = Sam2Masker(args.sam2_model, device=args.device)
    out = run_condition(OracleBoxDetector(class_aware=True, spec=spec), masker, samples,
                        spec=spec, iou_thresh=args.iou)
    _write_summary(Path(args.out), out["summary"])
    _print_headline("oracle_box", out["summary"])


def cmd_eval_cache(args) -> None:
    from .pipeline.cache import load_detections
    from .pipeline.run import run_condition
    from .segment.sam2 import Sam2Masker

    precomputed = load_detections(args.detections)
    samples, spec = _load_samples(args.dataset, args.fold, list(precomputed), None, args.seed)
    masker = Sam2Masker(args.sam2_model, device=args.device)
    out = run_condition(None, masker, samples, spec=spec, precomputed=precomputed,
                        iou_thresh=args.iou)
    _write_summary(Path(args.out), out["summary"])
    _print_headline(Path(args.detections).stem, out["summary"])


def cmd_report(args) -> None:
    summary = json.loads(Path(args.summary).read_text())
    _print_headline(Path(args.summary).parent.name, summary)


def _print_headline(name: str, s: dict) -> None:
    d = s.get("detection_pooled", {})
    log.info(f"\n[{name}]  n={s.get('n_patches')}")
    log.info(f"  binary PQ={s.get('binary_pq', 0):.3f}  SQ={s.get('sq', 0):.3f}  "
          f"DQ={s.get('dq', 0):.3f}  AJI={s.get('aji', 0):.3f}  Dice={s.get('dice', 0):.3f}")
    log.info(f"  detection  P={d.get('precision', 0):.3f}  R={d.get('recall', 0):.3f}  "
          f"F1={d.get('f1', 0):.3f}")
    if "mpq" in s:
        log.info(f"  mPQ={s['mpq']:.3f}  per-class={ {k: round(v, 3) for k, v in s.get('per_class_pq', {}).items()} }")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlm-medseg", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="pannuke")
    common.add_argument("--fold", default="fold1")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--device", default="auto")
    common.add_argument("--sam2-model", default="facebook/sam2-hiera-large")
    common.add_argument("--iou", type=float, default=0.5)

    ps = sub.add_parser("sample", parents=[common], help="write a stratified patch sample")
    ps.add_argument("--n", type=int, required=True)
    ps.add_argument("--out", required=True)
    ps.set_defaults(func=cmd_sample)

    po = sub.add_parser("oracle", parents=[common], help="oracle GT-box -> SAM2 (ceiling)")
    po.add_argument("--n", type=int, default=30)
    po.add_argument("--out", required=True)
    po.set_defaults(func=cmd_oracle)

    pe = sub.add_parser("eval-cache", parents=[common],
                        help="SAM2 + eval on cached LocateAnything detections")
    pe.add_argument("--detections", required=True)
    pe.add_argument("--out", required=True)
    pe.set_defaults(func=cmd_eval_cache)

    pr = sub.add_parser("report", help="print a summary.json")
    pr.add_argument("summary")
    pr.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
