#!/usr/bin/env python
"""Curate a nucleus-crop feature dataset for the classifier.

Decodes PanNuke patches, crops every nucleus, embeds with a frozen encoder
(UNI2-h), and assigns stratified + patch-grouped splits (no patch leakage).
Caches features to an ``.npz`` (+ ``.json`` meta) for fast, reproducible training.

Use a *different* fold for curation than you evaluate on (default fold2) to keep
the classifier honest.

Examples:
    python scripts/curate_classifier_data.py --fold fold2 --n 150 --max-per-class 4000
    python scripts/curate_classifier_data.py --fold fold2            # all patches
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from vlm_medseg.log import configure_logging, get_logger

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO = Path(__file__).resolve().parent.parent
log = get_logger("curate")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="pannuke")
    p.add_argument("--fold", default="fold2")
    p.add_argument("--n", type=int, default=None, help="patches to sample (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder", default="uni2h",
                   help="feature encoder: uni2h (default, pathology ViT-H) | "
                        "dinov3 (= dinov3_vitl16) | dinov3_vitb16 | dinov3_vits16")
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--min-size", type=int, default=48)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--split-mode", choices=["stratified", "group"], default="stratified",
                   help="stratified: class-balanced per split (rare classes present); "
                        "group: no patch leakage but rare classes may be absent from val/test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    configure_logging()

    from vlm_medseg.classify.dataset import curate
    from vlm_medseg.classify.encoders import build_encoder
    from vlm_medseg.data import get_dataset

    ds = get_dataset(args.dataset, fold=args.fold)
    encoder = build_encoder(args.encoder, device=args.device, batch_size=args.batch_size)
    log.info(f"curating {args.dataset}/{args.fold} with {args.encoder} (dim={encoder.dim}) ...")

    data = curate(
        ds, encoder, n_patches=args.n, seed=args.seed,
        margin=args.margin, min_size=args.min_size,
        max_per_class=args.max_per_class, n_splits=args.n_splits,
        split_mode=args.split_mode,
    )

    out = Path(args.out) if args.out else REPO / "data" / "classifier" / f"{args.encoder}_{args.fold}.npz"
    data.save(out)

    names = data.class_names
    log.info(f"\n{data.features.shape[0]} nuclei, dim={data.features.shape[1]} -> {out}")
    for split in ("train", "val", "test"):
        m = data.mask(split)
        cc = Counter(int(c) for c in data.class_id[m])
        log.info(f"  {split:5s} n={int(m.sum()):5d}  " +
              " ".join(f"{names[c]}={cc.get(c,0)}" for c in range(len(names))))


if __name__ == "__main__":
    main()
