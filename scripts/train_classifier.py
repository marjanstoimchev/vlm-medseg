#!/usr/bin/env python
"""Train the nucleus-classifier head on curated features.

Loads a curated ``.npz`` (from curate_classifier_data.py), fits a head (default
MLP; see ``--head``) on the train split, reports val/test metrics, and saves a
loadable probe to ``models/classifiers/<encoder>_<head>.joblib`` for
``run_method.py --classifier`` / the pipeline hook.

Example:
    python scripts/train_classifier.py --features data/classifier/uni2h_fold2.npz            # MLP
    python scripts/train_classifier.py --features data/classifier/uni2h_fold2.npz --head svm
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vlm_medseg.log import configure_logging, get_logger

REPO = Path(__file__).resolve().parent.parent
log = get_logger("train_classifier")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="curated .npz from curate_classifier_data.py")
    p.add_argument("--head", default="mlp", choices=["linear", "svm", "mlp", "knn"],
                   help="classifier head (default mlp)")
    p.add_argument("--C", type=float, default=1.0, help="regularization for linear/svm heads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: models/classifiers/<encoder>_<head>.joblib")
    p.add_argument("--quiet", action="store_true", help="suppress per-stage logs")
    args = p.parse_args()
    configure_logging()

    from vlm_medseg.classify.dataset import CropDataset
    from vlm_medseg.classify.linear_probe import save_probe
    from vlm_medseg.classify.train import train_probe

    verbose = not args.quiet
    t0 = time.time()

    if verbose:
        log.info(f"[1/3] loading features: {args.features}")
    data = CropDataset.load(args.features)
    if verbose:
        log.info(f"      {data.features.shape[0]} nuclei · dim={data.features.shape[1]} · "
              f"encoder={data.encoder} · margin={data.margin} min_size={data.min_size}")
        log.info(f"      splits: {dict(Counter(data.split.tolist()))}")

    if verbose:
        log.info(f"[2/3] training head='{args.head}' ...")
    clf, metrics = train_probe(data, head=args.head, C=args.C, seed=args.seed, verbose=verbose)

    out = Path(args.out) if args.out else REPO / "models" / "classifiers" / f"{data.encoder}_{args.head}.joblib"
    if verbose:
        log.info(f"[3/3] saving probe -> {out}")
    save_probe(out, clf, data.class_names, encoder=data.encoder,
               margin=data.margin, min_size=data.min_size, metrics=metrics)

    log.info(f"\ndone in {time.time() - t0:.1f}s · {data.encoder} + {args.head} · "
          f"trained on {metrics['n_train']} nuclei · saved -> {out}\n")
    for split in ("val", "test"):
        if split not in metrics:
            continue
        m = metrics[split]
        log.info(f"[{split}] n={m['n']}  accuracy={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")
        log.info("   per-class F1: " + "  ".join(f"{k}={v:.2f}" for k, v in m["per_class_f1"].items()))
        log.info("   confusion[gt x pred]:\n" + str(np.array(m["confusion"])))


if __name__ == "__main__":
    main()
