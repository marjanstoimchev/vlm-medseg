"""Train a probe on frozen nucleus features.

``train_probe`` fits any head from :mod:`.heads` (default MLP) and reports
val/test accuracy, macro-F1, per-class F1, and the confusion matrix. The head is
class-agnostic to the metrics — only ``build_head`` changes.
"""

from __future__ import annotations

from ..log import get_logger
from .dataset import CropDataset
from .heads import DEFAULT_HEAD, build_head

log = get_logger(__name__)


def train_probe(
    data: CropDataset, *, head: str = DEFAULT_HEAD, seed: int = 0, C: float = 1.0, verbose: bool = False
):
    """Fit ``head`` on the train split; return ``(pipeline, metrics)``."""
    import time

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    X, y, split = data.features, data.class_id, data.split
    nc = len(data.class_names)
    labels = list(range(nc))
    tr = split == "train"

    if verbose:
        counts = {data.class_names[c]: int((y[tr] == c).sum()) for c in labels}
        log.info(f"  fitting head='{head}' on {int(tr.sum())} x {X.shape[1]} features")
        log.info(f"  train class counts: {counts}")

    clf = build_head(head, seed=seed, C=C)
    t0 = time.time()
    clf.fit(X[tr], y[tr])
    if verbose:
        log.info(f"  fit done in {time.time() - t0:.1f}s")

    metrics: dict = {
        "head": head, "n_train": int(tr.sum()), "C": C,
        "train_class_counts": {data.class_names[c]: int((y[tr] == c).sum()) for c in labels},
    }
    for name in ("val", "test"):
        m = split == name
        if not m.any():
            continue
        pred = clf.predict(X[m])
        per_f1 = f1_score(y[m], pred, average=None, labels=labels, zero_division=0)
        metrics[name] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y[m], pred)),
            "macro_f1": float(f1_score(y[m], pred, average="macro", labels=labels, zero_division=0)),
            "per_class_f1": {data.class_names[c]: float(f) for c, f in zip(labels, per_f1)},
            "confusion": confusion_matrix(y[m], pred, labels=labels).tolist(),
        }
        if verbose:
            log.info(f"  [{name}] acc={metrics[name]['accuracy']:.3f} "
                  f"macroF1={metrics[name]['macro_f1']:.3f}")
    return clf, metrics


def train_linear_probe(data: CropDataset, *, C: float = 1.0, max_iter: int = 3000,
                       seed: int = 0, verbose: bool = False):
    """Back-compat wrapper: the class-balanced linear (logistic-regression) head."""
    return train_probe(data, head="linear", seed=seed, C=C, verbose=verbose)
