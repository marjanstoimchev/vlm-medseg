"""Classifier heads for the probe — one source of truth (repo + notebooks).

Each head is a fresh sklearn pipeline ``StandardScaler -> estimator``. ``mlp`` is
the repo default. Note: ``logreg``/``svm`` are class-balanced (good for the rare
Dead class); sklearn's ``mlp`` has no class weighting, so its rare-class recall
is a floor — a torch MLP with focal loss is the balanced upgrade if needed.
"""

from __future__ import annotations

DEFAULT_HEAD = "mlp"
HEAD_NAMES = ("linear", "svm", "mlp", "knn")


def build_head(name: str = DEFAULT_HEAD, *, seed: int = 0, C: float = 1.0):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "linear":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
    elif name == "svm":
        est = SVC(C=C, kernel="rbf", class_weight="balanced", random_state=seed)
    elif name == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, max_iter=500,
                            early_stopping=True, random_state=seed)
    elif name == "knn":
        est = KNeighborsClassifier(n_neighbors=15)
    else:
        raise KeyError(f"unknown head '{name}'; available: {HEAD_NAMES}")
    return make_pipeline(StandardScaler(), est)
