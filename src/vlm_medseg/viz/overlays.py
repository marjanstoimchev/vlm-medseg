"""Qualitative overlays and summary plots (needs the ``viz`` extra)."""

from __future__ import annotations

import numpy as np

from ..constants import GENERIC_COLOR
from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..types import Detection, InstancePrediction, Sample


def _color_for(class_id: int | None, spec: DatasetSpec) -> tuple[int, int, int]:
    if class_id is None:
        return GENERIC_COLOR
    return spec.class_colors.get(class_id, GENERIC_COLOR)


def overlay_instances(
    image: np.ndarray,
    instances: list[InstancePrediction],
    *,
    spec: DatasetSpec = PANNUKE_SPEC,
    alpha: float = 0.45,
    draw_contours: bool = True,
) -> np.ndarray:
    """Blend class-colored instance masks onto the patch (returns RGB uint8)."""
    import cv2

    fill = image.copy()
    for ins in instances:
        fill[ins.mask] = _color_for(ins.class_id, spec)
    out = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)
    if draw_contours:
        for ins in instances:
            contours, _ = cv2.findContours(
                ins.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, _color_for(ins.class_id, spec), 1)
    return out


def overlay_gt(sample: Sample, *, spec: DatasetSpec = PANNUKE_SPEC, alpha: float = 0.45) -> np.ndarray:
    instances = [
        InstancePrediction(mask=(sample.inst_map == i), class_id=c)
        for i, c in sample.inst_classes.items()
    ]
    return overlay_instances(sample.image, instances, spec=spec, alpha=alpha)


def draw_boxes(image: np.ndarray, detections: list[Detection], *, spec: DatasetSpec = PANNUKE_SPEC) -> np.ndarray:
    import cv2

    out = image.copy()
    for d in detections:
        if d.box is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), _color_for(d.class_id, spec), 1)
    return out


def draw_boundary(
    image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int] = (255, 255, 0), thickness: int = 1
) -> np.ndarray:
    """Return a copy of ``image`` with the mask's external contour drawn (yellow)."""
    import cv2

    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def comparison_panel(
    sample: Sample,
    pred_by_condition: dict[str, list[InstancePrediction]],
    *,
    spec: DatasetSpec = PANNUKE_SPEC,
    figsize_scale: float = 3.2,
):
    """Row: [image | GT | each condition's predicted instances]."""
    import matplotlib.pyplot as plt

    panels = [("H&E", sample.image), ("Ground truth", overlay_gt(sample, spec=spec))]
    for name, inst in pred_by_condition.items():
        panels.append((name, overlay_instances(sample.image, inst, spec=spec)))

    fig, axes = plt.subplots(1, len(panels), figsize=(figsize_scale * len(panels), figsize_scale))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def gallery(
    samples: list[Sample],
    pred_by_condition: dict[str, list[list[InstancePrediction]]],
    *,
    spec: DatasetSpec = PANNUKE_SPEC,
    max_rows: int = 6,
):
    """One row per patch comparing GT against each condition."""
    import matplotlib.pyplot as plt

    rows = min(max_rows, len(samples))
    cond_names = list(pred_by_condition)
    cols = 2 + len(cond_names)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for r in range(rows):
        s = samples[r]
        cells = [("H&E", s.image), ("GT", overlay_gt(s, spec=spec))]
        for cn in cond_names:
            cells.append((cn, overlay_instances(s.image, pred_by_condition[cn][r], spec=spec)))
        for c, (title, img) in enumerate(cells):
            axes[r][c].imshow(img)
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(title, fontsize=10)
        axes[r][0].set_ylabel(spec.group_name(s.group), fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_bars(
    summaries: dict[str, dict],
    metrics: tuple[str, ...] = ("binary_pq", "aji", "dice", "matched_iou"),
):
    """Grouped bars comparing headline metrics across conditions."""
    import matplotlib.pyplot as plt

    conds = list(summaries)
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2, 4))
    for i, c in enumerate(conds):
        ax.bar(x + i * width, [summaries[c].get(m, 0.0) for m in metrics], width, label=c)
    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Instance-segmentation metrics by condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_per_group_pq(summary: dict, *, spec: DatasetSpec = PANNUKE_SPEC):
    """Bar of binary PQ per group (e.g. tissue) for one condition."""
    import matplotlib.pyplot as plt

    by_g = summary.get("by_group", {})
    names = list(by_g)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(names)), 3.5))
    ax.bar(names, [by_g[n]["binary_pq"] for n in names], color="#377eb8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("binary PQ")
    ax.set_title(f"PQ by {spec.group_label}")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    return fig


# Back-compat alias.
plot_per_tissue_pq = plot_per_group_pq


def plot_confusion(summary: dict, *, spec: DatasetSpec = PANNUKE_SPEC):
    """Heatmap of the class confusion matrix (class-aware runs only)."""
    import matplotlib.pyplot as plt

    conf = np.array(summary.get("confusion_matrix", []))
    if conf.size == 0:
        return None
    names = spec.class_names
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Class confusion (matched instances)")
    thresh = conf.max() / 2 if conf.max() else 0
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if conf[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
