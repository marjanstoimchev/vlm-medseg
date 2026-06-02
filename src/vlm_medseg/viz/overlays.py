"""Qualitative overlays and summary plots (needs the ``viz`` extra)."""

from __future__ import annotations

import numpy as np

from ..constants import GENERIC_COLOR
from ..data.base import DatasetSpec
from ..data.pannuke import PANNUKE_SPEC
from ..types import Detection, InstancePrediction, Sample
from .style import BOUNDARY_COLOR, METRIC_LABELS, condition_color


def _color_for(class_id: int | None, spec: DatasetSpec) -> tuple[int, int, int]:
    if class_id is None:
        return GENERIC_COLOR
    return spec.class_colors.get(class_id, GENERIC_COLOR)


def class_legend_handles(spec: DatasetSpec = PANNUKE_SPEC, *, include_generic: bool = False):
    """Matplotlib ``Patch`` handles mapping each class to its overlay fill colour."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=tuple(v / 255 for v in spec.class_colors.get(cid, GENERIC_COLOR)),
              edgecolor="none", label=name)
        for cid, name in enumerate(spec.class_names)
    ]
    if include_generic:
        handles.append(Patch(facecolor=tuple(v / 255 for v in GENERIC_COLOR),
                             edgecolor="none", label="nucleus"))
    return handles


def overlay_instances(
    image: np.ndarray,
    instances: list[InstancePrediction],
    *,
    spec: DatasetSpec = PANNUKE_SPEC,
    alpha: float = 0.45,
    draw_contours: bool = True,
    contour_color: tuple[int, int, int] = BOUNDARY_COLOR,
) -> np.ndarray:
    """Blend class-coloured instance fills onto the patch with a contrasting boundary.

    The translucent fill encodes the class; the boundary is drawn in a single
    contrasting colour (yellow by default) so individual nuclei stay legible over
    both the H&E background and the fill. Returns RGB uint8.
    """
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
            cv2.drawContours(out, contours, -1, contour_color, 1)
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

    fig, axes = plt.subplots(1, len(panels), figsize=(figsize_scale * len(panels), figsize_scale + 0.5))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.axis("off")
    _add_class_legend(fig, spec)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def _add_class_legend(fig, spec: DatasetSpec) -> None:
    """Figure-level legend: fill colour per class, plus the boundary convention."""
    handles = class_legend_handles(spec)
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, 0.0), handlelength=1.1, columnspacing=1.4,
        title="nucleus class — fill colour   ·   yellow outline = segmentation boundary",
        title_fontsize=9, fontsize=9,
    )


def gallery(
    samples: list[Sample],
    pred_by_condition: dict[str, list[list[InstancePrediction]]],
    *,
    spec: DatasetSpec = PANNUKE_SPEC,
    max_rows: int = 6,
):
    """One row per patch comparing GT against each condition (contact-sheet style)."""
    import matplotlib.pyplot as plt

    rows = min(max_rows, len(samples))
    cond_names = list(pred_by_condition)
    cols = 2 + len(cond_names)
    col_titles = ["H&E", "Ground truth", *cond_names]
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.7 * rows + 0.7), squeeze=False)
    for r in range(rows):
        s = samples[r]
        imgs = [s.image, overlay_gt(s, spec=spec)]
        imgs += [overlay_instances(s.image, pred_by_condition[cn][r], spec=spec) for cn in cond_names]
        for c, img in enumerate(imgs):
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight="semibold")
        axes[r][0].set_ylabel(spec.group_name(s.group), fontsize=9, rotation=90, labelpad=4)
    _add_class_legend(fig, spec)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 2.5, 4.3))
    for i, c in enumerate(conds):
        vals = [float(summaries[c].get(m, 0.0)) for m in metrics]
        is_oracle = str(c).startswith("oracle")
        bars = ax.bar(
            x + i * width, vals, width,
            label=(f"{c} (ceiling)" if is_oracle else c),
            color=condition_color(c, i), hatch="//" if is_oracle else None,
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Mask and detection quality by method")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(len(conds), 4), loc="upper center", bbox_to_anchor=(0.5, -0.10))
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
