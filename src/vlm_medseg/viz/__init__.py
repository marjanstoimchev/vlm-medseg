"""Qualitative overlays and summary plots (needs the ``viz`` extra)."""

from .overlays import (
    class_legend_handles,
    comparison_panel,
    draw_boundary,
    draw_boxes,
    gallery,
    overlay_gt,
    overlay_instances,
    plot_confusion,
    plot_metric_bars,
    plot_per_group_pq,
    plot_per_tissue_pq,
)
from .style import PALETTE, condition_color, set_paper_style

__all__ = [
    "overlay_instances", "overlay_gt", "draw_boxes", "draw_boundary",
    "comparison_panel", "gallery", "class_legend_handles",
    "plot_metric_bars", "plot_per_group_pq", "plot_per_tissue_pq", "plot_confusion",
    "set_paper_style", "condition_color", "PALETTE",
]
