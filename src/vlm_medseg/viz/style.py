"""Shared, publication-oriented figure styling and palettes."""

from __future__ import annotations

# Colour-blind-friendly qualitative palette for method/condition series.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
# The oracle is a reference ceiling, not a competitor -> render it in neutral grey.
ORACLE_COLOR = "#9AA0A6"
# Yellow nucleus boundary: reads clearly over both the H&E and the class-coloured fill.
BOUNDARY_COLOR = (255, 255, 0)

# Human-readable axis labels for the internal metric keys.
METRIC_LABELS = {
    "binary_pq": "PQ", "mpq": "mPQ", "aji": "AJI", "dice": "Dice",
    "matched_iou": "matched IoU", "sq": "SQ", "dq": "DQ",
}


def set_paper_style() -> None:
    """Apply clean matplotlib defaults suited to figures in a write-up (idempotent)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlepad": 7,
        "axes.labelsize": 11, "axes.labelcolor": "#1a1a1a",
        "axes.edgecolor": "#5a5a5a", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#dcdcdc", "grid.linewidth": 0.6,
        "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a",
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10,
        "figure.titlesize": 13, "figure.titleweight": "semibold",
        "image.interpolation": "nearest",
    })


def condition_color(name: str, index: int) -> str:
    """Stable series colour; the oracle is rendered neutrally as the ceiling."""
    if str(name).startswith("oracle"):
        return ORACLE_COLOR
    return PALETTE[index % len(PALETTE)]
