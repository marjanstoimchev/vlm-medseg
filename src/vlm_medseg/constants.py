"""Dataset-level constants for PanNuke and the LocateAnything prompt mapping.

Schema is the authoritative one from the RationAI/PanNuke `datasets` feature
definitions (verified against the datasets-server `info` endpoint):

  categories (per nucleus instance): 0..4
      0 Neoplastic, 1 Inflammatory, 2 Connective, 3 Dead, 4 Epithelial
  tissue (per patch): 0..18  (19 organ types)

All patches are 256x256 H&E. There is no explicit "background" class in the
per-instance `categories` field; background is simply absence of an instance.
"""

from __future__ import annotations

DATASET_ID = "RationAI/PanNuke"
FOLDS = ("fold1", "fold2", "fold3")
PATCH_SIZE = 256

# ── Nuclei classes (per-instance `categories` field) ──────────────────────
NUCLEI_CLASSES: tuple[str, ...] = (
    "Neoplastic",      # 0
    "Inflammatory",    # 1
    "Connective",      # 2
    "Dead",            # 3
    "Epithelial",      # 4
)
NUM_NUCLEI_CLASSES = len(NUCLEI_CLASSES)

# ── Tissue types (per-patch `tissue` ClassLabel) ──────────────────────────
TISSUE_TYPES: tuple[str, ...] = (
    "Adrenal Gland", "Bile Duct", "Bladder", "Breast", "Cervix", "Colon",
    "Esophagus", "Head & Neck", "Kidney", "Liver", "Lung", "Ovarian",
    "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
)
NUM_TISSUE_TYPES = len(TISSUE_TYPES)

# ── Visualization palette (RGB), mirrors the conventional PanNuke colours ──
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (228, 26, 28),    # Neoplastic   - red
    1: (55, 126, 184),   # Inflammatory - blue
    2: (77, 175, 74),    # Connective   - green
    3: (255, 215, 0),    # Dead         - gold
    4: (152, 78, 163),   # Epithelial   - purple
}
GENERIC_COLOR = (255, 127, 0)  # orange, for class-agnostic ("nucleus") runs

# ── LocateAnything prompt mapping ─────────────────────────────────────────
# The grounding VLM takes a natural-language category list. We expose two
# strategies: a single class-agnostic phrase (binary nucleus detection) and a
# per-class phrase set (multi-class). Phrases are deliberately descriptive
# because the model was trained on natural images, not histopathology.
GENERIC_PROMPT = "cell nucleus"

CLASS_PROMPTS: dict[int, str] = {
    0: "neoplastic tumor cell nucleus",
    1: "inflammatory immune cell nucleus",
    2: "connective or soft tissue cell nucleus",
    3: "dead or necrotic cell nucleus",
    4: "epithelial cell nucleus",
}

# SAM3 concept prompts — short noun phrases (SAM3's preferred form), targeting
# the nucleus rather than the whole cell.
GENERIC_CONCEPT = "cell nucleus"
CONCEPT_PROMPTS: dict[int, str] = {
    0: "tumor cell nucleus",
    1: "immune cell nucleus",
    2: "stromal cell nucleus",
    3: "necrotic cell nucleus",
    4: "epithelial cell nucleus",
}
