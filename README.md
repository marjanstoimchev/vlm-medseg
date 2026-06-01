# vlm-medseg

A benchmark for **how vision models perform at medical instance segmentation** —
specifically nuclei on **PanNuke**. It runs a family of detectors (grounding
VLMs, open-vocab detectors, an oracle) into **SAM2** for masks, plus **SAM3**
text-prompted concept segmentation, then scores everything with a decomposed
metric suite (PQ / AJI / Dice / detection F1) and a post-hoc nucleus classifier.

Everything is modular and dataset-agnostic: detectors, segmenters, encoders, and
classifier heads are pluggable, and PanNuke is the reference dataset.

## The idea: decompose, then compare

A VLM pipeline must **find** an object, **name** it, and **segment** it. A single
score hides which stage failed, so the framework separates them:

- **Detection** (recall / counting): can the model localize each nucleus?
- **Segmentation** (matched-IoU / SQ): given a box, can SAM2 mask the nucleus?
- **Classification** (confusion / mPQ): is the nucleus subtype right?

Every run is read against an **oracle** (ground-truth boxes → SAM2) that fixes
the SAM2 ceiling, so any shortfall is attributed to the right stage.

### Conditions (the detector ladder)

| condition | localizer → masker | role |
|---|---|---|
| `oracle_box` | GT boxes → SAM2 | SAM2 ceiling (perfect detection) |
| `gdino_box` | Grounding-DINO → SAM2 | open-vocab baseline |
| `owlv2_box` | OWLv2 → SAM2 | open-vocab baseline (localizes small objects) |
| `qwen_box` | Qwen2.5-VL → SAM2 | stock VLM grounding baseline |
| `la_box` | LocateAnything-3B → SAM2 | fine-tuned grounding VLM (Kaggle GPU) |
| `sam3_text` | SAM3 text concept | detector-free, end-to-end |

Any box detector can be wrapped with **SAHI** (tile → detect → merge) to rescue
it on dense tiny nuclei, and any condition's labels can be replaced by the
**post-hoc classifier** (frozen pathology encoder + trained head).

## Layout

```
src/vlm_medseg/
  data/        DatasetSpec + BaseDataset + registry; PanNuke reference impl
  detect/      oracle, grounding_dino, owlv2, qwen_vl, locate_anything; sahi (tiling wrapper)
  segment/     sam2 (box/point masker), sam3 (text concept, end to end)
  classify/    crops, encoders (uni2h | dinov3), heads (linear|svm|mlp|knn), dataset curation, train
  pipeline/    run_condition / run_segmenter_condition; classifier hook; detection cache
  eval/        matching, PQ/SQ/DQ + mPQ, AJI/Dice, detection, classification, report
  viz/         class-coloured overlays + summary plots
  prompts.py   per-model prompt templates + the PROMPTS registry
  cli.py
scripts/       run_method.py (driver) · curate_classifier_data.py · train_classifier.py
               make_notebook.py · push_to_kaggle.sh
notebooks/     locate_anything_pannuke_kaggle (Kaggle LA pipeline)
               explore_embeddings · compare_encoders (classifier diagnostics)
configs/ · tests/
data/ · models/classifiers/ · results/   (gitignored artifacts)
```

## Install

```bash
pip install -e ".[data,sam2,viz,classify,dev]"     # local dev / eval
```
LocateAnything runs on Kaggle (CUDA); see `requirements-kaggle.txt` and the
notebook. `torch`/`transformers` are extras so a bare install stays light and
Kaggle's pre-built CUDA torch isn't clobbered.

## Run it (`scripts/run_method.py`)

Each run writes `results/<method>/`: `summary.json`, per-patch `predictions.jsonl`
+ `patch_metrics.jsonl`, `config.json`, and `overlays/` (written **as the run
progresses** — `H&E | GT | prediction`, or `before | after` with a classifier).

```bash
# SAM2 ceiling (fast, local):
python scripts/run_method.py oracle_box  --n 30

# open-vocab / VLM detectors -> SAM2 (MPS); add --sahi to tile dense patches:
python scripts/run_method.py owlv2_box   --n 8 --device mps
python scripts/run_method.py gdino_box   --n 8 --sahi --tile 128
python scripts/run_method.py qwen_box    --n 8 --device mps        # slow; upscales to 896 by default

# SAM3 concept segmentation (end to end), with post-hoc class correction:
python scripts/run_method.py sam3_text   --n 8 --device mps --threshold 0.2 --classifier uni2h_mlp
```
Flags: `--n --fold --device --model --iou --threshold(sam3) --input-short-size --sahi/--tile/--overlap --classifier --no-class-aware`.
`la_box` runs only on a CUDA GPU (Kaggle notebook).

## Post-hoc nucleus classifier

SAM3/VLM masks can be great while their *labels* are not. Decouple it: segment
class-agnostically, then classify each mask from its pixels with a frozen
pathology encoder + a trained head.

```bash
# 1) curate features (UNI2-h default; --encoder dinov3 also available)
python scripts/curate_classifier_data.py --fold fold2 --n 80 --device mps --max-per-class 6000
# 2) train a head (default MLP) -> models/classifiers/<encoder>_<head>.joblib
python scripts/train_classifier.py --features data/classifier/uni2h_fold2.npz
# 3) use it anywhere: --classifier uni2h_mlp  (bare name resolves under models/classifiers/)
```
Curation uses sklearn `StratifiedKFold` (class-balanced) splits; the honest test
is a different fold run through the pipeline. Explore/compare the encoders and
heads in `notebooks/explore_embeddings.ipynb` and `compare_encoders.ipynb`.

## Metrics

Per patch and aggregated, stratified by class and tissue: **PQ / SQ / DQ**,
binary and per-class **mPQ**, **AJI**, **Dice**, **matched-IoU** (isolates SAM2),
detection **P/R/F1** + **counting error**, and a class **confusion** matrix.

## Findings so far (small-n probes, MPS)

- **`oracle_box` PQ ≈ 0.81** — SAM2-large's ceiling on nuclei given perfect boxes.
- **`gdino_box` / `qwen_box` ≈ 0** — stock open-vocab/VLM models return whole-image
  boxes (Qwen even at 896 px); they don't densely localize nuclei.
- **`owlv2_box` localizes** (nucleus-sized boxes → real PQ); **SAHI** lifts GDINO from
  PQ 0 → ~0.25 with matched-IoU ~0.83.
- **`sam3_text`** gives the best masks (matched-IoU ~0.84) but collapses on class —
  fixed by the classifier (**UNI2-h + MLP: ~0.77 macro-F1**, all classes ≥ 0.70 at 9k nuclei).

Net: general VLMs aren't dense nuclei detectors yet; the working recipes are
**SAM3 + a pathology classifier** locally and **LocateAnything → SAM2** on a GPU.

## Extending

- **Dataset:** subclass `BaseDataset` + a `DatasetSpec`, `register_dataset("name", Cls)` — works everywhere (`--dataset name`).
- **Detector:** a class with `detect(sample) -> [Detection]` + `.name`/`.spec`; add a `build_detector` branch.
- **Encoder / head:** add to `classify/encoders.py` `_ENCODERS` or `classify/heads.py` `build_head`.

## Models

SAM2 (`facebook/sam2-hiera-large`), SAM3 (`facebook/sam3`), LocateAnything-3B
(`nvidia/LocateAnything-3B`), Grounding-DINO, OWLv2, Qwen2.5-VL-3B; classifier
encoders UNI2-h (`MahmoodLab/UNI2-h`) and DINOv3 (`facebook/dinov3-*`).

## License & credits

Code: MIT. Datasets/models keep their own licenses — **PanNuke** is CC-BY-NC-SA-4.0,
**LocateAnything-3B** is NVIDIA academic / non-profit research only, and UNI2-h is
gated. Built on SAM2/SAM3, LocateAnything, RationAI/PanNuke, UNI2-h, and DINOv3.
