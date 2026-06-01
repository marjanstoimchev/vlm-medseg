# vlm-medseg

**Benchmarking vision-language and open-vocabulary models for nuclei instance segmentation on PanNuke.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Lint](https://img.shields.io/badge/lint-ruff-261230)

`vlm-medseg` pairs object localizers — grounding VLMs (LocateAnything-3B,
Qwen2.5-VL), open-vocabulary detectors (Grounding-DINO, OWLv2), and a
ground-truth oracle — with **SAM2** for masks, adds **SAM3** text-prompted
concept segmentation, and scores everything with a decomposed metric suite
(PQ / AJI / Dice / detection F1). An optional post-hoc classifier (frozen
pathology encoder + linear/MLP head) recovers nucleus subtypes. The pipeline,
detectors, segmenters, encoders, and classifier heads are all pluggable, and the
data layer is dataset-agnostic (PanNuke is the reference dataset).

---

## Why

A nuclei pipeline must **find**, **name**, and **segment** each nucleus; a single
score hides which stage fails. `vlm-medseg` separates them and reads every method
against a **ground-truth oracle** that fixes SAM2's segmentation ceiling — so any
shortfall is attributable to detection, segmentation, or classification.

## Features

- **Six conditions** behind one interface: `oracle_box`, `gdino_box`, `owlv2_box`, `qwen_box`, `la_box`, `sam3_text`.
- **SAHI** tiling wrapper for any box detector (helps on dense, tiny nuclei).
- **Decomposed metrics:** PQ / SQ / DQ, per-class mPQ, AJI, Dice, matched-IoU, detection P/R/F1, counting error, class confusion — stratified by class and tissue.
- **Post-hoc nucleus classifier:** frozen encoder (UNI2-h or DINOv3) + trained head; before/after label overlays.
- **Dataset-agnostic core:** add a dataset via `DatasetSpec` + `BaseDataset`.
- **Reproducible runs:** every run writes `results/<method>/` (metrics, predictions, per-patch overlays); detection caching; Kaggle notebooks + local CLI.

## Installation

```bash
git clone https://github.com/marjanstoimchev/vlm-medseg.git
cd vlm-medseg
pip install -e ".[data,sam2,viz,classify,dev]"
```

`torch` / `transformers` live in extras so a bare install stays light and a host's
pre-built CUDA `torch` (e.g. Kaggle) isn't clobbered. LocateAnything-3B is
CUDA-only — run it via the Kaggle notebook (see [`requirements-kaggle.txt`](requirements-kaggle.txt)).

## Quick start

```bash
# SAM2 ceiling on 30 PanNuke patches (local, fast)
python scripts/run_method.py oracle_box --n 30

# Open-vocabulary detector -> SAM2, with tiling
python scripts/run_method.py owlv2_box --n 8 --sahi

# SAM3 text-concept segmentation + post-hoc class correction
python scripts/run_method.py sam3_text --n 8 --threshold 0.2 --classifier uni2h_mlp
```

Each run writes `results/<method>/`: `summary.json`, per-patch `predictions.jsonl`
and `patch_metrics.jsonl`, `config.json`, and `overlays/` (`H&E | GT | prediction`,
or `before | after` with a classifier).

## Usage

### Conditions

| condition | localizer → masker | family |
|---|---|---|
| `oracle_box` | ground-truth boxes → SAM2 | upper bound |
| `gdino_box` | Grounding-DINO → SAM2 | open-vocabulary detection |
| `owlv2_box` | OWLv2 → SAM2 | open-vocabulary detection |
| `qwen_box` | Qwen2.5-VL → SAM2 | grounding VLM (stock) |
| `la_box` | LocateAnything-3B → SAM2 | grounding VLM (fine-tuned, CUDA) |
| `sam3_text` | SAM3 text concept | concept segmentation (end to end) |

### `run_method.py` arguments

| argument | default | description |
|---|---|---|
| `method` | — | one of the conditions above (positional) |
| `--dataset` | `pannuke` | registered dataset name |
| `--fold` | `fold1` | dataset split |
| `--n` | `20` | number of patches (stratified across tissues) |
| `--seed` | `0` | sampling seed |
| `--device` | `auto` | `auto` / `mps` / `cuda` / `cpu` |
| `--model` | per method | override the model id |
| `--sam2-model` | `facebook/sam2-hiera-large` | SAM2 checkpoint (box methods) |
| `--iou` | `0.5` | IoU threshold for matching / PQ |
| `--threshold` | `0.3` | SAM3 detection threshold |
| `--input-short-size` | per method | upscale short side before the VLM (`qwen` 896, `la` 1024) |
| `--sahi` | off | tile the patch before the detector and NMS-merge |
| `--tile` / `--overlap` | `128` / `0.25` | SAHI tile size / overlap |
| `--classifier` | none | probe path or bare name → `models/classifiers/<name>.joblib` |
| `--no-class-aware` | off | class-agnostic (binary) prompts |
| `--max-overlays` / `--gallery-rows` | `12` / `6` | overlay rendering limits |
| `--out` | `results/<method>` | output directory |

### Post-hoc nucleus classifier

Segment class-agnostically, then classify each mask from its pixels with a frozen
pathology encoder + a trained head:

```bash
# 1) curate features (UNI2-h default; --encoder dinov3 also available)
python scripts/curate_classifier_data.py --fold fold2 --n 80 --max-per-class 6000
# 2) train a head (default MLP) -> models/classifiers/<encoder>_<head>.joblib
python scripts/train_classifier.py --features data/classifier/uni2h_fold2.npz
# 3) use it: --classifier uni2h_mlp
```

Curation uses class-balanced `StratifiedKFold` splits; the held-out fold run
through the pipeline is the honest test. The CLI (`vlm-medseg`) also exposes
`sample`, `oracle`, `eval-cache`, and `report` subcommands.

### Notebooks ([`notebooks/`](notebooks/))

| notebook | purpose |
|---|---|
| `locate_anything_pannuke_kaggle` | run LocateAnything-3B → SAM2 on a Kaggle GPU |
| `method_comparison_kaggle` | side-by-side comparison of LA / OWLv2 / SAM3 vs oracle on N random patches |
| `explore_embeddings` | classifier feature diagnostics (projection, per-tissue, retrieval) |
| `compare_encoders` | encoder × head bench (UNI2-h vs DINOv3; linear/SVM/MLP) |

Private-repo notebooks `pip install` the package from GitHub; on Kaggle add a
`GITHUB_TOKEN` secret (or attach the repo as a dataset).

## Results

Indicative numbers on **PanNuke fold1, _n_ = 8 patches, SAM2-hiera-large** (a small
probe for sanity, not a leaderboard — reproduce/scale with `run_method.py` or the
notebooks). LocateAnything (`la_box`) requires a GPU and is reported from the
Kaggle notebook.

| method | binary PQ | matched-IoU | detection recall |
|---|---|---|---|
| `oracle_box` (ceiling) | 0.81 | 0.82 | 0.99 |
| `owlv2_box` | localizes (real PQ) | ~0.89 | low |
| `gdino_box` / `qwen_box` | ≈ 0 (whole-image boxes) | — | ≈ 0 |
| `gdino_box` + SAHI | ~0.25 | ~0.83 | improved |
| `sam3_text` | best masks | ~0.84 | moderate |

Post-hoc classifier (PanNuke fold2 held-out, ~9k nuclei): **UNI2-h + MLP — accuracy 0.78, macro-F1 0.77** (per-class F1 0.70–0.84).

**Takeaway.** Stock open-vocabulary/VLM detectors do not densely localize nuclei
zero-shot; SAHI and resolution help, but the working recipes are **SAM3 + a
pathology classifier** locally and **LocateAnything → SAM2** on a GPU.

## Project layout

```
src/vlm_medseg/
  data/        DatasetSpec + BaseDataset + registry; PanNuke reference impl
  detect/      oracle, grounding_dino, owlv2, qwen_vl, locate_anything; sahi (tiling)
  segment/     sam2 (box masker), sam3 (text concept, end to end)
  classify/    crops, encoders, heads, dataset curation, train
  pipeline/    run_condition / run_segmenter_condition; classifier hook; detection cache
  eval/        matching, PQ/SQ/DQ + mPQ, AJI/Dice, detection, classification, report
  viz/         class-coloured overlays + summary plots
  prompts.py · log.py · cli.py
scripts/   run_method · curate_classifier_data · train_classifier · make_notebook · push_to_kaggle
notebooks/ · configs/ · tests/
```

## Extending

- **Dataset:** subclass `BaseDataset` + define a `DatasetSpec`, then `register_dataset("name", Cls)` — usable everywhere via `--dataset name`.
- **Detector:** a class with `detect(sample) -> list[Detection]` plus `.name` / `.spec`; add a branch in `build_detector`.
- **Encoder / head:** register in `classify/encoders.py` or `classify/heads.py`.

## References

1. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* Meta AI, 2024. https://github.com/facebookresearch/sam2
2. Meta AI. *SAM 3: Segment Anything with Concepts.* 2025. https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
3. NVIDIA. *LocateAnything.* https://research.nvidia.com/labs/lpr/locate-anything/ · model: https://huggingface.co/nvidia/LocateAnything-3B
4. Liu et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* 2023.
5. Minderer, Gritsenko, Houlsby. *Scaling Open-Vocabulary Object Detection (OWLv2).* NeurIPS 2023.
6. Qwen Team, Alibaba. *Qwen2.5-VL.* 2025. https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
7. Chen et al. *Towards a general-purpose foundation model for computational pathology (UNI).* Nature Medicine, 2024. · https://huggingface.co/MahmoodLab/UNI2-h
8. Siméoni et al. *DINOv3.* Meta AI, 2025.
9. Gamper et al. *PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification.* 2019. · https://huggingface.co/datasets/RationAI/PanNuke
10. Kirillov et al. *Panoptic Segmentation (PQ).* CVPR 2019. · Graham et al. *HoVer-Net.* Medical Image Analysis, 2019 (nuclei PQ).
11. Kumar et al. *A Dataset and a Technique for Generalized Nuclear Segmentation (AJI).* IEEE TMI, 2017.
12. Akyon et al. *Slicing Aided Hyper Inference (SAHI).* ICIP 2022. https://github.com/obss/sahi

## Citation

```bibtex
@software{stoimchev_vlm_medseg,
  author  = {Stoimchev, Marjan},
  title   = {vlm-medseg: benchmarking vision-language models for nuclei instance segmentation},
  year    = {2026},
  url     = {https://github.com/marjanstoimchev/vlm-medseg}
}
```

## License

[MIT](LICENSE) © 2026 Marjan Stoimchev.

Datasets and models retain their own licenses — **PanNuke** is CC-BY-NC-SA-4.0,
**LocateAnything-3B** is NVIDIA academic / non-profit research only, and **UNI2-h**
is gated. Review each before use.
