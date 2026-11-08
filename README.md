# LABELGAP: code and derived data

Research code for *Attacks Labeled as Benign: Systematic Label Omission in CIC-IDS2017 and CSE-CIC-IDS2018 and Its Consequences for Learning, Evaluation, and Auditing* by Guizheng Yang, Kejiu Tan, and Chengdong Luo.

This repository contains the alignment, model training, evaluation, and figure scripts; configuration files; the result tables behind the manuscript; and derived data for checking the principal comparisons without retraining. The release is a cleaned export of the research code. The original submission supplement and its historical Git log are preserved separately by the authors; this new repository starts its own history.

## Quick check

The derived-data check does not need the source datasets or a GPU. From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 code/verify_from_predictions.py
```

The check recomputes the main model comparison, reversal counts, and partial label repair results from `derived_data/test_predictions_*.csv.gz`.

## Source data

The source datasets are **not** included. Obtain them from their providers:

| Source | Link |
|---|---|
| Original CIC-IDS2017 (GeneratedLabelledFlows CSVs) | https://www.unb.ca/cic/datasets/ids-2017.html |
| Original CSE-CIC-IDS2018 (TrafficForML CSVs) | https://www.unb.ca/cic/datasets/ids-2018.html |
| Liu et al. corrected CIC-IDS2017 and CSE-CIC-IDS2018 flows | https://intrusion-detection.distrinet-research.be/CNS2022/index.html |
| Engelen et al. corrected CIC-IDS2017 flows | Required by the three-version CIC-IDS2017 alignment scripts; obtain the release cited in the manuscript. |

The CIC-IDS2017 pipeline uses the Engelen files for the three-way alignment and the natural-noise mask. Its Liu-label comparison depends on that generated mask. The source file versions and directory layout must match the configuration and conversion scripts.

## Repository layout

| Path | Purpose |
|---|---|
| `code/alignment_cicids2017/` | Convert original and corrected CIC-IDS2017 CSVs to Parquet, align flows, and build the noise mask |
| `code/scripts/directionA_2018_alignment.py` | Align original and corrected CSE-CIC-IDS2018 flows |
| `code/scripts/directionA*_prepare.py` | Prepare paired learning data and labels |
| `code/scripts/train_baselines.py`, `code/src/` | Train and evaluate the model families and noise-robust baselines |
| `code/scripts/directionA*_analyze.py`, `code/scripts/revision_r1_*.py`, `code/scripts/transfer_rescore_20261107.py` | Re-score predictions and produce audit analyses |
| `code/scripts/labelgap_figures.py`, `code/figures/` | Generate article figures and training-cost summaries |
| `code/configs/` | Parameters and analysis criteria |
| `code/results/` | Saved CSV and JSON outputs behind article tables and figures |
| `derived_data/` | Aligned flow summaries and saved test predictions |
| `code/verify_from_predictions.py` | Recompute key published comparisons from derived data |
| `MANIFEST.csv` | SHA-256 checksum and byte size of each released file |

## Reproduce the full pipeline

The reported environment was Python 3.11.15 on an NVIDIA GeForce RTX 4090; `requirements.txt` records package versions. Run pipeline commands from `code/`, because the scripts resolve configuration and results paths relative to that directory. Set the source paths in `configs/directionA_common.yaml`, `configs/direction_a_2018_alignment.yaml`, and the `directionA2018_*common.yaml` files for your machine. The repository-relative defaults use `code/data/raw/`, `code/data/processed/`, and `code/reports/`. These large input and generated directories are ignored by Git.

1. Convert the three CIC-IDS2017 releases using `alignment_cicids2017/01_original_to_parquet.py` and `02_improved_to_parquet.py` (both take `--in-dir` and `--out-dir`). Place their outputs under `data/processed/{original,engelen,liu}/`.
2. Run `alignment_cicids2017/10_label_diff.py` and `11_natural_noise_mask.py`, then `scripts/directionA_prepare.py`. `LABELGAP_WORK_DIR` can override the default `code/` workspace used by the two alignment scripts.
3. Place the original and Liu-corrected CSE-CIC-IDS2018 files under the configured directories. Run `scripts/directionA_2018_alignment.py`, `scripts/directionA2018_prepare.py`, and `scripts/directionA2018_attempted_variants.py`.
4. Run the relevant `scripts/run_directionA*.sh` files. They use `python3` by default; set `PY=/path/to/python` to select another environment.
5. Run the analysis and figure scripts listed above. `code/figures/cost_from_logs_v20261021.py` needs the original training logs in `code/logs/`; those large logs are not included.

The configuration files retain the experimental choices and distinguish planned analyses from post hoc analyses. The cleaned export changes paths, comments, and documentation; it does not change saved predictions or results.

## Derived data

`cicids2017_alignment.csv.gz` records the aligned original rows, their original and corrected labels, alignment tier, and split. `cicids2018_learning_sample.csv.gz` records the 40% learning sample, both label conventions for attempted flows, alignment tier, and split. `test_predictions_*.csv.gz` records the test labels and predictions for all model and seed runs. `class_names_*.json` maps class indices to names. These files contain labels and identifiers originating from the cited public releases; cite the original datasets when reusing them.

## License

Code: MIT License (`LICENSE`). Source datasets remain under the terms of their providers.
