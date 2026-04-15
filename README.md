# sleepfm-repro

SHHS-first downstream reproduction scaffold for **A Multimodal Sleep Foundation Model for Disease Prediction**.

## Goal
Use the public `zou-group/sleepfm-clinical` codebase together with local NAS-hosted SHHS data to validate the paper's downstream disease-prediction setting as quickly as possible on the Windows desktop GPU (RTX 4070).

## Current strategy
We are **not** attempting exact replication of the Stanford SSC+EHR experiments first.
We start with **SHHS downstream validation**, which is the fastest public path:

1. Convert SHHS EDF to raw HDF5 expected by SleepFM.
2. Generate SHHS embeddings with the provided pretrained base checkpoint.
3. Build SHHS CoxPH labels (`is_event.csv`, `time_to_event.csv`) and demo features.
4. Fine-tune/evaluate CoxPH diagnosis model on SHHS.

## LOO-CL audit status
The current repo now includes a reproducible LOO-CL latent-space audit built on top of the SHHS paper-like SleepFM setup.

### Main notebooks
- `notebooks/loo_cl_analysis_walkthrough.ipynb`
  - Initial result-first walkthrough of the geometry audit.
- `notebooks/loo_cl_tail_metrics_walkthrough.ipynb`
  - Tail-aware extension with `split score`, `outlier score`, and `unstable_window_fraction`.
- `notebooks/loo_cl_code_structure_walkthrough.ipynb`
  - Code-structure walkthrough linking upstream SleepFM code to local audit scripts and outputs.

### Main scripts
- `scripts/analyze_loo_geometry.py`
  - Computes geometry metrics from `5-minute aggregated` SleepFM embeddings.
- `scripts/make_loo_diagnosis_figures.py`
  - Joins geometry summaries with downstream disease outputs and generates tables/figures.
- `scripts/RUN_LOO_CL_AUDIT.ps1`
  - Runs the geometry audit, downstream join, and lightweight GitHub bundle export.
- `scripts/export_loo_cl_github_bundle.py`
  - Builds `reports/loo_cl_audit/` with GitHub-safe summaries and figures.

### GitHub-safe outputs
- `reports/loo_cl_audit/`
  - Curated summaries and figures intended for repository sharing.
- `docs/loo_cl_audit/README.md`
  - File-structure guide for keeping the repository readable and lightweight.

## Project layout
- `notebooks/` - analysis walkthrough notebooks
- `scripts/` - reproducible entry points for SHHS prep, LOO-CL audit, and report export
- `docs/` - repository-facing documentation
- `reports/` - lightweight figures and summary tables meant for GitHub
- `configs/` - local config templates
- `notes/` - free-form local notes
- `artifacts/labels/` - generated `is_event.csv`, `time_to_event.csv`, `demo_labels.csv`
- `artifacts/splits/` - generated split JSON
- `artifacts/processed/` - raw HDF5 converted from EDF
- `artifacts/embeddings/` - SleepFM embedding HDF5 outputs
- `artifacts/audit/` - full local audit outputs, including large intermediate CSVs kept out of Git

## Recommended first-pass SHHS outcomes
Chosen to balance event count, availability of event dates, and closeness to cardiovascular downstream prediction:
- `mi`
- `stroke`
- `chf`
- `cvd_death`
- `chd_death`
- `revasc_proc`

## Planned commands
### 1) Prepare SHHS downstream artifacts
```powershell
cd C:\Projects\aiot\sleepfm-repro
python scripts\prepare_shhs_downstream.py `
  --shhs-summary "\\192.168.0.37\Data\datasets\nsrr\shhs\datasets\shhs-cvd-summary-dataset-0.21.0.csv" `
  --shhs-harmonized "\\192.168.0.37\Data\datasets\nsrr\shhs\datasets\shhs-harmonized-dataset-0.21.0.csv" `
  --edf-root "\\192.168.0.37\Data\datasets\nsrr\shhs\polysomnography\edfs" `
  --processed-root "C:\Projects\aiot\sleepfm-repro\artifacts\processed\shhs" `
  --output-root "C:\Projects\aiot\sleepfm-repro\artifacts\labels\shhs_6cvd" `
  --split-json "C:\Projects\aiot\sleepfm-repro\artifacts\splits\shhs_downstream_split.json" `
  --targets configs\shhs_downstream_targets.yaml
```

### 2) Convert EDF to HDF5
Use `sleepfm-clinical/sleepfm/preprocessing/preprocessing.py` after adapting local paths.

### 3) Generate embeddings
Use the pretrained base checkpoint from `sleepfm-clinical/sleepfm/checkpoints/model_base`.

### 4) Fine-tune CoxPH diagnosis model
Use `configs/config_finetune_diagnosis_coxph_shhs_local.yaml` after filling in final local paths.

## Notes
- The upstream repo's diagnosis pipeline expects **embedding HDF5 files** named by study ID and labels with a `Study ID` column.
- The split JSON should point to **existing processed/raw HDF5 paths**; the diagnosis dataset later matches them to embedding files by basename.
- SHHS demographics are available, so a demo+signal model is feasible for this downstream stage.
- Large local artifacts are intentionally excluded from Git. Use `reports/loo_cl_audit/` for the lightweight bundle and keep `artifacts/` local-only.
