# Scripts

This directory contains the reproducible entry points for the SHHS SleepFM reproduction and the LOO-CL audit.

## LOO-CL audit
- `analyze_loo_geometry.py`
  - Computes window-level and subject-level geometry metrics from `5-minute aggregated` SleepFM embeddings.
  - Includes `consensus`, `alignment`, `jackknife`, `split score`, `outlier score`, and tail-aware subject summaries.
- `make_loo_diagnosis_figures.py`
  - Joins geometry summaries with downstream diagnosis outputs and generates summary tables plus figures.
- `RUN_LOO_CL_AUDIT.ps1`
  - Convenience PowerShell entry point that runs the geometry audit, downstream join, and GitHub bundle export in sequence.
- `export_loo_cl_github_bundle.py`
  - Creates a lightweight `reports/loo_cl_audit/` bundle with summaries and figures suitable for GitHub.

## SHHS reproduction helpers
- `prepare_shhs_downstream.py`
  - Builds SHHS label/demo artifacts for downstream CoxPH evaluation.
- `RUN_SHHS_PILOT.ps1`
  - Existing pilot-oriented PowerShell helper.
