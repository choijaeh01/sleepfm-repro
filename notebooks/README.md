# Notebooks

This directory contains the narrative notebooks used to inspect the SleepFM LOO-CL analysis from different angles.

## Files
- `loo_cl_analysis_walkthrough.ipynb`
  - Result-first walkthrough of the initial geometry audit and downstream diagnosis join.
- `loo_cl_code_structure_walkthrough.ipynb`
  - Code-structure walkthrough showing where LOO-CL, embedding export, downstream evaluation, and audit scripts connect in code.
- `loo_cl_tail_metrics_walkthrough.ipynb`
  - Result-first walkthrough of the tail-aware extension with `split score`, `outlier score`, and `unstable_window_fraction`.

## Suggested reading order
1. `loo_cl_analysis_walkthrough.ipynb`
2. `loo_cl_tail_metrics_walkthrough.ipynb`
3. `loo_cl_code_structure_walkthrough.ipynb`

## Note
These notebooks read local outputs under `artifacts/` for full analysis, while the curated GitHub-safe bundle lives under `reports/loo_cl_audit/`.
