# LOO-CL Audit Guide

This folder documents how the current SleepFM LOO-CL analysis is organized for a GitHub-facing repository.

## Goal
Keep the repository readable and reproducible without committing large local-only data products such as processed HDF5 files, full embedding dumps, or oversized intermediate CSV joins.

## Recommended repository structure
```text
sleepfm-repro/
├─ README.md
├─ .gitignore
├─ notebooks/
│  ├─ README.md
│  ├─ loo_cl_analysis_walkthrough.ipynb
│  ├─ loo_cl_tail_metrics_walkthrough.ipynb
│  └─ loo_cl_code_structure_walkthrough.ipynb
├─ scripts/
│  ├─ README.md
│  ├─ analyze_loo_geometry.py
│  ├─ make_loo_diagnosis_figures.py
│  ├─ RUN_LOO_CL_AUDIT.ps1
│  └─ export_loo_cl_github_bundle.py
├─ reports/
│  └─ loo_cl_audit/
│     ├─ README.md
│     ├─ manifest.json
│     ├─ summaries/
│     └─ figures/
└─ artifacts/
   └─ ... local-only large outputs, ignored from Git
```

## What should go to GitHub
- Analysis notebooks in `notebooks/`
- Reproducible source scripts in `scripts/`
- Lightweight analysis outputs in `reports/loo_cl_audit/`
- Documentation in `docs/`

## What should stay local
- `artifacts/processed/`
- `artifacts/embeddings/`
- Full `artifacts/audit/.../window_metrics.csv`
- Full `artifacts/audit/.../diagnosis_geometry_join.csv`
- Raw dataset symlinks, NAS paths, and large logs

## Minimal workflow
1. Run `scripts/RUN_LOO_CL_AUDIT.ps1`
2. Verify that `reports/loo_cl_audit/` is refreshed
3. Commit `README.md`, `docs/`, `notebooks/`, `scripts/`, and `reports/`
4. Do not commit `artifacts/` or `raw/`
