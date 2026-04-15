# LOO-CL Audit Report Bundle

This directory is the lightweight, GitHub-friendly export of the current LOO-CL analysis state.

## Included
- `summaries/geometry_summary.json`
- `summaries/diagnosis_geometry_summary.json`
- `summaries/target_error_summary.csv`
- `summaries/target_metric_correlation.csv`
- `summaries/subject_tail_top_unstable.csv`
- `summaries/subject_tail_top_stable.csv`
- `summaries/dominant_split_subject_counts.csv`
- `summaries/dominant_outlier_subject_counts.csv`
- `figures/*.png`
- `manifest.json`

## Intentionally excluded
- Full `window_metrics.csv`
- Full `diagnosis_geometry_join.csv`
- Processed/raw HDF5 inputs
- Embedding HDF5 files

Those heavier files remain available only in the local `artifacts/` tree.
