from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GitHub-friendly lightweight bundle for the LOO-CL analysis."
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/audit/loo_geometry_model_base_paperexact_shhs"),
        help="Directory containing the full local audit outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/loo_cl_audit"),
        help="Directory where the lightweight GitHub bundle will be written.",
    )
    parser.add_argument(
        "--top-k-subjects",
        type=int,
        default=50,
        help="Number of top stable/unstable subjects to export as compact CSV tables.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path, manifest: list[dict[str, str]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "type": "copied",
            "source": str(src.resolve()),
            "destination": str(dst.resolve()),
        }
    )


def write_csv(df: pd.DataFrame, path: Path, manifest: list[dict[str, str]], description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    manifest.append(
        {
            "type": "derived_csv",
            "description": description,
            "destination": str(path.resolve()),
        }
    )


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    diagnosis_dir = audit_dir / "diagnosis_6y"
    output_dir = args.output_dir.resolve()
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []

    summary_files = {
        audit_dir / "summary.json": summaries_dir / "geometry_summary.json",
        diagnosis_dir / "diagnosis_geometry_summary.json": summaries_dir / "diagnosis_geometry_summary.json",
        diagnosis_dir / "target_error_summary.csv": summaries_dir / "target_error_summary.csv",
        diagnosis_dir / "target_metric_correlation.csv": summaries_dir / "target_metric_correlation.csv",
    }
    figure_files = {
        diagnosis_dir / "scatter_grid_6y.png": figures_dir / "scatter_grid_6y.png",
        diagnosis_dir / "chf_tail_boxplots_6y.png": figures_dir / "chf_tail_boxplots_6y.png",
        diagnosis_dir / "chf_risk_rank_vs_tail_metrics_6y.png": figures_dir / "chf_risk_rank_vs_tail_metrics_6y.png",
        diagnosis_dir / "chf_split_vs_outlier_6y.png": figures_dir / "chf_split_vs_outlier_6y.png",
        diagnosis_dir / "chf_example_constellations.png": figures_dir / "chf_example_constellations.png",
    }

    for src, dst in {**summary_files, **figure_files}.items():
        if not src.exists():
            raise FileNotFoundError(src)
        copy_file(src, dst, manifest)

    subject_df = pd.read_csv(audit_dir / "subject_metrics.csv")

    unstable_cols = [
        "subject",
        "n_windows",
        "unstable_window_fraction",
        "alignment_mean_p05",
        "jackknife_instability_p95",
        "split_score_p95",
        "outlier_score_p95",
        "dominant_best_split_partition",
        "dominant_outlier_modality",
    ]
    top_unstable = subject_df.sort_values(
        ["unstable_window_fraction", "split_score_p95", "jackknife_instability_p95"],
        ascending=False,
    )[unstable_cols].head(args.top_k_subjects)
    top_stable = subject_df.sort_values(
        ["unstable_window_fraction", "split_score_p95", "jackknife_instability_p95"],
        ascending=True,
    )[unstable_cols].head(args.top_k_subjects)

    write_csv(
        top_unstable,
        summaries_dir / "subject_tail_top_unstable.csv",
        manifest,
        "Top unstable subjects ranked by tail-aware instability summary.",
    )
    write_csv(
        top_stable,
        summaries_dir / "subject_tail_top_stable.csv",
        manifest,
        "Top stable subjects ranked by tail-aware instability summary.",
    )

    split_counts = (
        subject_df["dominant_best_split_partition"]
        .value_counts(dropna=False)
        .rename_axis("dominant_best_split_partition")
        .reset_index(name="count")
    )
    split_counts["fraction"] = split_counts["count"] / split_counts["count"].sum()
    write_csv(
        split_counts,
        summaries_dir / "dominant_split_subject_counts.csv",
        manifest,
        "Subject-level dominant split partition frequency table.",
    )

    outlier_counts = (
        subject_df["dominant_outlier_modality"]
        .value_counts(dropna=False)
        .rename_axis("dominant_outlier_modality")
        .reset_index(name="count")
    )
    outlier_counts["fraction"] = outlier_counts["count"] / outlier_counts["count"].sum()
    write_csv(
        outlier_counts,
        summaries_dir / "dominant_outlier_subject_counts.csv",
        manifest,
        "Subject-level dominant outlier modality frequency table.",
    )

    metrics_overview = {
        "source_audit_dir": str(audit_dir),
        "bundle_dir": str(output_dir),
        "omitted_large_local_artifacts": [
            str((audit_dir / "window_metrics.csv").resolve()),
            str((diagnosis_dir / "diagnosis_geometry_join.csv").resolve()),
        ],
        "notes": [
            "Large local artifacts are intentionally excluded from the GitHub bundle.",
            "The bundle keeps notebooks, analysis scripts, small summaries, and figures.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"bundle": metrics_overview, "files": manifest}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"bundle_dir": str(output_dir), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
