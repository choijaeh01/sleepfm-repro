from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


MODALITY_COLORS = {
    "BAS": "#1f77b4",
    "RESP": "#ff7f0e",
    "EKG": "#2ca02c",
    "EMG": "#d62728",
}

ERROR_COLORS = {
    "TP": "#2ca02c",
    "TN": "#1f77b4",
    "FP": "#ff7f0e",
    "FN": "#d62728",
}

ERROR_ORDER = ["TN", "TP", "FP", "FN"]

SUMMARY_METRICS = [
    "consensus_norm_mean",
    "alignment_mean",
    "alignment_mean_p05",
    "jackknife_instability_mean",
    "jackknife_instability_p95",
    "split_score_mean",
    "split_score_p95",
    "outlier_score_mean",
    "outlier_score_p95",
    "unstable_window_fraction",
]

TAIL_METRICS = [
    ("alignment_mean_p05", "Alignment p05"),
    ("jackknife_instability_p95", "Jackknife p95"),
    ("split_score_p95", "Split score p95"),
    ("outlier_score_p95", "Outlier score p95"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join LOO geometry metrics with disease prediction outputs and create initial figures."
    )
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon-days", type=float, default=6 * 365.25)
    parser.add_argument("--focus-target", type=str, default="chf")
    parser.add_argument("--modalities", nargs="+", default=["BAS", "RESP", "EKG", "EMG"])
    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def simple_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None

    wins = 0.0
    total = 0
    for i in pos:
        for j in neg:
            total += 1
            if scores[i] > scores[j]:
                wins += 1.0
            elif scores[i] == scores[j]:
                wins += 0.5
    return wins / total


def topk_predictions(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    k = int(labels.sum())
    pred = np.zeros_like(labels)
    if k > 0:
        top_idx = np.argsort(scores)[::-1][:k]
        pred[top_idx] = 1
    return pred


def error_type(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def compute_local_pca(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    return centered @ basis


def normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def make_scatter_grid(join_df: pd.DataFrame, targets: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, target in zip(axes, targets):
        subset = join_df[join_df["target"] == target].copy()
        for err in ["TN", "TP", "FP", "FN"]:
            part = subset[subset["error_type"] == err]
            if part.empty:
                continue
            ax.scatter(
                part["consensus_norm_mean"],
                part["jackknife_instability_mean"],
                s=20,
                alpha=0.7,
                c=ERROR_COLORS[err],
                label=err,
                edgecolors="none",
            )
        ax.set_title(f"{target} (n={len(subset)})")
        ax.set_xlabel("Consensus Norm")
        ax.set_ylabel("Jackknife Instability")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("LOO Geometry vs 6-Year Outcome Error Type", y=1.02, fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_target_boxplot(join_df: pd.DataFrame, target: str, output_path: Path) -> None:
    subset = join_df[join_df["target"] == target].copy()
    order = ERROR_ORDER
    data = [subset.loc[subset["error_type"] == err, "jackknife_instability_mean"].values for err in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch, err in zip(box["boxes"], order):
        patch.set_facecolor(ERROR_COLORS[err])
        patch.set_alpha(0.5)
    ax.set_ylabel("Subject-Level Jackknife Instability")
    ax.set_title(f"{target}: instability by 6-year error type")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_focus_risk_scatter(join_df: pd.DataFrame, target: str, output_path: Path) -> None:
    subset = join_df[join_df["target"] == target].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for err in ERROR_ORDER:
        part = subset[subset["error_type"] == err]
        if part.empty:
            continue
        ax.scatter(
            part["risk_rank_pct"],
            part["jackknife_instability_mean"],
            s=28,
            alpha=0.75,
            c=ERROR_COLORS[err],
            label=err,
            edgecolors="none",
        )
    ax.set_xlabel("Risk Percentile")
    ax.set_ylabel("Subject-Level Jackknife Instability")
    ax.set_title(f"{target}: risk rank vs instability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_tail_boxplot_grid(join_df: pd.DataFrame, target: str, output_path: Path) -> None:
    subset = join_df[join_df["target"] == target].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for ax, (metric, title) in zip(axes.ravel(), TAIL_METRICS):
        data = [subset.loc[subset["error_type"] == err, metric].values for err in ERROR_ORDER]
        box = ax.boxplot(data, tick_labels=ERROR_ORDER, patch_artist=True)
        for patch, err in zip(box["boxes"], ERROR_ORDER):
            patch.set_facecolor(ERROR_COLORS[err])
            patch.set_alpha(0.5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{target}: tail-aware geometry by 6-year error type", y=1.02, fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_tail_risk_scatter_grid(join_df: pd.DataFrame, target: str, output_path: Path) -> None:
    subset = join_df[join_df["target"] == target].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for ax, (metric, title) in zip(axes.ravel(), TAIL_METRICS):
        for err in ERROR_ORDER:
            part = subset[subset["error_type"] == err]
            if part.empty:
                continue
            ax.scatter(
                part["risk_rank_pct"],
                part[metric],
                s=28,
                alpha=0.75,
                c=ERROR_COLORS[err],
                label=err,
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("Risk Percentile")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{target}: risk rank vs tail-aware geometry", y=1.02, fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_split_outlier_scatter(join_df: pd.DataFrame, target: str, output_path: Path) -> None:
    subset = join_df[join_df["target"] == target].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for err in ERROR_ORDER:
        part = subset[subset["error_type"] == err]
        if part.empty:
            continue
        ax.scatter(
            part["split_score_p95"],
            part["outlier_score_p95"],
            s=32,
            alpha=0.75,
            c=ERROR_COLORS[err],
            label=err,
            edgecolors="none",
        )
    ax.set_xlabel("Split score p95")
    ax.set_ylabel("Outlier score p95")
    ax.set_title(f"{target}: 2+2 split vs 3+1 outlier tail")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_examples(join_df: pd.DataFrame, target: str) -> list[tuple[str, str, str]]:
    subset = join_df[join_df["target"] == target].copy()
    examples: list[tuple[str, str, str]] = []

    selectors = [
        ("TN", True, "stable_tn"),
        ("TP", True, "stable_tp"),
        ("FP", False, "unstable_fp"),
        ("FN", False, "unstable_fn"),
    ]

    sort_columns = [
        "unstable_window_fraction",
        "split_score_p95",
        "outlier_score_p95",
        "jackknife_instability_p95",
    ]

    for err, ascending, label in selectors:
        part = subset[subset["error_type"] == err]
        if part.empty:
            continue
        row = part.sort_values(sort_columns, ascending=[ascending] * len(sort_columns)).iloc[0]
        examples.append((row["subject"], err, label))
    return examples


def select_window(window_df: pd.DataFrame, subject: str, err: str) -> pd.Series:
    part = window_df[window_df["subject"] == subject].copy()
    part["instability_score"] = (
        part["jackknife_instability"] + part["split_score"] + part["outlier_score"] + (1.0 - part["alignment_mean"])
    )
    if err in {"FP", "FN"}:
        return part.sort_values(["instability_score", "jackknife_instability"], ascending=False).iloc[0]
    return part.sort_values(["instability_score", "jackknife_instability"], ascending=True).iloc[0]


def load_window_vectors(embedding_dir: Path, subject: str, window_idx: int, modalities: list[str]) -> np.ndarray:
    path = embedding_dir / f"{subject}.hdf5"
    with h5py.File(path, "r") as handle:
        vectors = [np.asarray(handle[modality][window_idx], dtype=np.float32) for modality in modalities]
    return np.stack(vectors, axis=0)


def make_example_panels(
    embedding_dir: Path,
    window_df: pd.DataFrame,
    examples: list[tuple[str, str, str]],
    modalities: list[str],
    output_path: Path,
) -> None:
    n_cols = len(examples)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8), constrained_layout=True)
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, (subject, err, label) in enumerate(examples):
        window_row = select_window(window_df, subject, err)
        window_idx = int(window_row["window_idx"])
        modality_vectors = load_window_vectors(embedding_dir, subject, window_idx, modalities)
        modality_vectors = normalize(modality_vectors)

        loo_means = []
        for idx in range(len(modalities)):
            others = [j for j in range(len(modalities)) if j != idx]
            loo_means.append(modality_vectors[others].mean(axis=0))
        loo_means = np.stack(loo_means, axis=0)

        coords = compute_local_pca(np.vstack([modality_vectors, loo_means]))
        mod_coords = coords[: len(modalities)]
        loo_coords = coords[len(modalities) :]

        ax = axes[0, col]
        for idx, modality in enumerate(modalities):
            color = MODALITY_COLORS.get(modality, "black")
            ax.scatter(mod_coords[idx, 0], mod_coords[idx, 1], s=70, c=color)
            ax.scatter(loo_coords[idx, 0], loo_coords[idx, 1], s=70, c=color, marker="x")
            ax.annotate(modality, (mod_coords[idx, 0], mod_coords[idx, 1]), textcoords="offset points", xytext=(4, 4), fontsize=9)
            ax.arrow(
                mod_coords[idx, 0],
                mod_coords[idx, 1],
                loo_coords[idx, 0] - mod_coords[idx, 0],
                loo_coords[idx, 1] - mod_coords[idx, 1],
                color=color,
                alpha=0.55,
                width=0.002,
                head_width=0.04,
                length_includes_head=True,
            )
        ax.set_title(
            f"{label}\n{subject} | w={window_idx}\ncons={window_row['consensus_norm']:.3f}, jack={window_row['jackknife_instability']:.3f}"
        )
        ax.set_xlabel("Local PC1")
        ax.set_ylabel("Local PC2")
        ax.grid(alpha=0.2)

        heat_ax = axes[1, col]
        cos = modality_vectors @ modality_vectors.T
        im = heat_ax.imshow(cos, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        heat_ax.set_xticks(range(len(modalities)), modalities, rotation=45, ha="right")
        heat_ax.set_yticks(range(len(modalities)), modalities)
        heat_ax.set_title(f"{label} pairwise cosine")
        for i in range(len(modalities)):
            for j in range(len(modalities)):
                heat_ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=axes[1, :].ravel().tolist(), shrink=0.7)
    cbar.set_label("Cosine")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_df = pd.read_csv(args.geometry_dir / "subject_metrics.csv")
    window_df = pd.read_csv(args.geometry_dir / "window_metrics.csv")
    target_names = [col for col in pd.read_csv(args.labels_dir / "is_event.csv", nrows=1).columns if col != "Study ID"]

    outputs = np.asarray(load_pickle(args.eval_dir / "all_outputs.pickle"), dtype=float)
    event_times = np.asarray(load_pickle(args.eval_dir / "all_event_times.pickle"), dtype=float)
    is_event = np.asarray(load_pickle(args.eval_dir / "all_is_event.pickle"), dtype=int)
    paths = np.asarray(load_pickle(args.eval_dir / "all_paths.pickle"))

    subjects = [Path(path).stem for path in paths]
    geometry_test = subject_df.set_index("subject").loc[subjects].reset_index()

    join_rows = []
    summary = {
        "horizon_days": args.horizon_days,
        "n_test_subjects": int(len(subjects)),
        "targets": {},
    }
    correlation_rows = []

    for idx, target in enumerate(target_names):
        risk = outputs[:, idx]
        event_within_h = ((is_event[:, idx] == 1) & (event_times[:, idx] <= args.horizon_days)).astype(int)
        pred = topk_predictions(risk, event_within_h)
        auc = simple_auc(event_within_h, risk)

        target_df = geometry_test.copy()
        target_df["target"] = target
        target_df["risk"] = risk
        target_df["event_time"] = event_times[:, idx]
        target_df["is_event"] = is_event[:, idx]
        target_df["event_within_horizon"] = event_within_h
        target_df["pred_positive_topk"] = pred
        target_df["error_type"] = [error_type(int(y), int(p)) for y, p in zip(event_within_h, pred)]
        target_df["risk_rank_pct"] = pd.Series(risk).rank(method="average", pct=True).values
        join_rows.append(target_df)

        err_summary = {}
        for err in ERROR_ORDER:
            part = target_df[target_df["error_type"] == err]
            err_stats: dict[str, object] = {"count": int(len(part))}
            for metric in SUMMARY_METRICS:
                if metric in target_df.columns:
                    err_stats[metric] = None if part.empty else float(part[metric].mean())
            if "dominant_best_split_partition" in target_df.columns:
                err_stats["dominant_best_split_partition_counts"] = (
                    {} if part.empty else {str(k): int(v) for k, v in part["dominant_best_split_partition"].value_counts().to_dict().items()}
                )
            if "dominant_outlier_modality" in target_df.columns:
                err_stats["dominant_outlier_modality_counts"] = (
                    {} if part.empty else {str(k): int(v) for k, v in part["dominant_outlier_modality"].value_counts().to_dict().items()}
                )
            err_summary[err] = err_stats

        risk_metric_correlation: dict[str, float | None] = {}
        for metric in SUMMARY_METRICS:
            if metric not in target_df.columns or target_df[metric].nunique() <= 1:
                risk_metric_correlation[metric] = None
                continue
            corr = np.corrcoef(target_df["risk_rank_pct"], target_df[metric])[0, 1]
            risk_metric_correlation[metric] = float(corr)
            correlation_rows.append({"target": target, "metric": metric, "correlation": float(corr)})

        summary["targets"][target] = {
            "positive_within_horizon": int(event_within_h.sum()),
            "all_events": int(is_event[:, idx].sum()),
            "auroc": None if auc is None else float(auc),
            "error_type_summary": err_summary,
            "risk_rank_correlation": risk_metric_correlation,
        }

    join_df = pd.concat(join_rows, ignore_index=True)
    join_csv = args.output_dir / "diagnosis_geometry_join.csv"
    join_df.to_csv(join_csv, index=False)

    flat_summary_rows = []
    for target, info in summary["targets"].items():
        for err, stats in info["error_type_summary"].items():
            row = {
                "target": target,
                "positive_within_horizon": info["positive_within_horizon"],
                "all_events": info["all_events"],
                "auroc": info["auroc"],
                "error_type": err,
                "count": stats["count"],
            }
            for metric in SUMMARY_METRICS:
                row[metric] = stats.get(metric)
            flat_summary_rows.append(row)
    pd.DataFrame(flat_summary_rows).to_csv(args.output_dir / "target_error_summary.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(args.output_dir / "target_metric_correlation.csv", index=False)

    summary_path = args.output_dir / "diagnosis_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_scatter_grid(join_df, target_names, args.output_dir / "scatter_grid_6y.png")
    make_target_boxplot(join_df, args.focus_target, args.output_dir / f"{args.focus_target}_jackknife_boxplot_6y.png")
    make_focus_risk_scatter(join_df, args.focus_target, args.output_dir / f"{args.focus_target}_risk_rank_vs_instability_6y.png")
    make_tail_boxplot_grid(join_df, args.focus_target, args.output_dir / f"{args.focus_target}_tail_boxplots_6y.png")
    make_tail_risk_scatter_grid(join_df, args.focus_target, args.output_dir / f"{args.focus_target}_risk_rank_vs_tail_metrics_6y.png")
    make_split_outlier_scatter(join_df, args.focus_target, args.output_dir / f"{args.focus_target}_split_vs_outlier_6y.png")

    examples = choose_examples(join_df, args.focus_target)
    if examples:
        make_example_panels(
            args.embedding_dir,
            window_df,
            examples,
            args.modalities,
            args.output_dir / f"{args.focus_target}_example_constellations.png",
        )

    print(json.dumps(summary, indent=2))
    print(f"JOIN_CSV {join_csv}")


if __name__ == "__main__":
    main()
