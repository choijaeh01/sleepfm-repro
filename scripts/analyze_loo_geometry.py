from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute window-level and subject-level LOO geometry metrics from 5-minute SleepFM embeddings."
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        type=Path,
        help="Directory containing per-subject HDF5 files with modality datasets shaped [n_windows, embed_dim].",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where summary.json, subject_metrics.csv, and window_metrics.csv will be written.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["BAS", "RESP", "EKG", "EMG"],
        help="Modalities to include, in a fixed order.",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Optional cap on the number of subject files to process.",
    )
    parser.add_argument(
        "--jackknife-scale",
        choices=["half", "full"],
        default="half",
        help="Use 'half' for 0.5 * (1 - cosine), or 'full' for 1 - cosine.",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.05,
        help="Low-tail quantile used for alignment/consensus summaries and thresholded fractions.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.95,
        help="High-tail quantile used for instability/split/outlier summaries and thresholded fractions.",
    )
    parser.add_argument(
        "--subject-top-k",
        type=int,
        default=10,
        help="Top-k windows to average for tail-aware subject summaries.",
    )
    return parser.parse_args()


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    dots = np.sum(a * b, axis=1)
    return dots / np.clip(denom, eps, None)


def mean_from_pairs(cosine_windows: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(cosine_windows.shape[0], dtype=np.float32)
    values = np.stack([cosine_windows[:, left, right] for left, right in pairs], axis=0)
    return values.mean(axis=0)


def balanced_partitions(m_count: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if m_count < 4 or m_count % 2 != 0:
        return []

    indices = tuple(range(m_count))
    half = m_count // 2
    partitions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for group_a in combinations(indices, half):
        if 0 not in group_a:
            continue
        group_b = tuple(idx for idx in indices if idx not in group_a)
        partitions.append((group_a, group_b))
    return partitions


def partition_label(group_a: tuple[int, ...], group_b: tuple[int, ...], modalities: list[str]) -> str:
    left = "+".join(modalities[idx] for idx in group_a)
    right = "+".join(modalities[idx] for idx in group_b)
    return f"{left}|{right}"


def compute_split_metrics(cosine_windows: np.ndarray, modalities: list[str]) -> tuple[np.ndarray, np.ndarray]:
    partitions = balanced_partitions(cosine_windows.shape[1])
    if not partitions:
        n_windows = cosine_windows.shape[0]
        empty_labels = np.full(n_windows, "", dtype=object)
        return np.zeros(n_windows, dtype=np.float32), empty_labels

    scores = []
    labels = []
    for group_a, group_b in partitions:
        within_pairs = list(combinations(group_a, 2)) + list(combinations(group_b, 2))
        between_pairs = [(left, right) for left in group_a for right in group_b]
        within = mean_from_pairs(cosine_windows, within_pairs)
        between = mean_from_pairs(cosine_windows, between_pairs)
        scores.append(within - between)
        labels.append(partition_label(group_a, group_b, modalities))

    score_arr = np.stack(scores, axis=0)
    best_idx = np.argmax(score_arr, axis=0)
    window_idx = np.arange(score_arr.shape[1])
    best_score = score_arr[best_idx, window_idx].astype(np.float32)
    best_labels = np.asarray([labels[idx] for idx in best_idx], dtype=object)
    return best_score, best_labels


def compute_outlier_metrics(cosine_windows: np.ndarray, modalities: list[str]) -> tuple[np.ndarray, np.ndarray]:
    m_count = cosine_windows.shape[1]
    if m_count < 3:
        n_windows = cosine_windows.shape[0]
        empty_labels = np.full(n_windows, "", dtype=object)
        return np.zeros(n_windows, dtype=np.float32), empty_labels

    scores = []
    for outlier_idx in range(m_count):
        others = [idx for idx in range(m_count) if idx != outlier_idx]
        within_pairs = list(combinations(others, 2))
        within_others = mean_from_pairs(cosine_windows, within_pairs)
        between = np.stack([cosine_windows[:, outlier_idx, idx] for idx in others], axis=0).mean(axis=0)
        scores.append(within_others - between)

    score_arr = np.stack(scores, axis=0)
    best_idx = np.argmax(score_arr, axis=0)
    window_idx = np.arange(score_arr.shape[1])
    best_score = score_arr[best_idx, window_idx].astype(np.float32)
    best_modalities = np.asarray([modalities[idx] for idx in best_idx], dtype=object)
    return best_score, best_modalities


def load_subject_embeddings(path: Path, modalities: list[str]) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        missing = [modality for modality in modalities if modality not in handle]
        if missing:
            raise ValueError(f"missing modalities: {missing}")

        arrays = [np.asarray(handle[modality], dtype=np.float32) for modality in modalities]

    n_windows = min(arr.shape[0] for arr in arrays)
    if n_windows == 0:
        raise ValueError("zero windows")

    trimmed = [arr[:n_windows] for arr in arrays]
    return np.stack(trimmed, axis=0)


def compute_metrics(
    embeddings: np.ndarray,
    modalities: list[str],
    jackknife_scale: str,
) -> dict[str, np.ndarray]:
    # embeddings: [M, W, D]
    normalized = np.stack([normalize_rows(modality) for modality in embeddings], axis=0)
    m_count = normalized.shape[0]
    cosine_windows = np.einsum("mwd,nwd->wmn", normalized, normalized).astype(np.float32, copy=False)

    consensus = normalized.mean(axis=0)
    consensus_norm = np.linalg.norm(consensus, axis=1)

    loo_norms = []
    alignments = []
    instabilities = []

    for query_idx in range(m_count):
        other_indices = [idx for idx in range(m_count) if idx != query_idx]
        loo_mean = normalized[other_indices].mean(axis=0)
        loo_norm = np.linalg.norm(loo_mean, axis=1)
        alignment = cosine_rows(normalized[query_idx], loo_mean)

        reduced_means = []
        if len(other_indices) >= 2:
            for removed_idx in other_indices:
                kept = [idx for idx in other_indices if idx != removed_idx]
                reduced_means.append(normalized[kept].mean(axis=0))

        if len(reduced_means) >= 2:
            pairwise = [cosine_rows(reduced_means[a], reduced_means[b]) for a, b in combinations(range(len(reduced_means)), 2)]
            instability = 1.0 - np.mean(np.stack(pairwise, axis=0), axis=0)
            if jackknife_scale == "half":
                instability = 0.5 * instability
        else:
            instability = np.zeros(normalized.shape[1], dtype=np.float32)

        loo_norms.append(loo_norm)
        alignments.append(alignment)
        instabilities.append(instability)

    loo_norms_arr = np.stack(loo_norms, axis=0)
    alignments_arr = np.stack(alignments, axis=0)
    instabilities_arr = np.stack(instabilities, axis=0)
    split_score, best_split_partition = compute_split_metrics(cosine_windows, modalities)
    outlier_score, outlier_modality = compute_outlier_metrics(cosine_windows, modalities)

    result: dict[str, np.ndarray] = {
        "consensus_norm": consensus_norm,
        "loo_mean_norm_mean": loo_norms_arr.mean(axis=0),
        "alignment_mean": alignments_arr.mean(axis=0),
        "jackknife_instability": instabilities_arr.mean(axis=0),
        "split_score": split_score,
        "outlier_score": outlier_score,
        "best_split_partition": best_split_partition,
        "outlier_modality": outlier_modality,
    }

    for idx, modality in enumerate(modalities):
        result[f"alignment_{modality}"] = alignments_arr[idx]
        result[f"loo_norm_{modality}"] = loo_norms_arr[idx]

    return result


def summarize(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def topk_mean(values: np.ndarray, k: int, largest: bool) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan")
    k = max(1, min(int(k), int(values.size)))
    if largest:
        picked = np.partition(values, values.size - k)[-k:]
    else:
        picked = np.partition(values, k - 1)[:k]
    return float(np.mean(picked))


def dominant_label(values: np.ndarray) -> tuple[str, float]:
    if values.size == 0:
        return "", float("nan")
    labels, counts = np.unique(values.astype(str), return_counts=True)
    best_idx = int(np.argmax(counts))
    return str(labels[best_idx]), float(counts[best_idx] / counts.sum())


def main() -> None:
    args = parse_args()
    input_dir = args.embeddings_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.hdf5"))
    if args.limit_subjects is not None:
        files = files[: args.limit_subjects]
    if not files:
        raise SystemExit(f"No HDF5 files found in {input_dir}")

    window_numeric_fields = [
        "subject",
        "window_idx",
        "consensus_norm",
        "loo_mean_norm_mean",
        "alignment_mean",
        "jackknife_instability",
        "split_score",
        "outlier_score",
    ]
    window_label_fields = ["best_split_partition", "outlier_modality"]
    subject_fields = [
        "subject",
        "n_windows",
        "consensus_norm_mean",
        "consensus_norm_std",
        "consensus_norm_p05",
        "loo_mean_norm_mean",
        "loo_mean_norm_std",
        "alignment_mean",
        "alignment_std",
        "alignment_mean_p05",
        "alignment_bottomk_mean",
        "jackknife_instability_mean",
        "jackknife_instability_std",
        "jackknife_instability_p95",
        "jackknife_topk_mean",
        "split_score_mean",
        "split_score_std",
        "split_score_p95",
        "split_score_topk_mean",
        "outlier_score_mean",
        "outlier_score_std",
        "outlier_score_p95",
        "outlier_score_topk_mean",
        "consensus_low_fraction",
        "alignment_low_fraction",
        "jackknife_high_fraction",
        "split_high_fraction",
        "outlier_high_fraction",
        "unstable_window_fraction",
        "dominant_best_split_partition",
        "dominant_best_split_fraction",
        "dominant_outlier_modality",
        "dominant_outlier_fraction",
    ]

    for modality in args.modalities:
        window_numeric_fields.extend([f"alignment_{modality}", f"loo_norm_{modality}"])
        subject_fields.append(f"alignment_{modality}")

    window_csv = output_dir / "window_metrics.csv"
    subject_csv = output_dir / "subject_metrics.csv"

    subject_rows: dict[str, dict[str, float | int | str]] = {}
    subject_order: list[str] = []
    subject_cache: dict[str, dict[str, np.ndarray]] = {}

    all_alignment = []
    all_consensus = []
    all_loo_norm = []
    all_instability = []
    all_split = []
    all_outlier = []
    modality_alignment_dataset: dict[str, list[float]] = {modality: [] for modality in args.modalities}
    split_partition_counts: dict[str, int] = {}
    outlier_modality_counts: dict[str, int] = {modality: 0 for modality in args.modalities}
    n_windows_total = 0

    with window_csv.open("w", newline="", encoding="utf-8") as window_handle:
        window_fields = window_numeric_fields + window_label_fields
        window_writer = csv.DictWriter(window_handle, fieldnames=window_fields)
        window_writer.writeheader()

        for path in files:
            subject = path.stem
            embeddings = load_subject_embeddings(path, args.modalities)
            metrics = compute_metrics(embeddings, args.modalities, args.jackknife_scale)
            n_windows = int(embeddings.shape[1])
            n_windows_total += n_windows

            for window_idx in range(n_windows):
                row: dict[str, float | int | str] = {
                    "subject": subject,
                    "window_idx": window_idx,
                }
                for field in window_numeric_fields[2:]:
                    value = float(metrics[field][window_idx])
                    row[field] = value
                for field in window_label_fields:
                    row[field] = str(metrics[field][window_idx])
                window_writer.writerow(row)

            subject_row: dict[str, float | int | str] = {
                "subject": subject,
                "n_windows": n_windows,
            }
            subject_stats = {
                "consensus_norm": ("consensus_norm_mean", "consensus_norm_std"),
                "loo_mean_norm_mean": ("loo_mean_norm_mean", "loo_mean_norm_std"),
                "alignment_mean": ("alignment_mean", "alignment_std"),
                "jackknife_instability": ("jackknife_instability_mean", "jackknife_instability_std"),
                "split_score": ("split_score_mean", "split_score_std"),
                "outlier_score": ("outlier_score_mean", "outlier_score_std"),
            }
            for metric_key, (mean_key, std_key) in subject_stats.items():
                mean_value, std_value = summarize(metrics[metric_key])
                subject_row[mean_key] = mean_value
                subject_row[std_key] = std_value

            subject_row["consensus_norm_p05"] = quantile(metrics["consensus_norm"], args.low_quantile)
            subject_row["alignment_mean_p05"] = quantile(metrics["alignment_mean"], args.low_quantile)
            subject_row["alignment_bottomk_mean"] = topk_mean(metrics["alignment_mean"], args.subject_top_k, largest=False)
            subject_row["jackknife_instability_p95"] = quantile(metrics["jackknife_instability"], args.high_quantile)
            subject_row["jackknife_topk_mean"] = topk_mean(metrics["jackknife_instability"], args.subject_top_k, largest=True)
            subject_row["split_score_p95"] = quantile(metrics["split_score"], args.high_quantile)
            subject_row["split_score_topk_mean"] = topk_mean(metrics["split_score"], args.subject_top_k, largest=True)
            subject_row["outlier_score_p95"] = quantile(metrics["outlier_score"], args.high_quantile)
            subject_row["outlier_score_topk_mean"] = topk_mean(metrics["outlier_score"], args.subject_top_k, largest=True)

            best_partition, best_partition_fraction = dominant_label(metrics["best_split_partition"])
            outlier_label, outlier_fraction = dominant_label(metrics["outlier_modality"])
            subject_row["dominant_best_split_partition"] = best_partition
            subject_row["dominant_best_split_fraction"] = best_partition_fraction
            subject_row["dominant_outlier_modality"] = outlier_label
            subject_row["dominant_outlier_fraction"] = outlier_fraction

            for modality in args.modalities:
                modality_mean = float(np.mean(metrics[f"alignment_{modality}"]))
                subject_row[f"alignment_{modality}"] = modality_mean
                modality_alignment_dataset[modality].extend(metrics[f"alignment_{modality}"].tolist())

            subject_rows[subject] = subject_row
            subject_order.append(subject)
            subject_cache[subject] = {
                "consensus_norm": np.asarray(metrics["consensus_norm"], dtype=np.float32),
                "alignment_mean": np.asarray(metrics["alignment_mean"], dtype=np.float32),
                "jackknife_instability": np.asarray(metrics["jackknife_instability"], dtype=np.float32),
                "split_score": np.asarray(metrics["split_score"], dtype=np.float32),
                "outlier_score": np.asarray(metrics["outlier_score"], dtype=np.float32),
            }

            all_alignment.extend(metrics["alignment_mean"].tolist())
            all_consensus.extend(metrics["consensus_norm"].tolist())
            all_loo_norm.extend(metrics["loo_mean_norm_mean"].tolist())
            all_instability.extend(metrics["jackknife_instability"].tolist())
            all_split.extend(metrics["split_score"].tolist())
            all_outlier.extend(metrics["outlier_score"].tolist())

            for label in np.asarray(metrics["best_split_partition"]).astype(str):
                split_partition_counts[label] = split_partition_counts.get(label, 0) + 1
            for label in np.asarray(metrics["outlier_modality"]).astype(str):
                outlier_modality_counts[label] = outlier_modality_counts.get(label, 0) + 1

    thresholds = {
        "consensus_norm_low": quantile(np.asarray(all_consensus, dtype=np.float32), args.low_quantile),
        "alignment_mean_low": quantile(np.asarray(all_alignment, dtype=np.float32), args.low_quantile),
        "jackknife_instability_high": quantile(np.asarray(all_instability, dtype=np.float32), args.high_quantile),
        "split_score_high": quantile(np.asarray(all_split, dtype=np.float32), args.high_quantile),
        "outlier_score_high": quantile(np.asarray(all_outlier, dtype=np.float32), args.high_quantile),
    }

    for subject in subject_order:
        row = subject_rows[subject]
        cached = subject_cache[subject]

        consensus_low = cached["consensus_norm"] <= thresholds["consensus_norm_low"]
        alignment_low = cached["alignment_mean"] <= thresholds["alignment_mean_low"]
        jackknife_high = cached["jackknife_instability"] >= thresholds["jackknife_instability_high"]
        split_high = cached["split_score"] >= thresholds["split_score_high"]
        outlier_high = cached["outlier_score"] >= thresholds["outlier_score_high"]
        unstable = alignment_low | jackknife_high | split_high | outlier_high

        row["consensus_low_fraction"] = float(consensus_low.mean())
        row["alignment_low_fraction"] = float(alignment_low.mean())
        row["jackknife_high_fraction"] = float(jackknife_high.mean())
        row["split_high_fraction"] = float(split_high.mean())
        row["outlier_high_fraction"] = float(outlier_high.mean())
        row["unstable_window_fraction"] = float(unstable.mean())

    with subject_csv.open("w", newline="", encoding="utf-8") as subject_handle:
        subject_writer = csv.DictWriter(subject_handle, fieldnames=subject_fields)
        subject_writer.writeheader()
        subject_writer.writerows([subject_rows[subject] for subject in subject_order])

    summary = {
        "base": str(input_dir),
        "modalities": args.modalities,
        "n_subjects": len(files),
        "n_windows_total": n_windows_total,
        "jackknife_scale": args.jackknife_scale,
        "low_quantile": args.low_quantile,
        "high_quantile": args.high_quantile,
        "subject_top_k": args.subject_top_k,
        "alignment_mean_dataset": float(np.mean(all_alignment)),
        "consensus_norm_mean_dataset": float(np.mean(all_consensus)),
        "loo_mean_norm_mean_dataset": float(np.mean(all_loo_norm)),
        "jackknife_instability_mean_dataset": float(np.mean(all_instability)),
        "split_score_mean_dataset": float(np.mean(all_split)),
        "outlier_score_mean_dataset": float(np.mean(all_outlier)),
        "thresholds": thresholds,
        "modality_alignment_dataset": {
            modality: float(np.mean(values)) for modality, values in modality_alignment_dataset.items()
        },
        "best_split_partition_fraction_dataset": {
            label: count / n_windows_total for label, count in sorted(split_partition_counts.items())
        },
        "outlier_modality_fraction_dataset": {
            label: count / n_windows_total for label, count in sorted(outlier_modality_counts.items())
        },
        "subject_csv": str(subject_csv),
        "window_csv": str(window_csv),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
