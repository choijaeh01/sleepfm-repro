from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
import yaml


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sigma = s.std()
    if pd.isna(sigma) or sigma == 0:
        return s.fillna(mu if pd.notna(mu) else 0.0) * 0.0
    return (s.fillna(mu) - mu) / sigma


def detect_edf_files(edf_root: Path) -> dict[str, str]:
    files = {}
    for p in edf_root.rglob("*.edf"):
        stem = p.stem
        parts = stem.split("-")
        if len(parts) >= 2 and parts[0].lower().startswith("shhs"):
            study_id = parts[-1]
        else:
            study_id = stem
        files[study_id] = str(p)
    for p in edf_root.rglob("*.EDF"):
        stem = p.stem
        parts = stem.split("-")
        if len(parts) >= 2 and parts[0].lower().startswith("shhs"):
            study_id = parts[-1]
        else:
            study_id = stem
        files[study_id] = str(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shhs-summary", required=True)
    ap.add_argument("--shhs-harmonized", required=True)
    ap.add_argument("--edf-root", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    random.seed(args.seed)

    summary = pd.read_csv(args.shhs_summary, low_memory=False)
    harmonized = pd.read_csv(args.shhs_harmonized, low_memory=False)
    with open(args.targets, "r", encoding="utf-8") as f:
        target_cfg = yaml.safe_load(f)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_json = Path(args.split_json)
    split_json.parent.mkdir(parents=True, exist_ok=True)
    processed_root = Path(args.processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    summary["nsrrid"] = summary["nsrrid"].astype(str)
    harmonized["nsrrid"] = harmonized["nsrrid"].astype(str)
    harmonized_v1 = harmonized[harmonized["visitnumber"].astype(str) == "1"].copy()

    df = summary.merge(
        harmonized_v1[["nsrrid", "nsrr_age", "nsrr_bmi", "nsrr_sex", "nsrr_race"]],
        on="nsrrid",
        how="left",
    )

    edf_map = detect_edf_files(Path(args.edf_root))
    df = df[df["nsrrid"].isin(edf_map.keys())].copy()

    target_names = []
    is_event_out = pd.DataFrame({"Study ID": df["nsrrid"]})
    time_out = pd.DataFrame({"Study ID": df["nsrrid"]})

    for target in target_cfg["targets"]:
        name = target["name"]
        event_col = target["event_col"]
        time_col = target["time_col"]
        target_names.append(name)

        event_series = df[event_col].fillna(0)
        event_series = event_series.astype(str).str.strip()
        is_event = (~event_series.isin(["", "0", "0.0", "False", "false"])).astype(int)

        time_series = pd.to_numeric(df[time_col], errors="coerce")
        # censdate is also coded in days; fallback to censdate if event date missing
        cens = pd.to_numeric(df.get("censdate"), errors="coerce")
        tte = time_series.where(is_event.astype(bool), cens)
        tte = tte.fillna(cens)

        is_event_out[name] = is_event.astype(int)
        time_out[name] = tte.astype(float)

    demo = pd.DataFrame({"Study ID": df["nsrrid"]})
    demo["age_z"] = normalize_series(df["nsrr_age"])
    demo["bmi_z"] = normalize_series(df["nsrr_bmi"])
    demo["male"] = (df["nsrr_sex"].astype(str).str.lower() == "male").astype(int)
    demo["race_white"] = (df["nsrr_race"].astype(str).str.lower() == "white").astype(int)

    is_event_out.to_csv(output_root / "is_event.csv", index=False)
    time_out.to_csv(output_root / "time_to_event.csv", index=False)
    demo.to_csv(output_root / "demo_labels.csv", index=False)

    study_ids = sorted(df["nsrrid"].tolist())
    random.shuffle(study_ids)
    n = len(study_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_ids = study_ids[:n_train]
    val_ids = study_ids[n_train:n_train + n_val]
    test_ids = study_ids[n_train + n_val:]

    def to_processed_paths(ids: list[str]) -> list[str]:
        return [str(processed_root / f"{sid}.hdf5") for sid in ids]

    split = {
        "train": to_processed_paths(train_ids),
        "validation": to_processed_paths(val_ids),
        "test": to_processed_paths(test_ids),
    }
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    summary_out = {
        "n_subjects": len(study_ids),
        "n_train": len(train_ids),
        "n_validation": len(val_ids),
        "n_test": len(test_ids),
        "targets": {},
    }
    for name in target_names:
        summary_out["targets"][name] = {
            "events": int(is_event_out[name].sum()),
            "missing_time_to_event": int(time_out[name].isna().sum()),
        }

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
