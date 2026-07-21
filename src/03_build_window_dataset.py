#!/usr/bin/env python3
"""Build non-overlapping, chronological CICIoV2024 CAN-message windows.

Optional window-size experiment:
    python src/03_build_window_dataset.py --window-size 50

The original six CSV files are read only and never changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


FILES = (
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
)

RAW_FEATURES = (
    "ID",
    "DATA_0",
    "DATA_1",
    "DATA_2",
    "DATA_3",
    "DATA_4",
    "DATA_5",
    "DATA_6",
    "DATA_7",
)

TARGETS = ("label", "specific_class")


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent.parent, script_path.parent, Path.cwd()):
        if (candidate / "data" / "ciciov2024" / "decimal").is_dir():
            return candidate
    print("ERROR: Could not find data/ciciov2024/decimal.")
    sys.exit(1)


def entropy_from_counts(counts: np.ndarray) -> float:
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_window_features(values: np.ndarray, window_size: int) -> pd.DataFrame:
    """Convert [windows, messages, ID+8 bytes] into one feature row per window."""

    window_count = values.shape[0]
    ids = values[:, :, 0]
    payload = values[:, :, 1:].astype(np.float64, copy=False)

    result: dict[str, np.ndarray] = {}
    unique_ids = np.empty(window_count, dtype=np.int32)
    id_entropy = np.empty(window_count, dtype=np.float64)
    dominant_id_fraction = np.empty(window_count, dtype=np.float64)
    unique_frames = np.empty(window_count, dtype=np.int32)

    for index in range(window_count):
        _, id_counts = np.unique(ids[index], return_counts=True)
        unique_ids[index] = len(id_counts)
        id_entropy[index] = entropy_from_counts(id_counts)
        dominant_id_fraction[index] = id_counts.max() / window_size
        unique_frames[index] = len(np.unique(values[index], axis=0))

    result["id_unique_count"] = unique_ids
    result["id_entropy"] = id_entropy
    result["dominant_id_fraction"] = dominant_id_fraction
    result["frame_unique_count"] = unique_frames
    result["frame_unique_fraction"] = unique_frames / window_size
    result["id_change_rate"] = (ids[:, 1:] != ids[:, :-1]).mean(axis=1)
    result["consecutive_frame_repeat_rate"] = np.all(
        values[:, 1:, :] == values[:, :-1, :], axis=2
    ).mean(axis=1)

    payload_mean = payload.mean(axis=1)
    payload_std = payload.std(axis=1)
    payload_min = payload.min(axis=1)
    payload_max = payload.max(axis=1)
    zero_fraction = (payload == 0).mean(axis=1)
    ff_fraction = (payload == 255).mean(axis=1)
    mean_absolute_change = np.abs(np.diff(payload, axis=1)).mean(axis=1)

    for byte_index in range(8):
        prefix = f"data_{byte_index}"
        result[f"{prefix}_mean"] = payload_mean[:, byte_index]
        result[f"{prefix}_std"] = payload_std[:, byte_index]
        result[f"{prefix}_min"] = payload_min[:, byte_index]
        result[f"{prefix}_max"] = payload_max[:, byte_index]
        result[f"{prefix}_zero_fraction"] = zero_fraction[:, byte_index]
        result[f"{prefix}_ff_fraction"] = ff_fraction[:, byte_index]
        result[f"{prefix}_mean_abs_change"] = mean_absolute_change[:, byte_index]

    return pd.DataFrame(result)


def chronological_splits(window_count: int) -> np.ndarray:
    """Make a 70/15/15 split while preserving message order."""

    train_end = int(np.floor(window_count * 0.70))
    validation_end = int(np.floor(window_count * 0.85))
    if train_end < 1 or validation_end <= train_end or validation_end >= window_count:
        raise ValueError(f"Need at least 7 windows for a 70/15/15 split; got {window_count}")

    split = np.empty(window_count, dtype=object)
    split[:train_end] = "train"
    split[train_end:validation_end] = "validation"
    split[validation_end:] = "test"
    return split


def main(window_size: int) -> None:
    if window_size < 2:
        raise ValueError("window size must be at least 2")

    root = find_project_root()
    data_dir = root / "data" / "ciciov2024" / "decimal"
    results_dir = root / "results"
    processed_dir = root / "data" / "processed"
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FILES if not (data_dir / name).is_file()]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        sys.exit(1)

    print("CICIoV2024 Non-Overlapping Window Builder")
    print(f"Window size: {window_size} consecutive messages")
    print("Split: first 70% train, next 15% validation, final 15% test per file\n")

    all_windows: list[pd.DataFrame] = []
    discarded_rows = 0

    for file_number, file_name in enumerate(FILES, start=1):
        print(f"[{file_number}/{len(FILES)}] Processing {file_name} ...")
        path = data_dir / file_name
        frame = pd.read_csv(path, usecols=list(RAW_FEATURES + TARGETS), low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()

        for target in TARGETS:
            frame[target] = frame[target].astype("string").str.strip()
        frame["label"] = frame["label"].str.upper()

        binary_values = frame["label"].dropna().unique()
        class_values = frame["specific_class"].dropna().unique()
        if len(binary_values) != 1 or len(class_values) != 1:
            raise ValueError(
                f"{file_name} should contain one label/class, got "
                f"labels={binary_values.tolist()}, classes={class_values.tolist()}"
            )

        binary_label = str(binary_values[0])
        specific_class = str(class_values[0])
        binary_target = 0 if binary_label == "BENIGN" else 1

        usable_rows = len(frame) - (len(frame) % window_size)
        removed = len(frame) - usable_rows
        discarded_rows += removed
        if usable_rows == 0:
            raise ValueError(f"{file_name} has fewer rows than window size {window_size}")

        raw = frame.loc[: usable_rows - 1, list(RAW_FEATURES)].to_numpy()
        raw = raw.reshape(-1, window_size, len(RAW_FEATURES))
        window_count = raw.shape[0]

        features = extract_window_features(raw, window_size)
        features.insert(0, "multiclass_target", specific_class)
        features.insert(0, "binary_target", binary_target)
        features.insert(0, "split", chronological_splits(window_count))
        features.insert(0, "end_row", (np.arange(window_count) + 1) * window_size - 1)
        features.insert(0, "start_row", np.arange(window_count) * window_size)
        features.insert(0, "window_index", np.arange(window_count))
        features.insert(0, "source_file", file_name)
        all_windows.append(features)

        counts = features["split"].value_counts().reindex(
            ["train", "validation", "test"], fill_value=0
        )
        print(
            f"    class={specific_class}, windows={window_count:,}, "
            f"train={counts['train']:,}, validation={counts['validation']:,}, "
            f"test={counts['test']:,}, discarded raw rows={removed}"
        )
        del frame, raw, features

    dataset = pd.concat(all_windows, ignore_index=True)
    del all_windows

    prefix = f"ciciov2024_windows_w{window_size}"
    all_path = processed_dir / f"{prefix}_all.csv"
    dataset.to_csv(all_path, index=False)

    split_paths: dict[str, Path] = {}
    for split_name in ("train", "validation", "test"):
        split_path = processed_dir / f"{prefix}_{split_name}.csv"
        dataset.loc[dataset["split"] == split_name].to_csv(split_path, index=False)
        split_paths[split_name] = split_path

    summary = (
        dataset.groupby(["split", "multiclass_target"], observed=True)
        .size()
        .rename("windows")
        .reset_index()
    )
    summary["window_size"] = window_size
    summary_path = results_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Verify that raw row ranges never overlap within a source file.
    overlap_errors = 0
    for _, group in dataset.groupby("source_file", observed=True):
        ordered = group.sort_values("window_index")
        overlap_errors += int((ordered["start_row"].iloc[1:].to_numpy() <= ordered["end_row"].iloc[:-1].to_numpy()).sum())

    expected_classes = set(dataset["multiclass_target"].unique())
    missing_by_split: dict[str, list[str]] = {}
    for split_name in ("train", "validation", "test"):
        present = set(dataset.loc[dataset["split"] == split_name, "multiclass_target"].unique())
        missing_by_split[split_name] = sorted(expected_classes - present)

    print("\n" + "=" * 78)
    print(f"Total windows:             {len(dataset):,}")
    print(f"Engineered model features: {len(dataset.columns) - 7}")
    print(f"Discarded raw rows:        {discarded_rows:,}")
    print(f"Overlapping window pairs:  {overlap_errors} (must be 0)")
    print("\nWindow distribution:")
    print(summary.to_string(index=False))

    for split_name, missing_classes in missing_by_split.items():
        if missing_classes:
            print(f"WARNING: {split_name} is missing classes: {missing_classes}")

    print("\nSaved processed datasets:")
    print(f"  {all_path}")
    for split_name in ("train", "validation", "test"):
        print(f"  {split_paths[split_name]}")
    print(f"Summary: {summary_path}")
    print("\nWindow dataset construction completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="number of consecutive CAN messages in each non-overlapping window (default: 100)",
    )
    arguments = parser.parse_args()
    main(arguments.window_size)
