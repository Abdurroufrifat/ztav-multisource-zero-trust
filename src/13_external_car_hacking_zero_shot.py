#!/usr/bin/env python3
"""Zero-shot CICIoV2024-model evaluation on the HCRL Car-Hacking Dataset.

The four attack CSVs are headerless and contain:

    timestamp, hex CAN ID, DLC, eight hex payload bytes, R/T flag

The attack-free text capture uses a different ``Timestamp/ID/DLC`` layout.
Both formats are streamed and converted into non-overlapping 100-frame windows
with the exact 63 features used by the CICIoV2024 model. A window is positive
if it contains at least one injected (T) frame; attack-frame count/fraction are
retained for stratified analysis.

The existing group-disjoint CICIoV2024 logistic pipeline and its original
validation-selected threshold are applied without fitting, calibration, or
threshold selection on HCRL data. This is a genuine cross-dataset zero-shot
test of the CAN subsystem. It does not externally validate the simulated
GNSS/V2X/identity sources.

Run from D:\\ztav_project:

    .\\.venv\\Scripts\\python.exe src\\13_external_car_hacking_zero_shot.py

The source files are read only. Processing is chunked for a 16 GB computer.
This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


WINDOW_SIZE = 100
CHUNK_ROWS = 250_000
PREDICTION_BATCH_ROWS = 20_000
CSV_COLUMNS = (
    "timestamp",
    "can_id",
    "dlc",
    "data_0",
    "data_1",
    "data_2",
    "data_3",
    "data_4",
    "data_5",
    "data_6",
    "data_7",
    "flag",
)
RAW_COLUMNS = ("can_id",) + tuple(f"data_{index}" for index in range(8))
ATTACK_FILES: tuple[tuple[str, str], ...] = (
    ("DoS_dataset.csv", "DoS"),
    ("Fuzzy_dataset.csv", "FUZZY"),
    ("gear_dataset.csv", "GEAR_SPOOFING"),
    ("RPM_dataset.csv", "RPM_SPOOFING"),
)
NON_MODEL_COLUMNS = {
    "source_file",
    "window_index",
    "start_row",
    "end_row",
    "split",
    "chronological_split",
    "feature_signature",
    "binary_target",
    "multiclass_target",
}
NORMAL_LINE = re.compile(
    r"Timestamp:\s*([0-9.]+).*?ID:\s*([0-9A-Fa-f]+).*?"
    r"DLC:\s*(\d+)\s+(.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External zero-shot evaluation on HCRL Car-Hacking data."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument(
        "--skip-normal-capture",
        action="store_true",
        help="Skip normal_run_data.txt (not recommended for the final experiment).",
    )
    args = parser.parse_args()
    if args.chunk_rows < WINDOW_SIZE:
        parser.error(f"--chunk-rows must be at least {WINDOW_SIZE}")
    return args


def load_script(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_hex_values(values: Iterable[object]) -> np.ndarray:
    cleaned = (str(value).strip() if not pd.isna(value) else "0" for value in values)
    return np.fromiter(
        (int(value, 16) if value else 0 for value in cleaned),
        dtype=np.int64,
    )


def chunk_to_raw(chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Parse variable-DLC rows and find the trailing R/T marker dynamically.

    Headerless HCRL rows contain only ``DLC`` payload values.  Consequently,
    pandas places the R/T marker in ``data_<DLC>`` for short payloads and in
    the final ``flag`` column for eight-byte payloads.  Treating the twelfth
    field as an unconditional flag would incorrectly reject valid short rows.
    """

    payload_columns = [f"data_{index}" for index in range(8)]
    tail_columns = payload_columns + ["flag"]
    tail = (
        chunk[tail_columns]
        .fillna("")
        .astype("string")
        .apply(lambda column: column.str.strip().str.upper())
        .to_numpy(dtype=str)
    )
    marker_mask = (tail == "R") | (tail == "T")
    marker_counts = marker_mask.sum(axis=1)
    if np.any(marker_counts != 1):
        bad_row = int(np.flatnonzero(marker_counts != 1)[0])
        raise ValueError(
            "Expected exactly one HCRL R/T marker per row; "
            f"chunk row {bad_row} has tail={tail[bad_row].tolist()}"
        )
    marker_indices = marker_mask.argmax(axis=1)
    flags = tail[np.arange(len(tail)), marker_indices]

    dlc = pd.to_numeric(chunk["dlc"], errors="raise").to_numpy(dtype=np.int64)
    if np.any((dlc < 0) | (dlc > 8)):
        raise ValueError("HCRL DLC is outside [0, 8]")
    if np.any(marker_indices != dlc):
        bad_row = int(np.flatnonzero(marker_indices != dlc)[0])
        raise ValueError(
            "HCRL payload length does not match DLC; "
            f"chunk row {bad_row} has DLC={dlc[bad_row]}, "
            f"marker position={marker_indices[bad_row]}"
        )

    raw = np.zeros((len(chunk), 9), dtype=np.int64)
    raw[:, 0] = parse_hex_values(chunk["can_id"].to_numpy())
    for byte_index in range(8):
        present = byte_index < dlc
        values = np.where(present, tail[:, byte_index], "0")
        raw[:, byte_index + 1] = parse_hex_values(values)
    if np.any(raw[:, 1:] > 255) or np.any(raw < 0):
        raise ValueError("Parsed CAN values are outside their valid range")
    return raw, flags


def window_frame(
    raw: np.ndarray,
    flags: np.ndarray,
    source_file: str,
    source_class: str,
    first_window_index: int,
    feature_builder: ModuleType,
) -> pd.DataFrame:
    if len(raw) % WINDOW_SIZE or len(raw) != len(flags):
        raise ValueError("Window batch is not aligned")
    reshaped = raw.reshape(-1, WINDOW_SIZE, raw.shape[1])
    flag_windows = flags.reshape(-1, WINDOW_SIZE)
    attack_counts = (flag_windows == "T").sum(axis=1)
    window_count = len(reshaped)
    indices = np.arange(first_window_index, first_window_index + window_count)
    features = feature_builder.extract_window_features(reshaped, WINDOW_SIZE)
    features.insert(0, "attack_frame_fraction", attack_counts / WINDOW_SIZE)
    features.insert(0, "attack_frame_count", attack_counts)
    features.insert(0, "binary_target", (attack_counts > 0).astype(np.uint8))
    features.insert(0, "source_capture_class", source_class)
    features.insert(0, "end_row", (indices + 1) * WINDOW_SIZE - 1)
    features.insert(0, "start_row", indices * WINDOW_SIZE)
    features.insert(0, "window_index", indices)
    features.insert(0, "source_file", source_file)
    return features


def process_attack_csv(
    path: Path,
    source_class: str,
    feature_builder: ModuleType,
    chunk_rows: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\nProcessing {path.name} ...")
    started = time.perf_counter()
    feature_batches: list[pd.DataFrame] = []
    carry_raw = np.empty((0, 9), dtype=np.int64)
    carry_flags = np.empty(0, dtype="U1")
    total_rows = 0
    attack_frames = 0
    window_index = 0

    reader = pd.read_csv(
        path,
        header=None,
        names=list(CSV_COLUMNS),
        dtype="string",
        chunksize=chunk_rows,
        skipinitialspace=True,
        on_bad_lines="error",
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        raw, flags = chunk_to_raw(chunk)
        total_rows += len(raw)
        attack_frames += int((flags == "T").sum())
        if len(carry_raw):
            raw = np.concatenate((carry_raw, raw), axis=0)
            flags = np.concatenate((carry_flags, flags))

        usable = len(raw) - len(raw) % WINDOW_SIZE
        if usable:
            batch = window_frame(
                raw[:usable],
                flags[:usable],
                path.name,
                source_class,
                window_index,
                feature_builder,
            )
            feature_batches.append(batch)
            window_index += len(batch)
        carry_raw = raw[usable:].copy()
        carry_flags = flags[usable:].copy()
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"  chunks={chunk_number}, raw rows={total_rows:,}, "
                f"windows={window_index:,}"
            )

    windows = pd.concat(feature_batches, ignore_index=True)
    attack_windows = int(windows["binary_target"].sum())
    summary = {
        "source_file": path.name,
        "source_capture_class": source_class,
        "input_lines": total_rows,
        "skipped_malformed_lines": 0,
        "skipped_line_fraction": 0.0,
        "skipped_line_examples": "",
        "raw_rows": total_rows,
        "normal_frames": total_rows - attack_frames,
        "attack_frames": attack_frames,
        "windows": len(windows),
        "benign_windows": len(windows) - attack_windows,
        "attack_windows": attack_windows,
        "mixed_windows": int(
            ((windows["attack_frame_count"] > 0) & (windows["attack_frame_count"] < WINDOW_SIZE)).sum()
        ),
        "discarded_tail_rows": len(carry_raw),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
    print(
        f"  completed rows={total_rows:,}, windows={len(windows):,}, "
        f"attack windows={attack_windows:,}, discarded={len(carry_raw)}"
    )
    return windows, summary


def parse_normal_line(line: str, line_number: int) -> list[int]:
    match = NORMAL_LINE.search(line)
    if not match:
        raise ValueError(f"Cannot parse normal capture line {line_number}: {line[:120]!r}")
    can_id = int(match.group(2), 16)
    dlc = int(match.group(3))
    tokens = re.findall(r"\b[0-9A-Fa-f]{1,2}\b", match.group(4))
    payload = [int(token, 16) for token in tokens[:dlc]]
    payload = (payload + [0] * 8)[:8]
    return [can_id, *payload]


def process_normal_capture(
    path: Path,
    feature_builder: ModuleType,
    batch_windows: int = 2_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\nProcessing {path.name} ...")
    started = time.perf_counter()
    feature_batches: list[pd.DataFrame] = []
    current_window: list[list[int]] = []
    raw_windows: list[list[list[int]]] = []
    total_rows = 0
    input_lines = 0
    skipped_malformed_lines = 0
    skipped_examples: list[str] = []
    window_index = 0

    def flush() -> None:
        nonlocal raw_windows, window_index
        if not raw_windows:
            return
        raw = np.asarray(raw_windows, dtype=np.int64).reshape(-1, 9)
        flags = np.full(len(raw), "R", dtype="U1")
        batch = window_frame(
            raw,
            flags,
            path.name,
            "ATTACK_FREE",
            window_index,
            feature_builder,
        )
        feature_batches.append(batch)
        window_index += len(batch)
        raw_windows = []

    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_lines += 1
            try:
                parsed = parse_normal_line(line, line_number)
            except ValueError:
                skipped_malformed_lines += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append(f"line {line_number}: {line.strip()[:100]}")
                continue
            current_window.append(parsed)
            total_rows += 1
            if len(current_window) == WINDOW_SIZE:
                raw_windows.append(current_window)
                current_window = []
                if len(raw_windows) >= batch_windows:
                    flush()
                    if window_index % 10_000 == 0:
                        print(f"  raw rows={total_rows:,}, windows={window_index:,}")
    flush()
    skipped_fraction = (
        skipped_malformed_lines / input_lines if input_lines else 0.0
    )
    if skipped_fraction > 0.01:
        raise ValueError(
            "More than 1% of non-empty normal-capture lines were malformed: "
            f"{skipped_malformed_lines:,}/{input_lines:,} ({skipped_fraction:.2%}). "
            f"Examples: {skipped_examples}"
        )
    windows = pd.concat(feature_batches, ignore_index=True)
    summary = {
        "source_file": path.name,
        "source_capture_class": "ATTACK_FREE",
        "input_lines": input_lines,
        "skipped_malformed_lines": skipped_malformed_lines,
        "skipped_line_fraction": skipped_fraction,
        "skipped_line_examples": " | ".join(skipped_examples),
        "raw_rows": total_rows,
        "normal_frames": total_rows,
        "attack_frames": 0,
        "windows": len(windows),
        "benign_windows": len(windows),
        "attack_windows": 0,
        "mixed_windows": 0,
        "discarded_tail_rows": len(current_window),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
    print(
        f"  completed rows={total_rows:,}, windows={len(windows):,}, "
        f"discarded={len(current_window)}, malformed lines skipped="
        f"{skipped_malformed_lines:,} ({skipped_fraction:.4%})"
    )
    return windows, summary


def locate_normal_capture(data_dir: Path) -> Path:
    candidates = list(data_dir.rglob("normal_run_data.txt"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find normal_run_data.txt below {data_dir}")
    # Prefer an extracted nested file rather than any archive-like sibling.
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[-1]


def load_model_assets(
    project_root: Path,
) -> tuple[Any, float, list[str], pd.DataFrame]:
    model_path = (
        project_root
        / "models"
        / "group_disjoint_w100"
        / "group_disjoint_logistic_regression.joblib"
    )
    threshold_path = (
        project_root
        / "results"
        / "group_disjoint_w100"
        / "group_disjoint_thresholds.json"
    )
    training_path = (
        project_root
        / "data"
        / "processed"
        / "ciciov2024_windows_w100_group_disjoint_train.csv"
    )
    for path in (model_path, threshold_path, training_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing CICIoV2024 asset: {path}")
    model = joblib.load(model_path)
    with threshold_path.open(encoding="utf-8") as handle:
        threshold = float(json.load(handle)["Logistic Regression"])
    training = pd.read_csv(training_path)
    feature_names = [
        column for column in training.columns if column not in NON_MODEL_COLUMNS
    ]
    return model, threshold, feature_names, training[feature_names]


def predict_in_batches(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(features), PREDICTION_BATCH_ROWS):
        end = min(start + PREDICTION_BATCH_ROWS, len(features))
        values = features.iloc[start:end].to_numpy(dtype=np.float64)
        probabilities.append(model.predict_proba(values)[:, 1])
    return np.concatenate(probabilities)


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    frame: pd.DataFrame,
    threshold: float,
    scope: str,
) -> dict[str, object]:
    truth = frame["binary_target"].to_numpy(dtype=np.uint8)
    probability = frame["attack_probability"].to_numpy(dtype=np.float64)
    prediction = (probability >= threshold).astype(np.uint8)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    both_classes = len(np.unique(truth)) == 2
    return {
        "scope": scope,
        "threshold_source": "CICIoV2024 validation",
        "threshold": threshold,
        "windows": len(frame),
        "benign_windows": int((truth == 0).sum()),
        "attack_windows": int((truth == 1).sum()),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": (
            balanced_accuracy_score(truth, prediction) if both_classes else math.nan
        ),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "mcc": matthews_corrcoef(truth, prediction) if both_classes else math.nan,
        "roc_auc": roc_auc_score(truth, probability) if both_classes else math.nan,
        "pr_auc": average_precision_score(truth, probability) if both_classes else math.nan,
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "probability_mean_benign": (
            float(probability[truth == 0].mean()) if np.any(truth == 0) else math.nan
        ),
        "probability_mean_attack": (
            float(probability[truth == 1].mean()) if np.any(truth == 1) else math.nan
        ),
    }


def attack_density_rows(frame: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
    attacked = frame[frame["binary_target"] == 1].copy()
    bins = (
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-20", 6, 20),
        ("21-50", 21, 50),
        ("51-99", 51, 99),
        ("100", 100, 100),
    )
    output: list[dict[str, object]] = []
    for label, lower, upper in bins:
        group = attacked[
            attacked["attack_frame_count"].between(lower, upper, inclusive="both")
        ]
        probability = group["attack_probability"].to_numpy(dtype=float)
        detected = probability >= threshold
        output.append(
            {
                "attack_frames_per_window": label,
                "windows": len(group),
                "recall": float(detected.mean()) if len(group) else math.nan,
                "mean_attack_probability": (
                    float(probability.mean()) if len(group) else math.nan
                ),
            }
        )
    return output


def feature_shift_rows(
    external_features: pd.DataFrame,
    training_features: pd.DataFrame,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for column in external_features.columns:
        train = training_features[column].to_numpy(dtype=float)
        external = external_features[column].to_numpy(dtype=float)
        train_mean = float(train.mean())
        train_std = float(train.std())
        outside = (external < train.min()) | (external > train.max())
        output.append(
            {
                "feature": column,
                "ciciov_train_mean": train_mean,
                "ciciov_train_std": train_std,
                "ciciov_train_min": float(train.min()),
                "ciciov_train_max": float(train.max()),
                "external_mean": float(external.mean()),
                "external_std": float(external.std()),
                "external_min": float(external.min()),
                "external_max": float(external.max()),
                "standardized_mean_difference": (
                    (float(external.mean()) - train_mean) / train_std
                    if train_std > 0
                    else math.nan
                ),
                "external_fraction_outside_ciciov_train_range": float(outside.mean()),
            }
        )
    return sorted(
        output,
        key=lambda row: abs(float(row["standardized_mean_difference"]))
        if not pd.isna(row["standardized_mean_difference"])
        else -1,
        reverse=True,
    )


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_dir = project_root / "data" / "external" / "car_hacking"
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find external dataset directory: {data_dir}")

    step03_path = project_root / "src" / "03_build_window_dataset.py"
    if not step03_path.exists():
        step03_path = project_root / step03_path.name
    feature_builder = load_script(step03_path, "ztav_step03_external")
    model, threshold, feature_names, training_features = load_model_assets(project_root)

    all_windows: list[pd.DataFrame] = []
    parser_summaries: list[dict[str, object]] = []
    for file_name, source_class in ATTACK_FILES:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing HCRL attack file: {path}")
        windows, summary = process_attack_csv(
            path,
            source_class,
            feature_builder,
            chunk_rows=args.chunk_rows,
        )
        all_windows.append(windows)
        parser_summaries.append(summary)

    if not args.skip_normal_capture:
        normal_path = locate_normal_capture(data_dir)
        windows, summary = process_normal_capture(normal_path, feature_builder)
        all_windows.append(windows)
        parser_summaries.append(summary)

    dataset = pd.concat(all_windows, ignore_index=True)
    del all_windows
    missing_features = set(feature_names) - set(dataset.columns)
    unexpected_features = set(dataset.columns) - set(feature_names) - {
        "source_file",
        "window_index",
        "start_row",
        "end_row",
        "source_capture_class",
        "binary_target",
        "attack_frame_count",
        "attack_frame_fraction",
    }
    if missing_features or unexpected_features:
        raise ValueError(
            f"External feature schema mismatch: missing={sorted(missing_features)}, "
            f"unexpected={sorted(unexpected_features)}"
        )

    external_features = dataset[feature_names]
    probability = predict_in_batches(model, external_features)
    dataset["attack_probability"] = probability
    dataset["model_prediction"] = (probability >= threshold).astype(np.uint8)

    processed_dir = project_root / "data" / "processed" / "external_car_hacking"
    results_dir = project_root / "results" / "external_car_hacking_zero_shot"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / "car_hacking_windows_w100_predictions.csv"
    dataset.to_csv(dataset_path, index=False)

    metric_rows = [metric_row(dataset, threshold, "all_external_windows")]
    for source_file, group in dataset.groupby("source_file", sort=False):
        metric_rows.append(metric_row(group, threshold, f"source:{source_file}"))
    density = attack_density_rows(dataset, threshold)
    shift = feature_shift_rows(external_features, training_features)
    manifest: list[dict[str, object]] = [
        {"item": "evaluation_type", "value": "cross-dataset zero-shot CAN evaluation"},
        {"item": "training_dataset", "value": "CICIoV2024 group-disjoint training split"},
        {"item": "external_dataset", "value": "HCRL Car-Hacking Dataset"},
        {"item": "model", "value": "group_disjoint_logistic_regression.joblib"},
        {"item": "window_size", "value": WINDOW_SIZE},
        {
            "item": "external_window_label",
            "value": "attack if at least one T frame occurs in 100 consecutive frames",
        },
        {"item": "threshold", "value": threshold},
        {"item": "threshold_source", "value": "CICIoV2024 validation; not external data"},
        {"item": "retraining_on_external", "value": "none"},
        {"item": "external_threshold_tuning", "value": "none"},
        {"item": "overlap", "value": "non-overlapping consecutive windows per source file"},
        {
            "item": "scope_limitation",
            "value": "validates CAN subsystem only; other context sources remain simulated",
        },
    ]

    write_csv(results_dir / "external_parser_summary.csv", parser_summaries)
    write_csv(results_dir / "external_zero_shot_metrics.csv", metric_rows)
    write_csv(results_dir / "external_attack_density_recall.csv", density)
    write_csv(results_dir / "external_feature_domain_shift.csv", shift)
    write_csv(results_dir / "external_evaluation_manifest.csv", manifest)
    with (results_dir / "external_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    overall = metric_rows[0]
    top_shift = shift[0]
    print("\n" + "=" * 86)
    print("External HCRL zero-shot evaluation completed successfully.")
    print(
        f"Windows={int(overall['windows']):,}, benign={int(overall['benign_windows']):,}, "
        f"attack={int(overall['attack_windows']):,}"
    )
    print(
        f"Threshold={threshold:.6f}, precision={float(overall['precision']):.4f}, "
        f"recall={float(overall['recall']):.4f}, F1={float(overall['f1']):.4f}, "
        f"PR-AUC={float(overall['pr_auc']):.4f}"
    )
    print(
        "Largest standardized feature mean shift: "
        f"{top_shift['feature']} ({float(top_shift['standardized_mean_difference']):.3f})"
    )
    print(f"Processed windows: {dataset_path}")
    print(f"Results directory: {results_dir}")
    print("\nNext: diagnose domain shift and run source-ablation analysis.")


if __name__ == "__main__":
    main()
