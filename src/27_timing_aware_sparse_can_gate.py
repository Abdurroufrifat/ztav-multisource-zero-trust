#!/usr/bin/env python3
"""Evaluate CAN inter-arrival timing as a new sparse-attack evidence family.

Step 26 showed that temporal memory over the existing decisions cannot recover
low-density CAN injections while keeping both enforcement and healthy-recovery
false-positive rates below 5%.  This experiment therefore adds new evidence:
20-frame CAN inter-arrival timing features from the raw HCRL timestamps.

Scientific protocol
-------------------
* The frozen Step 17 100-frame gate and Step 21 structural micro-gate are not
  retrained or recalibrated.
* Timing baselines use the first 50 clean 20-frame windows of each session.
* The timing threshold is selected from attack-free normal calibration
  pseudo-sessions only, targeting a 0.5% micro-window FPR.
* Normal holdout sessions and all attack captures are evaluation-only.
* HCRL R/T flags are used only to calculate evaluation labels and metrics.
* Results compare frozen CAN, structural, timing, and structural+timing fusion.

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


MICRO_WINDOW_SIZE = 20
PARENT_WINDOW_SIZE = 100
MICRO_PER_PARENT = PARENT_WINDOW_SIZE // MICRO_WINDOW_SIZE
BOOTSTRAP_MICRO_WINDOWS = 50
PSEUDO_SESSION_MICRO_WINDOWS = 1_250
TIMING_TARGET_FPR = 0.005
GLOBAL_SCALE_FLOOR_FRACTION = 0.25
EPSILON = 1e-12
NORMAL_SOURCE = "normal_run_data.txt"
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
ATTACK_FILES = (
    ("DoS_dataset.csv", "DoS"),
    ("Fuzzy_dataset.csv", "FUZZY"),
    ("gear_dataset.csv", "GEAR_SPOOFING"),
    ("RPM_dataset.csv", "RPM_SPOOFING"),
)
NORMAL_LINE = re.compile(
    r"Timestamp:\s*([0-9.]+).*?ID:\s*([0-9A-Fa-f]+).*?"
    r"DLC:\s*(\d+)\s+(.*)$"
)
TIMING_FEATURES = (
    "log_window_span_us",
    "log_iat_mean_us",
    "log_iat_std_us",
    "log_iat_median_us",
    "log_iat_p10_us",
    "log_iat_p90_us",
    "log_message_rate_hz",
    "iat_nonpositive_fraction",
    "iat_burst_fraction",
)
META_COLUMNS = (
    "source_file",
    "window_index",
    "start_row",
    "end_row",
    "source_capture_class",
    "binary_target",
    "attack_frame_count",
    "attack_frame_fraction",
)
METHOD_COLUMNS = {
    "w100_instant": "w100_alarm_instant",
    "structural_multiscale_instant": "structural_multiscale_alarm",
    "timing_multiscale_instant": "timing_multiscale_alarm",
    "combined_multiscale_instant": "combined_multiscale_alarm",
    "combined_multiscale_persistent_2": "combined_persistent_multiscale_alarm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a session-normalized timing-aware sparse-CAN gate."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows < MICRO_WINDOW_SIZE:
        parser.error(f"--chunk-rows must be at least {MICRO_WINDOW_SIZE}")
    return args


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def locate_normal_capture(data_dir: Path) -> Path:
    candidates = list(data_dir.rglob(NORMAL_SOURCE))
    if not candidates:
        raise FileNotFoundError(f"Cannot find {NORMAL_SOURCE} below {data_dir}")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[-1]


def extract_flags(chunk: pd.DataFrame) -> np.ndarray:
    """Find the variable-position HCRL R/T marker without parsing payload bytes."""
    tail_columns = [f"data_{index}" for index in range(8)] + ["flag"]
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
        bad = int(np.flatnonzero(marker_counts != 1)[0])
        raise ValueError(
            "Expected exactly one HCRL R/T marker per row; "
            f"chunk row {bad} has tail={tail[bad].tolist()}"
        )
    marker_indices = marker_mask.argmax(axis=1)
    dlc = pd.to_numeric(chunk["dlc"], errors="raise").to_numpy(dtype=np.int64)
    if np.any((dlc < 0) | (dlc > 8)) or np.any(marker_indices != dlc):
        bad_mask = ((dlc < 0) | (dlc > 8)) | (marker_indices != dlc)
        bad = int(np.flatnonzero(bad_mask)[0])
        raise ValueError(
            "HCRL payload length/marker does not match DLC; "
            f"chunk row {bad} has DLC={dlc[bad]}, marker={marker_indices[bad]}"
        )
    return tail[np.arange(len(tail)), marker_indices]


def timing_features(timestamps: np.ndarray) -> pd.DataFrame:
    """Return robust, log-scaled timing descriptors for aligned windows."""
    if timestamps.ndim != 2 or timestamps.shape[1] != MICRO_WINDOW_SIZE:
        raise ValueError("Timestamp matrix is not aligned to 20-frame windows")
    iat = np.diff(timestamps, axis=1)
    positive = np.where(iat > 0, iat, np.nan)
    positive_count = np.sum(np.isfinite(positive), axis=1)
    safe = positive.copy()
    empty = positive_count == 0
    safe[empty, 0] = 0.0
    mean = np.nanmean(safe, axis=1)
    std = np.nanstd(safe, axis=1)
    median = np.nanmedian(safe, axis=1)
    p10 = np.nanquantile(safe, 0.10, axis=1)
    p90 = np.nanquantile(safe, 0.90, axis=1)
    span = np.maximum(timestamps[:, -1] - timestamps[:, 0], 0.0)
    rate = (MICRO_WINDOW_SIZE - 1) / np.maximum(span, 1e-6)
    burst_limit = np.maximum(0.25 * median, 1e-9)
    burst = np.sum((iat > 0) & (iat <= burst_limit[:, None]), axis=1)
    burst_fraction = burst / np.maximum(positive_count, 1)

    def log_microseconds(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(values, 0.0) * 1_000_000.0)

    return pd.DataFrame(
        {
            "log_window_span_us": log_microseconds(span),
            "log_iat_mean_us": log_microseconds(mean),
            "log_iat_std_us": log_microseconds(std),
            "log_iat_median_us": log_microseconds(median),
            "log_iat_p10_us": log_microseconds(p10),
            "log_iat_p90_us": log_microseconds(p90),
            "log_message_rate_hz": np.log1p(rate),
            "iat_nonpositive_fraction": np.mean(iat <= 0, axis=1),
            "iat_burst_fraction": burst_fraction,
        }
    )


def make_window_frame(
    timestamps: np.ndarray,
    flags: np.ndarray,
    source_file: str,
    source_class: str,
    first_window_index: int,
) -> pd.DataFrame:
    if len(timestamps) != len(flags) or len(timestamps) % MICRO_WINDOW_SIZE:
        raise ValueError("Timing batch is not window-aligned")
    matrix = timestamps.reshape(-1, MICRO_WINDOW_SIZE)
    flag_windows = flags.reshape(-1, MICRO_WINDOW_SIZE)
    counts = (flag_windows == "T").sum(axis=1)
    indices = np.arange(first_window_index, first_window_index + len(matrix))
    frame = timing_features(matrix)
    frame.insert(0, "attack_frame_fraction", counts / MICRO_WINDOW_SIZE)
    frame.insert(0, "attack_frame_count", counts)
    frame.insert(0, "binary_target", (counts > 0).astype(np.uint8))
    frame.insert(0, "source_capture_class", source_class)
    frame.insert(0, "end_row", (indices + 1) * MICRO_WINDOW_SIZE - 1)
    frame.insert(0, "start_row", indices * MICRO_WINDOW_SIZE)
    frame.insert(0, "window_index", indices)
    frame.insert(0, "source_file", source_file)
    return frame


def process_attack_capture(
    path: Path,
    source_class: str,
    chunk_rows: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"Building timing windows: {path.name}")
    started = time.perf_counter()
    batches: list[pd.DataFrame] = []
    carry_timestamps = np.empty(0, dtype=float)
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
        timestamps = pd.to_numeric(chunk["timestamp"], errors="raise").to_numpy(float)
        flags = extract_flags(chunk)
        if not np.isfinite(timestamps).all():
            raise ValueError(f"Non-finite timestamps in {path.name}")
        total_rows += len(chunk)
        attack_frames += int((flags == "T").sum())
        if len(carry_timestamps):
            timestamps = np.concatenate((carry_timestamps, timestamps))
            flags = np.concatenate((carry_flags, flags))
        usable = len(timestamps) - len(timestamps) % MICRO_WINDOW_SIZE
        if usable:
            batch = make_window_frame(
                timestamps[:usable],
                flags[:usable],
                path.name,
                source_class,
                window_index,
            )
            batches.append(batch)
            window_index += len(batch)
        carry_timestamps = timestamps[usable:].copy()
        carry_flags = flags[usable:].copy()
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"  chunks={chunk_number}, rows={total_rows:,}, "
                f"windows={window_index:,}"
            )
    windows = pd.concat(batches, ignore_index=True)
    return windows, {
        "source_file": path.name,
        "source_capture_class": source_class,
        "raw_rows": total_rows,
        "attack_frames": attack_frames,
        "windows": len(windows),
        "attack_windows": int(windows["binary_target"].sum()),
        "discarded_tail_rows": len(carry_timestamps),
        "skipped_malformed_lines": 0,
        "processing_seconds": round(time.perf_counter() - started, 3),
    }


def parse_normal_timestamp(line: str, line_number: int) -> float:
    match = NORMAL_LINE.search(line)
    if not match:
        raise ValueError(f"Cannot parse normal capture line {line_number}: {line[:120]!r}")
    return float(match.group(1))


def process_normal_capture(
    path: Path,
    batch_windows: int = 5_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"Building timing windows: {path.name}")
    started = time.perf_counter()
    batches: list[pd.DataFrame] = []
    current: list[float] = []
    timestamp_windows: list[list[float]] = []
    total_rows = 0
    input_lines = 0
    skipped = 0
    skipped_examples: list[str] = []
    window_index = 0

    def flush() -> None:
        nonlocal timestamp_windows, window_index
        if not timestamp_windows:
            return
        timestamps = np.asarray(timestamp_windows, dtype=float).reshape(-1)
        flags = np.full(len(timestamps), "R", dtype="U1")
        batch = make_window_frame(
            timestamps,
            flags,
            path.name,
            "ATTACK_FREE",
            window_index,
        )
        batches.append(batch)
        window_index += len(batch)
        timestamp_windows = []

    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_lines += 1
            try:
                timestamp = parse_normal_timestamp(line, line_number)
            except ValueError:
                skipped += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append(f"line {line_number}: {line.strip()[:100]}")
                continue
            current.append(timestamp)
            total_rows += 1
            if len(current) == MICRO_WINDOW_SIZE:
                timestamp_windows.append(current)
                current = []
                if len(timestamp_windows) >= batch_windows:
                    flush()
    flush()
    skipped_fraction = skipped / input_lines if input_lines else 0.0
    if skipped_fraction > 0.01:
        raise ValueError(
            f"More than 1% of normal lines were malformed: {skipped}/{input_lines}"
        )
    windows = pd.concat(batches, ignore_index=True)
    return windows, {
        "source_file": path.name,
        "source_capture_class": "ATTACK_FREE",
        "raw_rows": total_rows,
        "attack_frames": 0,
        "windows": len(windows),
        "attack_windows": 0,
        "discarded_tail_rows": len(current),
        "skipped_malformed_lines": skipped,
        "skipped_line_fraction": skipped_fraction,
        "skipped_line_examples": " | ".join(skipped_examples),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }


def build_timing_cache(
    project_root: Path,
    cache_path: Path,
    parser_summary_path: Path,
    chunk_rows: int,
) -> pd.DataFrame:
    data_dir = project_root / "data" / "external" / "car_hacking"
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find HCRL directory: {data_dir}")
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for file_name, source_class in ATTACK_FILES:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing HCRL capture: {path}")
        frame, summary = process_attack_capture(path, source_class, chunk_rows)
        frames.append(frame)
        summaries.append(summary)
    normal_path = locate_normal_capture(data_dir)
    frame, summary = process_normal_capture(normal_path)
    frames.append(frame)
    summaries.append(summary)
    data = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, index=False)
    write_csv(parser_summary_path, summaries)
    print(f"Saved timing cache: {cache_path}")
    return data


def robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    center = np.median(values, axis=axis)
    return 1.4826 * np.median(np.abs(values - center), axis=axis)


def timing_scale_floors(normal_calibration_source: pd.DataFrame) -> np.ndarray:
    values = normal_calibration_source[list(TIMING_FEATURES)].to_numpy(float)
    global_scales = robust_scale(values, axis=0)
    absolute = np.asarray(
        [0.01 if "fraction" not in feature else 0.005 for feature in TIMING_FEATURES],
        dtype=float,
    )
    return np.maximum(GLOBAL_SCALE_FLOOR_FRACTION * global_scales, absolute)


def score_session(
    frame: pd.DataFrame,
    session_id: str,
    floors: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = frame.sort_values("window_index").copy()
    if len(ordered) <= BOOTSTRAP_MICRO_WINDOWS:
        raise ValueError(f"Session {session_id} is too short")
    bootstrap = ordered.iloc[:BOOTSTRAP_MICRO_WINDOWS]
    baseline = bootstrap[list(TIMING_FEATURES)].to_numpy(float)
    center = np.median(baseline, axis=0)
    local_scale = robust_scale(baseline, axis=0)
    scale = np.maximum(local_scale, floors)
    evaluation = ordered.iloc[BOOTSTRAP_MICRO_WINDOWS:].copy()
    values = evaluation[list(TIMING_FEATURES)].to_numpy(float)
    deviations = np.abs(values - center) / np.maximum(scale, EPSILON)
    evaluation["timing_session_id"] = session_id
    evaluation["timing_deviation"] = np.quantile(deviations, 0.75, axis=1)
    audit: dict[str, object] = {
        "timing_session_id": session_id,
        "source_file": str(ordered.iloc[0]["source_file"]),
        "total_micro_windows": len(ordered),
        "bootstrap_micro_windows": BOOTSTRAP_MICRO_WINDOWS,
        "evaluated_micro_windows": len(evaluation),
        "bootstrap_attack_micro_windows": int(bootstrap["binary_target"].sum()),
        "bootstrap_attack_frames": int(bootstrap["attack_frame_count"].sum()),
        "scale_floor_features": int((local_scale < floors).sum()),
    }
    for index, feature in enumerate(TIMING_FEATURES):
        audit[f"{feature}_center"] = float(center[index])
        audit[f"{feature}_scale"] = float(scale[index])
    return evaluation, audit


def build_normal_sessions(
    normal: pd.DataFrame,
    floors: np.ndarray,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], int]:
    ordered = normal.sort_values("window_index")
    complete = len(ordered) // PSEUDO_SESSION_MICRO_WINDOWS
    if complete < 4:
        raise ValueError(f"Only {complete} complete normal pseudo-sessions are available")
    sessions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for index in range(complete):
        start = index * PSEUDO_SESSION_MICRO_WINDOWS
        end = start + PSEUDO_SESSION_MICRO_WINDOWS
        session, audit = score_session(
            ordered.iloc[start:end],
            f"normal_timing_session_{index:03d}",
            floors,
        )
        sessions.append(session)
        audits.append(audit)
    return sessions, audits, len(ordered) - complete * PSEUDO_SESSION_MICRO_WINDOWS


def add_timing_alarms(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["timing_alarm_instant"] = output["timing_deviation"] >= threshold
    output["timing_alarm_persistent_2"] = output.groupby(
        "timing_session_id", sort=False
    )["timing_alarm_instant"].transform(
        lambda values: values & values.shift(1, fill_value=False)
    )
    output["timing_continuous_trust"] = 1.0 / (
        1.0 + np.square(output["timing_deviation"] / max(threshold, EPSILON))
    )
    return output


def merge_structural_predictions(
    timing: pd.DataFrame,
    structural_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not structural_path.exists():
        raise FileNotFoundError(f"Missing Step 21 predictions: {structural_path}")
    structural = pd.read_csv(structural_path)
    columns = [
        "source_file",
        "window_index",
        "binary_target",
        "attack_frame_count",
        "micro_alarm_instant",
        "micro_alarm_persistent_2",
        "micro_structural_deviation",
    ]
    missing = set(columns) - set(structural.columns)
    if missing:
        raise ValueError(f"Step 21 predictions are missing: {sorted(missing)}")
    structural = structural[columns].rename(
        columns={
            "binary_target": "structural_binary_target",
            "attack_frame_count": "structural_attack_frame_count",
            "micro_alarm_instant": "structural_alarm_instant",
            "micro_alarm_persistent_2": "structural_alarm_persistent_2",
        }
    )
    merged = timing.merge(
        structural,
        on=["source_file", "window_index"],
        how="inner",
        validate="one_to_one",
    )
    mismatches = int(
        (
            (merged["binary_target"] != merged["structural_binary_target"])
            | (
                merged["attack_frame_count"]
                != merged["structural_attack_frame_count"]
            )
        ).sum()
    )
    if len(merged) != len(timing) or len(merged) != len(structural):
        raise RuntimeError(
            "Timing/structural evaluation rows do not align exactly: "
            f"timing={len(timing)}, structural={len(structural)}, merged={len(merged)}"
        )
    if mismatches:
        raise RuntimeError(f"Timing/structural labels disagree for {mismatches} rows")
    merged["combined_micro_alarm_instant"] = (
        merged["timing_alarm_instant"].astype(bool)
        | merged["structural_alarm_instant"].astype(bool)
    )
    merged["combined_micro_alarm_persistent_2"] = merged.groupby(
        "timing_session_id", sort=False
    )["combined_micro_alarm_instant"].transform(
        lambda values: values & values.shift(1, fill_value=False)
    )
    return merged, {
        "timing_evaluation_micro_windows": len(timing),
        "structural_evaluation_micro_windows": len(structural),
        "matched_micro_windows": len(merged),
        "label_mismatches": mismatches,
    }


def aggregate_to_parent(
    micro: pd.DataFrame,
    w100_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not w100_path.exists():
        raise FileNotFoundError(f"Missing Step 17 predictions: {w100_path}")
    micro = micro.copy()
    micro["parent_window_index"] = micro["start_row"].astype(np.int64) // PARENT_WINDOW_SIZE
    parent = micro.groupby(
        ["source_file", "parent_window_index"], sort=True, observed=True
    ).agg(
        micro_windows=("window_index", "size"),
        micro_attack_frames=("attack_frame_count", "sum"),
        structural_any_instant=("structural_alarm_instant", "max"),
        timing_any_instant=("timing_alarm_instant", "max"),
        timing_any_persistent_2=("timing_alarm_persistent_2", "max"),
        combined_any_instant=("combined_micro_alarm_instant", "max"),
        combined_any_persistent_2=("combined_micro_alarm_persistent_2", "max"),
        timing_deviation_max=("timing_deviation", "max"),
    ).reset_index()
    incomplete = int((parent["micro_windows"] != MICRO_PER_PARENT).sum())
    parent = parent[parent["micro_windows"] == MICRO_PER_PARENT].copy()
    w100 = pd.read_csv(w100_path)
    required = {
        "source_file",
        "window_index",
        "binary_target",
        "attack_frame_count",
        "alarm_instant",
        "alarm_persistent_2",
    }
    missing = required - set(w100.columns)
    if missing:
        raise ValueError(f"Step 17 predictions are missing: {sorted(missing)}")
    right = w100[list(required)].rename(
        columns={
            "window_index": "parent_window_index",
            "binary_target": "w100_binary_target",
            "attack_frame_count": "w100_attack_frame_count",
            "alarm_instant": "w100_alarm_instant",
            "alarm_persistent_2": "w100_alarm_persistent_2",
        }
    )
    merged = parent.merge(
        right,
        on=["source_file", "parent_window_index"],
        how="inner",
        validate="one_to_one",
    )
    mismatches = int(
        (
            (merged["micro_attack_frames"] != merged["w100_attack_frame_count"])
            | (
                (merged["micro_attack_frames"] > 0).astype(np.uint8)
                != merged["w100_binary_target"].astype(np.uint8)
            )
        ).sum()
    )
    if mismatches:
        raise RuntimeError(f"Micro/parent labels disagree for {mismatches} rows")
    merged["structural_multiscale_alarm"] = (
        merged["w100_alarm_instant"].astype(bool)
        | merged["structural_any_instant"].astype(bool)
    )
    merged["timing_multiscale_alarm"] = (
        merged["w100_alarm_instant"].astype(bool)
        | merged["timing_any_instant"].astype(bool)
    )
    merged["combined_multiscale_alarm"] = (
        merged["w100_alarm_instant"].astype(bool)
        | merged["combined_any_instant"].astype(bool)
    )
    merged["combined_persistent_multiscale_alarm"] = (
        merged["w100_alarm_instant"].astype(bool)
        | merged["combined_any_persistent_2"].astype(bool)
    )
    return merged, {
        "micro_parent_groups": len(parent) + incomplete,
        "incomplete_groups_discarded": incomplete,
        "complete_parent_groups": len(parent),
        "parents_matched_to_step17": len(merged),
        "parent_label_mismatches": mismatches,
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    frame: pd.DataFrame,
    method: str,
    alarm_column: str,
    scope: str,
) -> dict[str, object]:
    truth = frame["w100_binary_target"].to_numpy(dtype=np.uint8)
    prediction = frame[alarm_column].astype(bool).to_numpy(dtype=np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(truth)) == 2
    return {
        "method": method,
        "scope": scope,
        "parent_windows": len(frame),
        "benign_parent_windows": int((truth == 0).sum()),
        "attack_parent_windows": int((truth == 1).sum()),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction) if both else math.nan,
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "mcc": matthews_corrcoef(truth, prediction) if both else math.nan,
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluation_rows(
    parent: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    per_source: list[dict[str, object]] = []
    density: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        metrics.append(metric_row(parent, method, column, "all_parent_evaluation"))
        for source_file, source in parent.groupby("source_file", sort=True):
            per_source.append(metric_row(source, method, column, f"source:{source_file}"))
    attacked = parent[parent["w100_binary_target"] == 1]
    bins = (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100))
    for method, column in METHOD_COLUMNS.items():
        for label, lower, upper in bins:
            group = attacked[attacked["w100_attack_frame_count"].between(lower, upper)]
            density.append(
                {
                    "method": method,
                    "attack_frames_per_100": label,
                    "parent_windows": len(group),
                    "recall": float(group[column].mean()) if len(group) else math.nan,
                    "mean_timing_deviation_max": float(group["timing_deviation_max"].mean()) if len(group) else math.nan,
                }
            )
    return metrics, per_source, density


def plot_results(
    metrics: pd.DataFrame,
    density: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = (
        "w100_instant",
        "structural_multiscale_instant",
        "timing_multiscale_instant",
        "combined_multiscale_instant",
        "combined_multiscale_persistent_2",
    )
    labels = {
        "w100_instant": "Frozen w100",
        "structural_multiscale_instant": "w100 + structure",
        "timing_multiscale_instant": "w100 + timing",
        "combined_multiscale_instant": "w100 + structure + timing",
        "combined_multiscale_persistent_2": "Combined evidence, persistent-2",
    }
    bins = ("1", "2-5", "6-20", "21-100")
    x = np.arange(len(bins))
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), constrained_layout=True)
    for method in selected:
        group = density[density["method"] == method].set_index("attack_frames_per_100")
        axes[0].plot(
            x,
            [group.loc[label, "recall"] for label in bins],
            marker="o",
            linewidth=2,
            label=labels[method],
        )
    metric_group = metrics.set_index("method")
    positions = np.arange(len(selected))
    width = 0.22
    for offset, (metric, label) in enumerate(
        (("f1", "F1"), ("false_positive_rate", "FPR"), ("recall", "Recall"))
    ):
        axes[1].bar(
            positions + (offset - 1) * width,
            [metric_group.loc[method, metric] for method in selected],
            width,
            label=label,
        )
    axes[0].set(
        title="Sparse-attack recall",
        xlabel="Malicious frames per 100-frame parent window",
        ylabel="Recall",
        ylim=(0, 1.03),
        xticks=x,
        xticklabels=bins,
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        title="Overall operating points",
        ylabel="Metric",
        ylim=(0, 1.03),
        xticks=positions,
        xticklabels=["w100", "structure", "timing", "combined", "persistent"],
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Timing-aware sparse-CAN evidence ablation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    processed_dir = root / "data" / "processed"
    results_dir = root / "results" / "timing_aware_sparse_can_gate"
    cache_path = processed_dir / "car_hacking_windows_w20_timing.csv"
    parser_summary_path = results_dir / "timing_parser_summary.csv"
    if args.rebuild_cache or not cache_path.exists():
        data = build_timing_cache(root, cache_path, parser_summary_path, args.chunk_rows)
    else:
        print(f"Loading cached timing windows: {cache_path}")
        data = pd.read_csv(cache_path)
    required = set(META_COLUMNS + TIMING_FEATURES)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Timing cache is missing: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Timing cache contains missing required values")

    normal = data[data["source_file"] == NORMAL_SOURCE].copy()
    attacks = data[data["source_file"] != NORMAL_SOURCE].copy()
    if normal.empty or attacks.empty:
        raise ValueError("Both normal and attack timing windows are required")
    complete_normal_sessions = len(normal) // PSEUDO_SESSION_MICRO_WINDOWS
    calibration_session_count = complete_normal_sessions // 2
    calibration_floor_rows = calibration_session_count * PSEUDO_SESSION_MICRO_WINDOWS
    if calibration_session_count < 2:
        raise ValueError("Too few normal sessions for calibration/holdout separation")
    floors = timing_scale_floors(normal.sort_values("window_index").iloc[:calibration_floor_rows])
    normal_sessions, audits, discarded_normal = build_normal_sessions(normal, floors)
    split = len(normal_sessions) // 2
    for index, session in enumerate(normal_sessions):
        partition = "normal_calibration" if index < split else "normal_holdout"
        session["timing_protocol_partition"] = partition
        audits[index]["timing_protocol_partition"] = partition
    normal_calibration = pd.concat(normal_sessions[:split], ignore_index=True)
    normal_holdout = pd.concat(normal_sessions[split:], ignore_index=True)

    attack_evaluations: list[pd.DataFrame] = []
    for source_file, source in attacks.groupby("source_file", sort=True):
        evaluation, audit = score_session(
            source,
            f"attack_timing_session::{source_file}",
            floors,
        )
        if int(audit["bootstrap_attack_frames"]) != 0:
            raise ValueError(f"Trusted timing bootstrap for {source_file} contains attacks")
        evaluation["timing_protocol_partition"] = "attack_capture_evaluation"
        audit["timing_protocol_partition"] = "attack_capture_evaluation"
        attack_evaluations.append(evaluation)
        audits.append(audit)
    attack_evaluation = pd.concat(attack_evaluations, ignore_index=True)

    threshold = float(
        np.quantile(
            normal_calibration["timing_deviation"].to_numpy(float),
            1.0 - TIMING_TARGET_FPR,
        )
    )
    normal_calibration = add_timing_alarms(normal_calibration, threshold)
    normal_holdout = add_timing_alarms(normal_holdout, threshold)
    attack_evaluation = add_timing_alarms(attack_evaluation, threshold)
    timing_evaluation = pd.concat([attack_evaluation, normal_holdout], ignore_index=True)

    structural_path = (
        root / "results" / "multiscale_sparse_can_gate" / "w20_micro_gate_predictions.csv"
    )
    micro, micro_alignment = merge_structural_predictions(timing_evaluation, structural_path)
    w100_path = (
        root / "results" / "session_normalized_can_gate_w100" / "session_gate_predictions.csv"
    )
    parent, parent_alignment = aggregate_to_parent(micro, w100_path)
    metrics, per_source, density = evaluation_rows(parent)

    calibration = {
        "micro_window_size": MICRO_WINDOW_SIZE,
        "bootstrap_micro_windows": BOOTSTRAP_MICRO_WINDOWS,
        "normal_pseudo_session_micro_windows": PSEUDO_SESSION_MICRO_WINDOWS,
        "timing_target_fpr": TIMING_TARGET_FPR,
        "timing_deviation_threshold": threshold,
        "normal_calibration_micro_windows": len(normal_calibration),
        "normal_calibration_instant_fpr": float(normal_calibration["timing_alarm_instant"].mean()),
        "normal_calibration_persistent_2_fpr": float(normal_calibration["timing_alarm_persistent_2"].mean()),
        "normal_holdout_micro_windows": len(normal_holdout),
        "normal_holdout_instant_fpr": float(normal_holdout["timing_alarm_instant"].mean()),
        "normal_holdout_persistent_2_fpr": float(normal_holdout["timing_alarm_persistent_2"].mean()),
    }
    floor_rows = [
        {"feature": feature, "global_scale_floor": float(floors[index])}
        for index, feature in enumerate(TIMING_FEATURES)
    ]
    manifest = [
        {"item": "experiment_type", "value": "exploratory timing-aware sparse-CAN evidence ablation"},
        {"item": "new_evidence_family", "value": "CAN inter-arrival timing"},
        {"item": "timing_features", "value": ";".join(TIMING_FEATURES)},
        {"item": "timing_threshold_source", "value": "attack-free normal calibration pseudo-sessions only"},
        {"item": "timing_target_micro_fpr", "value": TIMING_TARGET_FPR},
        {"item": "frozen_inputs", "value": "Step 17 w100 gate; Step 21 structural micro-gate"},
        {"item": "fusion", "value": "OR ablation plus combined-evidence consecutive-2"},
        {"item": "label_usage", "value": "R/T labels used for evaluation only"},
        {"item": "discarded_normal_tail_micro_windows", "value": discarded_normal},
        {"item": "external_validity", "value": "HCRL exploratory; confirmation on another timestamped CAN capture required"},
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    micro.to_csv(results_dir / "timing_micro_gate_predictions.csv", index=False)
    parent.to_csv(results_dir / "timing_parent_predictions.csv", index=False)
    write_csv(results_dir / "timing_gate_overall_metrics.csv", metrics)
    write_csv(results_dir / "timing_gate_per_source_metrics.csv", per_source)
    write_csv(results_dir / "timing_gate_density_recall.csv", density)
    write_csv(results_dir / "timing_calibration_summary.csv", [calibration])
    write_csv(results_dir / "timing_session_bootstrap_audit.csv", audits)
    write_csv(results_dir / "timing_feature_scale_floors.csv", floor_rows)
    write_csv(results_dir / "timing_micro_alignment_audit.csv", [micro_alignment])
    write_csv(results_dir / "timing_parent_alignment_audit.csv", [parent_alignment])
    write_csv(results_dir / "timing_gate_manifest.csv", manifest)
    plot_results(
        pd.DataFrame(metrics),
        pd.DataFrame(density),
        results_dir / "timing_sparse_recall.png",
    )

    metric_frame = pd.DataFrame(metrics).set_index("method")
    density_frame = pd.DataFrame(density)
    print("\n" + "=" * 86)
    print("Timing-aware sparse-CAN evidence ablation completed successfully.")
    print(f"Timing deviation threshold: {threshold:.6f}")
    print(
        "Normal holdout timing micro FPR: "
        f"instant={calibration['normal_holdout_instant_fpr']:.4f}, "
        f"persistent_2={calibration['normal_holdout_persistent_2_fpr']:.4f}"
    )
    print("\nParent-window operating points:")
    for method in METHOD_COLUMNS:
        row = metric_frame.loc[method]
        print(
            f"  {method:<36} precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, F1={row['f1']:.4f}, "
            f"FPR={row['false_positive_rate']:.4f}"
        )
    print("\nSparse recall (combined persistent evidence):")
    selected = density_frame[
        density_frame["method"] == "combined_multiscale_persistent_2"
    ].set_index("attack_frames_per_100")
    for label in ("1", "2-5", "6-20", "21-100"):
        print(f"  frames={label:<6} recall={selected.loc[label, 'recall']:.4f}")
    eligible = metric_frame[metric_frame["false_positive_rate"] <= 0.05].index.tolist()
    print(f"\nMethods at or below 5% parent FPR: {eligible}")
    print(f"Results directory: {results_dir}")
    print(
        "\nNext: compare sparse recall and FPR. Confirm any promising timing fusion "
        "on a separate timestamped capture before changing the frozen policy."
    )


if __name__ == "__main__":
    main()
