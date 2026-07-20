#!/usr/bin/env python3
"""Frozen-threshold confirmation on the independent ROAD CAN dataset.

This stage applies the previously developed HCRL/CICIoV CAN gates to ROAD
without selecting a threshold, feature family, or fusion rule on ROAD labels.
Only the short, attack-free start of each capture is used for session
normalization.  All thresholds and scale floors are loaded from Steps 17, 21,
and 27 and remain frozen.

Primary endpoint
----------------
ROAD captures whose first 1,000 frames precede the documented injection
interval are eligible for the primary confirmation.  The four accelerator
captures have no clean injection interval (the compromised state spans the
capture), so they are reported separately as compromised-start negative
controls.  Ambient captures are used only to measure false alarms.

Run from D:\\ztav_project after extracting ROAD to data\\external\\road\\road:

    .\\.venv\\Scripts\\python.exe src\\28_road_frozen_external_confirmation.py

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
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
MICRO_BOOTSTRAP_WINDOWS = 50
PARENT_BOOTSTRAP_WINDOWS = 10
EPSILON = 1e-12
STRUCTURAL_FEATURES = (
    "id_unique_count",
    "id_entropy",
    "dominant_id_fraction",
    "frame_unique_count",
    "frame_unique_fraction",
    "id_change_rate",
    "consecutive_frame_repeat_rate",
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
METHOD_COLUMNS = {
    "w100_instant": "w100_alarm_instant",
    "structural_multiscale_instant": "structural_multiscale_alarm",
    "timing_multiscale_instant": "timing_multiscale_alarm",
    "combined_multiscale_instant": "combined_multiscale_alarm",
    "combined_multiscale_persistent_2": "combined_persistent_multiscale_alarm",
}
CANDUMP = re.compile(
    r"^\((?P<timestamp>[0-9]+(?:\.[0-9]+)?)\)\s+"
    r"(?P<channel>\S+)\s+(?P<can_id>[0-9A-Fa-f]+)#"
    r"(?P<payload>[0-9A-Fa-f]*)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen ROAD external confirmation for the CAN trust gates."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-parent-windows", type=int, default=2_000)
    args = parser.parse_args()
    if args.batch_parent_windows < 100:
        parser.error("--batch-parent-windows must be at least 100")
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


def locate_script(root: Path, name: str) -> Path:
    for candidate in (root / "src" / name, root / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find required script: {name}")


def load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def locate_road_root(root: Path) -> Path:
    candidates = (
        root / "data" / "external" / "road" / "road",
        root / "data" / "external" / "road",
    )
    for candidate in candidates:
        if (
            (candidate / "ambient" / "capture_metadata.json").exists()
            and (candidate / "attacks" / "capture_metadata.json").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Cannot find ROAD ambient/attacks metadata below data/external/road"
    )


def load_json(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty JSON object: {path}")
    return data


def capture_family(name: str, metadata: dict[str, object]) -> str:
    if name.startswith("accelerator_attack"):
        return "accelerator_compromised_state"
    if bool(metadata.get("modified")) or name.endswith("_masquerade"):
        return "masquerade"
    if name.startswith("fuzzing_attack"):
        return "fabrication_fuzzing"
    return "fabrication_targeted"


def list_capture_logs(
    directory: Path,
    metadata: dict[str, dict[str, object]],
) -> list[tuple[Path, str, dict[str, object]]]:
    logs = {
        path.stem: path
        for path in directory.rglob("*.log")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    }
    missing = sorted(set(metadata) - set(logs))
    unexpected = sorted(set(logs) - set(metadata))
    if missing:
        raise FileNotFoundError(
            f"ROAD logs missing for metadata entries in {directory}: {missing}"
        )
    if unexpected:
        print(f"WARNING: ignoring ROAD logs without metadata: {unexpected}")
    return [(logs[name], name, metadata[name]) for name in sorted(metadata)]


def parse_injection_id(value: object) -> int | None:
    if value is None or str(value).upper() == "XXX":
        return None
    return int(str(value), 16)


def signature_matches(
    can_id: int,
    payload_hex: str,
    injection_id: int | None,
    pattern: object,
) -> bool:
    if pattern is None:
        return False
    if injection_id is not None and can_id != injection_id:
        return False
    expected = str(pattern).upper()
    actual = payload_hex.upper().ljust(16, "0")[:16]
    if len(expected) != 16:
        return False
    return all(left == "X" or left == right for left, right in zip(expected, actual))


def robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    center = np.median(values, axis=axis)
    return 1.4826 * np.median(np.abs(values - center), axis=axis)


def probability_to_log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def read_feature_floors(
    path: Path,
    features: Sequence[str],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen feature floors: {path}")
    frame = pd.read_csv(path)
    if not {"feature", "global_scale_floor"}.issubset(frame.columns):
        raise ValueError(f"Invalid feature-floor file: {path}")
    mapping = frame.set_index("feature")["global_scale_floor"]
    missing = [feature for feature in features if feature not in mapping.index]
    if missing:
        raise ValueError(f"Frozen floors are missing features: {missing}")
    return np.asarray([float(mapping.loc[feature]) for feature in features])


def read_single_value(path: Path, column: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen calibration file: {path}")
    frame = pd.read_csv(path)
    if column not in frame or len(frame) != 1:
        raise ValueError(f"Expected one {column} value in {path}")
    return float(frame.iloc[0][column])


def read_w100_parameters(root: Path) -> tuple[float, float]:
    result_dir = root / "results" / "session_normalized_can_gate_w100"
    thresholds = pd.read_csv(result_dir / "session_gate_thresholds.csv")
    target_column = "calibration_target_fpr"
    rows = thresholds[np.isclose(thresholds[target_column].astype(float), 0.05)]
    if len(rows) != 1:
        raise ValueError("Cannot identify the frozen Step 17 5% threshold")
    threshold = float(rows.iloc[0]["deviation_threshold"])
    audits = pd.read_csv(result_dir / "session_gate_session_bootstrap_audit.csv")
    if "scale_floor" not in audits:
        raise ValueError("Step 17 bootstrap audit has no scale_floor")
    unique = np.unique(np.round(audits["scale_floor"].to_numpy(float), 12))
    if len(unique) != 1:
        raise ValueError(f"Step 17 contains multiple scale floors: {unique}")
    return threshold, float(unique[0])


def metric_row(
    frame: pd.DataFrame,
    method: str,
    column: str,
    scope: str,
) -> dict[str, object]:
    truth = frame["phase_target"].to_numpy(dtype=np.uint8)
    prediction = frame[column].astype(bool).to_numpy(dtype=np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(truth)) == 2
    return {
        "method": method,
        "scope": scope,
        "parent_windows": len(frame),
        "benign_windows": int((truth == 0).sum()),
        "attack_windows": int((truth == 1).sum()),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction) if both else math.nan,
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "mcc": matthews_corrcoef(truth, prediction) if both else math.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def process_capture(
    path: Path,
    name: str,
    metadata: dict[str, object],
    role: str,
    feature_builder: ModuleType,
    timing_module: ModuleType,
    batch_parent_windows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    interval_raw = metadata.get("injection_interval")
    interval = tuple(float(value) for value in interval_raw) if interval_raw is not None else None
    injection_id = parse_injection_id(metadata.get("injection_id"))
    pattern = metadata.get("injection_data_str")
    family = "ambient" if role == "ambient" else capture_family(name, metadata)
    micro_batches: list[pd.DataFrame] = []
    parent_batches: list[pd.DataFrame] = []
    raw_parents: list[list[list[int]]] = []
    timestamp_parents: list[list[float]] = []
    phase_parents: list[list[bool]] = []
    signature_parents: list[list[bool]] = []
    current_raw: list[list[int]] = []
    current_timestamps: list[float] = []
    current_phase: list[bool] = []
    current_signature: list[bool] = []
    first_timestamp: float | None = None
    input_lines = 0
    malformed = 0
    parent_index = 0

    def flush() -> None:
        nonlocal raw_parents, timestamp_parents, phase_parents, signature_parents, parent_index
        if not raw_parents:
            return
        raw = np.asarray(raw_parents, dtype=np.int64)
        timestamps = np.asarray(timestamp_parents, dtype=float)
        phase = np.asarray(phase_parents, dtype=bool)
        signature = np.asarray(signature_parents, dtype=bool)
        parent_count = len(raw)
        parent_indices = np.arange(parent_index, parent_index + parent_count)

        parent_features = feature_builder.extract_window_features(raw, PARENT_WINDOW_SIZE)
        parent_features.insert(0, "signature_frame_count", signature.sum(axis=1))
        parent_features.insert(0, "phase_frame_count", phase.sum(axis=1))
        parent_features.insert(0, "phase_target", phase.any(axis=1).astype(np.uint8))
        parent_features.insert(0, "window_end_elapsed", timestamps[:, -1])
        parent_features.insert(0, "window_start_elapsed", timestamps[:, 0])
        parent_features.insert(0, "window_index", parent_indices)
        parent_features.insert(0, "attack_family", family)
        parent_features.insert(0, "capture_role", role)
        parent_features.insert(0, "source_file", path.name)
        parent_features.insert(0, "capture_name", name)
        parent_batches.append(parent_features)

        raw_micro = raw.reshape(-1, MICRO_WINDOW_SIZE, 9)
        timestamps_micro = timestamps.reshape(-1, MICRO_WINDOW_SIZE)
        phase_micro = phase.reshape(-1, MICRO_WINDOW_SIZE)
        signature_micro = signature.reshape(-1, MICRO_WINDOW_SIZE)
        micro_indices = np.arange(parent_index * MICRO_PER_PARENT, (parent_index + parent_count) * MICRO_PER_PARENT)
        micro_features = feature_builder.extract_window_features(raw_micro, MICRO_WINDOW_SIZE)
        timing_features = timing_module.timing_features(timestamps_micro)
        for feature in TIMING_FEATURES:
            micro_features[feature] = timing_features[feature].to_numpy()
        micro_features.insert(0, "signature_frame_count", signature_micro.sum(axis=1))
        micro_features.insert(0, "phase_frame_count", phase_micro.sum(axis=1))
        micro_features.insert(0, "phase_target", phase_micro.any(axis=1).astype(np.uint8))
        micro_features.insert(0, "window_end_elapsed", timestamps_micro[:, -1])
        micro_features.insert(0, "window_start_elapsed", timestamps_micro[:, 0])
        micro_features.insert(0, "start_row", micro_indices * MICRO_WINDOW_SIZE)
        micro_features.insert(0, "window_index", micro_indices)
        micro_features.insert(0, "attack_family", family)
        micro_features.insert(0, "capture_role", role)
        micro_features.insert(0, "source_file", path.name)
        micro_features.insert(0, "capture_name", name)
        micro_batches.append(micro_features)
        parent_index += parent_count
        raw_parents = []
        timestamp_parents = []
        phase_parents = []
        signature_parents = []

    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line.strip():
                continue
            input_lines += 1
            match = CANDUMP.match(line.strip())
            if not match:
                malformed += 1
                continue
            timestamp = float(match.group("timestamp"))
            if first_timestamp is None:
                first_timestamp = timestamp
            elapsed = timestamp - first_timestamp
            can_id = int(match.group("can_id"), 16)
            payload_hex = match.group("payload").upper()
            if len(payload_hex) % 2 or len(payload_hex) > 16:
                malformed += 1
                continue
            payload = [int(payload_hex[index:index + 2], 16) for index in range(0, len(payload_hex), 2)]
            payload = (payload + [0] * 8)[:8]
            if role == "ambient":
                in_phase = False
            elif interval is None:
                in_phase = True
            else:
                in_phase = interval[0] <= elapsed <= interval[1]
            is_signature = in_phase and signature_matches(
                can_id, payload_hex, injection_id, pattern
            )
            current_raw.append([can_id, *payload])
            current_timestamps.append(elapsed)
            current_phase.append(in_phase)
            current_signature.append(is_signature)
            if len(current_raw) == PARENT_WINDOW_SIZE:
                raw_parents.append(current_raw)
                timestamp_parents.append(current_timestamps)
                phase_parents.append(current_phase)
                signature_parents.append(current_signature)
                current_raw, current_timestamps = [], []
                current_phase, current_signature = [], []
                if len(raw_parents) >= batch_parent_windows:
                    flush()
                    print(f"    {name}: parent windows={parent_index:,}")
    flush()
    if malformed / max(input_lines, 1) > 0.001:
        raise ValueError(
            f"More than 0.1% malformed candump lines in {path}: {malformed}/{input_lines}"
        )
    if not micro_batches or not parent_batches:
        raise ValueError(f"No complete windows parsed from {path}")
    micro = pd.concat(micro_batches, ignore_index=True)
    parent = pd.concat(parent_batches, ignore_index=True)
    summary = {
        "capture_name": name,
        "source_file": path.name,
        "capture_role": role,
        "attack_family": family,
        "metadata_elapsed_sec": metadata.get("elapsed_sec"),
        "parsed_elapsed_sec": float(parent["window_end_elapsed"].max()),
        "injection_interval_start": interval[0] if interval else math.nan,
        "injection_interval_end": interval[1] if interval else math.nan,
        "modified_masquerade": bool(metadata.get("modified", False)),
        "input_lines": input_lines,
        "malformed_lines": malformed,
        "parent_windows": len(parent),
        "micro_windows": len(micro),
        "discarded_tail_frames": len(current_raw),
        "phase_parent_windows": int(parent["phase_target"].sum()),
        "signature_frames": int(parent["signature_frame_count"].sum()),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
    return micro, parent, summary


def score_capture(
    micro: pd.DataFrame,
    parent: pd.DataFrame,
    identifier_model: object,
    structural_floors: np.ndarray,
    timing_floors: np.ndarray,
    structural_threshold: float,
    timing_threshold: float,
    w100_threshold: float,
    w100_scale_floor: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(micro) <= MICRO_BOOTSTRAP_WINDOWS or len(parent) <= PARENT_BOOTSTRAP_WINDOWS:
        raise ValueError(f"Capture {micro.iloc[0]['capture_name']} is too short")
    micro = micro.sort_values("window_index").copy()
    parent = parent.sort_values("window_index").copy()
    micro_bootstrap = micro.iloc[:MICRO_BOOTSTRAP_WINDOWS]
    parent_bootstrap = parent.iloc[:PARENT_BOOTSTRAP_WINDOWS]
    capture_name = str(micro.iloc[0]["capture_name"])

    def deviation(
        frame: pd.DataFrame,
        bootstrap: pd.DataFrame,
        features: Sequence[str],
        floors: np.ndarray,
    ) -> np.ndarray:
        baseline = bootstrap[list(features)].to_numpy(float)
        center = np.median(baseline, axis=0)
        scale = np.maximum(robust_scale(baseline, axis=0), floors)
        values = frame[list(features)].to_numpy(float)
        absolute_z = np.abs(values - center) / np.maximum(scale, EPSILON)
        return np.quantile(absolute_z, 0.75, axis=1)

    micro_eval = micro.iloc[MICRO_BOOTSTRAP_WINDOWS:].copy()
    micro_eval["structural_deviation"] = deviation(
        micro_eval, micro_bootstrap, STRUCTURAL_FEATURES, structural_floors
    )
    micro_eval["timing_deviation"] = deviation(
        micro_eval, micro_bootstrap, TIMING_FEATURES, timing_floors
    )
    micro_eval["structural_alarm"] = micro_eval["structural_deviation"] >= structural_threshold
    micro_eval["timing_alarm"] = micro_eval["timing_deviation"] >= timing_threshold
    micro_eval["combined_micro_alarm"] = micro_eval["structural_alarm"] | micro_eval["timing_alarm"]
    micro_eval["combined_micro_persistent_2"] = (
        micro_eval["combined_micro_alarm"]
        & micro_eval["combined_micro_alarm"].shift(1, fill_value=False)
    )
    micro_eval["parent_window_index"] = micro_eval["start_row"].astype(np.int64) // PARENT_WINDOW_SIZE
    micro_parent = micro_eval.groupby("parent_window_index", sort=True).agg(
        micro_windows=("window_index", "size"),
        structural_any=("structural_alarm", "max"),
        timing_any=("timing_alarm", "max"),
        combined_any=("combined_micro_alarm", "max"),
        combined_persistent_any=("combined_micro_persistent_2", "max"),
        structural_deviation_max=("structural_deviation", "max"),
        timing_deviation_max=("timing_deviation", "max"),
    ).reset_index()
    micro_parent = micro_parent[micro_parent["micro_windows"] == MICRO_PER_PARENT]

    probabilities = identifier_model.predict_proba(
        parent[list(STRUCTURAL_FEATURES)].to_numpy(float)
    )[:, 1]
    parent["identifier_log_odds"] = probability_to_log_odds(probabilities)
    bootstrap_values = parent.iloc[:PARENT_BOOTSTRAP_WINDOWS]["identifier_log_odds"].to_numpy(float)
    center = float(np.median(bootstrap_values))
    local_scale = float(robust_scale(bootstrap_values, axis=0))
    scale = max(local_scale, w100_scale_floor)
    parent_eval = parent.iloc[PARENT_BOOTSTRAP_WINDOWS:].copy()
    parent_eval["w100_deviation"] = np.abs(parent_eval["identifier_log_odds"] - center) / max(scale, EPSILON)
    parent_eval["w100_alarm_instant"] = parent_eval["w100_deviation"] >= w100_threshold
    parent_eval = parent_eval.merge(
        micro_parent,
        left_on="window_index",
        right_on="parent_window_index",
        how="inner",
        validate="one_to_one",
    )
    parent_eval["structural_multiscale_alarm"] = parent_eval["w100_alarm_instant"] | parent_eval["structural_any"]
    parent_eval["timing_multiscale_alarm"] = parent_eval["w100_alarm_instant"] | parent_eval["timing_any"]
    parent_eval["combined_multiscale_alarm"] = parent_eval["w100_alarm_instant"] | parent_eval["combined_any"]
    parent_eval["combined_persistent_multiscale_alarm"] = parent_eval["w100_alarm_instant"] | parent_eval["combined_persistent_any"]

    bootstrap_phase_frames = int(parent_bootstrap["phase_frame_count"].sum())
    audit = {
        "capture_name": capture_name,
        "capture_role": str(parent.iloc[0]["capture_role"]),
        "attack_family": str(parent.iloc[0]["attack_family"]),
        "micro_bootstrap_windows": MICRO_BOOTSTRAP_WINDOWS,
        "parent_bootstrap_windows": PARENT_BOOTSTRAP_WINDOWS,
        "bootstrap_frames": PARENT_BOOTSTRAP_WINDOWS * PARENT_WINDOW_SIZE,
        "bootstrap_end_elapsed": float(parent_bootstrap["window_end_elapsed"].max()),
        "bootstrap_phase_frames": bootstrap_phase_frames,
        "secure_start_valid": bootstrap_phase_frames == 0,
        "evaluated_parent_windows": len(parent_eval),
        "w100_local_scale": local_scale,
        "w100_scale_used": scale,
    }
    return parent_eval, audit


def evaluation_tables(
    primary: pd.DataFrame,
    compromised: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall: list[dict[str, object]] = []
    capture_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    ambient_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        overall.append(metric_row(primary, method, column, "primary_eligible_captures"))
        for name, group in primary.groupby("capture_name", sort=True):
            capture_rows.append(metric_row(group, method, column, f"capture:{name}"))
        for family, group in primary.groupby("attack_family", sort=True):
            family_rows.append(metric_row(group, method, column, f"family:{family}"))
        ambient = primary[primary["capture_role"] == "ambient"]
        for name, group in ambient.groupby("capture_name", sort=True):
            row = metric_row(group, method, column, f"ambient:{name}")
            low, high = wilson_interval(int(row["fp"]), int(row["benign_windows"]))
            row["fpr_wilson_95_low"] = low
            row["fpr_wilson_95_high"] = high
            ambient_rows.append(row)
        attacked = primary[(primary["phase_target"] == 1) & (primary["capture_role"] == "attack")]
        for label, lower, upper in (("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100)):
            group = attacked[attacked["signature_frame_count"].between(lower, upper)]
            density_rows.append({
                "method": method,
                "signature_frames_per_100": label,
                "parent_windows": len(group),
                "phase_recall": float(group[column].mean()) if len(group) else math.nan,
            })
    if not compromised.empty:
        for method, column in METHOD_COLUMNS.items():
            family_rows.append(metric_row(compromised, method, column, "family:accelerator_compromised_start_negative_control"))
    return overall, capture_rows, family_rows, ambient_rows, density_rows


def latency_rows(primary: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attacks = primary[(primary["capture_role"] == "attack") & (primary["phase_target"] == 1)]
    for name, group in attacks.groupby("capture_name", sort=True):
        ordered = group.sort_values("window_index")
        start = float(ordered["window_start_elapsed"].min())
        for method, column in METHOD_COLUMNS.items():
            detected = ordered[ordered[column]]
            rows.append({
                "capture_name": name,
                "attack_family": str(ordered.iloc[0]["attack_family"]),
                "method": method,
                "detected": not detected.empty,
                "latency_seconds_upper_bound": float(detected.iloc[0]["window_end_elapsed"] - start) if not detected.empty else math.nan,
                "latency_parent_windows": int(detected.iloc[0]["window_index"] - ordered.iloc[0]["window_index"]) if not detected.empty else math.nan,
            })
    return rows


def acceptance_rows(
    primary: pd.DataFrame,
    overall: pd.DataFrame,
    family: pd.DataFrame,
    ambient: pd.DataFrame,
    audits: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Apply criteria declared before the real ROAD results are observed."""
    method = "combined_multiscale_persistent_2"
    ambient_parent = primary[primary["capture_role"] == "ambient"]
    pooled = metric_row(
        ambient_parent,
        method,
        METHOD_COLUMNS[method],
        "all_ambient_captures",
    )
    _, pooled_upper = wilson_interval(int(pooled["fp"]), int(pooled["benign_windows"]))
    selected_ambient = ambient[ambient["method"] == method]
    macro_fpr = float(selected_ambient["false_positive_rate"].mean())
    captures_at_limit = int((selected_ambient["false_positive_rate"] <= 0.05).sum())
    capture_target = max(len(selected_ambient) - 2, 1)
    selected_overall = overall[overall["method"] == method].iloc[0]

    def family_recall(name: str) -> float:
        row = family[
            (family["method"] == method)
            & (family["scope"] == f"family:{name}")
        ]
        return float(row.iloc[0]["recall"]) if len(row) else math.nan

    eligible_attacks = [
        row
        for row in audits
        if row["capture_role"] == "attack" and bool(row["primary_eligible"])
    ]
    valid_bootstraps = sum(bool(row["secure_start_valid"]) for row in eligible_attacks)
    criteria = [
        ("persistent_overall_attack_recall", float(selected_overall["recall"]), ">=", 0.80),
        ("persistent_targeted_fabrication_recall", family_recall("fabrication_targeted"), ">=", 0.80),
        ("persistent_masquerade_recall", family_recall("masquerade"), ">=", 0.70),
        ("persistent_pooled_ambient_fpr", float(pooled["false_positive_rate"]), "<=", 0.05),
        ("persistent_pooled_ambient_fpr_wilson95_upper", pooled_upper, "<=", 0.05),
        ("persistent_macro_ambient_fpr", macro_fpr, "<=", 0.05),
        ("ambient_captures_at_or_below_5pct_fpr", float(captures_at_limit), ">=", float(capture_target)),
        ("eligible_attack_captures_with_valid_secure_start", float(valid_bootstraps), "==", float(len(eligible_attacks))),
    ]
    rows: list[dict[str, object]] = []
    for criterion, value, operator, target in criteria:
        if operator == ">=":
            passed = value >= target
        elif operator == "<=":
            passed = value <= target
        else:
            passed = value == target
        rows.append(
            {
                "criterion": criterion,
                "observed_value": value,
                "operator": operator,
                "required_value": target,
                "passed": bool(passed),
            }
        )
    overall_pass = all(bool(row["passed"]) for row in rows)
    rows.append(
        {
            "criterion": "all_predeclared_freeze_criteria",
            "observed_value": int(overall_pass),
            "operator": "==",
            "required_value": 1,
            "passed": overall_pass,
        }
    )
    return rows


def plot_results(
    family_metrics: pd.DataFrame,
    ambient_metrics: pd.DataFrame,
    output: Path,
) -> None:
    methods = ("w100_instant", "structural_multiscale_instant", "timing_multiscale_instant", "combined_multiscale_persistent_2")
    labels = ("w100", "structure", "timing", "persistent fusion")
    families = ("fabrication_fuzzing", "fabrication_targeted", "masquerade")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    x = np.arange(len(families))
    width = 0.19
    for index, (method, label) in enumerate(zip(methods, labels)):
        values = []
        for family in families:
            row = family_metrics[(family_metrics["method"] == method) & (family_metrics["scope"] == f"family:{family}")]
            values.append(float(row.iloc[0]["recall"]) if len(row) else math.nan)
        axes[0].bar(x + (index - 1.5) * width, values, width, label=label)
        ambient = ambient_metrics[ambient_metrics["method"] == method]
        axes[1].bar(index, float(ambient["false_positive_rate"].mean()), label=label)
    axes[0].set(title="Attack-phase recall", ylabel="Recall", ylim=(0, 1.03), xticks=x, xticklabels=["fuzzing", "targeted", "masquerade"])
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1, label="5% limit")
    axes[1].set(title="Macro ambient-capture FPR", ylabel="FPR", xticks=np.arange(len(methods)), xticklabels=labels)
    axes[1].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen ROAD external confirmation")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    road_root = locate_road_root(root)
    output_dir = root / "results" / "road_frozen_external_confirmation"
    print(f"ROAD root: {road_root}")

    feature_builder = load_script(locate_script(root, "03_build_window_dataset.py"), "ztav_step03_road")
    timing_module = load_script(locate_script(root, "27_timing_aware_sparse_can_gate.py"), "ztav_step27_road")
    identifier_model_path = root / "models" / "feature_family_ablation_w100" / "only_identifier_and_frame_structure.joblib"
    if not identifier_model_path.exists():
        raise FileNotFoundError(f"Missing frozen identifier model: {identifier_model_path}")
    identifier_model = joblib.load(identifier_model_path)
    if int(getattr(identifier_model, "n_features_in_", len(STRUCTURAL_FEATURES))) != len(STRUCTURAL_FEATURES):
        raise ValueError("Frozen identifier model does not expect seven structural features")

    structural_result = root / "results" / "multiscale_sparse_can_gate"
    timing_result = root / "results" / "timing_aware_sparse_can_gate"
    structural_floors = read_feature_floors(structural_result / "micro_feature_scale_floors.csv", STRUCTURAL_FEATURES)
    timing_floors = read_feature_floors(timing_result / "timing_feature_scale_floors.csv", TIMING_FEATURES)
    structural_threshold = read_single_value(structural_result / "micro_calibration_summary.csv", "micro_deviation_threshold")
    timing_threshold = read_single_value(timing_result / "timing_calibration_summary.csv", "timing_deviation_threshold")
    w100_threshold, w100_scale_floor = read_w100_parameters(root)

    ambient_metadata = load_json(road_root / "ambient" / "capture_metadata.json")
    attack_metadata = load_json(road_root / "attacks" / "capture_metadata.json")
    captures = [
        *( (path, name, meta, "ambient") for path, name, meta in list_capture_logs(road_root / "ambient", ambient_metadata) ),
        *( (path, name, meta, "attack") for path, name, meta in list_capture_logs(road_root / "attacks", attack_metadata) ),
    ]
    print(f"Captures: ambient={len(ambient_metadata)}, attacks={len(attack_metadata)}")

    predictions: list[pd.DataFrame] = []
    parser_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for index, (path, name, metadata, role) in enumerate(captures, start=1):
        print(f"[{index}/{len(captures)}] Processing {role} capture: {name}")
        micro, parent, parser_summary = process_capture(
            path, name, metadata, role, feature_builder, timing_module, args.batch_parent_windows
        )
        scored, audit = score_capture(
            micro, parent, identifier_model, structural_floors, timing_floors,
            structural_threshold, timing_threshold, w100_threshold, w100_scale_floor,
        )
        interval = metadata.get("injection_interval")
        if role == "ambient":
            eligible = True
            exclusion = ""
        elif interval is None:
            eligible = False
            exclusion = "no clean injection interval; compromised state spans capture"
        elif not bool(audit["secure_start_valid"]):
            eligible = False
            exclusion = "documented attack overlaps secure-start bootstrap"
        else:
            eligible = True
            exclusion = ""
        scored["primary_eligible"] = eligible
        scored["primary_exclusion_reason"] = exclusion
        audit["primary_eligible"] = eligible
        audit["primary_exclusion_reason"] = exclusion
        predictions.append(scored)
        parser_rows.append(parser_summary)
        audit_rows.append(audit)
        del micro, parent, scored

    prediction = pd.concat(predictions, ignore_index=True)
    primary = prediction[prediction["primary_eligible"]].copy()
    compromised = prediction[
        (~prediction["primary_eligible"])
        & (prediction["attack_family"] == "accelerator_compromised_state")
    ].copy()
    overall, per_capture, per_family, ambient_fpr, density = evaluation_tables(primary, compromised)
    latencies = latency_rows(primary)
    acceptance = acceptance_rows(
        primary,
        pd.DataFrame(overall),
        pd.DataFrame(per_family),
        pd.DataFrame(ambient_fpr),
        audit_rows,
    )
    manifest = [
        {"item": "experiment_type", "value": "frozen-threshold external ROAD confirmation"},
        {"item": "road_root", "value": str(road_root)},
        {"item": "ambient_captures", "value": len(ambient_metadata)},
        {"item": "attack_captures", "value": len(attack_metadata)},
        {"item": "primary_eligible_attack_captures", "value": sum(bool(row["primary_eligible"]) and row["capture_role"] == "attack" for row in audit_rows)},
        {"item": "compromised_start_negative_controls", "value": sum(not bool(row["primary_eligible"]) for row in audit_rows)},
        {"item": "threshold_source", "value": "Steps 17, 21, and 27 HCRL calibration outputs; frozen"},
        {"item": "road_labels_used_for_calibration", "value": "none"},
        {"item": "session_adaptation", "value": "first 1000 documented pre-attack frames; frozen thereafter"},
        {"item": "phase_label", "value": "ROAD metadata injection interval overlap"},
        {"item": "signature_count", "value": "metadata injection ID/payload wildcard match within injection interval"},
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_dir / "road_parent_predictions.csv", index=False)
    write_csv(output_dir / "road_overall_metrics.csv", overall)
    write_csv(output_dir / "road_per_capture_metrics.csv", per_capture)
    write_csv(output_dir / "road_per_family_metrics.csv", per_family)
    write_csv(output_dir / "road_ambient_fpr.csv", ambient_fpr)
    write_csv(output_dir / "road_signature_density_recall.csv", density)
    write_csv(output_dir / "road_detection_latencies.csv", latencies)
    write_csv(output_dir / "road_acceptance_criteria.csv", acceptance)
    write_csv(output_dir / "road_parser_summary.csv", parser_rows)
    write_csv(output_dir / "road_bootstrap_audit.csv", audit_rows)
    write_csv(output_dir / "road_confirmation_manifest.csv", manifest)
    plot_results(pd.DataFrame(per_family), pd.DataFrame(ambient_fpr), output_dir / "road_confirmation_summary.png")

    overall_frame = pd.DataFrame(overall).set_index("method")
    ambient_frame = pd.DataFrame(ambient_fpr)
    family_frame = pd.DataFrame(per_family)
    acceptance_frame = pd.DataFrame(acceptance).set_index("criterion")
    print("\n" + "=" * 88)
    print("Frozen ROAD external confirmation completed successfully.")
    print(f"Primary eligible attack captures: {sum(bool(row['primary_eligible']) and row['capture_role'] == 'attack' for row in audit_rows)}")
    print(f"Compromised-start negative controls: {sum(not bool(row['primary_eligible']) for row in audit_rows)}")
    print("\nPrimary operating points:")
    for method in METHOD_COLUMNS:
        row = overall_frame.loc[method]
        macro_ambient_fpr = float(
            ambient_frame[ambient_frame["method"] == method]["false_positive_rate"].mean()
        )
        print(
            f"  {method:<36} precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, pooled FPR={row['false_positive_rate']:.4f}, "
            f"ambient macro FPR={macro_ambient_fpr:.4f}"
        )
    print("\nPersistent-fusion recall by attack family:")
    selected = family_frame[family_frame["method"] == "combined_multiscale_persistent_2"]
    for family in ("fabrication_fuzzing", "fabrication_targeted", "masquerade"):
        row = selected[selected["scope"] == f"family:{family}"]
        if len(row):
            print(f"  {family:<24} recall={float(row.iloc[0]['recall']):.4f}")
    final_pass = bool(
        acceptance_frame.loc["all_predeclared_freeze_criteria", "passed"]
    )
    print(f"\nPredeclared freeze criteria passed: {final_pass}")
    print(f"\nResults directory: {output_dir}")
    print("\nNext: freeze only if ambient FPR and difficult-family recall satisfy the predeclared criteria.")


if __name__ == "__main__":
    main()
