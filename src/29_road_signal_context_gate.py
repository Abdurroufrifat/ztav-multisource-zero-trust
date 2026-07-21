#!/usr/bin/env python3
"""Ambient-trained semantic context gate for ROAD signal extractions.

Step 28 showed that CAN-structure and timing thresholds frozen on HCRL do not
transfer safely to ROAD.  This post-confirmation *development* stage therefore
uses ROAD's decoded signal time series to learn vehicle-specific healthy
context.  Attack labels are never used to fit a profile, calibrate a threshold,
or select a rule.

Leakage controls
----------------
* Only files named ``ambient_*.csv`` are eligible for healthy enrollment.
  Some ROAD archives contain attack-named CSV duplicates below the ambient
  directory; those files are deliberately ignored.
* Twelve ambient captures are assigned by a stable SHA-256 ordering to eight
  profile-fit, two benign-calibration, and two untouched benign-holdout files.
* Thresholds use only the two benign-calibration captures.
* The primary rule is predeclared as a corroborated, two-window-persistent
  semantic-context risk score.
* Signal-translated masquerade captures are used only after the gate is frozen.
  Accelerator captures have no clean attack interval and are reported as
  compromised-state controls, not included in the primary endpoint.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


SIGNAL_COLUMNS = tuple(f"Signal_{index}_of_ID" for index in range(1, 23))
REQUIRED_COLUMNS = ("Label", "Time", "ID", *SIGNAL_COLUMNS)
SPLIT_SALT = "ztav-road-signal-v1"
FIT_CAPTURES = 8
CALIBRATION_CAPTURES = 2
HOLDOUT_CAPTURES = 2
TARGET_CALIBRATION_FPR = 0.05
MAX_CALIBRATION_CAPTURE_FPR = 0.10
UNKNOWN_SCORE = 50.0
EPSILON = 1e-12
PRIMARY_METHOD = "semantic_context_persistent_2"
METHODS = {
    "semantic_any_instant": ("risk_any", 1),
    "semantic_context_instant": ("risk_context", 1),
    "semantic_context_persistent_2": ("risk_context", 2),
    "semantic_consensus_persistent_2": ("risk_consensus", 2),
}


@dataclass
class SignalProfile:
    can_id: int
    active_indices: np.ndarray
    value_center: np.ndarray
    value_scale: np.ndarray
    value_precision: np.ndarray
    delta_center: np.ndarray
    delta_scale: np.ndarray
    training_rows: int
    delta_rows: int


@dataclass
class Reservoir:
    values: np.ndarray
    priorities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROAD ambient-trained semantic signal context gate."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--window-rows", type=int, default=100)
    parser.add_argument("--reservoir-per-id", type=int, default=3_000)
    parser.add_argument("--min-id-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=2027)
    args = parser.parse_args()
    if args.chunk_rows < 1_000:
        parser.error("--chunk-rows must be at least 1000")
    if args.window_rows < 20:
        parser.error("--window-rows must be at least 20")
    if args.reservoir_per_id < 200:
        parser.error("--reservoir-per-id must be at least 200")
    if args.min_id_samples < 30:
        parser.error("--min-id-samples must be at least 30")
    return args


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def locate_signal_root(root: Path) -> Path:
    candidates = (
        root / "data" / "external" / "road" / "road" / "signal_extractions",
        root / "data" / "external" / "road" / "signal_extractions",
    )
    for candidate in candidates:
        if (candidate / "ambient").is_dir() and (candidate / "attacks").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find ROAD signal_extractions/ambient and signal_extractions/attacks"
    )


def locate_attack_metadata(signal_root: Path) -> Path:
    candidates = (
        signal_root / "attacks" / "metadata.json",
        signal_root.parent / "attacks" / "capture_metadata.json",
        signal_root.parent / "attacks" / "metadata.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find ROAD signal attack metadata JSON")


def load_metadata(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty metadata object: {path}")
    return data


def csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def validate_schema(path: Path) -> None:
    header = csv_header(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")


def list_files(
    signal_root: Path,
    metadata: dict[str, dict[str, object]],
) -> tuple[dict[str, Path], dict[str, Path]]:
    ambient = {
        path.stem: path
        for path in (signal_root / "ambient").glob("ambient_*.csv")
        if not path.name.startswith("._")
    }
    if len(ambient) != 12:
        raise ValueError(
            f"Expected exactly 12 ROAD ambient_*.csv captures; found {len(ambient)}"
        )
    attack_all = {
        path.stem: path
        for path in (signal_root / "attacks").glob("*.csv")
        if not path.name.startswith("._")
    }
    missing = sorted(set(metadata) - set(attack_all))
    if missing:
        raise FileNotFoundError(f"Signal attack CSVs missing for metadata: {missing}")
    attacks = {name: attack_all[name] for name in sorted(metadata)}
    for path in (*ambient.values(), *attacks.values()):
        validate_schema(path)
    return ambient, attacks


def stable_ambient_split(names: Iterable[str]) -> dict[str, str]:
    ordered = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{SPLIT_SALT}:{name}".encode("utf-8")
        ).hexdigest(),
    )
    expected = FIT_CAPTURES + CALIBRATION_CAPTURES + HOLDOUT_CAPTURES
    if len(ordered) != expected:
        raise ValueError(f"Expected {expected} ambient captures; found {len(ordered)}")
    split: dict[str, str] = {}
    for name in ordered[:FIT_CAPTURES]:
        split[name] = "profile_fit"
    for name in ordered[FIT_CAPTURES : FIT_CAPTURES + CALIBRATION_CAPTURES]:
        split[name] = "threshold_calibration"
    for name in ordered[-HOLDOUT_CAPTURES:]:
        split[name] = "benign_holdout"
    return split


def read_chunks(path: Path, chunk_rows: int) -> Iterable[pd.DataFrame]:
    dtype = {"Label": "float32", "Time": "float64", "ID": "float64"}
    dtype.update({column: "float32" for column in SIGNAL_COLUMNS})
    for chunk in pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype=dtype,
        chunksize=chunk_rows,
        low_memory=False,
    ):
        chunk = chunk.reset_index(drop=True)
        chunk["ID"] = pd.to_numeric(chunk["ID"], errors="coerce")
        chunk = chunk[chunk["ID"].notna()].reset_index(drop=True)
        if len(chunk):
            yield chunk


def update_reservoir(
    store: dict[int, Reservoir],
    can_id: int,
    values: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    valid = np.any(np.isfinite(values), axis=1)
    values = values[valid]
    if not len(values):
        return
    priorities = rng.random(len(values))
    previous = store.get(can_id)
    if previous is not None:
        values = np.vstack((previous.values, values))
        priorities = np.concatenate((previous.priorities, priorities))
    if len(values) > capacity:
        selected = np.argpartition(priorities, -capacity)[-capacity:]
        values = values[selected]
        priorities = priorities[selected]
    store[can_id] = Reservoir(values=values.astype(np.float32), priorities=priorities)


def group_positions(ids: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(sorted_ids)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        yield int(sorted_ids[left]), order[left:right]


def fit_reservoirs(
    files: Sequence[tuple[str, Path]],
    chunk_rows: int,
    capacity: int,
    seed: int,
) -> tuple[dict[int, Reservoir], dict[int, Reservoir], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    value_store: dict[int, Reservoir] = {}
    delta_store: dict[int, Reservoir] = {}
    audit: list[dict[str, object]] = []
    for file_index, (capture_name, path) in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] Fitting ambient profile: {capture_name}")
        last_by_id: dict[int, np.ndarray] = {}
        rows = 0
        for chunk in read_chunks(path, chunk_rows):
            ids = chunk["ID"].to_numpy(np.int64)
            matrix = chunk[list(SIGNAL_COLUMNS)].to_numpy(np.float32)
            rows += len(chunk)
            for can_id, positions in group_positions(ids):
                values = matrix[positions]
                update_reservoir(value_store, can_id, values, capacity, rng)
                previous = np.empty_like(values)
                previous[1:] = values[:-1]
                previous[0] = last_by_id.get(
                    can_id, np.full(values.shape[1], np.nan, dtype=np.float32)
                )
                deltas = values - previous
                update_reservoir(delta_store, can_id, deltas, capacity, rng)
                last_by_id[can_id] = values[-1].copy()
        audit.append(
            {
                "capture_name": capture_name,
                "split": "profile_fit",
                "rows_read": rows,
                "file_bytes": path.stat().st_size,
            }
        )
    return value_store, delta_store, audit


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    absolute = np.abs(values - center)
    mad = np.nanmedian(absolute, axis=0) * 1.4826
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    magnitude_floor = np.maximum(0.05, 1e-4 * np.maximum(np.abs(center), 1.0))
    scale = np.maximum.reduce((mad, iqr_scale, magnitude_floor))
    return center.astype(float), scale.astype(float)


def row_nanquantile(values: np.ndarray, quantile: float) -> np.ndarray:
    """Row quantile with zero for rows that contain no finite evidence."""
    output = np.zeros(len(values), dtype=float)
    valid = np.any(np.isfinite(values), axis=1)
    if np.any(valid):
        output[valid] = np.nanquantile(values[valid], quantile, axis=1)
    return output


def build_profiles(
    value_store: dict[int, Reservoir],
    delta_store: dict[int, Reservoir],
    min_samples: int,
) -> tuple[dict[int, SignalProfile], list[dict[str, object]]]:
    profiles: dict[int, SignalProfile] = {}
    rows: list[dict[str, object]] = []
    for can_id in sorted(value_store):
        values = value_store[can_id].values.astype(float)
        counts = np.isfinite(values).sum(axis=0)
        active = np.flatnonzero(counts >= min_samples)
        if not len(active):
            continue
        selected = values[:, active]
        value_center, value_scale = robust_center_scale(selected)
        standardized = (selected - value_center) / value_scale
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        if len(active) == 1:
            precision = np.ones((1, 1), dtype=float)
        else:
            precision = LedoitWolf(assume_centered=False).fit(standardized).precision_

        delta_values = delta_store.get(can_id)
        if delta_values is None:
            delta_selected = np.zeros((1, len(active)), dtype=float)
        else:
            delta_selected = delta_values.values[:, active].astype(float)
        delta_counts = np.isfinite(delta_selected).sum(axis=0)
        usable_delta = delta_counts >= max(30, min_samples // 2)
        delta_center = np.zeros(len(active), dtype=float)
        delta_scale = np.full(len(active), np.inf, dtype=float)
        if np.any(usable_delta):
            center, scale = robust_center_scale(delta_selected[:, usable_delta])
            delta_center[usable_delta] = center
            delta_scale[usable_delta] = scale

        profiles[can_id] = SignalProfile(
            can_id=can_id,
            active_indices=active.astype(np.int16),
            value_center=value_center,
            value_scale=value_scale,
            value_precision=np.asarray(precision, dtype=float),
            delta_center=delta_center,
            delta_scale=delta_scale,
            training_rows=len(values),
            delta_rows=len(delta_selected),
        )
        rows.append(
            {
                "can_id": can_id,
                "active_signals": len(active),
                "value_training_rows": len(values),
                "delta_training_rows": len(delta_selected),
                "active_signal_indices": "|".join(str(int(index) + 1) for index in active),
            }
        )
    if not profiles:
        raise RuntimeError("No signal profiles met the minimum sample requirement")
    return profiles, rows


def score_chunk(
    chunk: pd.DataFrame,
    profiles: dict[int, SignalProfile],
    last_by_id: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    ids = chunk["ID"].to_numpy(np.int64)
    matrix = chunk[list(SIGNAL_COLUMNS)].to_numpy(np.float32)
    length = len(chunk)
    marginal = np.full(length, UNKNOWN_SCORE, dtype=float)
    context = np.full(length, UNKNOWN_SCORE, dtype=float)
    transition = np.full(length, UNKNOWN_SCORE, dtype=float)
    unknown = np.ones(length, dtype=float)

    for can_id, positions in group_positions(ids):
        values = matrix[positions].astype(float)
        profile = profiles.get(can_id)
        if profile is None:
            last_by_id[can_id] = values[-1].copy()
            continue
        active = profile.active_indices.astype(int)
        selected = values[:, active]
        z = (selected - profile.value_center) / profile.value_scale
        finite = np.isfinite(z)
        abs_z = np.where(finite, np.abs(z), np.nan)
        row_marginal = row_nanquantile(abs_z, 0.90)
        row_marginal = np.nan_to_num(row_marginal, nan=0.0, posinf=UNKNOWN_SCORE)
        z_filled = np.where(finite, z, 0.0)
        mahal = np.einsum(
            "ij,jk,ik->i", z_filled, profile.value_precision, z_filled, optimize=True
        )
        row_context = np.sqrt(np.maximum(mahal, 0.0) / max(len(active), 1))

        previous = np.empty_like(values)
        previous[1:] = values[:-1]
        previous[0] = last_by_id.get(
            can_id, np.full(values.shape[1], np.nan, dtype=float)
        )
        deltas = values[:, active] - previous[:, active]
        delta_z = (deltas - profile.delta_center) / profile.delta_scale
        delta_abs = np.where(np.isfinite(delta_z), np.abs(delta_z), np.nan)
        row_transition = row_nanquantile(delta_abs, 0.90)
        row_transition = np.nan_to_num(row_transition, nan=0.0, posinf=UNKNOWN_SCORE)

        inactive = np.ones(values.shape[1], dtype=bool)
        inactive[active] = False
        novel_signal = np.any(np.isfinite(values[:, inactive]), axis=1) if np.any(inactive) else np.zeros(len(values), dtype=bool)
        no_known_signal = ~np.any(np.isfinite(selected), axis=1)
        row_unknown = novel_signal | no_known_signal
        row_marginal[row_unknown] = np.maximum(row_marginal[row_unknown], UNKNOWN_SCORE)
        row_context[row_unknown] = np.maximum(row_context[row_unknown], UNKNOWN_SCORE)
        row_transition[row_unknown] = np.maximum(row_transition[row_unknown], UNKNOWN_SCORE)

        marginal[positions] = row_marginal
        context[positions] = row_context
        transition[positions] = row_transition
        unknown[positions] = row_unknown.astype(float)
        last_by_id[can_id] = values[-1].copy()

    return {
        "marginal_score": marginal,
        "context_score": context,
        "transition_score": transition,
        "unknown_signal": unknown,
    }


def windowize_capture(
    capture_name: str,
    path: Path,
    role: str,
    attack_family: str,
    profiles: dict[int, SignalProfile],
    chunk_rows: int,
    window_rows: int,
) -> pd.DataFrame:
    pending: dict[str, np.ndarray] = {
        "time": np.empty(0, dtype=float),
        "label": np.empty(0, dtype=np.int8),
        "marginal_score": np.empty(0, dtype=float),
        "context_score": np.empty(0, dtype=float),
        "transition_score": np.empty(0, dtype=float),
        "unknown_signal": np.empty(0, dtype=float),
    }
    last_by_id: dict[int, np.ndarray] = {}
    batches: list[pd.DataFrame] = []
    window_offset = 0
    for chunk in read_chunks(path, chunk_rows):
        scored = score_chunk(chunk, profiles, last_by_id)
        arrays = {
            "time": chunk["Time"].to_numpy(float),
            "label": pd.to_numeric(chunk["Label"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy(np.int8),
            **scored,
        }
        arrays = {
            key: np.concatenate((pending[key], value))
            for key, value in arrays.items()
        }
        complete = (len(arrays["time"]) // window_rows) * window_rows
        if complete:
            count = complete // window_rows
            reshaped = {
                key: value[:complete].reshape(count, window_rows)
                for key, value in arrays.items()
            }
            batch = pd.DataFrame(
                {
                    "capture_name": capture_name,
                    "capture_role": role,
                    "attack_family": attack_family,
                    "window_index": np.arange(window_offset, window_offset + count),
                    "window_start_time": reshaped["time"][:, 0],
                    "window_end_time": reshaped["time"][:, -1],
                    "attack_target": reshaped["label"].max(axis=1).astype(np.int8),
                    "attack_row_fraction": reshaped["label"].mean(axis=1),
                    "marginal_q95": np.quantile(reshaped["marginal_score"], 0.95, axis=1),
                    "context_q95": np.quantile(reshaped["context_score"], 0.95, axis=1),
                    "transition_q95": np.quantile(reshaped["transition_score"], 0.95, axis=1),
                    "unknown_signal_fraction": reshaped["unknown_signal"].mean(axis=1),
                }
            )
            batches.append(batch)
            window_offset += count
        pending = {key: value[complete:] for key, value in arrays.items()}
    if not batches:
        raise ValueError(f"Capture {capture_name} produced no complete windows")
    output = pd.concat(batches, ignore_index=True)
    output["discarded_tail_rows"] = len(pending["time"])
    return output


def balanced_reference(calibration: pd.DataFrame, column: str) -> np.ndarray:
    groups = [group[column].to_numpy(float) for _, group in calibration.groupby("capture_name")]
    target = min(len(values) for values in groups)
    if target < 10:
        raise ValueError("Calibration capture has too few windows")
    sampled: list[np.ndarray] = []
    for values in groups:
        if len(values) == target:
            sampled.append(values)
        else:
            positions = np.linspace(0, len(values) - 1, target).round().astype(int)
            sampled.append(values[positions])
    return np.sort(np.concatenate(sampled))


def ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference, values, side="right") / max(len(reference), 1)


def add_risk_columns(
    frames: Sequence[pd.DataFrame], calibration: pd.DataFrame
) -> dict[str, np.ndarray]:
    evidence = ("marginal_q95", "context_q95", "transition_q95")
    references = {column: balanced_reference(calibration, column) for column in evidence}
    for frame in frames:
        percentiles = np.column_stack(
            [ecdf(references[column], frame[column].to_numpy(float)) for column in evidence]
        )
        ordered = np.sort(percentiles, axis=1)
        frame["risk_any"] = ordered[:, 2]
        frame["risk_consensus"] = ordered[:, 1]
        frame["risk_context"] = (ordered[:, 1] + ordered[:, 2]) / 2.0
    return references


def alarm_for_group(values: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    instant = values >= threshold
    if persistence == 1:
        return instant
    if persistence != 2:
        raise ValueError(f"Unsupported persistence: {persistence}")
    return instant & np.r_[False, instant[:-1]]


def add_alarm(
    frame: pd.DataFrame,
    output_column: str,
    risk_column: str,
    threshold: float,
    persistence: int,
) -> None:
    alarm = np.zeros(len(frame), dtype=bool)
    for _, positions in frame.groupby("capture_name", sort=False).indices.items():
        positions = np.asarray(positions)
        ordered = positions[np.argsort(frame.iloc[positions]["window_index"].to_numpy())]
        alarm[ordered] = alarm_for_group(
            frame.iloc[ordered][risk_column].to_numpy(float), threshold, persistence
        )
    frame[output_column] = alarm


def select_threshold(
    calibration: pd.DataFrame,
    risk_column: str,
    persistence: int,
) -> dict[str, object]:
    grid = np.linspace(0.50, 1.000001, 2_001)
    selected: dict[str, object] | None = None
    for threshold in grid:
        rates: list[float] = []
        alarms = 0
        windows = 0
        for _, group in calibration.groupby("capture_name", sort=True):
            ordered = group.sort_values("window_index")
            alarm = alarm_for_group(
                ordered[risk_column].to_numpy(float), float(threshold), persistence
            )
            rates.append(float(alarm.mean()))
            alarms += int(alarm.sum())
            windows += len(alarm)
        pooled = alarms / max(windows, 1)
        macro = float(np.mean(rates))
        worst = float(np.max(rates))
        if (
            pooled <= TARGET_CALIBRATION_FPR
            and macro <= TARGET_CALIBRATION_FPR
            and worst <= MAX_CALIBRATION_CAPTURE_FPR
        ):
            selected = {
                "threshold": float(threshold),
                "calibration_pooled_fpr": pooled,
                "calibration_macro_fpr": macro,
                "calibration_worst_capture_fpr": worst,
                "calibration_windows": windows,
            }
            break
    if selected is None:
        raise RuntimeError("Could not find a benign-only threshold satisfying availability limits")
    return selected


def metric_row(
    frame: pd.DataFrame,
    method: str,
    prediction_column: str,
    scope: str,
) -> dict[str, object]:
    truth = frame["attack_target"].to_numpy(np.int8)
    prediction = frame[prediction_column].to_numpy(np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    if len(np.unique(truth)) == 2:
        balanced = balanced_accuracy_score(truth, prediction)
        mcc = matthews_corrcoef(truth, prediction)
    else:
        balanced = math.nan
        mcc = math.nan
    return {
        "method": method,
        "scope": scope,
        "windows": len(frame),
        "benign_windows": int((truth == 0).sum()),
        "attack_windows": int((truth == 1).sum()),
        "balanced_accuracy": balanced,
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "mcc": mcc,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def attack_family(name: str) -> str:
    if name.startswith("accelerator_attack"):
        return "accelerator_compromised_state"
    if name.startswith("correlated_signal"):
        return "correlated_signal_masquerade"
    if name.startswith("max_engine_coolant"):
        return "max_engine_coolant_masquerade"
    if name.startswith("max_speedometer"):
        return "max_speedometer_masquerade"
    if name.startswith("reverse_light_off"):
        return "reverse_light_off_masquerade"
    if name.startswith("reverse_light_on"):
        return "reverse_light_on_masquerade"
    return "other_signal_attack"


def evaluate(
    holdout: pd.DataFrame,
    attacks: pd.DataFrame,
    controls: pd.DataFrame,
    thresholds: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    endpoint = pd.concat((holdout, attacks), ignore_index=True)
    for method in METHODS:
        column = f"alarm_{method}"
        overall.append(metric_row(endpoint, method, column, "benign_holdout_plus_signal_attacks"))
        for name, group in holdout.groupby("capture_name", sort=True):
            row = metric_row(group, method, column, f"holdout:{name}")
            holdout_rows.append(row)
        for family, group in attacks.groupby("attack_family", sort=True):
            family_rows.append(metric_row(group, method, column, f"family:{family}"))
        if len(controls):
            for name, group in controls.groupby("capture_name", sort=True):
                control_rows.append(
                    {
                        "method": method,
                        "capture_name": name,
                        "windows": len(group),
                        "compromised_state_alarm_rate": float(group[column].mean()),
                        "note": "negative control; capture is compromised from startup",
                    }
                )
    return overall, holdout_rows, family_rows, control_rows


def acceptance_rows(
    overall: pd.DataFrame,
    holdout: pd.DataFrame,
    families: pd.DataFrame,
    attack_audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    primary_overall = overall[overall["method"] == PRIMARY_METHOD].iloc[0]
    primary_holdout = holdout[holdout["method"] == PRIMARY_METHOD]
    primary_families = families[families["method"] == PRIMARY_METHOD]
    pooled_holdout = primary_holdout["fp"].sum() / max(primary_holdout["benign_windows"].sum(), 1)
    macro_holdout = float(primary_holdout["false_positive_rate"].mean())
    worst_holdout = float(primary_holdout["false_positive_rate"].max())
    worst_family = float(primary_families["recall"].min())
    labeled_primary = [row for row in attack_audit if row["endpoint_role"] == "primary_signal_attack"]
    all_labeled = all(int(row["positive_windows"]) > 0 for row in labeled_primary)
    criteria = [
        ("holdout_pooled_fpr", pooled_holdout, "<=", 0.05),
        ("holdout_macro_fpr", macro_holdout, "<=", 0.05),
        ("holdout_worst_capture_fpr", worst_holdout, "<=", 0.10),
        ("primary_masquerade_recall", float(primary_overall["recall"]), ">=", 0.70),
        ("worst_attack_family_recall", worst_family, ">=", 0.50),
        ("primary_attack_captures_with_positive_labels", float(sum(int(row["positive_windows"]) > 0 for row in labeled_primary)), "==", float(len(labeled_primary))),
        ("attack_labels_used_for_training_or_calibration", 0.0, "==", 0.0),
    ]
    rows: list[dict[str, object]] = []
    passed_values: list[bool] = []
    for name, observed, operator, required in criteria:
        if operator == "<=":
            passed = observed <= required
        elif operator == ">=":
            passed = observed >= required
        else:
            passed = observed == required
        if name == "primary_attack_captures_with_positive_labels":
            passed = passed and all_labeled
        passed_values.append(bool(passed))
        rows.append(
            {
                "criterion": name,
                "observed_value": observed,
                "operator": operator,
                "required_value": required,
                "passed": bool(passed),
            }
        )
    rows.append(
        {
            "criterion": "all_predeclared_candidate_readiness_criteria",
            "observed_value": float(all(passed_values)),
            "operator": "==",
            "required_value": 1.0,
            "passed": bool(all(passed_values)),
        }
    )
    return rows


def plot_summary(
    family_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    path: Path,
) -> None:
    selected_family = family_metrics[family_metrics["method"] == PRIMARY_METHOD].copy()
    selected_family["family"] = selected_family["scope"].str.replace("family:", "", regex=False)
    selected_holdout = holdout_metrics[holdout_metrics["method"] == PRIMARY_METHOD].copy()
    selected_holdout["capture"] = selected_holdout["scope"].str.replace("holdout:", "", regex=False)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(selected_family["family"], selected_family["recall"], color="#2ca02c")
    axes[0].axhline(0.50, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Attack-window recall")
    axes[0].set_title("Signal-semantic recall by family")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(selected_holdout["capture"], selected_holdout["false_positive_rate"], color="#d62728")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, max(0.12, float(selected_holdout["false_positive_rate"].max()) * 1.15))
    axes[1].set_ylabel("False-positive rate")
    axes[1].set_title("Untouched benign-capture availability")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("ROAD ambient-trained semantic context gate")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    signal_root = locate_signal_root(root)
    metadata_path = locate_attack_metadata(signal_root)
    metadata = load_metadata(metadata_path)
    ambient_files, attack_files = list_files(signal_root, metadata)
    split = stable_ambient_split(ambient_files)

    output_dir = root / "results" / "road_signal_context_gate"
    model_dir = root / "models" / "road_signal_context_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    split_rows = [
        {
            "capture_name": name,
            "split": split[name],
            "path": str(ambient_files[name]),
            "split_rule": f"SHA256({SPLIT_SALT}:capture_name)",
        }
        for name in sorted(ambient_files)
    ]
    fit_files = sorted(
        ((name, ambient_files[name]) for name in ambient_files if split[name] == "profile_fit")
    )
    calibration_files = sorted(
        ((name, ambient_files[name]) for name in ambient_files if split[name] == "threshold_calibration")
    )
    holdout_files = sorted(
        ((name, ambient_files[name]) for name in ambient_files if split[name] == "benign_holdout")
    )

    print("Fitting ROAD healthy signal profiles (attack labels are not loaded) ...")
    value_store, delta_store, fit_audit = fit_reservoirs(
        fit_files,
        args.chunk_rows,
        args.reservoir_per_id,
        args.random_seed,
    )
    profiles, profile_rows = build_profiles(value_store, delta_store, args.min_id_samples)
    del value_store, delta_store
    print(f"Learned semantic profiles for {len(profiles)} arbitration IDs.")

    calibration_frames: list[pd.DataFrame] = []
    holdout_frames: list[pd.DataFrame] = []
    for name, path in calibration_files:
        print(f"Scoring benign calibration capture: {name}")
        calibration_frames.append(
            windowize_capture(
                name, path, "ambient_calibration", "ambient", profiles,
                args.chunk_rows, args.window_rows,
            )
        )
    for name, path in holdout_files:
        print(f"Scoring untouched benign holdout: {name}")
        holdout_frames.append(
            windowize_capture(
                name, path, "ambient_holdout", "ambient", profiles,
                args.chunk_rows, args.window_rows,
            )
        )
    calibration = pd.concat(calibration_frames, ignore_index=True)
    holdout = pd.concat(holdout_frames, ignore_index=True)

    attack_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    attack_audit: list[dict[str, object]] = []
    for index, name in enumerate(sorted(attack_files), start=1):
        family = attack_family(name)
        interval = metadata[name].get("injection_interval")
        endpoint_role = (
            "compromised_start_control" if interval is None else "primary_signal_attack"
        )
        print(f"[{index}/{len(attack_files)}] Scoring {endpoint_role}: {name}")
        scored = windowize_capture(
            name, attack_files[name], endpoint_role, family, profiles,
            args.chunk_rows, args.window_rows,
        )
        attack_audit.append(
            {
                "capture_name": name,
                "attack_family": family,
                "endpoint_role": endpoint_role,
                "windows": len(scored),
                "positive_windows": int(scored["attack_target"].sum()),
                "positive_rows_equivalent": float(scored["attack_row_fraction"].sum() * args.window_rows),
                "metadata_has_injection_interval": interval is not None,
            }
        )
        if endpoint_role == "primary_signal_attack":
            attack_frames.append(scored)
        else:
            control_frames.append(scored)
    attacks = pd.concat(attack_frames, ignore_index=True)
    controls = pd.concat(control_frames, ignore_index=True) if control_frames else pd.DataFrame()

    references = add_risk_columns(
        [calibration, holdout, attacks, *([controls] if len(controls) else [])],
        calibration,
    )
    thresholds: dict[str, dict[str, object]] = {}
    threshold_rows: list[dict[str, object]] = []
    all_frames = [calibration, holdout, attacks, *([controls] if len(controls) else [])]
    for method, (risk_column, persistence) in METHODS.items():
        selected = select_threshold(calibration, risk_column, persistence)
        thresholds[method] = selected
        threshold_rows.append(
            {
                "method": method,
                "risk_column": risk_column,
                "persistence_windows": persistence,
                **selected,
                "attack_labels_used": False,
            }
        )
        for frame in all_frames:
            add_alarm(
                frame,
                f"alarm_{method}",
                risk_column,
                float(selected["threshold"]),
                persistence,
            )

    overall, holdout_rows, family_rows, control_rows = evaluate(
        holdout, attacks, controls, thresholds
    )
    acceptance = acceptance_rows(
        pd.DataFrame(overall),
        pd.DataFrame(holdout_rows),
        pd.DataFrame(family_rows),
        attack_audit,
    )
    manifest = [
        {"item": "experiment_stage", "value": "post-confirmation ROAD signal-semantic development"},
        {"item": "signal_root", "value": str(signal_root)},
        {"item": "profile_fit_ambient_captures", "value": len(fit_files)},
        {"item": "threshold_calibration_ambient_captures", "value": len(calibration_files)},
        {"item": "untouched_benign_holdout_captures", "value": len(holdout_files)},
        {"item": "primary_signal_attack_captures", "value": len(attack_frames)},
        {"item": "compromised_start_controls", "value": len(control_frames)},
        {"item": "primary_method_predeclared", "value": PRIMARY_METHOD},
        {"item": "attack_labels_used_for_profile_fit", "value": False},
        {"item": "attack_labels_used_for_threshold_calibration", "value": False},
        {"item": "window_rows", "value": args.window_rows},
        {"item": "reservoir_per_id", "value": args.reservoir_per_id},
        {"item": "minimum_id_samples", "value": args.min_id_samples},
        {"item": "random_seed", "value": args.random_seed},
    ]

    prediction = pd.concat((calibration, holdout, attacks, controls), ignore_index=True)
    prediction.to_csv(output_dir / "signal_context_predictions.csv", index=False)
    write_csv(output_dir / "signal_context_split_manifest.csv", split_rows)
    write_csv(output_dir / "signal_context_fit_audit.csv", fit_audit)
    write_csv(output_dir / "signal_context_profile_summary.csv", profile_rows)
    write_csv(output_dir / "signal_context_thresholds.csv", threshold_rows)
    write_csv(output_dir / "signal_context_overall_metrics.csv", overall)
    write_csv(output_dir / "signal_context_holdout_fpr.csv", holdout_rows)
    write_csv(output_dir / "signal_context_per_family_metrics.csv", family_rows)
    write_csv(output_dir / "signal_context_attack_label_audit.csv", attack_audit)
    if control_rows:
        write_csv(output_dir / "signal_context_compromised_controls.csv", control_rows)
    write_csv(output_dir / "signal_context_acceptance_criteria.csv", acceptance)
    write_csv(output_dir / "signal_context_manifest.csv", manifest)
    plot_summary(
        pd.DataFrame(family_rows),
        pd.DataFrame(holdout_rows),
        output_dir / "signal_context_summary.png",
    )
    model_payload = {
        "profiles": {can_id: asdict(profile) for can_id, profile in profiles.items()},
        "thresholds": thresholds,
        "references": references,
        "ambient_split": split,
        "signal_columns": SIGNAL_COLUMNS,
        "window_rows": args.window_rows,
        "primary_method": PRIMARY_METHOD,
        "manifest": manifest,
    }
    joblib.dump(model_payload, model_dir / "road_signal_context_gate.joblib", compress=3)

    overall_frame = pd.DataFrame(overall).set_index("method")
    holdout_frame = pd.DataFrame(holdout_rows)
    family_frame = pd.DataFrame(family_rows)
    acceptance_frame = pd.DataFrame(acceptance).set_index("criterion")
    selected = overall_frame.loc[PRIMARY_METHOD]
    selected_holdout = holdout_frame[holdout_frame["method"] == PRIMARY_METHOD]
    selected_family = family_frame[family_frame["method"] == PRIMARY_METHOD]
    print("\n" + "=" * 88)
    print("ROAD ambient-trained signal context gate completed successfully.")
    print(f"Profiles learned: {len(profiles)} IDs")
    print(f"Primary signal attack captures: {len(attack_frames)}")
    print(f"Compromised-start controls: {len(control_frames)}")
    print(f"Primary rule: {PRIMARY_METHOD}")
    print(
        f"Endpoint precision={selected['precision']:.4f}, recall={selected['recall']:.4f}, "
        f"F1={selected['f1']:.4f}, FPR={selected['false_positive_rate']:.4f}"
    )
    print(
        f"Untouched ambient pooled FPR={selected_holdout['fp'].sum() / max(selected_holdout['benign_windows'].sum(), 1):.4f}, "
        f"macro FPR={selected_holdout['false_positive_rate'].mean():.4f}"
    )
    print("\nPrimary recall by signal attack family:")
    for _, row in selected_family.iterrows():
        print(f"  {row['scope'].replace('family:', ''):<40} recall={row['recall']:.4f}")
    ready = bool(
        acceptance_frame.loc[
            "all_predeclared_candidate_readiness_criteria", "passed"
        ]
    )
    print(f"\nPredeclared candidate-readiness criteria passed: {ready}")
    print(f"Results directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print("\nNext: compare semantic-context results with Step 28 before integration.")


if __name__ == "__main__":
    main()
