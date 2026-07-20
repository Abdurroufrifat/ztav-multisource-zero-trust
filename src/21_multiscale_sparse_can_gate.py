#!/usr/bin/env python3
"""Add a 20-frame structural micro-gate for sparse CAN attacks.

Step 20 proved that a physical source cannot detect a CAN injection that causes
no simulated physical effect.  It also showed that the existing 100-frame CAN
gate is nearly blind when only 1--20 frames in a window are malicious.  This
stage adds a smaller, unsupervised structural window alongside the frozen
100-frame gate.

Protocol
--------
* Stream the HCRL raw captures into non-overlapping 20-frame windows.
* Retain seven identifier/frame-structure features only.
* Use 50 trusted startup micro-windows (1,000 frames) per session.
* Freeze each session baseline; never learn from later traffic.
* Calibrate the micro threshold on attack-free normal pseudo-sessions only.
* Use a 1% micro-window target FPR because five micro-windows are combined into
  one 100-frame decision (approximately a 5% family-wise target).
* Fuse by OR: an anomaly from either the frozen 100-frame gate or the new
  20-frame gate requests VERIFY/RESTRICT; two consecutive micro anomalies can
  request SAFE_FALLBACK.

Attack flags are used only for evaluation.  They are not used for baseline
fitting, threshold calibration, or detector decisions.

This is exploratory HCRL development, not independent validation.  If the
micro-gate improves sparse recall, it still requires poisoning stress and a new
dataset/capture before the final policy is frozen.

Run from D:\\ztav_project after Steps 17 and 20:

    .\\.venv\\Scripts\\python.exe src\\21_multiscale_sparse_can_gate.py

The first run builds a reusable w20 structural cache and can take several
minutes.  Later runs reuse that cache unless ``--rebuild-cache`` is supplied.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


MICRO_WINDOW_SIZE = 20
PARENT_WINDOW_SIZE = 100
MICRO_PER_PARENT = PARENT_WINDOW_SIZE // MICRO_WINDOW_SIZE
BOOTSTRAP_MICRO_WINDOWS = 50
PSEUDO_SESSION_MICRO_WINDOWS = 1_250
MICRO_TARGET_FPR = 0.01
GLOBAL_SCALE_FLOOR_FRACTION = 0.25
EPSILON = 1e-12
NORMAL_SOURCE = "normal_run_data.txt"
STRUCTURAL_FEATURES = (
    "id_unique_count",
    "id_entropy",
    "dominant_id_fraction",
    "frame_unique_count",
    "frame_unique_fraction",
    "id_change_rate",
    "consecutive_frame_repeat_rate",
)
COUNT_FEATURES = {"id_unique_count", "frame_unique_count"}
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
    "w20_any_instant": "w20_any_alarm_instant",
    "multiscale_instant": "multiscale_alarm_instant",
    "w100_persistent_2": "w100_alarm_persistent_2",
    "w20_any_persistent_2": "w20_any_alarm_persistent_2",
    "multiscale_persistent_2": "multiscale_alarm_persistent_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a multiscale sparse-CAN trust gate."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows < MICRO_WINDOW_SIZE:
        parser.error(f"--chunk-rows must be at least {MICRO_WINDOW_SIZE}")
    return args


def locate_script(project_root: Path, name: str) -> Path:
    for candidate in (project_root / "src" / name, project_root / name):
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


def build_structural_cache(
    project_root: Path,
    cache_path: Path,
    parser_summary_path: Path,
    chunk_rows: int,
) -> pd.DataFrame:
    step03 = load_script(
        locate_script(project_root, "03_build_window_dataset.py"),
        "ztav_step03_multiscale",
    )
    step13 = load_script(
        locate_script(project_root, "13_external_car_hacking_zero_shot.py"),
        "ztav_step13_multiscale",
    )
    step13.WINDOW_SIZE = MICRO_WINDOW_SIZE
    data_dir = project_root / "data" / "external" / "car_hacking"
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find HCRL directory: {data_dir}")
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    keep = list(META_COLUMNS + STRUCTURAL_FEATURES)
    for file_name, source_class in step13.ATTACK_FILES:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing HCRL attack capture: {path}")
        print(f"Building {MICRO_WINDOW_SIZE}-frame structural windows: {file_name}")
        windows, summary = step13.process_attack_csv(
            path,
            source_class,
            step03,
            chunk_rows=chunk_rows,
        )
        frames.append(windows[keep].copy())
        summary["window_size"] = MICRO_WINDOW_SIZE
        summaries.append(summary)
        del windows
    normal_path = step13.locate_normal_capture(data_dir)
    print(f"Building {MICRO_WINDOW_SIZE}-frame structural windows: {normal_path.name}")
    windows, summary = step13.process_normal_capture(normal_path, step03)
    frames.append(windows[keep].copy())
    summary["window_size"] = MICRO_WINDOW_SIZE
    summaries.append(summary)
    del windows
    data = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, index=False)
    write_csv(parser_summary_path, summaries)
    print(f"Saved structural cache: {cache_path}")
    return data


def robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    center = np.median(values, axis=axis)
    return 1.4826 * np.median(np.abs(values - center), axis=axis)


def scale_floors(normal: pd.DataFrame) -> np.ndarray:
    values = normal[list(STRUCTURAL_FEATURES)].to_numpy(dtype=float)
    global_scales = robust_scale(values, axis=0)
    absolute = np.asarray(
        [0.25 if feature in COUNT_FEATURES else 0.01 for feature in STRUCTURAL_FEATURES],
        dtype=float,
    )
    return np.maximum(GLOBAL_SCALE_FLOOR_FRACTION * global_scales, absolute)


def score_session(
    frame: pd.DataFrame,
    session_id: str,
    bootstrap_windows: int,
    floors: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = frame.sort_values("window_index").copy()
    if len(ordered) <= bootstrap_windows:
        raise ValueError(
            f"Session {session_id} has {len(ordered)} micro-windows; "
            f"more than {bootstrap_windows} are required"
        )
    bootstrap = ordered.iloc[:bootstrap_windows]
    baseline = bootstrap[list(STRUCTURAL_FEATURES)].to_numpy(dtype=float)
    center = np.median(baseline, axis=0)
    local_scale = robust_scale(baseline, axis=0)
    scale = np.maximum(local_scale, floors)
    evaluation = ordered.iloc[bootstrap_windows:].copy()
    values = evaluation[list(STRUCTURAL_FEATURES)].to_numpy(dtype=float)
    deviations = np.abs(values - center) / np.maximum(scale, EPSILON)
    evaluation["micro_session_id"] = session_id
    evaluation["micro_structural_deviation"] = np.quantile(
        deviations,
        0.75,
        axis=1,
    )
    audit: dict[str, object] = {
        "micro_session_id": session_id,
        "source_file": str(ordered.iloc[0]["source_file"]),
        "total_micro_windows": len(ordered),
        "bootstrap_micro_windows": bootstrap_windows,
        "evaluated_micro_windows": len(evaluation),
        "bootstrap_attack_micro_windows": int(bootstrap["binary_target"].sum()),
        "bootstrap_attack_frames": int(bootstrap["attack_frame_count"].sum()),
        "bootstrap_first_window_index": int(bootstrap["window_index"].min()),
        "bootstrap_last_window_index": int(bootstrap["window_index"].max()),
        "scale_floor_features": int((local_scale < floors).sum()),
    }
    for index, feature in enumerate(STRUCTURAL_FEATURES):
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
        raise ValueError(
            f"Only {complete} complete normal micro pseudo-sessions are available"
        )
    sessions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for index in range(complete):
        start = index * PSEUDO_SESSION_MICRO_WINDOWS
        end = start + PSEUDO_SESSION_MICRO_WINDOWS
        session, audit = score_session(
            ordered.iloc[start:end],
            f"normal_micro_session_{index:03d}",
            BOOTSTRAP_MICRO_WINDOWS,
            floors,
        )
        session["micro_protocol_partition"] = "unassigned_normal"
        audit["micro_protocol_partition"] = "unassigned_normal"
        sessions.append(session)
        audits.append(audit)
    discarded = len(ordered) - complete * PSEUDO_SESSION_MICRO_WINDOWS
    return sessions, audits, discarded


def add_micro_alarms(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["micro_alarm_instant"] = (
        output["micro_structural_deviation"] >= threshold
    )
    output["micro_alarm_persistent_2"] = output.groupby(
        "micro_session_id", sort=False
    )["micro_alarm_instant"].transform(
        lambda values: values & values.shift(1, fill_value=False)
    )
    output["micro_continuous_can_trust"] = 1.0 / (
        1.0
        + np.square(
            output["micro_structural_deviation"] / max(threshold, EPSILON)
        )
    )
    return output


def aggregate_to_parent(
    micro: pd.DataFrame,
    w100: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    micro = micro.copy()
    micro["parent_window_index"] = (
        micro["start_row"].astype(np.int64) // PARENT_WINDOW_SIZE
    )
    grouped = micro.groupby(
        ["source_file", "parent_window_index"],
        sort=True,
        observed=True,
    )
    parent = grouped.agg(
        micro_windows=("window_index", "size"),
        micro_attack_frames=("attack_frame_count", "sum"),
        micro_attack_windows=("binary_target", "sum"),
        w20_any_alarm_instant=("micro_alarm_instant", "max"),
        w20_any_alarm_persistent_2=("micro_alarm_persistent_2", "max"),
        micro_deviation_mean=("micro_structural_deviation", "mean"),
        micro_deviation_max=("micro_structural_deviation", "max"),
        micro_can_trust_min=("micro_continuous_can_trust", "min"),
    ).reset_index()
    incomplete = int((parent["micro_windows"] != MICRO_PER_PARENT).sum())
    parent = parent[parent["micro_windows"] == MICRO_PER_PARENT].copy()

    w100_columns = [
        "source_file",
        "window_index",
        "binary_target",
        "attack_frame_count",
        "alarm_instant",
        "alarm_persistent_2",
        "continuous_can_trust",
    ]
    missing = set(w100_columns) - set(w100.columns)
    if missing:
        raise ValueError(f"Step 17 predictions are missing: {sorted(missing)}")
    right = w100[w100_columns].rename(
        columns={
            "window_index": "parent_window_index",
            "binary_target": "w100_binary_target",
            "attack_frame_count": "w100_attack_frame_count",
            "alarm_instant": "w100_alarm_instant",
            "alarm_persistent_2": "w100_alarm_persistent_2",
            "continuous_can_trust": "w100_continuous_can_trust",
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
        raise RuntimeError(
            f"Micro/parent label alignment failed for {mismatches} windows"
        )
    merged["multiscale_alarm_instant"] = (
        merged["w100_alarm_instant"].astype(bool)
        | merged["w20_any_alarm_instant"].astype(bool)
    )
    merged["multiscale_alarm_persistent_2"] = (
        merged["w100_alarm_persistent_2"].astype(bool)
        | merged["w20_any_alarm_persistent_2"].astype(bool)
    )
    merged["multiscale_continuous_can_trust"] = np.minimum(
        merged["w100_continuous_can_trust"],
        merged["micro_can_trust_min"],
    )
    audit = {
        "micro_parent_groups": len(parent) + incomplete,
        "incomplete_micro_parent_groups_discarded": incomplete,
        "complete_micro_parent_groups": len(parent),
        "parents_matched_to_step17": len(merged),
        "micro_parent_label_mismatches": mismatches,
    }
    return merged, audit


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    frame: pd.DataFrame,
    method: str,
    alarm_column: str,
    scope: str,
) -> dict[str, object]:
    truth = frame["w100_binary_target"].to_numpy(dtype=np.uint8)
    prediction = frame[alarm_column].to_numpy(dtype=bool).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(truth)) == 2
    return {
        "method": method,
        "scope": scope,
        "parent_windows": len(frame),
        "benign_parent_windows": int((truth == 0).sum()),
        "attack_parent_windows": int((truth == 1).sum()),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction)
        if both
        else math.nan,
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


def density_rows(parent: pd.DataFrame) -> list[dict[str, object]]:
    attacked = parent[parent["w100_binary_target"] == 1]
    bins = (
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-20", 6, 20),
        ("21-100", 21, 100),
    )
    output: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        for label, lower, upper in bins:
            group = attacked[
                attacked["w100_attack_frame_count"].between(lower, upper)
            ]
            output.append(
                {
                    "method": method,
                    "attack_frames_per_100": label,
                    "parent_windows": len(group),
                    "recall": float(group[column].mean()) if len(group) else math.nan,
                    "mean_micro_deviation_max": float(
                        group["micro_deviation_max"].mean()
                    )
                    if len(group)
                    else math.nan,
                }
            )
    return output


def per_source_rows(parent: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_file, source in parent.groupby("source_file", sort=True):
        for method, column in METHOD_COLUMNS.items():
            output.append(metric_row(source, method, column, f"source:{source_file}"))
    return output


def plot_results(
    metrics: pd.DataFrame,
    density: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = ("w100_instant", "w20_any_instant", "multiscale_instant")
    labels = {
        "w100_instant": "Frozen 100-frame gate",
        "w20_any_instant": "20-frame micro-gate",
        "multiscale_instant": "Multiscale fusion",
    }
    bins = ("1", "2-5", "6-20", "21-100")
    x = np.arange(len(bins))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    for method in selected:
        group = density[density["method"] == method].set_index(
            "attack_frames_per_100"
        )
        axes[0].plot(
            x,
            [group.loc[label, "recall"] for label in bins],
            marker="o",
            linewidth=2,
            label=labels[method],
        )
    metric_group = metrics[
        metrics["scope"].eq("all_parent_evaluation")
        & metrics["method"].isin(selected)
    ].set_index("method")
    positions = np.arange(len(selected))
    width = 0.25
    for offset, (metric, label) in enumerate(
        (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
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
    axes[0].legend()
    axes[1].set(
        title="Overall parent-window performance",
        ylabel="Metric",
        ylim=(0, 1.03),
        xticks=positions,
        xticklabels=["w100", "w20", "fusion"],
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Multiscale sparse-CAN trust gate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "multiscale_sparse_can_gate"
    cache_path = processed_dir / "car_hacking_windows_w20_structural.csv"
    parser_summary_path = results_dir / "w20_parser_summary.csv"
    if args.rebuild_cache or not cache_path.exists():
        micro = build_structural_cache(
            project_root,
            cache_path,
            parser_summary_path,
            args.chunk_rows,
        )
    else:
        print(f"Loading cached structural micro-windows: {cache_path}")
        micro = pd.read_csv(cache_path)
    required = set(META_COLUMNS + STRUCTURAL_FEATURES)
    missing = required - set(micro.columns)
    if missing:
        raise ValueError(f"Structural cache is missing: {sorted(missing)}")
    if micro[list(required)].isna().any().any():
        raise ValueError("Structural cache contains missing required values")

    normal = micro[micro["source_file"] == NORMAL_SOURCE].copy()
    attack_captures = micro[micro["source_file"] != NORMAL_SOURCE].copy()
    if normal.empty or attack_captures.empty:
        raise ValueError("Normal and attack-capture micro-windows are both required")
    floors = scale_floors(normal)
    normal_sessions, session_audits, discarded_normal = build_normal_sessions(
        normal,
        floors,
    )
    split = len(normal_sessions) // 2
    calibration_sessions = normal_sessions[:split]
    holdout_sessions = normal_sessions[split:]
    for index, session in enumerate(normal_sessions):
        partition = "normal_calibration" if index < split else "normal_holdout"
        session["micro_protocol_partition"] = partition
        session_audits[index]["micro_protocol_partition"] = partition
    normal_calibration = pd.concat(calibration_sessions, ignore_index=True)
    normal_holdout = pd.concat(holdout_sessions, ignore_index=True)

    attack_evaluations: list[pd.DataFrame] = []
    for source_file, source in attack_captures.groupby("source_file", sort=True):
        evaluation, audit = score_session(
            source,
            f"attack_micro_session::{source_file}",
            BOOTSTRAP_MICRO_WINDOWS,
            floors,
        )
        audit["micro_protocol_partition"] = "attack_capture_evaluation"
        if int(audit["bootstrap_attack_frames"]) != 0:
            raise ValueError(
                f"Trusted micro bootstrap for {source_file} contains attack frames"
            )
        evaluation["micro_protocol_partition"] = "attack_capture_evaluation"
        attack_evaluations.append(evaluation)
        session_audits.append(audit)
    attack_evaluation = pd.concat(attack_evaluations, ignore_index=True)

    threshold = float(
        np.quantile(
            normal_calibration["micro_structural_deviation"].to_numpy(dtype=float),
            1.0 - MICRO_TARGET_FPR,
        )
    )
    normal_calibration = add_micro_alarms(normal_calibration, threshold)
    normal_holdout = add_micro_alarms(normal_holdout, threshold)
    attack_evaluation = add_micro_alarms(attack_evaluation, threshold)
    evaluation = pd.concat([attack_evaluation, normal_holdout], ignore_index=True)

    w100_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_predictions.csv"
    )
    if not w100_path.exists():
        raise FileNotFoundError(f"Missing Step 17 predictions: {w100_path}")
    w100 = pd.read_csv(w100_path)
    parent, alignment_audit = aggregate_to_parent(evaluation, w100)

    metric_rows: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        metric_rows.append(
            metric_row(parent, method, column, "all_parent_evaluation")
        )
    source_metrics = per_source_rows(parent)
    density = density_rows(parent)
    calibration_row = {
        "micro_window_size": MICRO_WINDOW_SIZE,
        "parent_window_size": PARENT_WINDOW_SIZE,
        "bootstrap_micro_windows": BOOTSTRAP_MICRO_WINDOWS,
        "bootstrap_frames": BOOTSTRAP_MICRO_WINDOWS * MICRO_WINDOW_SIZE,
        "normal_pseudo_session_micro_windows": PSEUDO_SESSION_MICRO_WINDOWS,
        "normal_pseudo_session_frames": PSEUDO_SESSION_MICRO_WINDOWS
        * MICRO_WINDOW_SIZE,
        "micro_target_fpr": MICRO_TARGET_FPR,
        "micro_deviation_threshold": threshold,
        "normal_calibration_micro_windows": len(normal_calibration),
        "normal_calibration_instant_fpr": float(
            normal_calibration["micro_alarm_instant"].mean()
        ),
        "normal_calibration_persistent_2_fpr": float(
            normal_calibration["micro_alarm_persistent_2"].mean()
        ),
        "normal_holdout_micro_windows": len(normal_holdout),
        "normal_holdout_instant_fpr": float(
            normal_holdout["micro_alarm_instant"].mean()
        ),
        "normal_holdout_persistent_2_fpr": float(
            normal_holdout["micro_alarm_persistent_2"].mean()
        ),
    }
    scale_rows = [
        {"feature": feature, "global_scale_floor": float(floors[index])}
        for index, feature in enumerate(STRUCTURAL_FEATURES)
    ]
    manifest = [
        {"item": "experiment_type", "value": "exploratory multiscale sparse-CAN gate"},
        {"item": "micro_window_size", "value": MICRO_WINDOW_SIZE},
        {"item": "parent_window_size", "value": PARENT_WINDOW_SIZE},
        {"item": "structural_features", "value": ";".join(STRUCTURAL_FEATURES)},
        {"item": "micro_threshold_source", "value": "attack-free normal calibration pseudo-sessions only"},
        {"item": "micro_target_fpr", "value": MICRO_TARGET_FPR},
        {"item": "fusion_rule", "value": "OR of frozen w100 gate and w20 structural micro-gate"},
        {"item": "label_usage", "value": "evaluation only"},
        {"item": "discarded_normal_tail_micro_windows", "value": discarded_normal},
        {"item": "external_validity_limit", "value": "HCRL exploratory development; requires independent confirmation"},
        {"item": "next_required_test", "value": "micro-baseline poisoning stress before policy freeze"},
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        "source_file",
        "window_index",
        "start_row",
        "end_row",
        "source_capture_class",
        "binary_target",
        "attack_frame_count",
        "micro_session_id",
        "micro_protocol_partition",
        "micro_structural_deviation",
        "micro_alarm_instant",
        "micro_alarm_persistent_2",
        "micro_continuous_can_trust",
    ]
    evaluation[prediction_columns].to_csv(
        results_dir / "w20_micro_gate_predictions.csv", index=False
    )
    parent.to_csv(results_dir / "multiscale_parent_predictions.csv", index=False)
    write_csv(results_dir / "multiscale_overall_metrics.csv", metric_rows)
    write_csv(results_dir / "multiscale_per_source_metrics.csv", source_metrics)
    write_csv(results_dir / "multiscale_density_recall.csv", density)
    write_csv(results_dir / "micro_calibration_summary.csv", [calibration_row])
    write_csv(results_dir / "micro_session_bootstrap_audit.csv", session_audits)
    write_csv(results_dir / "micro_feature_scale_floors.csv", scale_rows)
    write_csv(results_dir / "micro_parent_alignment_audit.csv", [alignment_audit])
    write_csv(results_dir / "multiscale_manifest.csv", manifest)
    plot_results(
        pd.DataFrame(metric_rows),
        pd.DataFrame(density),
        results_dir / "multiscale_sparse_recall.png",
    )

    metrics = pd.DataFrame(metric_rows).set_index("method")
    density_frame = pd.DataFrame(density)
    print("\n" + "=" * 84)
    print("Multiscale sparse-CAN gate completed successfully.")
    print(f"Micro deviation threshold: {threshold:.6f}")
    print(
        "Normal holdout micro FPR: "
        f"instant={calibration_row['normal_holdout_instant_fpr']:.4f}, "
        f"persistent_2={calibration_row['normal_holdout_persistent_2_fpr']:.4f}"
    )
    print("\nParent-window performance:")
    for method in ("w100_instant", "w20_any_instant", "multiscale_instant"):
        row = metrics.loc[method]
        print(
            f"  {method:<24} precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, F1={row['f1']:.4f}, "
            f"FPR={row['false_positive_rate']:.4f}"
        )
    print("\nSparse parent-window recall:")
    for label in ("1", "2-5", "6-20", "21-100"):
        values = density_frame[density_frame["attack_frames_per_100"] == label]
        old = float(values.loc[values["method"] == "w100_instant", "recall"].iloc[0])
        fused = float(values.loc[values["method"] == "multiscale_instant", "recall"].iloc[0])
        print(f"  frames={label:<6} w100={old:.4f}, multiscale={fused:.4f}")
    print(f"\nResults directory: {results_dir}")
    print("\nNext: accept or reject the micro-gate, then stress-test its startup baseline.")


if __name__ == "__main__":
    main()
