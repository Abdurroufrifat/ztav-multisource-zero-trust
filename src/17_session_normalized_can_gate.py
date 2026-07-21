#!/usr/bin/env python3
"""Session-normalized CAN trust gate with graded Zero Trust actions.

Step 16 proved that a static healthy profile does not transfer across HCRL
capture sessions.  This script normalizes the frozen identifier-only model's
log-odds relative to a short, trusted startup baseline for each session.

Protocol
--------
* Calibrate deviation thresholds only on attack-free normal_run_data.txt.
* Split that capture into chronological pseudo-sessions of 250 windows.
* Use the first 10 windows of every pseudo-session as a frozen baseline.
* Calibrate on the first half of pseudo-sessions and keep the rest as healthy
  holdout sessions.
* For each attack capture, use exactly the first 10 windows after secure start
  as its frozen baseline.  Never update it during the session, which limits
  poisoning risk.
* A first anomalous window requests VERIFY/RESTRICT.  Two consecutive anomalous
  windows request SAFE_FALLBACK.

Attack labels are not used for baseline fitting or threshold selection.  They
are used only for evaluation and for an audit that the assumed secure-start
windows were attack-free in this dataset.

This remains exploratory: the 10-window bootstrap and operational calibration
target were chosen after inspecting HCRL.  Freeze the design and evaluate it on
an independent capture before making a confirmatory generalization claim.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
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


NORMAL_SOURCE = "normal_run_data.txt"
DEFAULT_BOOTSTRAP_WINDOWS = 10
DEFAULT_PSEUDO_SESSION_WINDOWS = 250
DEFAULT_OPERATIONAL_TARGET_FPR = 0.05
SENSITIVITY_TARGETS = (0.01, 0.02, 0.05, 0.10)
GLOBAL_SCALE_FLOOR_FRACTION = 0.25
ABSOLUTE_SCALE_FLOOR = 0.05
EPSILON = 1e-12
REQUIRED_COLUMNS = {
    "source_file",
    "source_capture_class",
    "window_index",
    "binary_target",
    "attack_frame_count",
    "attack_frame_fraction",
    "identifier_log_odds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Session-normalized CAN trust gate evaluation."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--bootstrap-windows",
        type=int,
        default=DEFAULT_BOOTSTRAP_WINDOWS,
    )
    parser.add_argument(
        "--pseudo-session-windows",
        type=int,
        default=DEFAULT_PSEUDO_SESSION_WINDOWS,
    )
    parser.add_argument(
        "--operational-target-fpr",
        type=float,
        default=DEFAULT_OPERATIONAL_TARGET_FPR,
    )
    args = parser.parse_args()
    if args.bootstrap_windows < 5:
        parser.error("--bootstrap-windows must be at least 5")
    if args.pseudo_session_windows <= args.bootstrap_windows * 2:
        parser.error("--pseudo-session-windows must exceed twice the bootstrap size")
    if args.operational_target_fpr not in SENSITIVITY_TARGETS:
        parser.error(
            "--operational-target-fpr must be one of "
            + ", ".join(str(value) for value in SENSITIVITY_TARGETS)
        )
    return args


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def robust_scale(values: np.ndarray) -> float:
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def score_session(
    frame: pd.DataFrame,
    session_id: str,
    bootstrap_windows: int,
    global_scale: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = frame.sort_values("window_index").copy()
    if len(ordered) <= bootstrap_windows:
        raise ValueError(
            f"Session {session_id} has {len(ordered)} windows; more than "
            f"{bootstrap_windows} are required"
        )
    bootstrap = ordered.iloc[:bootstrap_windows]
    bootstrap_values = bootstrap["identifier_log_odds"].to_numpy(dtype=float)
    center = float(np.median(bootstrap_values))
    session_scale = robust_scale(bootstrap_values)
    scale_floor = max(
        GLOBAL_SCALE_FLOOR_FRACTION * global_scale,
        ABSOLUTE_SCALE_FLOOR,
    )
    scale_used = max(session_scale, scale_floor)
    evaluation = ordered.iloc[bootstrap_windows:].copy()
    evaluation["session_id"] = session_id
    evaluation["session_center_log_odds"] = center
    evaluation["session_scale_log_odds"] = scale_used
    evaluation["session_deviation_score"] = np.abs(
        evaluation["identifier_log_odds"].to_numpy(dtype=float) - center
    ) / max(scale_used, EPSILON)
    audit = {
        "session_id": session_id,
        "source_file": str(ordered.iloc[0]["source_file"]),
        "total_windows": len(ordered),
        "bootstrap_windows": bootstrap_windows,
        "evaluated_windows": len(evaluation),
        "bootstrap_attack_windows": int(bootstrap["binary_target"].sum()),
        "bootstrap_first_window_index": int(bootstrap["window_index"].min()),
        "bootstrap_last_window_index": int(bootstrap["window_index"].max()),
        "bootstrap_center_log_odds": center,
        "bootstrap_raw_robust_scale": session_scale,
        "scale_floor": scale_floor,
        "scale_used": scale_used,
        "scale_floor_applied": bool(session_scale < scale_floor),
    }
    return evaluation, audit


def add_alarm_columns(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["alarm_instant"] = output["session_deviation_score"] >= threshold
    output["alarm_persistent_2"] = output.groupby("session_id", sort=False)[
        "alarm_instant"
    ].transform(lambda values: values & values.shift(1, fill_value=False))
    return output


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    target_fpr: float,
    method: str,
    scope: str,
    truth: np.ndarray,
    alarm: np.ndarray,
    anomaly_score: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    prediction = alarm.astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(truth)) == 2
    return {
        "calibration_target_fpr": target_fpr,
        "method": method,
        "scope": scope,
        "deviation_threshold": threshold,
        "windows": len(truth),
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
        "roc_auc": (
            roc_auc_score(truth, anomaly_score) if both_classes else math.nan
        ),
        "pr_auc": (
            average_precision_score(truth, anomaly_score) if both_classes else math.nan
        ),
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_normal_pseudo_sessions(
    normal: pd.DataFrame,
    bootstrap_windows: int,
    pseudo_session_windows: int,
    global_scale: float,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], int]:
    ordered = normal.sort_values("window_index")
    complete_sessions = len(ordered) // pseudo_session_windows
    if complete_sessions < 4:
        raise ValueError(
            f"Only {complete_sessions} complete normal pseudo-sessions are available"
        )
    scored: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for index in range(complete_sessions):
        start = index * pseudo_session_windows
        end = start + pseudo_session_windows
        session_id = f"normal_pseudo_session_{index:03d}"
        evaluation, audit = score_session(
            ordered.iloc[start:end],
            session_id,
            bootstrap_windows,
            global_scale,
        )
        evaluation["protocol_partition"] = "unassigned_normal_pseudo_session"
        scored.append(evaluation)
        audit["protocol_partition"] = "unassigned_normal_pseudo_session"
        audits.append(audit)
    discarded = len(ordered) - complete_sessions * pseudo_session_windows
    return scored, audits, discarded


def attack_density_rows(
    evaluation: pd.DataFrame,
    target_fpr: float,
    threshold: float,
) -> list[dict[str, object]]:
    attacked = evaluation[evaluation["binary_target"] == 1]
    bins = (
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-20", 6, 20),
        ("21-50", 21, 50),
        ("51-99", 51, 99),
        ("100", 100, 100),
    )
    output: list[dict[str, object]] = []
    for method, column in (
        ("instant_verify_restrict", "alarm_instant"),
        ("persistent_2_safe_fallback", "alarm_persistent_2"),
    ):
        for label, lower, upper in bins:
            group = attacked[
                attacked["attack_frame_count"].between(lower, upper, inclusive="both")
            ]
            output.append(
                {
                    "calibration_target_fpr": target_fpr,
                    "deviation_threshold": threshold,
                    "method": method,
                    "attack_frames_per_window": label,
                    "windows": len(group),
                    "recall": float(group[column].mean()) if len(group) else math.nan,
                    "mean_deviation_score": (
                        float(group["session_deviation_score"].mean())
                        if len(group)
                        else math.nan
                    ),
                }
            )
    return output


def latency_rows(
    evaluation: pd.DataFrame,
    target_fpr: float,
    threshold: float,
) -> list[dict[str, object]]:
    attack_capture = evaluation[
        evaluation["source_file"] != NORMAL_SOURCE
    ].copy()
    output: list[dict[str, object]] = []
    for source_file, source in attack_capture.groupby("source_file", sort=True):
        source = source.sort_values("window_index")
        run_ids = source["binary_target"].ne(source["binary_target"].shift()).cumsum()
        attack_runs = source[source["binary_target"] == 1].groupby(
            run_ids[source["binary_target"] == 1],
            sort=True,
        )
        for run_number, (_, run) in enumerate(attack_runs, start=1):
            for method, column in (
                ("instant_verify_restrict", "alarm_instant"),
                ("persistent_2_safe_fallback", "alarm_persistent_2"),
            ):
                detected = run[run[column]]
                detected_flag = not detected.empty
                latency = (
                    int(detected.iloc[0]["window_index"] - run.iloc[0]["window_index"])
                    if detected_flag
                    else math.nan
                )
                output.append(
                    {
                        "calibration_target_fpr": target_fpr,
                        "deviation_threshold": threshold,
                        "source_file": source_file,
                        "attack_run_number": run_number,
                        "attack_run_start_window": int(run["window_index"].min()),
                        "attack_run_end_window": int(run["window_index"].max()),
                        "attack_run_windows": len(run),
                        "method": method,
                        "detected": detected_flag,
                        "latency_windows": latency,
                        "latency_frames_lower_bound": (
                            latency * 100 if detected_flag else math.nan
                        ),
                        "latency_frames_upper_bound": (
                            (latency + 1) * 100 if detected_flag else math.nan
                        ),
                    }
                )
    return output


def latency_summary_rows(latencies: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (method, source_file), group in latencies.groupby(
        ["method", "source_file"], sort=True
    ):
        detected = group[group["detected"]]
        values = detected["latency_windows"].to_numpy(dtype=float)
        output.append(
            {
                "method": method,
                "source_file": source_file,
                "attack_runs": len(group),
                "detected_runs": len(detected),
                "missed_runs": int((~group["detected"]).sum()),
                "run_detection_rate": float(group["detected"].mean()),
                "latency_windows_mean": float(values.mean()) if len(values) else math.nan,
                "latency_windows_median": (
                    float(np.median(values)) if len(values) else math.nan
                ),
                "latency_windows_p95": (
                    float(np.quantile(values, 0.95)) if len(values) else math.nan
                ),
                "latency_windows_max": float(values.max()) if len(values) else math.nan,
            }
        )
    return output


def plot_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for method, group in summary.groupby("method", sort=True):
        group = group.sort_values("calibration_target_fpr")
        axes[0].plot(
            group["calibration_target_fpr"],
            group["recall"],
            marker="o",
            label=f"{method} recall",
        )
        axes[0].plot(
            group["calibration_target_fpr"],
            group["f1"],
            marker="s",
            linestyle="--",
            label=f"{method} F1",
        )
        axes[1].plot(
            group["calibration_target_fpr"],
            group["false_positive_rate"],
            marker="o",
            label=method,
        )
    axes[0].set(
        xlabel="Attack-free calibration target FPR",
        ylabel="Metric",
        ylim=(0, 1.03),
        title="Detection sensitivity",
    )
    axes[1].set(
        xlabel="Attack-free calibration target FPR",
        ylabel="Observed evaluation FPR",
        title="False-alarm transfer",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Session-normalized CAN gate sensitivity")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_path = (
        project_root
        / "results"
        / "drift_aware_can_gate_w100"
        / "drift_gate_predictions.csv"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Step 16 prediction file: {input_path}")
    print(f"Loading Step 16 predictions: {input_path}")
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Step 16 prediction file is missing: {sorted(missing)}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required Step 16 prediction columns contain missing values")

    normal = data[data["source_file"] == NORMAL_SOURCE].sort_values("window_index")
    attack_captures = data[data["source_file"] != NORMAL_SOURCE]
    if normal.empty or attack_captures.empty:
        raise ValueError("Both normal and attack-capture data are required")
    global_scale = max(
        robust_scale(normal["identifier_log_odds"].to_numpy(dtype=float)),
        ABSOLUTE_SCALE_FLOOR,
    )

    normal_sessions, session_audits, discarded_normal_windows = (
        build_normal_pseudo_sessions(
            normal,
            args.bootstrap_windows,
            args.pseudo_session_windows,
            global_scale,
        )
    )
    split_index = len(normal_sessions) // 2
    normal_calibration_sessions = normal_sessions[:split_index]
    normal_holdout_sessions = normal_sessions[split_index:]
    for index, frame in enumerate(normal_sessions):
        partition = (
            "normal_threshold_calibration"
            if index < split_index
            else "normal_healthy_holdout"
        )
        frame["protocol_partition"] = partition
        session_audits[index]["protocol_partition"] = partition
    normal_calibration = pd.concat(normal_calibration_sessions, ignore_index=True)
    normal_holdout = pd.concat(normal_holdout_sessions, ignore_index=True)

    attack_evaluations: list[pd.DataFrame] = []
    for source_file, source in attack_captures.groupby("source_file", sort=True):
        evaluation, audit = score_session(
            source,
            f"attack_capture::{source_file}",
            args.bootstrap_windows,
            global_scale,
        )
        audit["protocol_partition"] = "attack_capture_evaluation"
        if int(audit["bootstrap_attack_windows"]) != 0:
            raise ValueError(
                f"Secure-start bootstrap for {source_file} contains attack windows; "
                "the session-normalized protocol is invalid for this capture"
            )
        evaluation["protocol_partition"] = "attack_capture_evaluation"
        attack_evaluations.append(evaluation)
        session_audits.append(audit)

    evaluation_base = pd.concat(
        [*attack_evaluations, normal_holdout],
        ignore_index=True,
    )
    thresholds: dict[float, float] = {
        target: float(
            np.quantile(
                normal_calibration["session_deviation_score"].to_numpy(dtype=float),
                1.0 - target,
            )
        )
        for target in SENSITIVITY_TARGETS
    }

    sensitivity_rows: list[dict[str, object]] = []
    per_source_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    operational_evaluation: pd.DataFrame | None = None
    operational_threshold = thresholds[args.operational_target_fpr]
    for target in SENSITIVITY_TARGETS:
        threshold = thresholds[target]
        calibration_scored = add_alarm_columns(normal_calibration, threshold)
        evaluation = add_alarm_columns(evaluation_base, threshold)
        if target == args.operational_target_fpr:
            operational_evaluation = evaluation.copy()
        threshold_rows.append(
            {
                "calibration_target_fpr": target,
                "deviation_threshold": threshold,
                "calibration_windows": len(calibration_scored),
                "calibration_instant_fpr": float(
                    calibration_scored["alarm_instant"].mean()
                ),
                "calibration_persistent_2_fpr": float(
                    calibration_scored["alarm_persistent_2"].mean()
                ),
                "normal_holdout_windows": len(normal_holdout),
                "bootstrap_windows_per_session": args.bootstrap_windows,
                "pseudo_session_windows": args.pseudo_session_windows,
            }
        )
        truth = evaluation["binary_target"].to_numpy(dtype=np.uint8)
        anomaly_score = evaluation["session_deviation_score"].to_numpy(dtype=float)
        for method, column in (
            ("instant_verify_restrict", "alarm_instant"),
            ("persistent_2_safe_fallback", "alarm_persistent_2"),
        ):
            row = metric_row(
                target,
                method,
                "all_evaluation_windows",
                truth,
                evaluation[column].to_numpy(dtype=bool),
                anomaly_score,
                threshold,
            )
            sensitivity_rows.append(row)
            for source_file, source in evaluation.groupby("source_file", sort=True):
                per_source_rows.append(
                    metric_row(
                        target,
                        method,
                        f"source:{source_file}",
                        source["binary_target"].to_numpy(dtype=np.uint8),
                        source[column].to_numpy(dtype=bool),
                        source["session_deviation_score"].to_numpy(dtype=float),
                        threshold,
                    )
                )

    if operational_evaluation is None:
        raise RuntimeError("Operational evaluation was not created")
    operational_evaluation["continuous_can_trust"] = 1.0 / (
        1.0
        + np.square(
            operational_evaluation["session_deviation_score"]
            / max(operational_threshold, EPSILON)
        )
    )
    operational_evaluation["can_gate_action"] = np.select(
        [
            operational_evaluation["alarm_persistent_2"],
            operational_evaluation["alarm_instant"],
        ],
        ["SAFE_FALLBACK", "VERIFY_RESTRICT"],
        default="ALLOW",
    )

    latencies = pd.DataFrame(
        latency_rows(
            operational_evaluation,
            args.operational_target_fpr,
            operational_threshold,
        )
    )
    latency_summary = latency_summary_rows(latencies)
    density = attack_density_rows(
        operational_evaluation,
        args.operational_target_fpr,
        operational_threshold,
    )

    output_dir = project_root / "results" / "session_normalized_can_gate_w100"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "session_gate_thresholds.csv", threshold_rows)
    write_csv(output_dir / "session_gate_fpr_sensitivity.csv", sensitivity_rows)
    write_csv(output_dir / "session_gate_per_source_metrics.csv", per_source_rows)
    write_csv(output_dir / "session_gate_attack_density_recall.csv", density)
    write_csv(
        output_dir / "session_gate_session_bootstrap_audit.csv",
        session_audits,
    )
    latencies.to_csv(output_dir / "session_gate_attack_run_latency.csv", index=False)
    write_csv(output_dir / "session_gate_latency_summary.csv", latency_summary)

    prediction_columns = [
        "source_file",
        "source_capture_class",
        "window_index",
        "binary_target",
        "attack_frame_count",
        "attack_frame_fraction",
        "session_id",
        "protocol_partition",
        "identifier_log_odds",
        "session_center_log_odds",
        "session_scale_log_odds",
        "session_deviation_score",
        "alarm_instant",
        "alarm_persistent_2",
        "continuous_can_trust",
        "can_gate_action",
    ]
    operational_evaluation[prediction_columns].to_csv(
        output_dir / "session_gate_predictions.csv",
        index=False,
    )
    plot_sensitivity(
        pd.DataFrame(sensitivity_rows),
        output_dir / "session_gate_fpr_sensitivity.png",
    )

    manifest: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "session-normalized CAN trust gate"},
        {"item": "input_score", "value": "frozen identifier-only model log-odds"},
        {"item": "normal_threshold_source", "value": NORMAL_SOURCE},
        {"item": "bootstrap_windows_per_session", "value": args.bootstrap_windows},
        {"item": "pseudo_session_windows", "value": args.pseudo_session_windows},
        {"item": "normal_complete_pseudo_sessions", "value": len(normal_sessions)},
        {
            "item": "normal_threshold_calibration_sessions",
            "value": len(normal_calibration_sessions),
        },
        {
            "item": "normal_healthy_holdout_sessions",
            "value": len(normal_holdout_sessions),
        },
        {"item": "normal_tail_windows_discarded", "value": discarded_normal_windows},
        {"item": "operational_target_fpr", "value": args.operational_target_fpr},
        {"item": "operational_deviation_threshold", "value": operational_threshold},
        {"item": "attack_labels_used_for_threshold", "value": "none"},
        {
            "item": "baseline_update_policy",
            "value": "frozen after secure-start bootstrap; no online updates",
        },
        {
            "item": "graded_policy",
            "value": (
                "first anomaly=VERIFY_RESTRICT; two consecutive anomalies="
                "SAFE_FALLBACK"
            ),
        },
        {
            "item": "continuous_trust",
            "value": "1/(1+(deviation/operational_threshold)^2)",
        },
        {
            "item": "study_status",
            "value": "exploratory; bootstrap and target chosen after HCRL inspection",
        },
        {
            "item": "confirmatory_requirement",
            "value": "freeze and evaluate on an independent capture including startup attacks",
        },
    ]
    write_csv(output_dir / "session_gate_manifest.csv", manifest)

    operational_rows = pd.DataFrame(sensitivity_rows)
    operational_rows = operational_rows[
        operational_rows["calibration_target_fpr"] == args.operational_target_fpr
    ]
    display_columns = [
        "method",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positive_rate",
    ]
    print("\n" + "=" * 100)
    print("Session-normalized CAN gate completed successfully.")
    print(
        f"Normal pseudo-sessions={len(normal_sessions)}, "
        f"threshold-calibration={len(normal_calibration_sessions)}, "
        f"healthy-holdout={len(normal_holdout_sessions)}"
    )
    print(
        f"Operational target FPR={args.operational_target_fpr:.3f}, "
        f"deviation threshold={operational_threshold:.6f}"
    )
    print(operational_rows[display_columns].to_string(index=False))
    print(f"\nResults directory: {output_dir}")
    print(
        "\nNext: freeze the exploratory candidate, test startup-baseline poisoning, "
        "then integrate the CAN trust signal with GNSS/V2X/identity context."
    )


if __name__ == "__main__":
    main()
