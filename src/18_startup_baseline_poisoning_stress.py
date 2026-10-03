#!/usr/bin/env python3
"""Stress-test session startup-baseline poisoning and a consistency guard.

The Step 17 CAN gate assumes that ten windows collected after secure startup
are trustworthy.  This script evaluates how the frozen session baseline reacts
when 0--10 of those windows are synthetically replaced by attack windows from
the same HCRL source capture.

A bootstrap-consistency guard is calibrated only on attack-free
normal_run_data.txt pseudo-sessions.  It measures startup dispersion, the
difference between the first and second half, and the total log-odds range.
The guard is not a replacement for secure boot, identity attestation, or
multi-source context; it is an additional fail-closed check.

The poisoning is counterfactual resampling, not a real captured startup attack.
Treat every result as exploratory.  Confirm the frozen guard on an independent
capture containing genuine startup attacks or a controlled testbed injection.

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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


NORMAL_SOURCE = "normal_run_data.txt"
BOOTSTRAP_WINDOWS = 10
PSEUDO_SESSION_WINDOWS = 250
DEFAULT_REPETITIONS = 20
DEFAULT_SEED = 42
GUARD_CALIBRATION_FPR = 0.05
GLOBAL_SCALE_FLOOR_FRACTION = 0.25
ABSOLUTE_SCALE_FLOOR = 0.05
EPSILON = 1e-12
REQUIRED_COLUMNS = {
    "source_file",
    "window_index",
    "binary_target",
    "identifier_log_odds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Startup-baseline poisoning stress test."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("--repetitions must be at least 5")
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


def baseline_metrics(values: np.ndarray) -> np.ndarray:
    if len(values) != BOOTSTRAP_WINDOWS:
        raise ValueError(
            f"Expected {BOOTSTRAP_WINDOWS} bootstrap values; found {len(values)}"
        )
    midpoint = len(values) // 2
    return np.asarray(
        [
            robust_scale(values),
            abs(float(np.median(values[:midpoint]) - np.median(values[midpoint:]))),
            float(np.ptp(values)),
        ],
        dtype=float,
    )


def calibrate_guard(normal: pd.DataFrame) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    list[dict[str, object]],
]:
    ordered = normal.sort_values("window_index")
    complete_sessions = len(ordered) // PSEUDO_SESSION_WINDOWS
    if complete_sessions < 10:
        raise ValueError("Too few normal pseudo-sessions to calibrate the guard")
    metric_rows: list[np.ndarray] = []
    session_ids: list[str] = []
    for index in range(complete_sessions):
        start = index * PSEUDO_SESSION_WINDOWS
        bootstrap = ordered.iloc[start : start + BOOTSTRAP_WINDOWS]
        values = bootstrap["identifier_log_odds"].to_numpy(dtype=float)
        metric_rows.append(baseline_metrics(values))
        session_ids.append(f"normal_pseudo_session_{index:03d}")
    matrix = np.vstack(metric_rows)
    metric_median = np.median(matrix, axis=0)
    metric_mad_scale = 1.4826 * np.median(np.abs(matrix - metric_median), axis=0)
    standard_deviation = matrix.std(axis=0)
    metric_scale = np.where(
        metric_mad_scale > EPSILON,
        metric_mad_scale,
        np.where(standard_deviation > EPSILON, standard_deviation, 1.0),
    )
    guard_scores = np.max(np.abs((matrix - metric_median) / metric_scale), axis=1)
    guard_threshold = float(
        np.quantile(guard_scores, 1.0 - GUARD_CALIBRATION_FPR)
    )
    output: list[dict[str, object]] = []
    for index, session_id in enumerate(session_ids):
        output.append(
            {
                "session_id": session_id,
                "startup_robust_scale": matrix[index, 0],
                "startup_half_median_gap": matrix[index, 1],
                "startup_range": matrix[index, 2],
                "guard_score": guard_scores[index],
                "guard_threshold": guard_threshold,
                "guard_rejected": bool(guard_scores[index] >= guard_threshold),
            }
        )
    return metric_median, metric_scale, guard_threshold, output


def guard_score(
    bootstrap_values: np.ndarray,
    metric_median: np.ndarray,
    metric_scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    metrics = baseline_metrics(bootstrap_values)
    score = float(np.max(np.abs((metrics - metric_median) / metric_scale)))
    return score, metrics


def session_scores(
    evaluation: pd.DataFrame,
    bootstrap_values: np.ndarray,
    global_scale: float,
) -> tuple[np.ndarray, float, float, bool]:
    center = float(np.median(bootstrap_values))
    raw_scale = robust_scale(bootstrap_values)
    scale_floor = max(
        GLOBAL_SCALE_FLOOR_FRACTION * global_scale,
        ABSOLUTE_SCALE_FLOOR,
    )
    scale_used = max(raw_scale, scale_floor)
    score = np.abs(
        evaluation["identifier_log_odds"].to_numpy(dtype=float) - center
    ) / max(scale_used, EPSILON)
    return score, center, scale_used, bool(raw_scale < scale_floor)


def alarm_metrics(
    truth: np.ndarray,
    alarm: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(truth, alarm, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(truth, alarm, zero_division=0),
        "recall": recall_score(truth, alarm, zero_division=0),
        "f1": f1_score(truth, alarm, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_detection_summary(
    evaluation: pd.DataFrame,
    alarm: np.ndarray,
) -> tuple[int, int, float, float]:
    frame = evaluation.copy()
    frame["alarm"] = alarm
    run_ids = frame["binary_target"].ne(frame["binary_target"].shift()).cumsum()
    attack_runs = frame[frame["binary_target"] == 1].groupby(
        run_ids[frame["binary_target"] == 1],
        sort=True,
    )
    detected_runs = 0
    latencies: list[int] = []
    total_runs = 0
    for _, run in attack_runs:
        total_runs += 1
        detected = run[run["alarm"]]
        if detected.empty:
            continue
        detected_runs += 1
        latencies.append(
            int(detected.iloc[0]["window_index"] - run.iloc[0]["window_index"])
        )
    return (
        total_runs,
        detected_runs,
        detected_runs / total_runs if total_runs else math.nan,
        float(np.mean(latencies)) if latencies else math.nan,
    )


def aggregate_rows(runs: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    group_columns = [
        "source_file",
        "contaminated_bootstrap_windows",
        "contamination_fraction",
        "method",
    ]
    for keys, group in runs.groupby(group_columns, sort=True):
        source_file, contaminated, fraction, method = keys
        output.append(
            {
                "source_file": source_file,
                "contaminated_bootstrap_windows": contaminated,
                "contamination_fraction": fraction,
                "method": method,
                "repetitions": len(group),
                "guard_rejection_rate": group["guard_rejected"].mean(),
                "precision_mean": group["precision"].mean(),
                "precision_std": group["precision"].std(ddof=0),
                "recall_mean": group["recall"].mean(),
                "recall_std": group["recall"].std(ddof=0),
                "recall_min": group["recall"].min(),
                "f1_mean": group["f1"].mean(),
                "f1_std": group["f1"].std(ddof=0),
                "false_positive_rate_mean": group["false_positive_rate"].mean(),
                "false_positive_rate_max": group["false_positive_rate"].max(),
                "run_detection_rate_mean": group["run_detection_rate"].mean(),
                "latency_windows_mean": group["latency_windows_mean"].mean(),
                "bootstrap_center_mean": group["bootstrap_center_log_odds"].mean(),
                "bootstrap_scale_mean": group["bootstrap_scale_used"].mean(),
                "guard_score_mean": group["guard_score"].mean(),
                "guard_score_min": group["guard_score"].min(),
                "guard_score_max": group["guard_score"].max(),
            }
        )
    return output


def macro_rows(aggregate: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (contaminated, fraction, method), group in aggregate.groupby(
        ["contaminated_bootstrap_windows", "contamination_fraction", "method"],
        sort=True,
    ):
        output.append(
            {
                "contaminated_bootstrap_windows": contaminated,
                "contamination_fraction": fraction,
                "method": method,
                "sources": len(group),
                "guard_rejection_rate_macro": group["guard_rejection_rate"].mean(),
                "precision_macro": group["precision_mean"].mean(),
                "recall_macro": group["recall_mean"].mean(),
                "recall_worst_source": group["recall_mean"].min(),
                "f1_macro": group["f1_mean"].mean(),
                "false_positive_rate_macro": group[
                    "false_positive_rate_mean"
                ].mean(),
                "false_positive_rate_worst_source": group[
                    "false_positive_rate_mean"
                ].max(),
                "run_detection_rate_macro": group["run_detection_rate_mean"].mean(),
            }
        )
    return output


def failure_boundary_rows(aggregate: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (source_file, method), group in aggregate.groupby(
        ["source_file", "method"], sort=True
    ):
        group = group.sort_values("contaminated_bootstrap_windows")
        recall_fail = group[group["recall_mean"] < 0.90]
        fpr_fail = group[group["false_positive_rate_mean"] > 0.05]
        output.append(
            {
                "source_file": source_file,
                "method": method,
                "first_contamination_count_recall_below_0_90": (
                    int(recall_fail.iloc[0]["contaminated_bootstrap_windows"])
                    if not recall_fail.empty
                    else "not_reached"
                ),
                "first_contamination_fraction_recall_below_0_90": (
                    float(recall_fail.iloc[0]["contamination_fraction"])
                    if not recall_fail.empty
                    else "not_reached"
                ),
                "first_contamination_count_fpr_above_0_05": (
                    int(fpr_fail.iloc[0]["contaminated_bootstrap_windows"])
                    if not fpr_fail.empty
                    else "not_reached"
                ),
                "first_contamination_fraction_fpr_above_0_05": (
                    float(fpr_fail.iloc[0]["contamination_fraction"])
                    if not fpr_fail.empty
                    else "not_reached"
                ),
            }
        )
    return output


def plot_sensitivity(macro: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for method, group in macro.groupby("method", sort=True):
        group = group.sort_values("contamination_fraction")
        axes[0].plot(
            group["contamination_fraction"],
            group["recall_macro"],
            marker="o",
            label=method,
        )
        axes[1].plot(
            group["contamination_fraction"],
            group["false_positive_rate_macro"],
            marker="o",
            label=method,
        )
    guard = (
        macro.groupby("contamination_fraction", as_index=False)[
            "guard_rejection_rate_macro"
        ]
        .mean()
        .sort_values("contamination_fraction")
    )
    axes[2].plot(
        guard["contamination_fraction"],
        guard["guard_rejection_rate_macro"],
        marker="o",
        color="tab:red",
    )
    axes[0].set(title="Attack recall", ylabel="Macro recall", ylim=(0, 1.03))
    axes[1].set(title="False alarms", ylabel="Macro FPR", ylim=(0, 1.03))
    axes[2].set(
        title="Startup consistency guard",
        ylabel="Poisoned bootstrap rejection rate",
        ylim=(0, 1.03),
    )
    for axis in axes:
        axis.set_xlabel("Contaminated startup-window fraction")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Session-baseline poisoning sensitivity")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    step16_path = (
        project_root
        / "results"
        / "drift_aware_can_gate_w100"
        / "drift_gate_predictions.csv"
    )
    threshold_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_thresholds.csv"
    )
    for path in (step16_path, threshold_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 16/17 input: {path}")
    print(f"Loading Step 16 predictions: {step16_path}")
    data = pd.read_csv(step16_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Step 16 predictions are missing: {sorted(missing)}")
    thresholds = pd.read_csv(threshold_path)
    selected = thresholds[np.isclose(thresholds["calibration_target_fpr"], 0.05)]
    if len(selected) != 1:
        raise ValueError("Could not find the Step 17 operational 0.05 threshold")
    deviation_threshold = float(selected.iloc[0]["deviation_threshold"])

    normal = data[data["source_file"] == NORMAL_SOURCE].sort_values("window_index")
    attacks = data[data["source_file"] != NORMAL_SOURCE]
    if normal.empty or attacks.empty:
        raise ValueError("Both normal and attack captures are required")
    global_scale = max(
        robust_scale(normal["identifier_log_odds"].to_numpy(dtype=float)),
        ABSOLUTE_SCALE_FLOOR,
    )
    metric_median, metric_scale, guard_threshold, guard_calibration_rows = (
        calibrate_guard(normal)
    )

    run_rows: list[dict[str, object]] = []
    source_names = sorted(attacks["source_file"].unique())
    for source_index, source_file in enumerate(source_names):
        source = attacks[attacks["source_file"] == source_file].sort_values(
            "window_index"
        )
        clean_bootstrap = source.iloc[:BOOTSTRAP_WINDOWS]
        if int(clean_bootstrap["binary_target"].sum()) != 0:
            raise ValueError(f"Clean bootstrap for {source_file} contains attacks")
        clean_values = clean_bootstrap["identifier_log_odds"].to_numpy(dtype=float)
        attack_pool = source[source["binary_target"] == 1][
            "identifier_log_odds"
        ].to_numpy(dtype=float)
        if len(attack_pool) < BOOTSTRAP_WINDOWS:
            raise ValueError(f"Attack pool for {source_file} is too small")
        evaluation = source.iloc[BOOTSTRAP_WINDOWS:].copy()
        truth = evaluation["binary_target"].to_numpy(dtype=np.uint8)

        print(f"Stress testing {source_file} ...")
        for contaminated_count in range(BOOTSTRAP_WINDOWS + 1):
            for repetition in range(args.repetitions):
                rng = np.random.default_rng(
                    args.seed
                    + source_index * 100_000
                    + contaminated_count * 1_000
                    + repetition
                )
                poisoned = clean_values.copy()
                if contaminated_count:
                    positions = rng.choice(
                        BOOTSTRAP_WINDOWS,
                        size=contaminated_count,
                        replace=False,
                    )
                    poisoned[positions] = rng.choice(
                        attack_pool,
                        size=contaminated_count,
                        replace=False,
                    )
                startup_guard_score, startup_metrics = guard_score(
                    poisoned,
                    metric_median,
                    metric_scale,
                )
                guard_rejected = startup_guard_score >= guard_threshold
                score, center, scale_used, floor_applied = session_scores(
                    evaluation,
                    poisoned,
                    global_scale,
                )
                instant = score >= deviation_threshold
                persistent = instant & np.r_[False, instant[:-1]]
                for method, alarm in (
                    ("instant_verify_restrict", instant),
                    ("persistent_2_safe_fallback", persistent),
                ):
                    metrics = alarm_metrics(truth, alarm)
                    total_runs, detected_runs, run_rate, latency = (
                        run_detection_summary(evaluation, alarm)
                    )
                    run_rows.append(
                        {
                            "source_file": source_file,
                            "contaminated_bootstrap_windows": contaminated_count,
                            "contamination_fraction": contaminated_count
                            / BOOTSTRAP_WINDOWS,
                            "repetition": repetition,
                            "random_seed": (
                                args.seed
                                + source_index * 100_000
                                + contaminated_count * 1_000
                                + repetition
                            ),
                            "method": method,
                            "guard_score": startup_guard_score,
                            "guard_threshold": guard_threshold,
                            "guard_rejected": guard_rejected,
                            "startup_robust_scale": startup_metrics[0],
                            "startup_half_median_gap": startup_metrics[1],
                            "startup_range": startup_metrics[2],
                            "bootstrap_center_log_odds": center,
                            "bootstrap_scale_used": scale_used,
                            "scale_floor_applied": floor_applied,
                            **metrics,
                            "attack_runs": total_runs,
                            "detected_attack_runs": detected_runs,
                            "run_detection_rate": run_rate,
                            "latency_windows_mean": latency,
                        }
                    )

    runs = pd.DataFrame(run_rows)
    aggregate = pd.DataFrame(aggregate_rows(runs))
    macro = pd.DataFrame(macro_rows(aggregate))
    failure_boundaries = failure_boundary_rows(aggregate)

    output_dir = project_root / "results" / "startup_poisoning_stress_w100"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "bootstrap_poisoning_runs.csv", index=False)
    aggregate.to_csv(output_dir / "bootstrap_poisoning_aggregate.csv", index=False)
    macro.to_csv(output_dir / "bootstrap_poisoning_macro_summary.csv", index=False)
    write_csv(
        output_dir / "bootstrap_guard_calibration.csv",
        guard_calibration_rows,
    )
    write_csv(
        output_dir / "bootstrap_poisoning_failure_boundaries.csv",
        failure_boundaries,
    )
    plot_sensitivity(macro, output_dir / "bootstrap_poisoning_sensitivity.png")

    manifest: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "startup baseline poisoning stress"},
        {"item": "bootstrap_windows", "value": BOOTSTRAP_WINDOWS},
        {"item": "contamination_counts", "value": "0 through 10"},
        {"item": "repetitions_per_count_source", "value": args.repetitions},
        {"item": "random_seed_base", "value": args.seed},
        {
            "item": "poisoning_method",
            "value": "replace clean startup log-odds with sampled same-source attack log-odds",
        },
        {"item": "step17_deviation_threshold", "value": deviation_threshold},
        {"item": "guard_threshold_source", "value": NORMAL_SOURCE},
        {"item": "guard_calibration_fpr", "value": GUARD_CALIBRATION_FPR},
        {"item": "guard_threshold", "value": guard_threshold},
        {
            "item": "guard_features",
            "value": "startup robust scale; half-median gap; total log-odds range",
        },
        {"item": "attack_labels_used_for_guard_calibration", "value": "none"},
        {
            "item": "guard_fail_action",
            "value": "reject enrollment and request SAFE_FALLBACK/re-authentication",
        },
        {
            "item": "limitation",
            "value": "counterfactual resampling; not a captured real-time startup attack",
        },
        {
            "item": "study_status",
            "value": "exploratory; confirm on independent startup-attack captures",
        },
    ]
    write_csv(output_dir / "bootstrap_poisoning_manifest.csv", manifest)

    clean = macro[macro["contaminated_bootstrap_windows"] == 0]
    half = macro[macro["contaminated_bootstrap_windows"] == 5]
    guard_by_count = (
        macro.groupby("contaminated_bootstrap_windows")[
            "guard_rejection_rate_macro"
        ]
        .mean()
        .sort_index()
    )
    print("\n" + "=" * 100)
    print("Startup-baseline poisoning stress test completed successfully.")
    print(f"Step 17 deviation threshold: {deviation_threshold:.6f}")
    print(
        f"Startup guard threshold: {guard_threshold:.6f}; "
        f"clean normal pseudo-session rejection="
        f"{np.mean([row['guard_rejected'] for row in guard_calibration_rows]):.4f}"
    )
    print("\nClean startup macro performance:")
    print(
        clean[
            [
                "method",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "false_positive_rate_macro",
                "guard_rejection_rate_macro",
            ]
        ].to_string(index=False)
    )
    print("\n50% contaminated startup macro performance:")
    print(
        half[
            [
                "method",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "false_positive_rate_macro",
                "guard_rejection_rate_macro",
            ]
        ].to_string(index=False)
    )
    print("\nGuard rejection rate by contaminated startup-window count:")
    print(guard_by_count.to_string())
    print(f"\nResults directory: {output_dir}")
    print(
        "\nNext: freeze the guarded session gate and integrate its trust/action "
        "signals with GNSS, V2X, identity, and vehicle-state context."
    )


if __name__ == "__main__":
    main()
