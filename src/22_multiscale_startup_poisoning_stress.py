#!/usr/bin/env python3
"""Stress-test startup poisoning of the Step 21 micro-gate.

Step 21 found that instantaneous micro-window fusion improves sparse recall but
creates too many false alarms.  The viable exploratory candidate is therefore
the two-consecutive-anomaly multiscale gate.  Before accepting that candidate,
this script contaminates its 50-window trusted startup baseline with 0--50
attack micro-windows and measures:

* micro-gate and fused parent-window precision, recall, F1, and FPR;
* a startup-consistency guard calibrated on attack-free normal sessions only;
* the contamination point where detector performance fails; and
* clean-start guard rejection on normal and attack-capture sessions.

Attack labels are used only to construct the controlled poisoning experiment
and to evaluate decisions.  They are not used to fit the gate threshold or the
startup-consistency guard.

Run from D:\\ztav_project after Step 21:

    .\\.venv\\Scripts\\python.exe src\\22_multiscale_startup_poisoning_stress.py

This is an exploratory stress test, not production automotive software.
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


CONTAMINATION_COUNTS = (0, 1, 2, 5, 10, 20, 25, 30, 40, 50)
REPETITIONS = 5
GUARD_TARGET_FPR = 0.05
EPSILON = 1e-12
METHOD_COLUMNS = {
    "micro_persistent_2": "w20_any_alarm_persistent_2",
    "multiscale_persistent_2": "multiscale_alarm_persistent_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test Step 21 micro-baseline poisoning."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
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


def startup_metrics(
    bootstrap: np.ndarray,
    floors: np.ndarray,
) -> np.ndarray:
    midpoint = len(bootstrap) // 2
    local_scale = 1.4826 * np.median(
        np.abs(bootstrap - np.median(bootstrap, axis=0)),
        axis=0,
    )
    half_gap = np.abs(
        np.median(bootstrap[:midpoint], axis=0)
        - np.median(bootstrap[midpoint:], axis=0)
    )
    value_range = np.ptp(bootstrap, axis=0)
    return np.asarray(
        [
            np.quantile(local_scale / np.maximum(floors, EPSILON), 0.75),
            np.quantile(half_gap / np.maximum(floors, EPSILON), 0.75),
            np.quantile(value_range / np.maximum(floors, EPSILON), 0.75),
        ],
        dtype=float,
    )


def calibrate_guard(
    normal: pd.DataFrame,
    floors: np.ndarray,
    step21: ModuleType,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    ordered = normal.sort_values("window_index")
    complete = len(ordered) // step21.PSEUDO_SESSION_MICRO_WINDOWS
    split = complete // 2
    if split < 10:
        raise ValueError("Too few normal sessions to calibrate startup guard")
    matrices: list[np.ndarray] = []
    session_ids: list[str] = []
    partitions: list[str] = []
    for index in range(complete):
        start = index * step21.PSEUDO_SESSION_MICRO_WINDOWS
        bootstrap = ordered.iloc[
            start : start + step21.BOOTSTRAP_MICRO_WINDOWS
        ][list(step21.STRUCTURAL_FEATURES)].to_numpy(dtype=float)
        matrices.append(startup_metrics(bootstrap, floors))
        session_ids.append(f"normal_micro_session_{index:03d}")
        partitions.append("guard_calibration" if index < split else "guard_holdout")
    matrix = np.vstack(matrices)
    calibration = matrix[:split]
    median = np.median(calibration, axis=0)
    mad = 1.4826 * np.median(np.abs(calibration - median), axis=0)
    standard = calibration.std(axis=0)
    scale = np.where(mad > EPSILON, mad, np.where(standard > EPSILON, standard, 1.0))
    scores = np.max(np.abs((matrix - median) / scale), axis=1)
    threshold = float(np.quantile(scores[:split], 1.0 - GUARD_TARGET_FPR))
    audit = [
        {
            "micro_session_id": session_ids[index],
            "partition": partitions[index],
            "startup_dispersion_ratio": matrix[index, 0],
            "startup_half_gap_ratio": matrix[index, 1],
            "startup_range_ratio": matrix[index, 2],
            "guard_score": scores[index],
            "guard_threshold": threshold,
            "guard_rejected": bool(scores[index] >= threshold),
        }
        for index in range(len(scores))
    ]
    return median, scale, threshold, audit


def guard_score(
    bootstrap: np.ndarray,
    floors: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    metrics = startup_metrics(bootstrap, floors)
    score = float(np.max(np.abs((metrics - median) / scale)))
    return score, metrics


def score_with_bootstrap(
    source: pd.DataFrame,
    bootstrap: np.ndarray,
    floors: np.ndarray,
    threshold: float,
    step21: ModuleType,
) -> pd.DataFrame:
    evaluation = source.sort_values("window_index").iloc[
        step21.BOOTSTRAP_MICRO_WINDOWS :
    ].copy()
    center = np.median(bootstrap, axis=0)
    local_scale = 1.4826 * np.median(np.abs(bootstrap - center), axis=0)
    scale = np.maximum(local_scale, floors)
    values = evaluation[list(step21.STRUCTURAL_FEATURES)].to_numpy(dtype=float)
    deviations = np.abs(values - center) / np.maximum(scale, EPSILON)
    evaluation["micro_structural_deviation"] = np.quantile(
        deviations,
        0.75,
        axis=1,
    )
    evaluation["micro_session_id"] = "poisoning_stress_session"
    evaluation["micro_alarm_instant"] = (
        evaluation["micro_structural_deviation"] >= threshold
    )
    instant = evaluation["micro_alarm_instant"].to_numpy(dtype=bool)
    evaluation["micro_alarm_persistent_2"] = instant & np.r_[False, instant[:-1]]
    evaluation["micro_continuous_can_trust"] = 1.0 / (
        1.0
        + np.square(
            evaluation["micro_structural_deviation"] / max(threshold, EPSILON)
        )
    )
    return evaluation


def metric_values(
    step21: ModuleType,
    parent: pd.DataFrame,
    method: str,
    column: str,
) -> dict[str, object]:
    row = step21.metric_row(parent, method, column, "poisoning_stress")
    return {
        key: row[key]
        for key in (
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "tn",
            "fp",
            "fn",
            "tp",
        )
    }


def aggregate_runs(runs: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups = runs.groupby(
        [
            "source_file",
            "contaminated_bootstrap_windows",
            "contamination_fraction",
            "method",
        ],
        sort=True,
    )
    for keys, group in groups:
        source, count, fraction, method = keys
        output.append(
            {
                "source_file": source,
                "contaminated_bootstrap_windows": count,
                "contamination_fraction": fraction,
                "method": method,
                "repetitions": len(group),
                "guard_rejection_rate": float(group["guard_rejected"].mean()),
                "precision_mean": float(group["precision"].mean()),
                "precision_std": float(group["precision"].std(ddof=0)),
                "recall_mean": float(group["recall"].mean()),
                "recall_std": float(group["recall"].std(ddof=0)),
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=0)),
                "false_positive_rate_mean": float(
                    group["false_positive_rate"].mean()
                ),
                "false_positive_rate_std": float(
                    group["false_positive_rate"].std(ddof=0)
                ),
                "guard_score_mean": float(group["guard_score"].mean()),
                "guard_score_min": float(group["guard_score"].min()),
                "guard_score_max": float(group["guard_score"].max()),
            }
        )
    return output


def macro_rows(aggregate: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for keys, group in aggregate.groupby(
        ["contaminated_bootstrap_windows", "contamination_fraction", "method"],
        sort=True,
    ):
        count, fraction, method = keys
        output.append(
            {
                "contaminated_bootstrap_windows": count,
                "contamination_fraction": fraction,
                "method": method,
                "sources": len(group),
                "guard_rejection_rate_macro": float(
                    group["guard_rejection_rate"].mean()
                ),
                "precision_macro": float(group["precision_mean"].mean()),
                "recall_macro": float(group["recall_mean"].mean()),
                "f1_macro": float(group["f1_mean"].mean()),
                "false_positive_rate_macro": float(
                    group["false_positive_rate_mean"].mean()
                ),
            }
        )
    return output


def failure_boundaries(aggregate: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (source, method), group in aggregate.groupby(
        ["source_file", "method"], sort=True
    ):
        group = group.sort_values("contaminated_bootstrap_windows")
        recall_fail = group[group["recall_mean"] < 0.90]
        fpr_fail = group[group["false_positive_rate_mean"] > 0.05]
        guard_success = group[group["guard_rejection_rate"] >= 0.90]
        output.append(
            {
                "source_file": source,
                "method": method,
                "first_count_recall_below_0_90": int(
                    recall_fail.iloc[0]["contaminated_bootstrap_windows"]
                )
                if len(recall_fail)
                else -1,
                "first_count_fpr_above_0_05": int(
                    fpr_fail.iloc[0]["contaminated_bootstrap_windows"]
                )
                if len(fpr_fail)
                else -1,
                "first_count_guard_rejection_at_least_0_90": int(
                    guard_success.iloc[0]["contaminated_bootstrap_windows"]
                )
                if len(guard_success)
                else -1,
            }
        )
    return output


def plot_results(macro: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
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
    axes[0].set(title="Attack recall", ylabel="Macro recall")
    axes[1].set(title="False alarms", ylabel="Macro FPR")
    axes[2].set(title="Startup consistency guard", ylabel="Rejection rate")
    for axis in axes:
        axis.set_xlabel("Poisoned startup-window fraction")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Multiscale micro-baseline poisoning sensitivity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    step21 = load_script(
        locate_script(project_root, "21_multiscale_sparse_can_gate.py"),
        "ztav_step21_poisoning",
    )
    cache_path = (
        project_root
        / "data"
        / "processed"
        / "car_hacking_windows_w20_structural.csv"
    )
    calibration_path = (
        project_root
        / "results"
        / "multiscale_sparse_can_gate"
        / "micro_calibration_summary.csv"
    )
    w100_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_predictions.csv"
    )
    for path in (cache_path, calibration_path, w100_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Step 21 asset: {path}")
    data = pd.read_csv(cache_path)
    normal = data[data["source_file"] == step21.NORMAL_SOURCE].copy()
    attacks = data[data["source_file"] != step21.NORMAL_SOURCE].copy()
    floors = step21.scale_floors(normal)
    calibration = pd.read_csv(calibration_path)
    if len(calibration) != 1:
        raise ValueError("Expected one Step 21 calibration row")
    micro_threshold = float(calibration.iloc[0]["micro_deviation_threshold"])
    w100 = pd.read_csv(w100_path)

    guard_median, guard_scale, guard_threshold, guard_audit = calibrate_guard(
        normal,
        floors,
        step21,
    )
    normal_guard = pd.DataFrame(guard_audit)
    normal_holdout_rejection = float(
        normal_guard.loc[
            normal_guard["partition"] == "guard_holdout", "guard_rejected"
        ].mean()
    )

    run_rows: list[dict[str, object]] = []
    clean_source_guards: list[dict[str, object]] = []
    total_runs = (
        attacks["source_file"].nunique()
        * len(CONTAMINATION_COUNTS)
        * args.repetitions
    )
    run_number = 0
    for source_file, source in attacks.groupby("source_file", sort=True):
        source = source.sort_values("window_index").reset_index(drop=True)
        clean_bootstrap_frame = source.iloc[: step21.BOOTSTRAP_MICRO_WINDOWS]
        if int(clean_bootstrap_frame["attack_frame_count"].sum()) != 0:
            raise ValueError(f"Clean bootstrap for {source_file} contains attacks")
        clean_bootstrap = clean_bootstrap_frame[
            list(step21.STRUCTURAL_FEATURES)
        ].to_numpy(dtype=float)
        attack_pool = source[source["binary_target"] == 1]
        if len(attack_pool) < step21.BOOTSTRAP_MICRO_WINDOWS:
            raise ValueError(f"Too few attack micro-windows for {source_file}")
        source_w100 = w100[w100["source_file"] == source_file]
        clean_score, clean_metrics = guard_score(
            clean_bootstrap,
            floors,
            guard_median,
            guard_scale,
        )
        clean_source_guards.append(
            {
                "source_file": source_file,
                "guard_score": clean_score,
                "guard_threshold": guard_threshold,
                "guard_rejected": clean_score >= guard_threshold,
                "startup_dispersion_ratio": clean_metrics[0],
                "startup_half_gap_ratio": clean_metrics[1],
                "startup_range_ratio": clean_metrics[2],
            }
        )
        attack_values = attack_pool[
            list(step21.STRUCTURAL_FEATURES)
        ].to_numpy(dtype=float)
        for count in CONTAMINATION_COUNTS:
            for repetition in range(args.repetitions):
                run_number += 1
                print(
                    f"[{run_number}/{total_runs}] source={source_file}, "
                    f"poisoned={count}/{step21.BOOTSTRAP_MICRO_WINDOWS}, "
                    f"rep={repetition + 1}"
                )
                rng = np.random.default_rng(
                    42 + repetition + count * 1_000 + sum(source_file.encode("utf-8"))
                )
                contaminated = clean_bootstrap.copy()
                if count:
                    positions = rng.choice(
                        step21.BOOTSTRAP_MICRO_WINDOWS,
                        size=count,
                        replace=False,
                    )
                    samples = rng.choice(len(attack_values), size=count, replace=False)
                    contaminated[positions] = attack_values[samples]
                startup_score, startup_values = guard_score(
                    contaminated,
                    floors,
                    guard_median,
                    guard_scale,
                )
                scored = score_with_bootstrap(
                    source,
                    contaminated,
                    floors,
                    micro_threshold,
                    step21,
                )
                parent, alignment = step21.aggregate_to_parent(scored, source_w100)
                for method, column in METHOD_COLUMNS.items():
                    metrics = metric_values(step21, parent, method, column)
                    run_rows.append(
                        {
                            "source_file": source_file,
                            "contaminated_bootstrap_windows": count,
                            "contamination_fraction": count
                            / step21.BOOTSTRAP_MICRO_WINDOWS,
                            "repetition": repetition,
                            "random_seed": int(
                                42
                                + repetition
                                + count * 1_000
                                + sum(source_file.encode("utf-8"))
                            ),
                            "method": method,
                            "guard_score": startup_score,
                            "guard_threshold": guard_threshold,
                            "guard_rejected": startup_score >= guard_threshold,
                            "startup_dispersion_ratio": startup_values[0],
                            "startup_half_gap_ratio": startup_values[1],
                            "startup_range_ratio": startup_values[2],
                            "parents_matched": alignment[
                                "parents_matched_to_step17"
                            ],
                            **metrics,
                        }
                    )

    runs = pd.DataFrame(run_rows)
    aggregate = pd.DataFrame(aggregate_runs(runs))
    macro = pd.DataFrame(macro_rows(aggregate))
    boundaries = failure_boundaries(aggregate)
    output_dir = (
        project_root / "results" / "multiscale_startup_poisoning_stress"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "micro_poisoning_runs.csv", index=False)
    aggregate.to_csv(output_dir / "micro_poisoning_aggregate.csv", index=False)
    macro.to_csv(output_dir / "micro_poisoning_macro_summary.csv", index=False)
    write_csv(output_dir / "micro_guard_normal_audit.csv", guard_audit)
    write_csv(output_dir / "micro_clean_source_guard_audit.csv", clean_source_guards)
    write_csv(output_dir / "micro_poisoning_failure_boundaries.csv", boundaries)
    plot_results(macro, output_dir / "micro_poisoning_sensitivity.png")
    manifest = [
        {"item": "experiment_type", "value": "multiscale micro-baseline poisoning stress"},
        {"item": "candidate_policy", "value": "two-hit persistent multiscale gate"},
        {"item": "bootstrap_micro_windows", "value": step21.BOOTSTRAP_MICRO_WINDOWS},
        {"item": "contamination_counts", "value": ";".join(map(str, CONTAMINATION_COUNTS))},
        {"item": "repetitions", "value": args.repetitions},
        {"item": "micro_deviation_threshold_frozen", "value": micro_threshold},
        {"item": "guard_target_fpr", "value": GUARD_TARGET_FPR},
        {"item": "guard_threshold", "value": guard_threshold},
        {"item": "normal_guard_holdout_rejection_rate", "value": normal_holdout_rejection},
        {"item": "label_usage", "value": "controlled contamination construction and evaluation only"},
        {"item": "external_validity_limit", "value": "exploratory HCRL stress test"},
    ]
    write_csv(output_dir / "micro_poisoning_manifest.csv", manifest)

    clean = macro[macro["contaminated_bootstrap_windows"] == 0]
    half = macro[macro["contaminated_bootstrap_windows"] == 25]
    print("\n" + "=" * 84)
    print("Multiscale startup-poisoning stress test completed successfully.")
    print(f"Micro threshold remained frozen at {micro_threshold:.6f}")
    print(f"Startup guard threshold: {guard_threshold:.6f}")
    print(f"Normal guard holdout rejection rate: {normal_holdout_rejection:.4f}")
    print("\nClean-start macro performance:")
    print(
        clean[["method", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro"]]
        .to_string(index=False)
    )
    print("\n50% contaminated-start macro performance:")
    print(
        half[["method", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro", "guard_rejection_rate_macro"]]
        .to_string(index=False)
    )
    print(f"\nResults directory: {output_dir}")
    print("\nNext: accept, modify, or reject the persistent micro-gate before final policy freeze.")


if __name__ == "__main__":
    main()
