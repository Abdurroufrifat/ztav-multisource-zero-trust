#!/usr/bin/env python3
"""Falsify the Step 19 result with a physical-only vehicle-state signal.

Step 08 deliberately capped ``sensor_control_consistency_score`` at 0.20 in
the CAN-injection and combined-attack phases.  That engineering shortcut is
useful for exercising a policy, but it also makes the vehicle-state source a
near-perfect proxy for the experimental phase.  Step 19 showed that this source
dominates sparse-CAN detection.

This diagnostic repeats the exact Step 19 replay and fixed thresholds twice:

1. reported context: retain the Step 08 vehicle-state score;
2. physical-only context: recompute the score solely from SUMO acceleration
   and the finite difference of speed, without reading phase or attack labels.

No threshold is fitted or changed.  This is a counterfactual/falsification
audit, not an independent test.  It must be completed before freezing the
integrated policy for confirmatory evaluation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "guarded_can_only": ("physical", "guarded_can_alarm"),
    "physical_without_vehicle_state": (
        "physical",
        "proposed_without_vehicle_state_alarm",
    ),
    "physical_proposed_multisource": ("physical", "proposed_multisource_alarm"),
    "reported_proposed_multisource": ("reported", "proposed_multisource_alarm"),
}
PHYSICAL_SCORE_THRESHOLD = 0.35
SCORE_DECAY_MPS2 = 2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a physical-only vehicle-state counterfactual audit."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/physical_vehicle_state_counterfactual_w100"),
    )
    return parser.parse_args()


def locate(project_root: Path, name: str) -> Path:
    for candidate in (project_root / "src" / name, project_root / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find required script: {name}")


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


def recompute_physical_scores(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    required = {
        "simulation_time_s",
        "speed_mps",
        "acceleration_mps2",
        "sensor_control_consistency_score",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            "SUMO rows cannot support the physical counterfactual; missing "
            f"columns: {sorted(missing)}"
        )
    output: list[dict[str, str]] = []
    audits: list[dict[str, object]] = []
    previous_time: float | None = None
    previous_speed: float | None = None
    for row in rows:
        current = dict(row)
        time_s = float(row["simulation_time_s"])
        speed = float(row["speed_mps"])
        acceleration = float(row["acceleration_mps2"])
        if previous_time is None or previous_speed is None:
            derived_acceleration = acceleration
        else:
            delta_t = time_s - previous_time
            derived_acceleration = (
                (speed - previous_speed) / delta_t if delta_t > 0 else acceleration
            )
        error = abs(acceleration - derived_acceleration)
        physical_score = max(0.0, min(1.0, math.exp(-error / SCORE_DECAY_MPS2)))
        reported_score = float(row["sensor_control_consistency_score"])
        current["reported_sensor_control_consistency_score"] = f"{reported_score:.12g}"
        current["sensor_control_consistency_score"] = f"{physical_score:.12g}"
        output.append(current)
        audits.append(
            {
                "simulation_time_s": time_s,
                "phase": row["phase"],
                "ground_truth_attack": int(row["ground_truth_attack"]),
                "reported_sensor_control_score": reported_score,
                "physical_sensor_control_score": physical_score,
                "reported_below_monitor_threshold": int(
                    reported_score < PHYSICAL_SCORE_THRESHOLD
                ),
                "physical_below_monitor_threshold": int(
                    physical_score < PHYSICAL_SCORE_THRESHOLD
                ),
                "sumo_acceleration_mps2": acceleration,
                "speed_derived_acceleration_mps2": derived_acceleration,
                "absolute_acceleration_error_mps2": error,
            }
        )
        previous_time = time_s
        previous_speed = speed
    return output, audits


def metric_rows(
    step19: ModuleType,
    seed: int,
    source_file: str,
    density: str,
    reported: Sequence[dict[str, object]],
    physical: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    truth = [int(row["ground_truth_attack"]) for row in physical]
    frames = {"reported": reported, "physical": physical}
    output: list[dict[str, object]] = []
    for method, (variant, column) in METHODS.items():
        predictions = [int(row[column]) for row in frames[variant]]
        output.append(
            {
                "seed": seed,
                "hcrl_source_file": source_file,
                "density_scenario": density,
                "method": method,
                **step19.binary_metrics(truth, predictions),
            }
        )
    return output


def phase_rows(
    seed: int,
    source_file: str,
    density: str,
    reported: Sequence[dict[str, object]],
    physical: Sequence[dict[str, object]],
    sensor_audit: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    phase_order: list[str] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(physical):
        phase = str(row["phase"])
        if phase not in grouped_indices:
            phase_order.append(phase)
        grouped_indices[phase].append(index)
    variants = {"reported": reported, "physical": physical}
    for phase in phase_order:
        indices = grouped_indices[phase]
        audit_values = [sensor_audit[index] for index in indices]
        row: dict[str, object] = {
            "seed": seed,
            "hcrl_source_file": source_file,
            "density_scenario": density,
            "phase": phase,
            "ground_truth_attack": int(physical[indices[0]]["ground_truth_attack"]),
            "rows": len(indices),
            "mean_hcrl_attack_frames": statistics.fmean(
                int(physical[index]["hcrl_attack_frame_count"]) for index in indices
            ),
            "reported_sensor_score_mean": statistics.fmean(
                float(item["reported_sensor_control_score"]) for item in audit_values
            ),
            "physical_sensor_score_mean": statistics.fmean(
                float(item["physical_sensor_control_score"]) for item in audit_values
            ),
            "reported_sensor_below_threshold_rate": statistics.fmean(
                int(item["reported_below_monitor_threshold"]) for item in audit_values
            ),
            "physical_sensor_below_threshold_rate": statistics.fmean(
                int(item["physical_below_monitor_threshold"]) for item in audit_values
            ),
        }
        for method, (variant, column) in METHODS.items():
            row[f"{method}_alarm_rate"] = statistics.fmean(
                int(variants[variant][index][column]) for index in indices
            )
        output.append(row)
    return output


def latency_rows(
    seed: int,
    source_file: str,
    density: str,
    reported: Sequence[dict[str, object]],
    physical: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    variants = {"reported": reported, "physical": physical}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(physical):
        if int(row["ground_truth_attack"]):
            grouped[str(row["phase"])].append(index)
    output: list[dict[str, object]] = []
    for phase, indices in grouped.items():
        for method, (variant, column) in METHODS.items():
            offsets = [
                offset
                for offset, index in enumerate(indices)
                if int(variants[variant][index][column])
            ]
            output.append(
                {
                    "seed": seed,
                    "hcrl_source_file": source_file,
                    "density_scenario": density,
                    "phase": phase,
                    "method": method,
                    "phase_rows": len(indices),
                    "detected": int(bool(offsets)),
                    "first_detection_offset_windows": offsets[0] if offsets else -1,
                }
            )
    return output


def aggregate_metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    metrics = (
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "accuracy",
    )
    output: list[dict[str, object]] = []
    for (density, method), group in frame.groupby(
        ["density_scenario", "method"], sort=False
    ):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            output.append(
                {
                    "density_scenario": density,
                    "method": method,
                    "metric": metric,
                    "runs": len(values),
                    "mean": float(np.mean(values)),
                    "population_std": float(np.std(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return output


def aggregate_phases(frame: pd.DataFrame) -> list[dict[str, object]]:
    numeric_columns = [
        "mean_hcrl_attack_frames",
        "reported_sensor_score_mean",
        "physical_sensor_score_mean",
        "reported_sensor_below_threshold_rate",
        "physical_sensor_below_threshold_rate",
        *(f"{method}_alarm_rate" for method in METHODS),
    ]
    output: list[dict[str, object]] = []
    for (density, phase), group in frame.groupby(
        ["density_scenario", "phase"], sort=False
    ):
        row: dict[str, object] = {
            "density_scenario": density,
            "phase": phase,
            "ground_truth_attack": int(group["ground_truth_attack"].iloc[0]),
            "runs": len(group),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        output.append(row)
    return output


def plot_results(
    metrics: pd.DataFrame,
    phases: pd.DataFrame,
    density_order: Sequence[str],
    output_path: Path,
) -> None:
    labels = {
        "representative_all": "All densities",
        "low_1_5": "1-5",
        "medium_6_20": "6-20",
        "high_21_100": "21-100",
    }
    styles = {
        "reported_proposed_multisource": ("Reported Step 19", "o"),
        "physical_proposed_multisource": ("Physical-only vehicle state", "s"),
        "guarded_can_only": ("Guarded CAN only", "^"),
    }
    x = np.arange(len(density_order))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    can_phase = phases[phases["phase"] == "can_injection"]
    for method, (label, marker) in styles.items():
        f1_values: list[float] = []
        phase_values: list[float] = []
        for density in density_order:
            f1_values.append(
                float(
                    metrics[
                        metrics["density_scenario"].eq(density)
                        & metrics["method"].eq(method)
                    ]["f1"].mean()
                )
            )
            phase_values.append(
                float(
                    can_phase[can_phase["density_scenario"].eq(density)][
                        f"{method}_alarm_rate"
                    ].mean()
                )
            )
        axes[0].plot(x, f1_values, marker=marker, linewidth=2, label=label)
        axes[1].plot(x, phase_values, marker=marker, linewidth=2, label=label)
    for axis in axes:
        axis.set_xticks(x, [labels[item] for item in density_order])
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.3)
        axis.set_xlabel("HCRL malicious frames per 100-frame CAN window")
    axes[0].set_title("End-to-end F1")
    axes[0].set_ylabel("Mean F1")
    axes[1].set_title("CAN-injection phase recall")
    axes[1].set_ylabel("Mean alarm rate / recall")
    axes[1].legend(loc="lower right")
    figure.suptitle("Vehicle-state counterfactual falsification audit")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else project_root / args.results_dir
    )
    step19_path = locate(
        project_root, "19_integrate_guarded_can_multisource_policy.py"
    )
    spec_name = "ztav_step19_counterfactual"
    spec = importlib.util.spec_from_file_location(spec_name, step19_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Step 19: {step19_path}")
    step19 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = step19
    spec.loader.exec_module(step19)
    trust = step19.load_script(
        step19.locate_script(project_root, "ztav_phase0.py"),
        "ztav_core_step20",
    )

    can_predictions_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_predictions.csv"
    )
    can_threshold_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_thresholds.csv"
    )
    guard_path = (
        project_root
        / "results"
        / "startup_poisoning_stress_w100"
        / "bootstrap_poisoning_runs.csv"
    )
    sumo_paths = sorted(
        (project_root / "data" / "processed" / "sumo_repeated_seeds").glob(
            "sumo_context_attacks_seed_*.csv"
        )
    )
    if not sumo_paths:
        raise FileNotFoundError("No Step 10 repeated-seed SUMO datasets found")
    can_predictions = pd.read_csv(can_predictions_path)
    step19.validate_columns(
        can_predictions.columns,
        step19.REQUIRED_CAN_COLUMNS,
        "Step 17 predictions",
    )
    can_threshold = step19.load_operational_can_threshold(can_threshold_path)
    guards = step19.load_clean_startup_guard(guard_path)
    sources = sorted(
        source
        for source in can_predictions.loc[
            can_predictions["source_file"] != "normal_run_data.txt", "source_file"
        ].unique()
    )

    per_run: list[dict[str, object]] = []
    per_phase: list[dict[str, object]] = []
    latencies: list[dict[str, object]] = []
    score_audit: list[dict[str, object]] = []
    run_audit: list[dict[str, object]] = []
    total_runs = len(sumo_paths) * len(sources) * len(step19.DENSITY_ORDER)
    run_number = 0

    for sumo_path in sumo_paths:
        seed = step19.seed_from_filename(sumo_path)
        sumo_rows = step19.read_csv_rows(sumo_path)
        step19.validate_columns(sumo_rows[0], step19.REQUIRED_SUMO_COLUMNS, sumo_path.name)
        physical_sumo, sensor_audit = recompute_physical_scores(sumo_rows)
        for source_file in sources:
            for density in step19.DENSITY_ORDER:
                run_number += 1
                print(
                    f"[{run_number}/{total_runs}] seed={seed}, source={source_file}, "
                    f"density={density}"
                )
                reported_replay, replay_audit = step19.build_replay_rows(
                    sumo_rows,
                    can_predictions,
                    source_file,
                    density,
                    seed,
                )
                physical_replay, physical_replay_audit = step19.build_replay_rows(
                    physical_sumo,
                    can_predictions,
                    source_file,
                    density,
                    seed,
                )
                if [row["hcrl_window_index"] for row in reported_replay] != [
                    row["hcrl_window_index"] for row in physical_replay
                ]:
                    raise RuntimeError("Reported and physical replays are not paired")
                reported_evaluation = step19.evaluate_replay(
                    reported_replay,
                    can_threshold,
                    guards[source_file],
                    trust,
                )
                physical_evaluation = step19.evaluate_replay(
                    physical_replay,
                    can_threshold,
                    guards[source_file],
                    trust,
                )
                per_run.extend(
                    metric_rows(
                        step19,
                        seed,
                        source_file,
                        density,
                        reported_evaluation,
                        physical_evaluation,
                    )
                )
                per_phase.extend(
                    phase_rows(
                        seed,
                        source_file,
                        density,
                        reported_evaluation,
                        physical_evaluation,
                        sensor_audit,
                    )
                )
                latencies.extend(
                    latency_rows(
                        seed,
                        source_file,
                        density,
                        reported_evaluation,
                        physical_evaluation,
                    )
                )
                run_audit.append(
                    {
                        **replay_audit,
                        "paired_replay_verified": replay_audit
                        == physical_replay_audit,
                        "can_deviation_threshold": can_threshold,
                        "physical_score_formula": "exp(-abs(sumo_acceleration-speed_difference_acceleration)/2.5)",
                    }
                )
        for item in sensor_audit:
            score_audit.append({"seed": seed, **item})

    per_run_frame = pd.DataFrame(per_run)
    phase_frame = pd.DataFrame(per_phase)
    aggregate = aggregate_metrics(per_run_frame)
    phase_aggregate = aggregate_phases(phase_frame)
    manifest: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "vehicle-state counterfactual falsification audit"},
        {"item": "total_paired_runs", "value": total_runs},
        {"item": "threshold_policy", "value": "all Step 19 thresholds frozen; no refitting"},
        {"item": "reported_variant", "value": "retains Step 08 phase-imposed sensor-control cap"},
        {"item": "physical_variant", "value": "uses acceleration vs speed-difference acceleration only"},
        {"item": "physical_score_decay_mps2", "value": SCORE_DECAY_MPS2},
        {"item": "sensor_monitor_threshold", "value": PHYSICAL_SCORE_THRESHOLD},
        {"item": "label_usage", "value": "labels used only for replay construction and evaluation"},
        {"item": "interpretation_limit", "value": "counterfactual audit on existing SUMO/HCRL data; not independent validation"},
    ]

    write_csv(results_dir / "counterfactual_per_run_metrics.csv", per_run)
    write_csv(results_dir / "counterfactual_aggregate_metrics.csv", aggregate)
    write_csv(results_dir / "counterfactual_per_phase_metrics.csv", per_phase)
    write_csv(
        results_dir / "counterfactual_phase_summary.csv", phase_aggregate
    )
    write_csv(results_dir / "counterfactual_detection_latencies.csv", latencies)
    write_csv(results_dir / "counterfactual_sensor_score_audit.csv", score_audit)
    write_csv(results_dir / "counterfactual_replay_audit.csv", run_audit)
    write_csv(results_dir / "counterfactual_manifest.csv", manifest)
    plot_results(
        per_run_frame,
        phase_frame,
        step19.DENSITY_ORDER,
        results_dir / "counterfactual_performance.png",
    )

    print("\n" + "=" * 80)
    print("Physical vehicle-state counterfactual completed successfully.")
    print(f"Paired replay runs: {total_runs}")
    print("All Step 19 thresholds remained frozen.")
    print("\nMean end-to-end F1 and counterfactual change:")
    for density in step19.DENSITY_ORDER:
        values = per_run_frame[per_run_frame["density_scenario"] == density]
        reported_f1 = values.loc[
            values["method"] == "reported_proposed_multisource", "f1"
        ].mean()
        physical_f1 = values.loc[
            values["method"] == "physical_proposed_multisource", "f1"
        ].mean()
        print(
            f"  {density:<21} reported={reported_f1:.4f}, "
            f"physical-only={physical_f1:.4f}, delta={physical_f1-reported_f1:+.4f}"
        )
    print(f"\nResults directory: {results_dir}")
    print("\nNext: decide whether the integrated policy survives falsification before freezing it.")


if __name__ == "__main__":
    main()
