#!/usr/bin/env python3
"""Evaluate subtle-to-full continuous-source attack severity.

Step 10 found identical results for source thresholds 0.2, 0.3 and 0.4,
indicating that the original injected attacks were strongly separated from
healthy evidence.  This stage performs an offline counterfactual severity
sweep for isolated GNSS, V2X and CAN attacks across all preserved random seeds.

For an attacked evidence column, severity ``s`` is defined as:

    counterfactual = healthy_median + s * (full_attack - healthy_median)

Thus, 0.0 is the seed-specific healthy median and 1.0 is the full-strength
observation generated in Step 08.  This is a robustness/ablation analysis, not
a substitute for physical attack experiments and not an independent test set.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Sequence


DEFAULT_SEVERITIES = tuple(round(value / 10.0, 1) for value in range(1, 11))
ATTACK_EPISODES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gps_spoofing": (
        "gps_spoofing",
        "recovery_after_gps",
        ("gnss_imu_consistency_score",),
    ),
    "v2x_falsification": (
        "v2x_falsification",
        "recovery_after_v2x",
        ("v2x_consistency_score",),
    ),
    "can_injection": (
        "can_injection",
        "recovery_after_can",
        ("can_behavior_score", "sensor_control_consistency_score"),
    ),
}

WEIGHTS: dict[str, tuple[str, float]] = {
    "identity": ("identity_score", 0.15),
    "device_posture": ("device_posture_score", 0.13),
    "can_behavior": ("can_behavior_score", 0.20),
    "gnss_imu_consistency": ("gnss_imu_consistency_score", 0.17),
    "v2x_consistency": ("v2x_consistency_score", 0.10),
    "sensor_control_consistency": ("sensor_control_consistency_score", 0.20),
    "freshness": ("freshness_score", 0.05),
}
WEIGHTED_MEMORY = 0.65
WEIGHTED_INITIAL_TRUST = 0.80
WEIGHTED_ALARM_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an offline counterfactual attack-severity sweep."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--severities",
        type=float,
        nargs="+",
        default=list(DEFAULT_SEVERITIES),
        help="Severity values in (0, 1] (default: 0.1 through 1.0).",
    )
    args = parser.parse_args()
    if any(not 0.0 < value <= 1.0 for value in args.severities):
        parser.error("Every severity must be in (0, 1]")
    args.severities = sorted(set(round(value, 6) for value in args.severities))
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty results: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_from_filename(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot extract seed from {path.name}") from exc


def healthy_medians(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    healthy = [row for row in rows if row["phase"] == "healthy_baseline"]
    if not healthy:
        raise ValueError("Dataset has no healthy_baseline rows")
    score_columns = {column for column, _ in WEIGHTS.values()}
    return {
        column: statistics.median(float(row[column]) for row in healthy)
        for column in score_columns
    }


def build_counterfactual_episode(
    rows: Sequence[dict[str, str]],
    attack_phase: str,
    recovery_phase: str,
    attacked_columns: Sequence[str],
    severity: float,
    medians: dict[str, float],
) -> list[dict[str, str]]:
    selected_phases = {"healthy_baseline", attack_phase, recovery_phase}
    episode: list[dict[str, str]] = []
    for source in rows:
        if source["phase"] not in selected_phases:
            continue
        row = dict(source)
        if row["phase"] == attack_phase:
            for column in attacked_columns:
                healthy_value = medians[column]
                full_attack_value = float(row[column])
                counterfactual = healthy_value + severity * (
                    full_attack_value - healthy_value
                )
                row[column] = f"{max(0.0, min(1.0, counterfactual)):.9f}"
            row["ground_truth_attack"] = "1"
        else:
            row["ground_truth_attack"] = "0"
        episode.append(row)
    if not episode:
        raise ValueError(f"No rows selected for {attack_phase}")
    return episode


def weighted_alarms(rows: Sequence[dict[str, str]]) -> list[int]:
    trust = WEIGHTED_INITIAL_TRUST
    alarms: list[int] = []
    total_weight = sum(weight for _, weight in WEIGHTS.values())
    for row in rows:
        instantaneous = sum(
            float(row[column]) * weight for column, weight in WEIGHTS.values()
        ) / total_weight
        trust = WEIGHTED_MEMORY * trust + (1.0 - WEIGHTED_MEMORY) * instantaneous
        alarms.append(int(trust < WEIGHTED_ALARM_THRESHOLD))
    return alarms


def context_aware_alarms(
    rows: Sequence[dict[str, str]], step09: ModuleType
) -> list[int]:
    monitors = {
        source: step09.PersistentMonitor(
            threshold=threshold,
            attack_persistence=step09.ATTACK_PERSISTENCE,
            recovery_persistence=step09.RECOVERY_PERSISTENCE,
        )
        for source, (_, threshold) in step09.SOURCE_THRESHOLDS.items()
    }
    identity = step09.BooleanCriticalMonitor(
        recovery_persistence=step09.RECOVERY_PERSISTENCE
    )
    device = step09.BooleanCriticalMonitor(
        recovery_persistence=step09.RECOVERY_PERSISTENCE
    )
    alarms: list[int] = []
    for row in rows:
        source_states = [
            monitors[source].update(float(row[column]))
            for source, (column, _) in step09.SOURCE_THRESHOLDS.items()
        ]
        identity_state = identity.update(bool(int(row["identity_critical_failure"])))
        device_state = device.update(bool(int(row["device_critical_failure"])))
        alarms.append(int(any(source_states) or identity_state or device_state))
    return alarms


def calculate_episode_metrics(
    rows: Sequence[dict[str, str]], alarms: Sequence[int], attack_phase: str
) -> dict[str, float | int | str]:
    if len(rows) != len(alarms):
        raise ValueError("rows and alarms must have equal lengths")
    attack_indices = [index for index, row in enumerate(rows) if row["phase"] == attack_phase]
    normal_indices = [index for index, row in enumerate(rows) if row["phase"] != attack_phase]
    if not attack_indices or not normal_indices:
        raise ValueError(f"Incomplete episode for {attack_phase}")

    detected = sum(alarms[index] for index in attack_indices)
    false_alarms = sum(alarms[index] for index in normal_indices)
    first_alarm = next((index for index in attack_indices if alarms[index]), None)
    start_index = attack_indices[0]
    latency: float | str = (
        "not_detected" if first_alarm is None else float(first_alarm - start_index)
    )
    return {
        "attack_rows": len(attack_indices),
        "normal_and_recovery_rows": len(normal_indices),
        "detected_attack_rows": detected,
        "false_alarm_rows": false_alarms,
        "recall": detected / len(attack_indices),
        "false_positive_rate": false_alarms / len(normal_indices),
        "detection_latency_s": latency,
    }


def aggregate(per_seed: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        key = (str(row["attack_type"]), float(row["severity"]), str(row["method"]))
        grouped[key].append(row)

    output: list[dict[str, object]] = []
    for (attack_type, severity, method), values in sorted(grouped.items()):
        recalls = [float(row["recall"]) for row in values]
        fprs = [float(row["false_positive_rate"]) for row in values]
        numeric_latencies = [
            float(row["detection_latency_s"])
            for row in values
            if row["detection_latency_s"] != "not_detected"
        ]
        output.append(
            {
                "attack_type": attack_type,
                "severity": severity,
                "method": method,
                "runs": len(values),
                "recall_mean": round(statistics.fmean(recalls), 6),
                "recall_population_std": round(statistics.pstdev(recalls), 6),
                "recall_minimum": round(min(recalls), 6),
                "recall_maximum": round(max(recalls), 6),
                "false_positive_rate_mean": round(statistics.fmean(fprs), 6),
                "detected_runs": len(numeric_latencies),
                "latency_mean_s": (
                    "not_detected"
                    if not numeric_latencies
                    else round(statistics.fmean(numeric_latencies), 6)
                ),
            }
        )
    return output


def detection_boundaries(
    aggregate_rows: Sequence[dict[str, object]], target_recall: float = 0.95
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in aggregate_rows:
        groups[(str(row["attack_type"]), str(row["method"]))].append(row)
    output: list[dict[str, object]] = []
    for (attack_type, method), values in sorted(groups.items()):
        values = sorted(values, key=lambda row: float(row["severity"]))
        qualifying = [row for row in values if float(row["recall_mean"]) >= target_recall]
        boundary = "not_reached" if not qualifying else qualifying[0]["severity"]
        output.append(
            {
                "attack_type": attack_type,
                "method": method,
                "target_mean_recall": target_recall,
                "minimum_tested_severity_meeting_target": boundary,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_dir = project_root / "data" / "processed" / "sumo_repeated_seeds"
    dataset_paths = sorted(dataset_dir.glob("sumo_context_attacks_seed_*.csv"))
    if not dataset_paths:
        raise FileNotFoundError(
            f"No repeated-seed datasets found in {dataset_dir}. Run Step 10 first."
        )

    step09_path = project_root / "src" / "09_evaluate_context_aware_zero_trust.py"
    if not step09_path.exists():
        step09_path = project_root / step09_path.name
    step09 = load_script(step09_path, "ztav_step09_severity")

    datasets = {
        seed_from_filename(path): read_csv(path) for path in dataset_paths
    }
    per_seed: list[dict[str, object]] = []
    for seed, source_rows in datasets.items():
        medians = healthy_medians(source_rows)
        for attack_type, (attack_phase, recovery_phase, columns) in ATTACK_EPISODES.items():
            for severity in args.severities:
                episode = build_counterfactual_episode(
                    source_rows,
                    attack_phase,
                    recovery_phase,
                    columns,
                    severity,
                    medians,
                )
                methods = {
                    "weighted_counterfactual": weighted_alarms(episode),
                    "context_aware": context_aware_alarms(episode, step09),
                }
                for method, alarms in methods.items():
                    metrics = calculate_episode_metrics(episode, alarms, attack_phase)
                    per_seed.append(
                        {
                            "seed": seed,
                            "attack_type": attack_type,
                            "severity": severity,
                            "method": method,
                            **{
                                key: round(value, 6) if isinstance(value, float) else value
                                for key, value in metrics.items()
                            },
                        }
                    )

    aggregate_rows = aggregate(per_seed)
    boundary_rows = detection_boundaries(aggregate_rows)
    definition_rows: list[dict[str, object]] = [
        {
            "item": "severity_definition",
            "value": "healthy_median + severity * (full_attack - healthy_median)",
        },
        {
            "item": "scope",
            "value": "offline counterfactual GNSS, V2X and CAN robustness analysis",
        },
        {
            "item": "seeds",
            "value": ";".join(map(str, sorted(datasets))),
        },
        {
            "item": "tested_severities",
            "value": ";".join(map(str, args.severities)),
        },
    ]

    results_dir = project_root / "results" / "attack_severity_sweep"
    write_csv(results_dir / "severity_sweep_per_seed.csv", per_seed)
    write_csv(results_dir / "severity_sweep_aggregate.csv", aggregate_rows)
    write_csv(results_dir / "detection_boundaries.csv", boundary_rows)
    write_csv(results_dir / "severity_definition.csv", definition_rows)

    print("\n" + "=" * 78)
    print("Attack-severity robustness sweep completed successfully.")
    print(f"Seeds: {len(datasets)}")
    print(f"Severity levels: {len(args.severities)}")
    print(f"Evaluations: {len(per_seed):,}")
    print("\nMinimum tested severity reaching mean recall >= 0.95:")
    for row in boundary_rows:
        print(
            f"  {str(row['attack_type']):18s} {str(row['method']):25s} "
            f"{row['minimum_tested_severity_meeting_target']}"
        )
    print(f"\nResults directory: {results_dir}")
    print("\nNext: review detection boundaries and design external validation.")


if __name__ == "__main__":
    main()
