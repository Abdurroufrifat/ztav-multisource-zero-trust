#!/usr/bin/env python3
"""Repeat the SUMO attack experiment and perform sensitivity analysis.

This stage runs Step 08 with multiple deterministic random seeds, preserves each
generated context dataset, and compares:

1. the original global weighted-threshold alarm, and
2. the proposed persistent source-aware alarm from Step 09.

It also evaluates a declared grid of source thresholds and persistence values.
The grid is reported as a robustness analysis; it must not be described as an
independent test-set optimization.

Run from D:\\ztav_project:

    .\\.venv\\Scripts\\python.exe src\\10_repeated_seed_and_sensitivity.py

The default five runs use seeds 2027 through 2031.  To re-analyze already saved
per-seed datasets without running SUMO again:

    .\\.venv\\Scripts\\python.exe src\\10_repeated_seed_and_sensitivity.py \
        --skip-simulation

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Sequence


DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)
SENSITIVITY_THRESHOLDS = (0.20, 0.30, 0.40)
SENSITIVITY_ATTACK_PERSISTENCE = (1, 2, 3)
SENSITIVITY_RECOVERY_PERSISTENCE = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated-seed validation and threshold sensitivity."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 2027 2028 2029 2030 2031).",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Analyze previously preserved per-seed context CSV files.",
    )
    return parser.parse_args()


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
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def calculate_metrics(truth: Sequence[int], alarms: Sequence[int]) -> dict[str, float | int]:
    if len(truth) != len(alarms):
        raise ValueError("truth and alarms must have equal lengths")
    tp = fp = tn = fn = 0
    for expected, predicted in zip(truth, alarms):
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fn += 1
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": safe_divide(fp, fp + tn),
        "accuracy": safe_divide(tp + tn, len(truth)),
    }


def source_aware_alarms_complete(
    rows: Sequence[dict[str, str]],
    step09: ModuleType,
    source_thresholds: dict[str, tuple[str, float]],
    attack_persistence: int,
    recovery_persistence: int,
) -> list[int]:
    """Update all source monitors without Boolean short-circuiting."""

    monitors = {
        source: step09.PersistentMonitor(
            threshold=threshold,
            attack_persistence=attack_persistence,
            recovery_persistence=recovery_persistence,
        )
        for source, (_, threshold) in source_thresholds.items()
    }
    identity = step09.BooleanCriticalMonitor(
        recovery_persistence=recovery_persistence
    )
    device = step09.BooleanCriticalMonitor(
        recovery_persistence=recovery_persistence
    )
    alarms: list[int] = []
    for row in rows:
        source_states = [
            monitors[source].update(float(row[column]))
            for source, (column, _) in source_thresholds.items()
        ]
        identity_state = identity.update(bool(int(row["identity_critical_failure"])))
        device_state = device.update(bool(int(row["device_critical_failure"])))
        alarms.append(int(any(source_states) or identity_state or device_state))
    return alarms


def latency_rows(
    seed: int,
    rows: Sequence[dict[str, str]],
    alarms_by_method: dict[str, Sequence[int]],
) -> list[dict[str, object]]:
    phases: dict[str, list[int]] = defaultdict(list)
    phase_order: list[str] = []
    for index, row in enumerate(rows):
        if not int(row["ground_truth_attack"]):
            continue
        phase = row["phase"]
        if phase not in phases:
            phase_order.append(phase)
        phases[phase].append(index)

    output: list[dict[str, object]] = []
    for phase in phase_order:
        indices = phases[phase]
        start_time = float(rows[indices[0]]["simulation_time_s"])
        result: dict[str, object] = {"seed": seed, "phase": phase}
        for method, alarms in alarms_by_method.items():
            first_index = next((index for index in indices if alarms[index]), None)
            result[f"{method}_latency_s"] = (
                "not_detected"
                if first_index is None
                else round(float(rows[first_index]["simulation_time_s"]) - start_time, 3)
            )
        output.append(result)
    return output


def run_seed_simulation(project_root: Path, step08_path: Path, seed: int) -> Path:
    command = [
        sys.executable,
        str(step08_path),
        "--project-root",
        str(project_root),
        "--seed",
        str(seed),
        "--rebuild-scenario",
    ]
    print("\n" + "=" * 76)
    print(f"Running SUMO attack experiment for seed {seed}")
    subprocess.run(command, check=True)
    generated = project_root / "data" / "processed" / "sumo_context_attacks.csv"
    if not generated.exists():
        raise FileNotFoundError(f"Step 08 did not create {generated}")
    return generated


def aggregate_metrics(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    metric_names = ("precision", "recall", "f1", "false_positive_rate", "accuracy")
    for row in rows:
        for metric in metric_names:
            grouped[(str(row["method"]), metric)].append(float(row[metric]))

    output: list[dict[str, object]] = []
    for (method, metric), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "metric": metric,
                "runs": len(values),
                "mean": round(statistics.fmean(values), 6),
                "population_std": round(statistics.pstdev(values), 6),
                "minimum": round(min(values), 6),
                "maximum": round(max(values), 6),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    src_dir = project_root / "src"
    step08_path = src_dir / "08_run_sumo_attack_experiments.py"
    step09_path = src_dir / "09_evaluate_context_aware_zero_trust.py"
    if not step08_path.exists():
        step08_path = project_root / step08_path.name
    if not step09_path.exists():
        step09_path = project_root / step09_path.name
    step09 = load_script(step09_path, "ztav_step09_sensitivity")

    dataset_dir = project_root / "data" / "processed" / "sumo_repeated_seeds"
    results_dir = project_root / "results" / "repeated_seed_validation"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[int, list[dict[str, str]]] = {}
    for seed in args.seeds:
        preserved = dataset_dir / f"sumo_context_attacks_seed_{seed}.csv"
        if args.skip_simulation:
            if not preserved.exists():
                raise FileNotFoundError(
                    f"--skip-simulation requested but {preserved} does not exist"
                )
        else:
            generated = run_seed_simulation(project_root, step08_path, seed)
            shutil.copy2(generated, preserved)
            print(f"Preserved seed {seed} dataset: {preserved}")
        datasets[seed] = read_csv(preserved)

    seed_metrics: list[dict[str, object]] = []
    all_latencies: list[dict[str, object]] = []
    for seed, rows in datasets.items():
        truth = [int(row["ground_truth_attack"]) for row in rows]
        weighted = [int(row["security_alarm"]) for row in rows]
        proposed = source_aware_alarms_complete(
            rows,
            step09,
            source_thresholds=step09.SOURCE_THRESHOLDS,
            attack_persistence=step09.ATTACK_PERSISTENCE,
            recovery_persistence=step09.RECOVERY_PERSISTENCE,
        )
        alarms_by_method = {
            "weighted": weighted,
            "context_aware": proposed,
        }
        all_latencies.extend(latency_rows(seed, rows, alarms_by_method))
        for method, alarms in alarms_by_method.items():
            metrics = calculate_metrics(truth, alarms)
            seed_metrics.append(
                {
                    "seed": seed,
                    "method": method,
                    **{
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in metrics.items()
                    },
                }
            )

    sensitivity_rows: list[dict[str, object]] = []
    for threshold, attack_persistence, recovery_persistence in itertools.product(
        SENSITIVITY_THRESHOLDS,
        SENSITIVITY_ATTACK_PERSISTENCE,
        SENSITIVITY_RECOVERY_PERSISTENCE,
    ):
        per_seed: list[dict[str, float | int]] = []
        thresholds = {
            source: (column, threshold)
            for source, (column, _) in step09.SOURCE_THRESHOLDS.items()
        }
        for rows in datasets.values():
            truth = [int(row["ground_truth_attack"]) for row in rows]
            alarms = source_aware_alarms_complete(
                rows,
                step09,
                source_thresholds=thresholds,
                attack_persistence=attack_persistence,
                recovery_persistence=recovery_persistence,
            )
            per_seed.append(calculate_metrics(truth, alarms))
        sensitivity_rows.append(
            {
                "common_source_threshold": threshold,
                "attack_persistence_rows": attack_persistence,
                "recovery_persistence_rows": recovery_persistence,
                "runs": len(per_seed),
                "precision_mean": round(
                    statistics.fmean(float(item["precision"]) for item in per_seed), 6
                ),
                "recall_mean": round(
                    statistics.fmean(float(item["recall"]) for item in per_seed), 6
                ),
                "f1_mean": round(
                    statistics.fmean(float(item["f1"]) for item in per_seed), 6
                ),
                "f1_population_std": round(
                    statistics.pstdev(float(item["f1"]) for item in per_seed), 6
                ),
                "false_positive_rate_mean": round(
                    statistics.fmean(
                        float(item["false_positive_rate"]) for item in per_seed
                    ),
                    6,
                ),
            }
        )

    aggregate = aggregate_metrics(seed_metrics)
    write_csv(results_dir / "per_seed_metrics.csv", seed_metrics)
    write_csv(results_dir / "aggregate_metrics.csv", aggregate)
    write_csv(results_dir / "per_seed_detection_latencies.csv", all_latencies)
    write_csv(results_dir / "threshold_sensitivity.csv", sensitivity_rows)

    print("\n" + "=" * 76)
    print("Repeated-seed validation completed successfully.")
    for method in ("weighted", "context_aware"):
        f1_row = next(
            row
            for row in aggregate
            if row["method"] == method and row["metric"] == "f1"
        )
        print(
            f"{method:15s} F1 mean={float(f1_row['mean']):.4f}, "
            f"std={float(f1_row['population_std']):.4f}, "
            f"range=[{float(f1_row['minimum']):.4f}, "
            f"{float(f1_row['maximum']):.4f}]"
        )
    print(f"Results directory: {results_dir}")
    print("\nNext: vary attack severity and test subtle/slow attacks.")


if __name__ == "__main__":
    main()
