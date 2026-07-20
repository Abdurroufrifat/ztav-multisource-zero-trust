#!/usr/bin/env python3
"""Integrate a soft startup-quality warning with the multiscale Zero Trust policy.

Step 23 proved that the startup guard is not a reliable attack detector across
captures: it falsely rejected a clean DoS capture and missed Fuzzy poisoning.
This stage therefore does not let the guard declare an attack or directly force
a safety fallback.  Instead, it treats rejection as an enrollment-quality
warning:

* the frozen 100-frame and 20-frame CAN evidence remains active;
* a warning alone enters restricted/local operation and requests re-verification;
* a warning plus independent CAN evidence can trigger a corroborated fallback;
* persistent CAN, identity/device failure, GNSS inconsistency, or physical
  sensor/control inconsistency can still trigger safety action independently.

The script compares this soft policy with the rejected hard-guard policy, the
100-frame policy, CAN-only multiscale detection, and context-only detection.
The Step 21 thresholds and Step 23 guard scores remain frozen.

Run from D:\\ztav_project after Steps 21 and 23:

    .\\.venv\\Scripts\\python.exe src\\24_integrate_soft_guarded_multiscale_policy.py

This is an exploratory hybrid HCRL/SUMO evaluation, not production software.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAN_ATTACK_PHASES = {"can_injection", "combined_attack"}
ATTACK_DENSITY_SCENARIOS = {
    "representative_all": (1, 100),
    "low_1_5": (1, 5),
    "medium_6_20": (6, 20),
    "high_21_100": (21, 100),
}
DENSITY_ORDER = tuple(ATTACK_DENSITY_SCENARIOS)
METHOD_COLUMNS = {
    "legacy_weighted_threshold": "legacy_alarm",
    "hard_guard_multisource": "hard_guard_multisource_alarm",
    "frozen_w100_multisource": "w100_multisource_alarm",
    "multiscale_can_only": "multiscale_can_only_alarm",
    "context_without_can": "context_without_can_alarm",
    "soft_guard_without_vehicle_state": "soft_without_vehicle_state_alarm",
    "proposed_soft_guard_multisource": "soft_guard_multisource_alarm",
}
REQUIRED_PARENT_COLUMNS = {
    "source_file",
    "parent_window_index",
    "micro_attack_frames",
    "w100_binary_target",
    "w100_attack_frame_count",
    "w100_alarm_instant",
    "w100_continuous_can_trust",
    "multiscale_alarm_instant",
    "multiscale_continuous_can_trust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate a soft startup-quality warning with multiscale CAN."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/soft_guarded_multiscale_policy"),
    )
    return parser.parse_args()


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


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_columns(actual: Iterable[str], required: set[str], label: str) -> None:
    missing = required - set(actual)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_guard(path: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"source_file", "guard_score", "guard_threshold", "guard_rejected"}
    validate_columns(frame.columns, required, "Step 23 clean-source audit")
    if frame["source_file"].duplicated().any():
        raise ValueError("Duplicate source rows in Step 23 clean-source audit")
    return {
        str(row.source_file): {
            "score": float(row.guard_score),
            "threshold": float(row.guard_threshold),
            "warning": as_bool(row.guard_rejected),
        }
        for row in frame.itertuples(index=False)
    }


def build_replay(
    sumo_rows: Sequence[dict[str, str]],
    parents: pd.DataFrame,
    source_file: str,
    density: str,
    seed: int,
    step19: ModuleType,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = parents[parents["source_file"] == source_file].copy()
    lower, upper = ATTACK_DENSITY_SCENARIOS[density]
    benign_pool = source[source["w100_binary_target"] == 0]
    attack_pool = source[
        (source["w100_binary_target"] == 1)
        & source["w100_attack_frame_count"].between(lower, upper)
    ]
    if benign_pool.empty or attack_pool.empty:
        raise ValueError(
            f"Empty replay pool for {source_file}, density={density}: "
            f"benign={len(benign_pool)}, attack={len(attack_pool)}"
        )
    attack_count = sum(row["phase"] in CAN_ATTACK_PHASES for row in sumo_rows)
    benign_count = len(sumo_rows) - attack_count
    benign, benign_replacement = step19.sample_frame(
        benign_pool,
        benign_count,
        step19.stable_seed(seed, source_file, density, "multiscale_benign"),
    )
    attack, attack_replacement = step19.sample_frame(
        attack_pool,
        attack_count,
        step19.stable_seed(seed, source_file, density, "multiscale_attack"),
    )
    output: list[dict[str, object]] = []
    benign_i = attack_i = 0
    for sumo_row in sumo_rows:
        if sumo_row["phase"] in CAN_ATTACK_PHASES:
            can_row = attack.iloc[attack_i]
            attack_i += 1
        else:
            can_row = benign.iloc[benign_i]
            benign_i += 1
        row: dict[str, object] = dict(sumo_row)
        row.update(
            {
                "hcrl_source_file": source_file,
                "hcrl_parent_window_index": int(can_row["parent_window_index"]),
                "hcrl_binary_target": int(can_row["w100_binary_target"]),
                "hcrl_attack_frame_count": int(can_row["w100_attack_frame_count"]),
                "w100_alarm_instant_input": as_bool(can_row["w100_alarm_instant"]),
                "w100_continuous_can_trust": float(
                    can_row["w100_continuous_can_trust"]
                ),
                "multiscale_alarm_instant_input": as_bool(
                    can_row["multiscale_alarm_instant"]
                ),
                "multiscale_continuous_can_trust": float(
                    can_row["multiscale_continuous_can_trust"]
                ),
            }
        )
        output.append(row)
    audit = {
        "seed": seed,
        "source_file": source_file,
        "density_scenario": density,
        "density_min_attack_frames": lower,
        "density_max_attack_frames": upper,
        "replay_rows": len(output),
        "benign_pool_windows": len(benign_pool),
        "attack_pool_windows": len(attack_pool),
        "benign_sampling_with_replacement": benign_replacement,
        "attack_sampling_with_replacement": attack_replacement,
        "unique_replayed_benign_windows": int(benign["parent_window_index"].nunique()),
        "unique_replayed_attack_windows": int(attack["parent_window_index"].nunique()),
    }
    return output, audit


def evaluate_replay(
    rows: Sequence[dict[str, object]],
    guard: dict[str, object],
    step19: ModuleType,
) -> list[dict[str, object]]:
    monitors = {
        source: step19.PersistentMonitor(threshold)
        for source, (_, threshold) in step19.SOURCE_THRESHOLDS.items()
    }
    identity_monitor = step19.BooleanCriticalMonitor()
    device_monitor = step19.BooleanCriticalMonitor()
    previous_multiscale = False
    previous_w100 = False
    warning = bool(guard["warning"])
    output: list[dict[str, object]] = []

    for row in rows:
        multiscale_instant = bool(row["multiscale_alarm_instant_input"])
        w100_instant = bool(row["w100_alarm_instant_input"])
        multiscale_persistent = multiscale_instant and previous_multiscale
        w100_persistent = w100_instant and previous_w100
        previous_multiscale = multiscale_instant
        previous_w100 = w100_instant

        noncan: set[str] = set()
        for source, (column, _) in step19.SOURCE_THRESHOLDS.items():
            if monitors[source].update(float(row[column])):
                noncan.add(source)
        identity_failure = bool(int(row["identity_critical_failure"]))
        device_failure = bool(int(row["device_critical_failure"]))
        if identity_monitor.update(identity_failure):
            noncan.add("identity")
        if device_monitor.update(device_failure):
            noncan.add("device_posture")

        independent_critical = bool(
            noncan & {"identity", "device_posture", "gnss", "sensor_control"}
        )
        corroborated_warning = warning and (multiscale_instant or bool(noncan))
        if multiscale_persistent or independent_critical:
            local_action = "SAFE_FALLBACK"
            action_basis = "persistent_or_independent_critical_evidence"
        elif corroborated_warning and multiscale_instant:
            local_action = "SAFE_FALLBACK"
            action_basis = "startup_warning_plus_can_evidence"
        elif multiscale_instant:
            local_action = "VERIFY_RESTRICT"
            action_basis = "single_multiscale_can_evidence"
        elif warning:
            local_action = "ALLOW_LOCAL_RESTRICTED"
            action_basis = "startup_quality_warning_only"
        elif "v2x" in noncan:
            local_action = "ALLOW_LOCAL_ONLY"
            action_basis = "v2x_evidence"
        else:
            local_action = "ALLOW"
            action_basis = "no_active_evidence"

        cooperative_action = (
            "DENY_COOPERATIVE_ACTION"
            if "v2x" in noncan or identity_failure or device_failure
            else "REQUIRE_REVERIFICATION"
            if warning
            else "ALLOW"
        )
        telemetry_action = (
            "DENY"
            if noncan & {"identity", "device_posture"}
            else "RESTRICT"
            if warning or multiscale_instant
            else "ALLOW"
        )

        without_vehicle_state = noncan - {"sensor_control"}
        multiscale_alarm = int(multiscale_instant)
        noncan_alarm = int(bool(noncan))
        output.append(
            {
                "simulation_time_s": float(row["simulation_time_s"]),
                "phase": str(row["phase"]),
                "ground_truth_attack": int(row["ground_truth_attack"]),
                "source_file": str(row["hcrl_source_file"]),
                "parent_window_index": int(row["hcrl_parent_window_index"]),
                "attack_frame_count": int(row["hcrl_attack_frame_count"]),
                "w100_alarm_instant": int(w100_instant),
                "w100_alarm_persistent_2": int(w100_persistent),
                "multiscale_alarm_instant": multiscale_alarm,
                "multiscale_alarm_persistent_2": int(multiscale_persistent),
                "startup_quality_warning": int(warning),
                "startup_guard_score": float(guard["score"]),
                "startup_guard_threshold": float(guard["threshold"]),
                "active_noncan_sources": ";".join(sorted(noncan)),
                "legacy_alarm": int(row["security_alarm"]),
                "hard_guard_multisource_alarm": int(
                    warning or multiscale_instant or bool(noncan)
                ),
                "w100_multisource_alarm": int(w100_instant or bool(noncan)),
                "multiscale_can_only_alarm": multiscale_alarm,
                "context_without_can_alarm": noncan_alarm,
                "soft_without_vehicle_state_alarm": int(
                    multiscale_instant or bool(without_vehicle_state)
                ),
                "soft_guard_multisource_alarm": int(
                    multiscale_instant or bool(noncan)
                ),
                "local_control_action": local_action,
                "local_action_basis": action_basis,
                "cooperative_action": cooperative_action,
                "telemetry_action": telemetry_action,
                "operating_mode": (
                    "BASELINE_UNTRUSTED" if warning else "BASELINE_ACCEPTED"
                ),
            }
        )
    return output


def metric_rows(
    seed: int,
    source_file: str,
    density: str,
    evaluated: Sequence[dict[str, object]],
    step19: ModuleType,
) -> list[dict[str, object]]:
    truth = [int(row["ground_truth_attack"]) for row in evaluated]
    output = []
    for method, column in METHOD_COLUMNS.items():
        prediction = [int(row[column]) for row in evaluated]
        output.append(
            {
                "seed": seed,
                "source_file": source_file,
                "density_scenario": density,
                "method": method,
                **step19.binary_metrics(truth, prediction),
            }
        )
    return output


def phase_rows(
    seed: int,
    source_file: str,
    density: str,
    evaluated: Sequence[dict[str, object]],
    step19: ModuleType,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(evaluated)
    output: list[dict[str, object]] = []
    for phase, group in frame.groupby("phase", sort=False):
        truth = group["ground_truth_attack"].astype(int).tolist()
        for method, column in METHOD_COLUMNS.items():
            prediction = group[column].astype(int).tolist()
            output.append(
                {
                    "seed": seed,
                    "source_file": source_file,
                    "density_scenario": density,
                    "phase": phase,
                    "method": method,
                    **step19.binary_metrics(truth, prediction),
                }
            )
    return output


def aggregate_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = ["precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]
    rows = []
    for (density, method), group in per_run.groupby(
        ["density_scenario", "method"], sort=True
    ):
        row: dict[str, object] = {
            "density_scenario": density,
            "method": method,
            "runs": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(aggregate: pd.DataFrame, output_path: Path) -> None:
    methods = [
        "hard_guard_multisource",
        "frozen_w100_multisource",
        "multiscale_can_only",
        "proposed_soft_guard_multisource",
    ]
    labels = {
        "hard_guard_multisource": "Hard guard",
        "frozen_w100_multisource": "W100 multisource",
        "multiscale_can_only": "Multiscale CAN only",
        "proposed_soft_guard_multisource": "Proposed soft guard",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(DENSITY_ORDER))
    width = 0.19
    for index, method in enumerate(methods):
        group = aggregate[aggregate["method"] == method].set_index(
            "density_scenario"
        ).reindex(DENSITY_ORDER)
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, group["f1_mean"], width, label=labels[method])
        axes[1].bar(
            x + offset,
            group["false_positive_rate_mean"],
            width,
            label=labels[method],
        )
    for axis in axes:
        axis.set_xticks(x, DENSITY_ORDER, rotation=15)
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set(title="End-to-end detection", ylabel="Mean F1")
    axes[1].set(title="Availability cost", ylabel="Mean false-positive rate")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Soft startup-quality warning in the multiscale Zero Trust policy")
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
    step19 = load_script(
        locate_script(project_root, "19_integrate_guarded_can_multisource_policy.py"),
        "ztav_step19_soft_guard",
    )
    parent_path = (
        project_root
        / "results"
        / "multiscale_sparse_can_gate"
        / "multiscale_parent_predictions.csv"
    )
    guard_path = (
        project_root
        / "results"
        / "reference_anchored_startup_guard"
        / "reference_guard_clean_source_audit.csv"
    )
    if not parent_path.exists() or not guard_path.exists():
        raise FileNotFoundError("Step 21 parent predictions or Step 23 guard audit is missing")
    parents = pd.read_csv(parent_path)
    validate_columns(parents.columns, REQUIRED_PARENT_COLUMNS, "Step 21 parents")
    guards = load_guard(guard_path)
    sources = sorted(
        source
        for source in parents["source_file"].unique()
        if source != "normal_run_data.txt"
    )
    missing_guard = set(sources) - set(guards)
    if missing_guard:
        raise ValueError(f"Step 23 guard rows missing for: {sorted(missing_guard)}")

    sumo_dir = project_root / "data" / "processed" / "sumo_repeated_seeds"
    sumo_paths = sorted(sumo_dir.glob("sumo_context_attacks_seed_*.csv"))
    if not sumo_paths:
        raise FileNotFoundError(f"No repeated-seed SUMO files found in {sumo_dir}")

    per_run_rows: list[dict[str, object]] = []
    per_phase_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    total = len(sumo_paths) * len(sources) * len(DENSITY_ORDER)
    run_number = 0
    for sumo_path in sumo_paths:
        seed = step19.seed_from_filename(sumo_path)
        sumo_rows = step19.read_csv_rows(sumo_path)
        step19.validate_columns(sumo_rows[0], step19.REQUIRED_SUMO_COLUMNS, sumo_path.name)
        for source_file in sources:
            for density in DENSITY_ORDER:
                run_number += 1
                print(
                    f"[{run_number}/{total}] seed={seed}, source={source_file}, "
                    f"density={density}"
                )
                replay, audit = build_replay(
                    sumo_rows, parents, source_file, density, seed, step19
                )
                evaluated = evaluate_replay(replay, guards[source_file], step19)
                per_run_rows.extend(
                    metric_rows(seed, source_file, density, evaluated, step19)
                )
                per_phase_rows.extend(
                    phase_rows(seed, source_file, density, evaluated, step19)
                )
                replay_rows.append(
                    {
                        **audit,
                        "startup_guard_score": guards[source_file]["score"],
                        "startup_guard_threshold": guards[source_file]["threshold"],
                        "startup_quality_warning": guards[source_file]["warning"],
                    }
                )
                for row in evaluated:
                    decision_rows.append(
                        {"seed": seed, "density_scenario": density, **row}
                    )
                counts = Counter(row["local_control_action"] for row in evaluated)
                for action, count in sorted(counts.items()):
                    distribution_rows.append(
                        {
                            "seed": seed,
                            "source_file": source_file,
                            "density_scenario": density,
                            "local_control_action": action,
                            "count": count,
                            "percentage": 100.0 * count / len(evaluated),
                        }
                    )

    per_run = pd.DataFrame(per_run_rows)
    per_phase = pd.DataFrame(per_phase_rows)
    aggregate = aggregate_metrics(per_run)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(results_dir / "soft_guard_per_run_metrics.csv", index=False)
    aggregate.to_csv(results_dir / "soft_guard_aggregate_metrics.csv", index=False)
    per_phase.to_csv(results_dir / "soft_guard_per_phase_metrics.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(
        results_dir / "soft_guard_policy_decisions.csv", index=False
    )
    pd.DataFrame(replay_rows).to_csv(
        results_dir / "soft_guard_replay_manifest.csv", index=False
    )
    pd.DataFrame(distribution_rows).to_csv(
        results_dir / "soft_guard_decision_distribution.csv", index=False
    )
    manifest = pd.DataFrame(
        [
            {"item": "experiment", "value": "soft-guarded multiscale hybrid replay"},
            {"item": "startup_guard_role", "value": "quality warning; never an attack alarm alone"},
            {"item": "warning_only_action", "value": "ALLOW_LOCAL_RESTRICTED and re-verification"},
            {"item": "warning_plus_can_action", "value": "corroborated SAFE_FALLBACK"},
            {"item": "can_gate", "value": "frozen Step 21 multiscale"},
            {"item": "replay_runs", "value": total},
            {"item": "hcrl_sources", "value": ";".join(sources)},
            {"item": "sumo_seeds", "value": ";".join(str(step19.seed_from_filename(p)) for p in sumo_paths)},
            {"item": "external_validity", "value": "exploratory HCRL/SUMO hybrid; not independent confirmation"},
        ]
    )
    manifest.to_csv(results_dir / "soft_guard_manifest.csv", index=False)
    plot_results(aggregate, results_dir / "soft_guard_policy_comparison.png")

    print("\n" + "=" * 88)
    print("Soft-guarded multiscale Zero Trust integration completed successfully.")
    warning_sources = [source for source in sources if guards[source]["warning"]]
    print(f"Replay runs: {total}")
    print(
        f"Startup-quality warnings: {len(warning_sources)}/{len(sources)} "
        f"({'; '.join(warning_sources) if warning_sources else 'none'})"
    )
    print("\nMean end-to-end metrics by density:")
    chosen = aggregate[
        aggregate["method"].isin(
            ["hard_guard_multisource", "proposed_soft_guard_multisource"]
        )
    ]
    print(
        chosen[
            [
                "density_scenario",
                "method",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "false_positive_rate_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults directory: {results_dir}")
    print("\nNext: use availability and sparse-recall results to freeze or reject the soft policy.")


if __name__ == "__main__":
    main()
