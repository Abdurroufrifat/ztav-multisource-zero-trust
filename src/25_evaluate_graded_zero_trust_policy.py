#!/usr/bin/env python3
"""Evaluate graded advisory and enforcement operating points for the policy.

Step 24 counted every instantaneous 20-frame anomaly as a binary attack alarm.
That contradicted the Step 21 decision to accept only the two-hit persistent
multiscale gate for operational enforcement, and produced an 11--12% FPR.
It also showed that simulated vehicle-state context dominates CAN-injection
recall, although Step 20 found that this signal does not survive a physical
counterfactual.

This stage reuses the frozen Step 24 replay rows and reports two separate Zero
Trust operating points:

* advisory: an instantaneous anomaly can request verification/monitoring;
* enforcement: a persistent anomaly or independent context can declare an
  attack and trigger a safety response.

The vehicle-state-free enforcement policy is the primary endpoint.  The
vehicle-state-inclusive policy is reported only as an ablation.  Startup guard
rejection remains a quality warning and never becomes an attack label alone.

Run from D:\\ztav_project after Step 24:

    .\\.venv\\Scripts\\python.exe src\\25_evaluate_graded_zero_trust_policy.py

No raw HCRL or SUMO data is rebuilt.  This is exploratory research software.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DENSITY_ORDER = (
    "representative_all",
    "low_1_5",
    "medium_6_20",
    "high_21_100",
)
METHOD_COLUMNS = {
    "legacy_weighted_threshold": "legacy_alarm",
    "hard_startup_guard": "hard_guard_alarm",
    "context_only_without_vehicle_state": "context_only_no_state_alarm",
    "frozen_w100_persistent_without_vehicle_state": "w100_persistent_no_state_alarm",
    "advisory_instant_without_vehicle_state": "advisory_instant_no_state_alarm",
    "proposed_persistent_without_vehicle_state": "persistent_no_state_alarm",
    "persistent_with_vehicle_state_ablation": "persistent_with_state_alarm",
}
PRIMARY_METHOD = "proposed_persistent_without_vehicle_state"
REQUIRED_COLUMNS = {
    "seed",
    "density_scenario",
    "simulation_time_s",
    "phase",
    "ground_truth_attack",
    "source_file",
    "multiscale_alarm_instant",
    "multiscale_alarm_persistent_2",
    "w100_alarm_persistent_2",
    "startup_quality_warning",
    "active_noncan_sources",
    "legacy_alarm",
}
CRITICAL_LOCAL_SOURCES = {"identity", "device_posture", "gnss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate graded Zero Trust advisory/enforcement policy."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/soft_guarded_multiscale_policy/soft_guard_policy_decisions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/graded_zero_trust_policy"),
    )
    return parser.parse_args()


def resolve(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def validate_columns(actual: Iterable[str], required: set[str]) -> None:
    missing = required - set(actual)
    if missing:
        raise ValueError(f"Step 24 decisions are missing: {sorted(missing)}")


def split_sources(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item for item in str(value).split(";") if item}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    truth: Sequence[int], prediction: Sequence[int]
) -> dict[str, float | int]:
    if len(truth) != len(prediction):
        raise ValueError("Truth and prediction lengths differ")
    tp = fp = tn = fn = 0
    for expected, predicted in zip(truth, prediction):
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
        "false_negative_rate": safe_divide(fn, fn + tp),
        "accuracy": safe_divide(tp + tn, len(truth)),
    }


def add_policy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    noncan_sets = output["active_noncan_sources"].map(split_sources)
    without_state = noncan_sets.map(lambda values: values - {"sensor_control"})
    context_no_state = without_state.map(bool)
    context_with_state = noncan_sets.map(bool)
    critical_no_state = without_state.map(
        lambda values: bool(values & CRITICAL_LOCAL_SOURCES)
    )
    v2x_active = without_state.map(lambda values: "v2x" in values)

    instant = output["multiscale_alarm_instant"].astype(bool)
    persistent = output["multiscale_alarm_persistent_2"].astype(bool)
    w100_persistent = output["w100_alarm_persistent_2"].astype(bool)
    warning = output["startup_quality_warning"].astype(bool)

    output["hard_guard_alarm"] = (
        warning | persistent | context_no_state
    ).astype(int)
    output["context_only_no_state_alarm"] = context_no_state.astype(int)
    output["w100_persistent_no_state_alarm"] = (
        w100_persistent | context_no_state
    ).astype(int)
    output["advisory_instant_no_state_alarm"] = (
        instant | context_no_state
    ).astype(int)
    output["persistent_no_state_alarm"] = (
        persistent | context_no_state
    ).astype(int)
    output["persistent_with_state_alarm"] = (
        persistent | context_with_state
    ).astype(int)

    local_actions = []
    action_bases = []
    cooperative_actions = []
    telemetry_actions = []
    for index in range(len(output)):
        if persistent.iloc[index] or critical_no_state.iloc[index]:
            local = "SAFE_FALLBACK"
            basis = "persistent_can_or_independent_critical_context"
        elif instant.iloc[index]:
            local = "MONITOR_VERIFY"
            basis = "instant_can_advisory"
        elif warning.iloc[index]:
            local = "DEGRADED_MONITORING"
            basis = "startup_quality_warning_only"
        elif v2x_active.iloc[index]:
            local = "ALLOW_LOCAL_ONLY"
            basis = "v2x_context_only"
        else:
            local = "ALLOW"
            basis = "no_active_evidence"
        local_actions.append(local)
        action_bases.append(basis)
        cooperative_actions.append(
            "DENY_COOPERATIVE_ACTION"
            if v2x_active.iloc[index]
            else "REQUIRE_REVERIFICATION"
            if warning.iloc[index]
            else "ALLOW"
        )
        telemetry_actions.append(
            "DENY"
            if bool(without_state.iloc[index] & {"identity", "device_posture"})
            else "RESTRICT"
            if instant.iloc[index] or warning.iloc[index]
            else "ALLOW"
        )
    output["graded_local_control_action"] = local_actions
    output["graded_action_basis"] = action_bases
    output["graded_cooperative_action"] = cooperative_actions
    output["graded_telemetry_action"] = telemetry_actions
    return output


def run_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seed", "source_file", "density_scenario"]
    for key, group in frame.groupby(keys, sort=True):
        seed, source, density = key
        truth = group["ground_truth_attack"].astype(int).tolist()
        for method, column in METHOD_COLUMNS.items():
            rows.append(
                {
                    "seed": seed,
                    "source_file": source,
                    "density_scenario": density,
                    "method": method,
                    **binary_metrics(truth, group[column].astype(int).tolist()),
                }
            )
    return pd.DataFrame(rows)


def phase_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seed", "source_file", "density_scenario", "phase"]
    for key, group in frame.groupby(keys, sort=True):
        seed, source, density, phase = key
        truth = group["ground_truth_attack"].astype(int).tolist()
        for method, column in METHOD_COLUMNS.items():
            rows.append(
                {
                    "seed": seed,
                    "source_file": source,
                    "density_scenario": density,
                    "phase": phase,
                    "method": method,
                    **binary_metrics(truth, group[column].astype(int).tolist()),
                }
            )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = [
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
    ]
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(keys, values))
        row["runs"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def action_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seed", "source_file", "density_scenario"]
    for key, group in frame.groupby(keys, sort=True):
        seed, source, density = key
        counts = Counter(group["graded_local_control_action"])
        for action, count in sorted(counts.items()):
            rows.append(
                {
                    "seed": seed,
                    "source_file": source,
                    "density_scenario": density,
                    "local_control_action": action,
                    "count": count,
                    "percentage": 100.0 * count / len(group),
                }
            )
    return pd.DataFrame(rows)


def primary_summary(
    run_aggregate: pd.DataFrame,
    phase_aggregate: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    primary = run_aggregate[run_aggregate["method"] == PRIMARY_METHOD].copy()
    can = phase_aggregate[
        (phase_aggregate["method"] == PRIMARY_METHOD)
        & (phase_aggregate["phase"] == "can_injection")
    ][["density_scenario", "recall_mean"]].rename(
        columns={"recall_mean": "can_injection_recall_mean"}
    )
    healthy_phases = {
        "healthy_baseline",
        "recovery_after_gps",
        "recovery_after_can",
        "recovery_after_v2x",
        "recovery_after_identity",
        "final_recovery",
    }
    healthy = phase_aggregate[
        (phase_aggregate["method"] == PRIMARY_METHOD)
        & phase_aggregate["phase"].isin(healthy_phases)
    ].groupby("density_scenario", as_index=False)["false_positive_rate_mean"].mean()
    healthy = healthy.rename(
        columns={"false_positive_rate_mean": "healthy_recovery_fpr_macro"}
    )
    output = primary.merge(can, on="density_scenario", how="left").merge(
        healthy, on="density_scenario", how="left"
    )
    output["startup_reverification_sessions"] = int(
        sessions["startup_quality_warning"].sum()
    )
    output["total_replay_sessions"] = len(sessions)
    return output


def plot_results(
    aggregate_frame: pd.DataFrame,
    phase_aggregate: pd.DataFrame,
    output_path: Path,
) -> None:
    methods = [
        "advisory_instant_without_vehicle_state",
        "proposed_persistent_without_vehicle_state",
        "persistent_with_vehicle_state_ablation",
        "frozen_w100_persistent_without_vehicle_state",
    ]
    labels = {
        "advisory_instant_without_vehicle_state": "Instant advisory",
        "proposed_persistent_without_vehicle_state": "Persistent primary",
        "persistent_with_vehicle_state_ablation": "Persistent + vehicle state",
        "frozen_w100_persistent_without_vehicle_state": "W100 persistent",
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    x = np.arange(len(DENSITY_ORDER))
    width = 0.19
    for index, method in enumerate(methods):
        group = aggregate_frame[aggregate_frame["method"] == method].set_index(
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
        can = phase_aggregate[
            (phase_aggregate["method"] == method)
            & (phase_aggregate["phase"] == "can_injection")
        ].set_index("density_scenario").reindex(DENSITY_ORDER)
        axes[2].bar(x + offset, can["recall_mean"], width, label=labels[method])
    for axis in axes:
        axis.set_xticks(x, DENSITY_ORDER, rotation=18)
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set(title="End-to-end enforcement", ylabel="Mean F1")
    axes[1].set(title="False-alarm cost", ylabel="Mean FPR")
    axes[2].set(title="CAN-injection phase", ylabel="Mean recall")
    axes[0].legend(fontsize=8)
    figure.suptitle("Graded Zero Trust operating-point confirmation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_path = resolve(project_root, args.input)
    results_dir = resolve(project_root, args.results_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Step 24 decision file is missing: {input_path}")
    raw = pd.read_csv(input_path)
    validate_columns(raw.columns, REQUIRED_COLUMNS)
    graded = add_policy_columns(raw)

    per_run = run_metrics(graded)
    per_phase = phase_metrics(graded)
    run_aggregate = aggregate(per_run, ["density_scenario", "method"])
    phase_aggregate = aggregate(
        per_phase, ["density_scenario", "phase", "method"]
    )
    sessions = graded[
        ["seed", "source_file", "density_scenario", "startup_quality_warning"]
    ].drop_duplicates()
    primary = primary_summary(run_aggregate, phase_aggregate, sessions)
    actions = action_distribution(graded)

    results_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(results_dir / "graded_per_run_metrics.csv", index=False)
    per_phase.to_csv(results_dir / "graded_per_phase_metrics.csv", index=False)
    run_aggregate.to_csv(results_dir / "graded_aggregate_metrics.csv", index=False)
    phase_aggregate.to_csv(
        results_dir / "graded_phase_aggregate_metrics.csv", index=False
    )
    primary.to_csv(results_dir / "graded_primary_endpoint_summary.csv", index=False)
    actions.to_csv(results_dir / "graded_action_distribution.csv", index=False)
    sessions.to_csv(results_dir / "graded_startup_session_actions.csv", index=False)
    graded.to_csv(results_dir / "graded_policy_decisions.csv", index=False)
    manifest = pd.DataFrame(
        [
            {"item": "primary_endpoint", "value": PRIMARY_METHOD},
            {"item": "advisory_rule", "value": "instant multiscale anomaly requests verification"},
            {"item": "enforcement_rule", "value": "persistent multiscale or independent context"},
            {"item": "vehicle_state_role", "value": "ablation only; excluded from primary"},
            {"item": "startup_guard_role", "value": "quality warning and one re-verification request"},
            {"item": "threshold_status", "value": "all Step 21 and Step 23 thresholds frozen"},
            {"item": "external_validity", "value": "exploratory replay, not independent confirmation"},
        ]
    )
    manifest.to_csv(results_dir / "graded_manifest.csv", index=False)
    plot_results(
        run_aggregate,
        phase_aggregate,
        results_dir / "graded_policy_operating_points.png",
    )

    columns = [
        "density_scenario",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "false_positive_rate_mean",
        "can_injection_recall_mean",
        "healthy_recovery_fpr_macro",
    ]
    print("\n" + "=" * 92)
    print("Graded Zero Trust policy evaluation completed successfully.")
    print(f"Replay decision rows: {len(graded):,}")
    print(
        f"Startup re-verification sessions: "
        f"{int(sessions['startup_quality_warning'].sum())}/{len(sessions)}"
    )
    print("\nPrimary vehicle-state-free persistent policy:")
    print(primary[columns].to_string(index=False))
    print(f"\nResults directory: {results_dir}")
    print("\nNext: freeze the graded policy only if enforcement FPR and sparse-CAN recall are acceptable.")


if __name__ == "__main__":
    main()
