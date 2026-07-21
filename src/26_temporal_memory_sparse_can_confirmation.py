#!/usr/bin/env python3
"""Select and confirm temporal memory for sparse multiscale CAN evidence.

Step 25 achieved an enforcement FPR below 5%, but the strict two-consecutive
parent-window rule detected only about 30% of low-density CAN-injection rows.
This stage evaluates a small, prespecified family of rolling-hit/hold rules.

To reduce selection leakage, replay seeds are split chronologically:

* first three seeds: development and rule selection;
* final two seeds: untouched confirmation.

Only rules meeting both a 5% overall enforcement FPR and a 5% macro
healthy/recovery FPR on development are eligible.  Among eligible rules, the
one with the highest low-density CAN-injection recall is frozen.  Confirmation
metrics are then computed without changing the selected rule or thresholds.

Vehicle-state context remains excluded from every primary policy.  Startup
quality warnings never become attack alarms.  This script reuses Step 24 rows
and does not rebuild HCRL or SUMO data.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_FPR = 0.05
LOW_DENSITY = "low_1_5"
DENSITY_ORDER = (
    "representative_all",
    "low_1_5",
    "medium_6_20",
    "high_21_100",
)
HEALTHY_PHASES = {
    "healthy_baseline",
    "recovery_after_gps",
    "recovery_after_can",
    "recovery_after_v2x",
    "recovery_after_identity",
    "final_recovery",
}
RULES = {
    "strict_consecutive_2": (2, 2, 0),
    "two_of_3_no_hold": (2, 3, 0),
    "two_of_3_hold_2": (2, 3, 2),
    "two_of_3_hold_3": (2, 3, 3),
    "two_of_4_hold_2": (2, 4, 2),
    "three_of_5_hold_2": (3, 5, 2),
    "instant_advisory_reference": (1, 1, 0),
}
REQUIRED_COLUMNS = {
    "seed",
    "source_file",
    "density_scenario",
    "simulation_time_s",
    "phase",
    "ground_truth_attack",
    "multiscale_alarm_instant",
    "active_noncan_sources",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and confirm sparse-CAN temporal memory."
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
        default=Path("results/temporal_memory_sparse_can_confirmation"),
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


def temporal_memory(
    values: Sequence[int | bool],
    required_hits: int,
    history_windows: int,
    hold_clean_windows: int,
) -> np.ndarray:
    history: deque[bool] = deque(maxlen=history_windows)
    active = False
    clean_streak = 0
    output: list[bool] = []
    for raw in values:
        current = bool(raw)
        history.append(current)
        triggered = sum(history) >= required_hits
        if hold_clean_windows == 0:
            active = triggered
        else:
            if triggered:
                active = True
                clean_streak = 0
            elif active:
                if current:
                    clean_streak = 0
                else:
                    clean_streak += 1
                    if clean_streak >= hold_clean_windows:
                        active = False
                        clean_streak = 0
        output.append(active)
    return np.asarray(output, dtype=bool)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    truth: Sequence[int], prediction: Sequence[int]
) -> dict[str, float | int]:
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
    }


def build_rule_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output_parts = []
    keys = ["seed", "source_file", "density_scenario"]
    for _, group in frame.groupby(keys, sort=True):
        group = group.sort_values("simulation_time_s").copy()
        noncan_no_state = group["active_noncan_sources"].map(split_sources).map(
            lambda values: bool(values - {"sensor_control"})
        ).to_numpy(dtype=bool)
        instant = group["multiscale_alarm_instant"].to_numpy(dtype=bool)
        for rule, (hits, history, hold) in RULES.items():
            can_alarm = temporal_memory(instant, hits, history, hold)
            group[f"can__{rule}"] = can_alarm.astype(int)
            group[f"policy__{rule}"] = (can_alarm | noncan_no_state).astype(int)
        output_parts.append(group)
    return pd.concat(output_parts, ignore_index=True)


def metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    phase_rows = []
    run_keys = ["seed", "source_file", "density_scenario"]
    for key, group in frame.groupby(run_keys, sort=True):
        seed, source, density = key
        truth = group["ground_truth_attack"].astype(int).tolist()
        for rule in RULES:
            run_rows.append(
                {
                    "seed": seed,
                    "source_file": source,
                    "density_scenario": density,
                    "rule": rule,
                    **binary_metrics(
                        truth, group[f"policy__{rule}"].astype(int).tolist()
                    ),
                }
            )
        for phase, phase_group in group.groupby("phase", sort=False):
            phase_truth = phase_group["ground_truth_attack"].astype(int).tolist()
            for rule in RULES:
                phase_rows.append(
                    {
                        "seed": seed,
                        "source_file": source,
                        "density_scenario": density,
                        "phase": phase,
                        "rule": rule,
                        **binary_metrics(
                            phase_truth,
                            phase_group[f"policy__{rule}"].astype(int).tolist(),
                        ),
                        "can_alarm_rate": float(
                            phase_group[f"can__{rule}"].mean()
                        ),
                    }
                )
    return pd.DataFrame(run_rows), pd.DataFrame(phase_rows)


def candidate_summary(
    per_run: pd.DataFrame,
    per_phase: pd.DataFrame,
    seeds: set[int],
    partition: str,
) -> pd.DataFrame:
    runs = per_run[per_run["seed"].isin(seeds)]
    phases = per_phase[per_phase["seed"].isin(seeds)]
    rows = []
    for rule, group in runs.groupby("rule", sort=True):
        low_can = phases[
            (phases["rule"] == rule)
            & (phases["density_scenario"] == LOW_DENSITY)
            & (phases["phase"] == "can_injection")
        ]
        healthy = phases[
            (phases["rule"] == rule) & phases["phase"].isin(HEALTHY_PHASES)
        ]
        rows.append(
            {
                "partition": partition,
                "rule": rule,
                "seeds": ";".join(map(str, sorted(seeds))),
                "runs": len(group),
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "f1_mean": float(group["f1"].mean()),
                "overall_fpr_mean": float(group["false_positive_rate"].mean()),
                "overall_fpr_max": float(group["false_positive_rate"].max()),
                "low_can_injection_recall_mean": float(low_can["recall"].mean()),
                "low_can_injection_recall_min": float(low_can["recall"].min()),
                "healthy_recovery_fpr_macro": float(
                    healthy["false_positive_rate"].mean()
                ),
                "healthy_recovery_fpr_max": float(
                    healthy["false_positive_rate"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def select_rule(development: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    ranked = development.copy()
    ranked["eligible_overall_fpr"] = ranked["overall_fpr_mean"] <= TARGET_FPR
    ranked["eligible_healthy_recovery_fpr"] = (
        ranked["healthy_recovery_fpr_macro"] <= TARGET_FPR
    )
    ranked["eligible"] = (
        ranked["eligible_overall_fpr"]
        & ranked["eligible_healthy_recovery_fpr"]
        & (ranked["rule"] != "instant_advisory_reference")
    )
    eligible = ranked[ranked["eligible"]].copy()
    if eligible.empty:
        # Preserve a result rather than silently relaxing the constraint.
        selected = "strict_consecutive_2"
    else:
        eligible = eligible.sort_values(
            ["low_can_injection_recall_mean", "f1_mean", "overall_fpr_mean"],
            ascending=[False, False, True],
        )
        selected = str(eligible.iloc[0]["rule"])
    ranked["selected_on_development"] = ranked["rule"] == selected
    return selected, ranked


def aggregate_confirmation(
    per_run: pd.DataFrame,
    per_phase: pd.DataFrame,
    confirmation_seeds: set[int],
    selected: str,
) -> pd.DataFrame:
    runs = per_run[
        per_run["seed"].isin(confirmation_seeds) & (per_run["rule"] == selected)
    ]
    phases = per_phase[
        per_phase["seed"].isin(confirmation_seeds)
        & (per_phase["rule"] == selected)
    ]
    rows = []
    for density, group in runs.groupby("density_scenario", sort=True):
        can = phases[
            (phases["density_scenario"] == density)
            & (phases["phase"] == "can_injection")
        ]
        healthy = phases[
            (phases["density_scenario"] == density)
            & phases["phase"].isin(HEALTHY_PHASES)
        ]
        rows.append(
            {
                "partition": "confirmation",
                "selected_rule": selected,
                "density_scenario": density,
                "seeds": ";".join(map(str, sorted(confirmation_seeds))),
                "runs": len(group),
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "f1_mean": float(group["f1"].mean()),
                "overall_fpr_mean": float(group["false_positive_rate"].mean()),
                "overall_fpr_max": float(group["false_positive_rate"].max()),
                "can_injection_recall_mean": float(can["recall"].mean()),
                "can_injection_recall_min": float(can["recall"].min()),
                "healthy_recovery_fpr_macro": float(
                    healthy["false_positive_rate"].mean()
                ),
                "healthy_recovery_fpr_max": float(
                    healthy["false_positive_rate"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_results(
    development: pd.DataFrame,
    confirmation_all: pd.DataFrame,
    primary: pd.DataFrame,
    selected: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    colors = [
        "tab:red" if rule == selected else "tab:blue"
        for rule in development["rule"]
    ]
    axes[0].scatter(
        development["overall_fpr_mean"],
        development["low_can_injection_recall_mean"],
        c=colors,
        s=70,
    )
    for row in development.itertuples(index=False):
        axes[0].annotate(
            row.rule,
            (row.overall_fpr_mean, row.low_can_injection_recall_mean),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[0].axvline(TARGET_FPR, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        title="Development selection",
        xlabel="Mean enforcement FPR",
        ylabel="Low-density CAN recall",
    )

    ordered = confirmation_all.sort_values("low_can_injection_recall_mean")
    axes[1].barh(
        ordered["rule"],
        ordered["low_can_injection_recall_mean"],
        color=["tab:red" if rule == selected else "tab:gray" for rule in ordered["rule"]],
    )
    axes[1].set(title="Confirmation rule sensitivity", xlabel="Low-density CAN recall")
    axes[1].set_xlim(0.0, 1.02)

    primary = primary.set_index("density_scenario").reindex(DENSITY_ORDER)
    x = np.arange(len(DENSITY_ORDER))
    axes[2].bar(x - 0.18, primary["overall_fpr_mean"], 0.36, label="FPR")
    axes[2].bar(
        x + 0.18,
        primary["can_injection_recall_mean"],
        0.36,
        label="CAN recall",
    )
    axes[2].set_xticks(x, DENSITY_ORDER, rotation=18)
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set(title=f"Frozen confirmation: {selected}")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Sparse-CAN temporal-memory selection and confirmation")
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
    seeds = sorted(int(seed) for seed in raw["seed"].unique())
    if len(seeds) < 5:
        raise ValueError(
            f"Five replay seeds are required for a 3/2 split; found {seeds}"
        )
    development_seeds = set(seeds[:3])
    confirmation_seeds = set(seeds[-2:])
    if development_seeds & confirmation_seeds:
        raise RuntimeError("Development and confirmation seeds overlap")

    predictions = build_rule_predictions(raw)
    per_run, per_phase = metric_tables(predictions)
    development = candidate_summary(
        per_run, per_phase, development_seeds, "development"
    )
    confirmation_all = candidate_summary(
        per_run, per_phase, confirmation_seeds, "confirmation"
    )
    selected, development_ranked = select_rule(development)
    primary = aggregate_confirmation(
        per_run, per_phase, confirmation_seeds, selected
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    development_ranked.to_csv(
        results_dir / "temporal_candidate_development_summary.csv", index=False
    )
    confirmation_all.to_csv(
        results_dir / "temporal_candidate_confirmation_summary.csv", index=False
    )
    primary.to_csv(
        results_dir / "temporal_confirmation_primary_endpoint.csv", index=False
    )
    per_run.to_csv(results_dir / "temporal_all_per_run_metrics.csv", index=False)
    per_phase.to_csv(results_dir / "temporal_all_per_phase_metrics.csv", index=False)
    selected_parameters = RULES[selected]
    selection = pd.DataFrame(
        [
            {
                "selected_rule": selected,
                "required_hits": selected_parameters[0],
                "history_windows": selected_parameters[1],
                "hold_clean_windows": selected_parameters[2],
                "development_seeds": ";".join(map(str, sorted(development_seeds))),
                "confirmation_seeds": ";".join(map(str, sorted(confirmation_seeds))),
                "overall_fpr_constraint": TARGET_FPR,
                "healthy_recovery_fpr_constraint": TARGET_FPR,
                "selection_objective": "maximize low-density CAN-injection recall, then F1",
            }
        ]
    )
    selection.to_csv(results_dir / "temporal_selected_rule.csv", index=False)
    manifest = pd.DataFrame(
        [
            {"item": "experiment", "value": "temporal-memory sparse-CAN confirmation"},
            {"item": "development_seeds", "value": ";".join(map(str, sorted(development_seeds)))},
            {"item": "confirmation_seeds", "value": ";".join(map(str, sorted(confirmation_seeds)))},
            {"item": "selected_rule", "value": selected},
            {"item": "vehicle_state", "value": "excluded"},
            {"item": "startup_guard", "value": "quality signal only"},
            {"item": "threshold_status", "value": "frozen"},
            {"item": "external_validity", "value": "internal seed confirmation; independent data still required"},
        ]
    )
    manifest.to_csv(results_dir / "temporal_manifest.csv", index=False)
    plot_results(
        development_ranked,
        confirmation_all,
        primary,
        selected,
        results_dir / "temporal_memory_confirmation.png",
    )

    print("\n" + "=" * 96)
    print("Temporal-memory sparse-CAN selection and confirmation completed successfully.")
    print(f"Development seeds: {sorted(development_seeds)}")
    print(f"Confirmation seeds: {sorted(confirmation_seeds)}")
    print(f"Selected rule: {selected} {RULES[selected]}")
    print("\nDevelopment candidates:")
    print(
        development_ranked[
            [
                "rule",
                "f1_mean",
                "overall_fpr_mean",
                "healthy_recovery_fpr_macro",
                "low_can_injection_recall_mean",
                "eligible",
                "selected_on_development",
            ]
        ].to_string(index=False)
    )
    print("\nFrozen confirmation endpoint:")
    print(
        primary[
            [
                "density_scenario",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "overall_fpr_mean",
                "can_injection_recall_mean",
                "healthy_recovery_fpr_macro",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults directory: {results_dir}")
    print("\nNext: freeze only if confirmation preserves FPR and materially improves sparse recall.")


if __name__ == "__main__":
    main()
