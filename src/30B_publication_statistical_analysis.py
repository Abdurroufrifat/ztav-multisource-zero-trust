"""
This script is additive and read-only with respect to existing artifacts. It
uses independent replay units (seed x external source x density), not windows,
for confidence intervals and paired comparisons. It never trains a model or
changes a threshold.

"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT_ROOT = RESULTS / "publication_statistical_analysis"
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUT = AUDIT_ROOT / RUN_ID

RANDOM_SEED = 314159
BOOTSTRAP_REPLICATES = 20_000
PERMUTATION_REPLICATES = 100_000
ALPHA = 0.05

PROPOSED_RUN_LEVEL_METHOD = "proposed_without_vehicle_state"
FINAL_GRADED_METHOD = "proposed_persistent_without_vehicle_state"
BASELINES = [
    "guarded_can_only",
    "context_without_can",
    "legacy_weighted_threshold",
    "proposed_multisource",
]
PRIMARY_SINGLE_SOURCE_BASELINES = {"guarded_can_only", "context_without_can"}
METRICS = ["f1", "recall", "false_positive_rate"]
DENSITY_ORDER = ["representative_all", "low_1_5", "medium_6_20", "high_21_100"]


INPUT_SPECS = {
    "run_level": {
        "preferred": "guarded_multisource_zero_trust_w100/integrated_per_run_metrics.csv",
        "filename": "integrated_per_run_metrics.csv",
    },
    "replay_audit": {
        "preferred": "guarded_multisource_zero_trust_w100/integrated_replay_audit.csv",
        "filename": "integrated_replay_audit.csv",
    },
    "graded_summary": {
        "preferred": "graded_zero_trust_policy/graded_primary_endpoint_summary.csv",
        "filename": "graded_primary_endpoint_summary.csv",
    },
    "temporal_confirmation": {
        "preferred": "temporal_memory_sparse_can_confirmation/temporal_candidate_confirmation_summary.csv",
        "filename": "temporal_candidate_confirmation_summary.csv",
    },
    "temporal_selected_rule": {
        "preferred": "temporal_memory_sparse_can_confirmation/temporal_selected_rule.csv",
        "filename": "temporal_selected_rule.csv",
    },
    "multiscale_density": {
        "preferred": "multiscale_sparse_can_gate/multiscale_density_recall.csv",
        "filename": "multiscale_density_recall.csv",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(value: Any) -> float:
    return float(value)


def locate(key: str) -> Path:
    spec = INPUT_SPECS[key]
    preferred = RESULTS / spec["preferred"]
    if preferred.exists():
        return preferred
    matches = sorted(
        path
        for path in RESULTS.rglob(spec["filename"])
        if AUDIT_ROOT not in path.parents
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Missing required input {spec['filename']!r}; expected {preferred}"
        )
    raise RuntimeError(
        f"Ambiguous input {spec['filename']!r}; found: "
        + ", ".join(str(path) for path in matches)
    )


def unit_key(row: dict[str, str]) -> tuple[str, str, str]:
    source = row.get("hcrl_source_file") or row.get("source_file") or "unknown_source"
    return row["seed"], source, row["density_scenario"]


def percentile_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return math.nan, math.nan
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    return float(low), float(high)


def sign_flip_pvalue(differences: np.ndarray, rng: np.random.Generator) -> float:
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return math.nan
    observed = abs(float(np.mean(differences)))
    if observed == 0 and np.all(differences == 0):
        return 1.0
    extreme = 0
    completed = 0
    batch = 10_000
    while completed < PERMUTATION_REPLICATES:
        size = min(batch, PERMUTATION_REPLICATES - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
        completed += size
    return (extreme + 1.0) / (PERMUTATION_REPLICATES + 1.0)


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["density_scenario"], row["metric"])].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: float(row["p_value_two_sided"]))
        running = 0.0
        m = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (m - index) * float(row["p_value_two_sided"]))
            running = max(running, adjusted)
            row["p_value_holm"] = running
            row["statistically_significant_holm"] = running < ALPHA


def summary_t_critical(n: int) -> tuple[float, str]:
    try:
        from scipy.stats import t
        return float(t.ppf(1 - ALPHA / 2, df=n - 1)), "summary_t_interval"
    except Exception:
        return 1.96, "summary_normal_approximation"


def descriptive_row(
    partition: str,
    density: str,
    method: str,
    metric: str,
    values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    low, high = percentile_ci(values, rng)
    return {
        "analysis_partition": partition,
        "density_scenario": density,
        "method": method,
        "metric": metric,
        "n_independent_units": len(values),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "ci95_low": low,
        "ci95_high": high,
        "ci_method": f"clustered_percentile_bootstrap_{BOOTSTRAP_REPLICATES}",
    }


def build_run_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    required_methods = {PROPOSED_RUN_LEVEL_METHOD, *BASELINES}
    available_methods = {row["method"] for row in rows}
    missing = sorted(required_methods - available_methods)
    if missing:
        raise RuntimeError(f"Missing declared run-level methods: {missing}")

    output = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row["method"] not in required_methods:
            continue
        seed, source, density = unit_key(row)
        key = seed, source, density, row["method"]
        if key in seen:
            raise RuntimeError(f"Duplicate run-level method record: {key}")
        seen.add(key)
        output.append(
            {
                "analysis_partition": "development_multisource_replay",
                "unit_id": f"seed={seed}|source={source}|density={density}",
                "seed": seed,
                "source_file": source,
                "density_scenario": density,
                "method": row["method"],
                "true_positive": row["true_positive"],
                "false_positive": row["false_positive"],
                "true_negative": row["true_negative"],
                "false_negative": row["false_negative"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "false_positive_rate": row["false_positive_rate"],
                "false_negative_rate": row["false_negative_rate"],
                "accuracy": row["accuracy"],
            }
        )
    return output


def paired_comparisons(
    run_rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    index = {
        (row["unit_id"], row["method"]): row
        for row in run_rows
    }
    comparisons: list[dict[str, Any]] = []
    for density in DENSITY_ORDER:
        proposed_units = sorted(
            row["unit_id"]
            for row in run_rows
            if row["density_scenario"] == density
            and row["method"] == PROPOSED_RUN_LEVEL_METHOD
        )
        if len(proposed_units) < 2:
            raise RuntimeError(f"Insufficient paired units for {density}: {len(proposed_units)}")
        for metric in METRICS:
            for baseline in BASELINES:
                missing_pairs = [
                    unit for unit in proposed_units if (unit, baseline) not in index
                ]
                if missing_pairs:
                    raise RuntimeError(
                        f"Unpaired records for density={density}, baseline={baseline}: {missing_pairs[:3]}"
                    )
                proposed = np.array(
                    [float(index[(unit, PROPOSED_RUN_LEVEL_METHOD)][metric]) for unit in proposed_units]
                )
                reference = np.array(
                    [float(index[(unit, baseline)][metric]) for unit in proposed_units]
                )
                differences = proposed - reference
                ci_low, ci_high = percentile_ci(differences, rng)
                sd_diff = float(np.std(differences, ddof=1))
                raw_effect = float(np.mean(differences))
                benefit_effect = -raw_effect if metric == "false_positive_rate" else raw_effect
                standardized = (
                    benefit_effect / sd_diff if sd_diff > 0 else (math.inf if benefit_effect > 0 else 0.0)
                )
                comparisons.append(
                    {
                        "analysis_partition": "development_multisource_replay",
                        "density_scenario": density,
                        "metric": metric,
                        "proposed_method": PROPOSED_RUN_LEVEL_METHOD,
                        "baseline_method": baseline,
                        "comparison_role": (
                            "single_source_baseline"
                            if baseline in PRIMARY_SINGLE_SOURCE_BASELINES
                            else "legacy_or_vehicle_state_ablation"
                        ),
                        "n_paired_units": len(differences),
                        "proposed_mean": float(np.mean(proposed)),
                        "baseline_mean": float(np.mean(reference)),
                        "raw_mean_difference_proposed_minus_baseline": raw_effect,
                        "benefit_oriented_effect": benefit_effect,
                        "paired_difference_ci95_low": ci_low,
                        "paired_difference_ci95_high": ci_high,
                        "standardized_paired_effect_dz_benefit_oriented": standardized,
                        "p_value_two_sided": sign_flip_pvalue(differences, rng),
                        "p_value_holm": math.nan,
                        "statistically_significant_holm": False,
                    }
                )
    holm_adjust(comparisons)
    return comparisons


def confidence_intervals(
    run_rows: list[dict[str, Any]], graded_rows: list[dict[str, str]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for density in DENSITY_ORDER:
        for method in [PROPOSED_RUN_LEVEL_METHOD, *BASELINES]:
            selected = [
                row for row in run_rows
                if row["density_scenario"] == density and row["method"] == method
            ]
            for metric in METRICS:
                values = np.array([float(row[metric]) for row in selected], dtype=float)
                output.append(
                    descriptive_row(
                        "development_multisource_replay", density, method, metric, values, rng
                    )
                )

    graded_by_density = {
        row["density_scenario"]: row
        for row in graded_rows
        if row.get("method") == FINAL_GRADED_METHOD
    }
    if set(DENSITY_ORDER) - set(graded_by_density):
        raise RuntimeError("Final graded summary is missing one or more density scenarios")
    summary_metrics = {
        "precision": ("precision_mean", "precision_std"),
        "recall": ("recall_mean", "recall_std"),
        "f1": ("f1_mean", "f1_std"),
        "false_positive_rate": ("false_positive_rate_mean", "false_positive_rate_std"),
        "can_injection_recall": ("can_injection_recall_mean", None),
        "healthy_recovery_fpr_macro": ("healthy_recovery_fpr_macro", None),
    }
    for density in DENSITY_ORDER:
        row = graded_by_density[density]
        n = int(float(row["runs"]))
        critical, ci_method = summary_t_critical(n)
        for metric, (mean_field, sd_field) in summary_metrics.items():
            mean = float(row[mean_field])
            if sd_field is not None and row.get(sd_field, "") != "":
                sd = float(row[sd_field])
                margin = critical * sd / math.sqrt(n)
                low, high = mean - margin, mean + margin
                method_name = ci_method
            else:
                sd = math.nan
                low, high = math.nan, math.nan
                method_name = "not_estimable_from_available_summary"
            output.append(
                {
                    "analysis_partition": "graded_policy_development_summary",
                    "density_scenario": density,
                    "method": FINAL_GRADED_METHOD,
                    "metric": metric,
                    "n_independent_units": n,
                    "mean": mean,
                    "standard_deviation": sd,
                    "median": math.nan,
                    "q1": math.nan,
                    "q3": math.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci_method": method_name,
                }
            )
    return output


def sample_size_audit(
    run_rows: list[dict[str, Any]], replay_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for density in DENSITY_ORDER:
        selected_units = {
            row["unit_id"] for row in run_rows
            if row["density_scenario"] == density
            and row["method"] == PROPOSED_RUN_LEVEL_METHOD
        }
        replay = [row for row in replay_rows if row["density_scenario"] == density]
        output.append(
            {
                "density_scenario": density,
                "independent_paired_units": len(selected_units),
                "seeds": len({row["seed"] for row in replay}),
                "source_captures": len({row.get("hcrl_source_file", "") for row in replay}),
                "replay_sessions": len(replay),
                "attack_sampling_with_replacement_sessions": sum(
                    str(row.get("attack_sampling_with_replacement", "")).lower() == "true"
                    for row in replay
                ),
                "minimum_unique_attack_windows": min(
                    (int(float(row["unique_replayed_attack_windows"])) for row in replay),
                    default=0,
                ),
                "inference_unit": "seed x external source capture x density",
                "overlapping_windows_treated_as_independent": False,
            }
        )
    return output


def hypothesis_assessment(
    comparisons: list[dict[str, Any]],
    graded_rows: list[dict[str, str]],
    temporal_rows: list[dict[str, str]],
    multiscale_rows: list[dict[str, str]],
    selected_rule: str,
) -> list[dict[str, Any]]:
    h1_density = []
    for density in DENSITY_ORDER:
        relevant = [
            row for row in comparisons
            if row["density_scenario"] == density
            and row["metric"] == "f1"
            and row["baseline_method"] in PRIMARY_SINGLE_SOURCE_BASELINES
        ]
        passed = all(
            float(row["benefit_oriented_effect"]) > 0
            and bool(row["statistically_significant_holm"])
            for row in relevant
        )
        h1_density.append((density, passed))

    final_rows = [row for row in graded_rows if row.get("method") == FINAL_GRADED_METHOD]
    h2_f1 = all(float(row["f1_mean"]) >= 0.90 for row in final_rows)
    h2_fpr = all(float(row["false_positive_rate_mean"]) <= 0.05 for row in final_rows)

    temporal_lookup = {row["rule"]: row for row in temporal_rows}
    selected = temporal_lookup.get(selected_rule)
    w100_low = [
        row for row in multiscale_rows
        if row.get("method") == "w100_instant"
        and row.get("attack_frames_per_100") in {"1", "2-5"}
    ]
    h3_descriptive = False
    h3_detail = "Required selected-rule or frozen-w100 summaries not found"
    if selected and w100_low:
        selected_recall = float(selected["low_can_injection_recall_mean"])
        total_windows = sum(float(row["parent_windows"]) for row in w100_low)
        w100_recall = sum(
            float(row["recall"]) * float(row["parent_windows"]) for row in w100_low
        ) / total_windows
        selected_fpr = float(selected["overall_fpr_mean"])
        h3_descriptive = selected_recall > w100_recall and selected_fpr <= 0.05
        h3_detail = (
            f"selected low-density CAN recall={selected_recall:.4f}, "
            f"frozen w100 low-density recall={w100_recall:.4f}, selected mean FPR={selected_fpr:.4f}; "
            "descriptive only because paired temporal run records are unavailable"
        )

    return [
        {
            "hypothesis": "H1",
            "assessment": "supported" if all(value for _, value in h1_density) else "not_supported_across_all_densities",
            "confirmatory": False,
            "evidence": "; ".join(f"{density}={passed}" for density, passed in h1_density),
            "interpretation": "Development replay inference; Step 30E is required for confirmation.",
        },
        {
            "hypothesis": "H2",
            "assessment": "development_criteria_met" if h2_f1 and h2_fpr else "development_criteria_not_met",
            "confirmatory": False,
            "evidence": f"all density F1>=0.90: {h2_f1}; all density mean FPR<=0.05: {h2_fpr}",
            "interpretation": "Based on graded-policy development summaries; Step 30E remains required.",
        },
        {
            "hypothesis": "H3",
            "assessment": "descriptively_supported" if h3_descriptive else "not_descriptively_supported",
            "confirmatory": False,
            "evidence": h3_detail,
            "interpretation": "No inferential claim without paired temporal run records.",
        },
        {
            "hypothesis": "H4",
            "assessment": "pending_30C",
            "confirmatory": False,
            "evidence": "Missing-source non-inferiority experiment not yet run.",
            "interpretation": "Do not claim source-loss robustness yet.",
        },
        {
            "hypothesis": "H5",
            "assessment": "pending_30C",
            "confirmatory": False,
            "evidence": "Compromised/conflicting-source unsafe-ALLOW experiment not yet run.",
            "interpretation": "Do not claim that one source cannot force ALLOW yet.",
        },
    ]


def build_plots(comparisons: list[dict[str, Any]], cis: list[dict[str, Any]]) -> bool:
    cache = OUT / ".matplotlib_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    single_source = [
        row for row in comparisons
        if row["metric"] == "f1" and row["baseline_method"] in PRIMARY_SINGLE_SOURCE_BASELINES
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    y_labels = [f"{row['density_scenario']} | {row['baseline_method']}" for row in single_source]
    effects = [float(row["benefit_oriented_effect"]) for row in single_source]
    low = [float(row["paired_difference_ci95_low"]) for row in single_source]
    high = [float(row["paired_difference_ci95_high"]) for row in single_source]
    y = np.arange(len(y_labels))
    axes[0].errorbar(
        effects,
        y,
        xerr=[np.array(effects) - np.array(low), np.array(high) - np.array(effects)],
        fmt="o",
        capsize=3,
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, y_labels, fontsize=8)
    axes[0].set_xlabel("Paired F1 difference: proposed - baseline")
    axes[0].set_title("Architecture-level effect sizes (development replay)")
    axes[0].grid(axis="x", alpha=0.25)

    graded = [
        row for row in cis
        if row["analysis_partition"] == "graded_policy_development_summary"
        and row["metric"] in {"f1", "false_positive_rate"}
    ]
    for metric, marker in [("f1", "o"), ("false_positive_rate", "s")]:
        selected = [row for row in graded if row["metric"] == metric]
        x = np.arange(len(selected))
        means = np.array([float(row["mean"]) for row in selected])
        lows = np.array([float(row["ci95_low"]) for row in selected])
        highs = np.array([float(row["ci95_high"]) for row in selected])
        axes[1].errorbar(
            x,
            means,
            yerr=[means - lows, highs - means],
            marker=marker,
            capsize=3,
            label=metric,
        )
    axes[1].axhline(0.90, color="green", linestyle="--", linewidth=1, label="F1 criterion")
    axes[1].axhline(0.05, color="red", linestyle=":", linewidth=1, label="FPR criterion")
    axes[1].set_xticks(np.arange(len(DENSITY_ORDER)), DENSITY_ORDER, rotation=18, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Final graded-policy development intervals")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Step 30B publication statistical analysis", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "publication_statistical_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.rmtree(cache, ignore_errors=True)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = {key: locate(key) for key in INPUT_SPECS}
    raw_run_rows = read_csv(inputs["run_level"])
    replay_rows = read_csv(inputs["replay_audit"])
    graded_rows = read_csv(inputs["graded_summary"])
    temporal_rows = read_csv(inputs["temporal_confirmation"])
    multiscale_rows = read_csv(inputs["multiscale_density"])
    selected_rule_rows = read_csv(inputs["temporal_selected_rule"])
    if len(selected_rule_rows) != 1:
        raise RuntimeError("Expected exactly one frozen temporal-rule record")
    selected_rule = selected_rule_rows[0]["selected_rule"]

    rng = np.random.default_rng(RANDOM_SEED)
    run_rows = build_run_table(raw_run_rows)
    comparisons = paired_comparisons(run_rows, rng)
    intervals = confidence_intervals(run_rows, graded_rows, rng)
    sample_sizes = sample_size_audit(run_rows, replay_rows)
    hypotheses = hypothesis_assessment(
        comparisons, graded_rows, temporal_rows, multiscale_rows, selected_rule
    )

    write_csv(
        OUT / "publication_run_level_metrics.csv",
        run_rows,
        [
            "analysis_partition", "unit_id", "seed", "source_file", "density_scenario", "method",
            "true_positive", "false_positive", "true_negative", "false_negative", "precision", "recall",
            "f1", "false_positive_rate", "false_negative_rate", "accuracy",
        ],
    )
    write_csv(
        OUT / "publication_statistical_comparisons.csv",
        comparisons,
        [
            "analysis_partition", "density_scenario", "metric", "proposed_method", "baseline_method",
            "comparison_role", "n_paired_units", "proposed_mean", "baseline_mean",
            "raw_mean_difference_proposed_minus_baseline", "benefit_oriented_effect",
            "paired_difference_ci95_low", "paired_difference_ci95_high",
            "standardized_paired_effect_dz_benefit_oriented", "p_value_two_sided", "p_value_holm",
            "statistically_significant_holm",
        ],
    )
    write_csv(
        OUT / "publication_confidence_intervals.csv",
        intervals,
        [
            "analysis_partition", "density_scenario", "method", "metric", "n_independent_units", "mean",
            "standard_deviation", "median", "q1", "q3", "ci95_low", "ci95_high", "ci_method",
        ],
    )
    write_csv(
        OUT / "publication_sample_size_audit.csv",
        sample_sizes,
        [
            "density_scenario", "independent_paired_units", "seeds", "source_captures", "replay_sessions",
            "attack_sampling_with_replacement_sessions", "minimum_unique_attack_windows", "inference_unit",
            "overlapping_windows_treated_as_independent",
        ],
    )
    write_csv(
        OUT / "publication_hypothesis_assessment.csv",
        hypotheses,
        ["hypothesis", "assessment", "confirmatory", "evidence", "interpretation"],
    )

    plot_saved = build_plots(comparisons, intervals)
    h1 = next(row for row in hypotheses if row["hypothesis"] == "H1")
    h2 = next(row for row in hypotheses if row["hypothesis"] == "H2")
    h3 = next(row for row in hypotheses if row["hypothesis"] == "H3")
    summary = {
        "analysis_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "existing_project_artifacts_changed": 0,
        "analysis_partition": "development_multisource_replay",
        "inference_unit": "seed x external source capture x density",
        "window_level_pseudoreplication_used": False,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "multiple_testing_correction": "Holm within density x metric comparison family",
        "proposed_run_level_method": PROPOSED_RUN_LEVEL_METHOD,
        "final_graded_method": FINAL_GRADED_METHOD,
        "frozen_temporal_rule": selected_rule,
        "run_level_records": len(run_rows),
        "paired_comparisons": len(comparisons),
        "confidence_interval_records": len(intervals),
        "h1_development_assessment": h1["assessment"],
        "h2_development_assessment": h2["assessment"],
        "h3_development_assessment": h3["assessment"],
        "confirmatory_claim_permitted": False,
        "reason_confirmation_not_permitted": "Step 30E untouched final confirmation has not been run.",
        "input_files": {key: str(path.relative_to(ROOT)) for key, path in inputs.items()},
        "plot_saved": plot_saved,
    }
    (OUT / "publication_statistical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# Step 30B publication statistical analysis

This analysis changed zero existing project artifacts. It used seed, source
capture and density as the independent paired unit; overlapping windows were
not treated as independent observations.

## Development assessments

- H1: {h1['assessment']}
- H2: {h2['assessment']}
- H3: {h3['assessment']}
- Confirmatory claim permitted: False

H1 is allowed to fail. A high context-only end-to-end F1 does not imply CAN
coverage, so the manuscript must also report CAN-injection recall and phase
coverage. No result in this folder is labelled final confirmation.

## Next

Review the effect sizes, adjusted p-values, sample-size audit and hypothesis
table. Then run Step 30C. Do not run Step 31.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    archive = Path(
        shutil.make_archive(
            str(AUDIT_ROOT / f"publication_statistical_analysis_{RUN_ID}"),
            "zip",
            root_dir=OUT,
        )
    )

    print("=" * 78)
    print("Step 30B publication statistical analysis completed successfully.")
    print("Existing project artifacts changed: 0")
    print(f"Independent run-level records: {len(run_rows)}")
    print(f"Paired comparisons: {len(comparisons)}")
    print(f"H1 development assessment: {h1['assessment']}")
    print(f"H2 development assessment: {h2['assessment']}")
    print(f"H3 development assessment: {h3['assessment']}")
    print("Confirmatory claim permitted: False (Step 30E has not been run)")
    print(f"Results directory: {OUT}")
    print(f"Results archive: {archive}")
    print("\nNext: send the terminal result and the publication CSV, JSON, and plot outputs.")
    print("Do not run Step 31 yet.")


if __name__ == "__main__":
    main()
