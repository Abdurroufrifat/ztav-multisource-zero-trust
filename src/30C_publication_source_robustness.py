#!/usr/bin/env python3
"""Step 30C: publication source-robustness and policy-safety audit.

This additive analysis replays the frozen Step 25 graded-policy decisions under
predeclared source perturbations.  It does not train a model, choose a threshold,
or overwrite an existing result.  The independent unit is one seed x external
CAN capture x attack-density replay, not an overlapping window.

Run from D:\\ztav_project after Steps 25 and 30B:

    .\\.venv\\Scripts\\python.exe .\\src\\30C_publication_source_robustness.py

Outputs are written to a new timestamped directory below:

    results/publication_source_robustness/

The experiment is development falsification evidence.  Step 30E remains the
only untouched confirmation stage, and Step 31 must not be run after this script.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT_ROOT = RESULTS / "publication_source_robustness"
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUT = AUDIT_ROOT / RUN_ID

INPUT_PREFERRED = RESULTS / "graded_zero_trust_policy" / "graded_policy_decisions.csv"
INPUT_FILENAME = "graded_policy_decisions.csv"

RANDOM_SEED = 271828
BOOTSTRAP_REPLICATES = 20_000
ALPHA = 0.05
H4_NONINFERIORITY_MARGIN = 0.10
STALE_LAG_ROWS = 5

DENSITY_ORDER = ["representative_all", "low_1_5", "medium_6_20", "high_21_100"]
SOURCES = ["can", "gnss", "v2x", "identity", "device_posture"]
H4_CONTEXT_SOURCES = ["gnss", "v2x", "identity", "device_posture"]
CRITICAL_LOCAL_SOURCES = {"can", "gnss", "identity", "device_posture"}

SCENARIOS = [
    "reference",
    "missing_full_session",
    "stale_delay_5",
    "compromised_detected_false_healthy",
    "compromised_undetected_false_healthy",
    "conflicting_high_risk",
]
H5_SCENARIOS = {
    "compromised_detected_false_healthy",
    "compromised_undetected_false_healthy",
    "conflicting_high_risk",
}

REQUIRED_COLUMNS = {
    "seed",
    "source_file",
    "density_scenario",
    "simulation_time_s",
    "phase",
    "ground_truth_attack",
    "multiscale_alarm_instant",
    "multiscale_alarm_persistent_2",
    "startup_quality_warning",
    "active_noncan_sources",
}


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario: str
    availability_known: bool
    integrity_failure_known: bool
    description: str
    role: str


SCENARIO_DEFINITIONS = {
    "reference": ScenarioDefinition(
        "reference", False, False,
        "Frozen Step 25 vehicle-state-free persistent policy without perturbation.",
        "paired_reference",
    ),
    "missing_full_session": ScenarioDefinition(
        "missing_full_session", True, False,
        "Target source is unavailable for the complete replay; its evidence is removed and re-verification is required.",
        "H4_primary_and_availability_cost",
    ),
    "stale_delay_5": ScenarioDefinition(
        "stale_delay_5", True, False,
        "Target source evidence is delayed by five ordered replay rows; stale quality is known and triggers re-verification.",
        "stale_and_delayed_evidence_diagnostic",
    ),
    "compromised_detected_false_healthy": ScenarioDefinition(
        "compromised_detected_false_healthy", True, True,
        "Target source falsely reports healthy but its integrity failure is detected; evidence is quarantined.",
        "H5_detected_compromise",
    ),
    "compromised_undetected_false_healthy": ScenarioDefinition(
        "compromised_undetected_false_healthy", False, False,
        "Worst-case Byzantine target falsely reports healthy while retaining apparently valid integrity; no quality alarm is available.",
        "H5_residual_vulnerability",
    ),
    "conflicting_high_risk": ScenarioDefinition(
        "conflicting_high_risk", True, False,
        "Target source reports high risk while other evidence may disagree; conflict is marked and cannot receive full ALLOW.",
        "H5_conflicting_evidence",
    ),
}


def locate_input() -> Path:
    if INPUT_PREFERRED.exists():
        return INPUT_PREFERRED
    matches = sorted(
        path for path in RESULTS.rglob(INPUT_FILENAME)
        if AUDIT_ROOT not in path.parents
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Missing {INPUT_FILENAME}; expected {INPUT_PREFERRED}. Run Step 25 first."
        )
    raise RuntimeError(
        f"Ambiguous {INPUT_FILENAME}; found: " + ", ".join(str(path) for path in matches)
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def split_sources(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item for item in str(value).split(";") if item} - {"sensor_control"}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fn = int(np.sum(truth & ~prediction))
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


def percentile_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    return float(low), float(high)


def source_signals(group: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    noncan = group["active_noncan_sources"].map(split_sources)
    risk = {
        "can": group["multiscale_alarm_persistent_2"].astype(bool).to_numpy(),
        "gnss": noncan.map(lambda value: "gnss" in value).to_numpy(dtype=bool),
        "v2x": noncan.map(lambda value: "v2x" in value).to_numpy(dtype=bool),
        "identity": noncan.map(lambda value: "identity" in value).to_numpy(dtype=bool),
        "device_posture": noncan.map(lambda value: "device_posture" in value).to_numpy(dtype=bool),
    }
    instant_can = group["multiscale_alarm_instant"].astype(bool).to_numpy()
    return risk, instant_can


def delayed(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.zeros(len(values), dtype=bool)
    if lag < len(values):
        output[lag:] = values[:-lag]
    return output


def perturb_signals(
    base: dict[str, np.ndarray],
    base_instant_can: np.ndarray,
    scenario: str,
    target_source: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    risk = {key: value.copy() for key, value in base.items()}
    instant_can = base_instant_can.copy()
    quality_issue = np.zeros(len(instant_can), dtype=bool)
    if scenario == "reference":
        return risk, instant_can, quality_issue
    if target_source not in SOURCES:
        raise ValueError(f"Unknown target source: {target_source}")

    if scenario == "missing_full_session":
        risk[target_source][:] = False
        if target_source == "can":
            instant_can[:] = False
        quality_issue[:] = True
    elif scenario == "stale_delay_5":
        risk[target_source] = delayed(risk[target_source], STALE_LAG_ROWS)
        if target_source == "can":
            instant_can = delayed(instant_can, STALE_LAG_ROWS)
        quality_issue[:] = True
    elif scenario == "compromised_detected_false_healthy":
        risk[target_source][:] = False
        if target_source == "can":
            instant_can[:] = False
        quality_issue[:] = True
    elif scenario == "compromised_undetected_false_healthy":
        risk[target_source][:] = False
        if target_source == "can":
            instant_can[:] = False
    elif scenario == "conflicting_high_risk":
        risk[target_source][:] = True
        if target_source == "can":
            instant_can[:] = True
        quality_issue[:] = True
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return risk, instant_can, quality_issue


def evaluate_variant(
    truth: np.ndarray,
    risk: dict[str, np.ndarray],
    instant_can: np.ndarray,
    quality_issue: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.logical_or.reduce([risk[source] for source in SOURCES])
    metrics = binary_metrics(truth, prediction)

    critical = np.logical_or.reduce([risk[source] for source in CRITICAL_LOCAL_SOURCES])
    v2x = risk["v2x"]
    safe_fallback = critical
    monitor_verify = ~safe_fallback & instant_can
    verify_restrict = ~safe_fallback & ~monitor_verify & quality_issue
    allow_local_only = ~safe_fallback & ~monitor_verify & ~verify_restrict & v2x
    full_allow = ~safe_fallback & ~monitor_verify & ~verify_restrict & ~allow_local_only

    attack_rows = int(np.sum(truth))
    benign_rows = int(np.sum(~truth))
    unsafe_allow_count = int(np.sum(truth & full_allow))
    high_risk_allow_count = int(np.sum(prediction & full_allow))
    quality_issue_allow_count = int(np.sum(quality_issue & full_allow))
    metrics.update(
        {
            "rows": len(truth),
            "attack_rows": attack_rows,
            "benign_rows": benign_rows,
            "unsafe_allow_count": unsafe_allow_count,
            "unsafe_allow_rate": safe_divide(unsafe_allow_count, attack_rows),
            "high_risk_allow_count": high_risk_allow_count,
            "quality_issue_allow_count": quality_issue_allow_count,
            "safe_fallback_rate": float(np.mean(safe_fallback)),
            "monitor_verify_rate": float(np.mean(monitor_verify)),
            "verify_restrict_rate": float(np.mean(verify_restrict)),
            "allow_local_only_rate": float(np.mean(allow_local_only)),
            "full_allow_rate": float(np.mean(full_allow)),
            "benign_restriction_rate": safe_divide(int(np.sum((~truth) & (~full_allow))), benign_rows),
        }
    )
    return metrics


def run_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["seed", "source_file", "density_scenario"]
    for (seed, capture, density), raw_group in frame.groupby(keys, sort=True):
        group = raw_group.sort_values("simulation_time_s", kind="stable").reset_index(drop=True)
        truth = group["ground_truth_attack"].astype(bool).to_numpy()
        base, instant = source_signals(group)
        reference = evaluate_variant(
            truth, base, instant, np.zeros(len(group), dtype=bool)
        )
        unit_id = f"seed={seed}|capture={capture}|density={density}"
        variants = [("reference", "none")]
        variants.extend(
            (scenario, source)
            for scenario in SCENARIOS if scenario != "reference"
            for source in SOURCES
        )
        for scenario, target in variants:
            risk, instant_variant, quality = perturb_signals(base, instant, scenario, target)
            values = evaluate_variant(truth, risk, instant_variant, quality)
            rows.append(
                {
                    "analysis_partition": "development_source_falsification",
                    "unit_id": unit_id,
                    "seed": seed,
                    "source_capture": capture,
                    "density_scenario": density,
                    "scenario": scenario,
                    "target_source": target,
                    "reference_f1": reference["f1"],
                    "absolute_f1_loss": float(reference["f1"]) - float(values["f1"]),
                    **values,
                }
            )
    return rows


def aggregate_metrics(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    keys = ["scenario", "target_source", "density_scenario"]
    for (scenario, target, density), group in frame.groupby(keys, sort=True):
        losses = group["absolute_f1_loss"].to_numpy(dtype=float)
        low, high = percentile_ci(losses, rng)
        attack_rows = int(group["attack_rows"].sum())
        unsafe_count = int(group["unsafe_allow_count"].sum())
        output.append(
            {
                "scenario": scenario,
                "target_source": target,
                "density_scenario": density,
                "independent_runs": len(group),
                "reference_f1_mean": float(group["reference_f1"].mean()),
                "perturbed_f1_mean": float(group["f1"].mean()),
                "absolute_f1_loss_mean": float(group["absolute_f1_loss"].mean()),
                "absolute_f1_loss_std": float(group["absolute_f1_loss"].std(ddof=1)),
                "f1_loss_ci95_low": low,
                "f1_loss_ci95_high": high,
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "false_positive_rate_mean": float(group["false_positive_rate"].mean()),
                "unsafe_allow_count": unsafe_count,
                "attack_rows": attack_rows,
                "unsafe_allow_rate_pooled": safe_divide(unsafe_count, attack_rows),
                "unsafe_allow_rate_run_mean": float(group["unsafe_allow_rate"].mean()),
                "unsafe_allow_rate_run_max": float(group["unsafe_allow_rate"].max()),
                "benign_restriction_rate_mean": float(group["benign_restriction_rate"].mean()),
                "safe_fallback_rate_mean": float(group["safe_fallback_rate"].mean()),
                "verify_restrict_rate_mean": float(group["verify_restrict_rate"].mean()),
                "full_allow_rate_mean": float(group["full_allow_rate"].mean()),
                "high_risk_allow_count": int(group["high_risk_allow_count"].sum()),
                "quality_issue_allow_count": int(group["quality_issue_allow_count"].sum()),
                "ci_method": f"run_level_percentile_bootstrap_{BOOTSTRAP_REPLICATES}",
            }
        )
    return output


def safety_checks(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in aggregate:
        scenario = row["scenario"]
        target = row["target_source"]
        if scenario == "missing_full_session" and target in H4_CONTEXT_SOURCES:
            upper = float(row["f1_loss_ci95_high"])
            passed = upper <= H4_NONINFERIORITY_MARGIN
            rows.append(
                {
                    "hypothesis": "H4",
                    "scenario": scenario,
                    "target_source": target,
                    "density_scenario": row["density_scenario"],
                    "criterion": f"upper 95% CI of paired mean F1 loss <= {H4_NONINFERIORITY_MARGIN:.2f}",
                    "observed_value": upper,
                    "passed": passed,
                    "unsafe_allow_count": row["unsafe_allow_count"],
                    "attack_rows": row["attack_rows"],
                    "availability_cost_benign_restriction_rate": row["benign_restriction_rate_mean"],
                    "interpretation": "non-inferior" if passed else "source-loss robustness margin exceeded",
                }
            )
        elif scenario in H5_SCENARIOS and target in SOURCES:
            unsafe = int(row["unsafe_allow_count"])
            high_risk_allow = int(row["high_risk_allow_count"])
            passed = unsafe == 0 and high_risk_allow == 0
            rows.append(
                {
                    "hypothesis": "H5",
                    "scenario": scenario,
                    "target_source": target,
                    "density_scenario": row["density_scenario"],
                    "criterion": "zero full-ALLOW attack rows and zero high-risk full-ALLOW rows",
                    "observed_value": row["unsafe_allow_rate_pooled"],
                    "passed": passed,
                    "unsafe_allow_count": unsafe,
                    "attack_rows": row["attack_rows"],
                    "availability_cost_benign_restriction_rate": row["benign_restriction_rate_mean"],
                    "interpretation": "safety condition met" if passed else "unsafe full ALLOW observed",
                }
            )
    return rows


def hypothesis_rows(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    h4 = [row for row in checks if row["hypothesis"] == "H4"]
    h5 = [row for row in checks if row["hypothesis"] == "H5"]
    h4_failed = [
        f"{row['target_source']}:{row['density_scenario']}"
        for row in h4 if not row["passed"]
    ]
    h5_failed = [
        f"{row['scenario']}:{row['target_source']}:{row['density_scenario']}"
        for row in h5 if not row["passed"]
    ]
    h5_undetected_only = bool(h5_failed) and all(
        item.startswith("compromised_undetected_false_healthy:") for item in h5_failed
    )
    return [
        {
            "hypothesis": "H4",
            "assessment": "development_supported" if h4 and not h4_failed else "development_not_supported",
            "confirmatory": False,
            "tests": len(h4),
            "failed_tests": len(h4_failed),
            "evidence": ";".join(h4_failed) if h4_failed else "all non-CAN source-loss margins met",
            "interpretation": "Step 30E remains required; failed source/density conditions must not be hidden.",
        },
        {
            "hypothesis": "H5",
            "assessment": (
                "development_supported" if h5 and not h5_failed
                else "not_supported_under_undetected_byzantine_source" if h5_undetected_only
                else "development_not_supported"
            ),
            "confirmatory": False,
            "tests": len(h5),
            "failed_tests": len(h5_failed),
            "evidence": ";".join(h5_failed[:40]) if h5_failed else "all compromised/conflicting-source safety checks met",
            "interpretation": "Detected compromise and undetected Byzantine compromise are reported separately.",
        },
    ]


def build_plot(aggregate: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (OUT / "plot_not_created.txt").write_text(
            f"Matplotlib unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        return False

    frame = pd.DataFrame(aggregate)
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    x = np.arange(len(H4_CONTEXT_SOURCES))
    width = 0.19
    missing = frame[frame["scenario"] == "missing_full_session"]
    for index, density in enumerate(DENSITY_ORDER):
        group = missing[missing["density_scenario"] == density].set_index("target_source").reindex(H4_CONTEXT_SOURCES)
        axes[0].bar(
            x + (index - 1.5) * width,
            group["absolute_f1_loss_mean"],
            width,
            label=density,
        )
    axes[0].axhline(H4_NONINFERIORITY_MARGIN, color="black", linestyle="--", label="H4 margin")
    axes[0].set_xticks(x, H4_CONTEXT_SOURCES, rotation=15)
    axes[0].set(title="Complete source-loss F1 degradation", ylabel="Paired mean absolute F1 loss")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    safety_scenarios = [
        "compromised_detected_false_healthy",
        "compromised_undetected_false_healthy",
        "conflicting_high_risk",
    ]
    x2 = np.arange(len(SOURCES))
    width2 = 0.25
    labels = {
        "compromised_detected_false_healthy": "Detected compromise",
        "compromised_undetected_false_healthy": "Undetected Byzantine",
        "conflicting_high_risk": "Conflicting high risk",
    }
    for index, scenario in enumerate(safety_scenarios):
        group = (
            frame[frame["scenario"] == scenario]
            .groupby("target_source", as_index=True)["unsafe_allow_rate_pooled"]
            .max()
            .reindex(SOURCES)
        )
        axes[1].bar(x2 + (index - 1) * width2, group, width2, label=labels[scenario])
    axes[1].set_xticks(x2, SOURCES, rotation=15)
    axes[1].set(title="Worst-density unsafe full-ALLOW rate", ylabel="Attack-row unsafe ALLOW rate")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Step 30C multi-source Zero Trust falsification audit", fontsize=15)
    figure.savefig(OUT / "source_robustness_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    input_path = locate_input()
    frame = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Step 25 decisions are missing columns: {sorted(missing)}")
    unknown_density = sorted(set(frame["density_scenario"]) - set(DENSITY_ORDER))
    if unknown_density:
        raise ValueError(f"Unexpected density scenarios: {unknown_density}")
    duplicate = frame.duplicated(
        ["seed", "source_file", "density_scenario", "simulation_time_s"], keep=False
    )
    if duplicate.any():
        raise ValueError("Duplicate ordered replay rows detected within an independent run")

    rng = np.random.default_rng(RANDOM_SEED)
    per_run = run_metrics(frame)
    aggregate = aggregate_metrics(per_run, rng)
    checks = safety_checks(aggregate)
    hypotheses = hypothesis_rows(checks)
    plot_saved = build_plot(aggregate)

    run_fields = [
        "analysis_partition", "unit_id", "seed", "source_capture", "density_scenario",
        "scenario", "target_source", "rows", "attack_rows", "benign_rows",
        "reference_f1", "absolute_f1_loss", "true_positive", "false_positive",
        "true_negative", "false_negative", "precision", "recall", "f1",
        "false_positive_rate", "false_negative_rate", "accuracy", "unsafe_allow_count",
        "unsafe_allow_rate", "high_risk_allow_count", "quality_issue_allow_count",
        "safe_fallback_rate", "monitor_verify_rate", "verify_restrict_rate",
        "allow_local_only_rate", "full_allow_rate", "benign_restriction_rate",
    ]
    write_csv(OUT / "source_robustness_run_metrics.csv", per_run, run_fields)
    aggregate_fields = [
        "scenario", "target_source", "density_scenario", "independent_runs",
        "reference_f1_mean", "perturbed_f1_mean", "absolute_f1_loss_mean",
        "absolute_f1_loss_std", "f1_loss_ci95_low", "f1_loss_ci95_high",
        "precision_mean", "recall_mean", "false_positive_rate_mean",
        "unsafe_allow_count", "attack_rows", "unsafe_allow_rate_pooled",
        "unsafe_allow_rate_run_mean", "unsafe_allow_rate_run_max",
        "benign_restriction_rate_mean", "safe_fallback_rate_mean",
        "verify_restrict_rate_mean", "full_allow_rate_mean", "high_risk_allow_count",
        "quality_issue_allow_count", "ci_method",
    ]
    write_csv(OUT / "source_robustness_aggregate_metrics.csv", aggregate, aggregate_fields)
    write_csv(
        OUT / "source_robustness_safety_checks.csv",
        checks,
        [
            "hypothesis", "scenario", "target_source", "density_scenario", "criterion",
            "observed_value", "passed", "unsafe_allow_count", "attack_rows",
            "availability_cost_benign_restriction_rate", "interpretation",
        ],
    )
    write_csv(
        OUT / "source_robustness_hypothesis_assessment.csv",
        hypotheses,
        ["hypothesis", "assessment", "confirmatory", "tests", "failed_tests", "evidence", "interpretation"],
    )
    definitions = [
        {
            "scenario": definition.scenario,
            "target_sources": "none" if key == "reference" else ";".join(SOURCES),
            "availability_known": definition.availability_known,
            "integrity_failure_known": definition.integrity_failure_known,
            "stale_lag_rows": STALE_LAG_ROWS if key == "stale_delay_5" else 0,
            "role": definition.role,
            "description": definition.description,
        }
        for key, definition in SCENARIO_DEFINITIONS.items()
    ]
    write_csv(
        OUT / "source_robustness_scenario_definitions.csv",
        definitions,
        [
            "scenario", "target_sources", "availability_known", "integrity_failure_known",
            "stale_lag_rows", "role", "description",
        ],
    )

    h4 = next(row for row in hypotheses if row["hypothesis"] == "H4")
    h5 = next(row for row in hypotheses if row["hypothesis"] == "H5")
    summary = {
        "analysis_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "existing_project_artifacts_changed": 0,
        "analysis_partition": "development_source_falsification",
        "input_file": str(input_path.relative_to(ROOT)),
        "independent_unit": "seed x external CAN capture x attack density",
        "window_level_pseudoreplication_used": False,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "h4_noninferiority_margin": H4_NONINFERIORITY_MARGIN,
        "stale_delay_rows": STALE_LAG_ROWS,
        "sources": SOURCES,
        "vehicle_state_in_primary_policy": False,
        "run_metric_records": len(per_run),
        "aggregate_records": len(aggregate),
        "safety_checks": len(checks),
        "h4_development_assessment": h4["assessment"],
        "h5_development_assessment": h5["assessment"],
        "confirmatory_claim_permitted": False,
        "reason_confirmation_not_permitted": "Step 30E untouched final confirmation has not been run.",
        "plot_saved": plot_saved,
    }
    (OUT / "source_robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    readme = f"""# Step 30C publication source-robustness audit

This timestamped analysis changed zero existing artifacts.  It evaluated the
frozen vehicle-state-free persistent policy.  Source availability and integrity
warnings changed the graded safety action but were not counted as attack labels;
this separates detection accuracy from availability cost.

## Development assessments

- H4: {h4['assessment']}
- H5: {h5['assessment']}
- Confirmatory claim permitted: False

An undetected Byzantine false-healthy source is deliberately separated from a
detected integrity failure.  Negative results are retained.  Review all failed
source/density cases before Step 30D, and do not run Step 31.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    archive = Path(
        shutil.make_archive(
            str(AUDIT_ROOT / f"publication_source_robustness_{RUN_ID}"),
            "zip",
            root_dir=OUT,
        )
    )

    failed_h4 = int(h4["failed_tests"])
    failed_h5 = int(h5["failed_tests"])
    print("=" * 82)
    print("Step 30C publication source-robustness audit completed successfully.")
    print("Existing project artifacts changed: 0")
    print(f"Independent replay units: {frame.groupby(['seed', 'source_file', 'density_scenario']).ngroups}")
    print(f"Perturbed run-level records: {len(per_run):,}")
    print(f"H4 development assessment: {h4['assessment']} (failed checks={failed_h4})")
    print(f"H5 development assessment: {h5['assessment']} (failed checks={failed_h5})")
    print("Confirmatory claim permitted: False (Step 30E has not been run)")
    print(f"Results directory: {OUT}")
    print(f"Results archive: {archive}")
    print("\nNext: send the terminal result, safety checks, hypothesis assessment, summary JSON, and plot.")
    print("Do not run Step 31 yet.")


if __name__ == "__main__":
    main()
