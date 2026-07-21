#!/usr/bin/env python3
"""
This additive audit explains the negative Step 30C results without changing a
model, threshold, policy, dataset, or prior result. It tests whether attack and
benign rows become observationally indistinguishable when one source falsely
reports healthy and no quality warning is available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE_AUDIT_ROOT = RESULTS / "publication_source_robustness"
OUT_ROOT = RESULTS / "publication_source_observability"
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUT = OUT_ROOT / RUN_ID

INPUT_PREFERRED = RESULTS / "graded_zero_trust_policy" / "graded_policy_decisions.csv"
INPUT_FILENAME = "graded_policy_decisions.csv"
RANDOM_SEED = 161803
BOOTSTRAP_REPLICATES = 20_000
ALPHA = 0.05
H4_MARGIN = 0.10

DENSITY_ORDER = ["representative_all", "low_1_5", "medium_6_20", "high_21_100"]
SOURCES = ["can", "gnss", "v2x", "identity", "device_posture"]
NONCAN_SOURCES = ["gnss", "v2x", "identity", "device_posture"]
REQUIRED_COLUMNS = {
    "seed", "source_file", "density_scenario", "simulation_time_s",
    "ground_truth_attack", "multiscale_alarm_instant",
    "multiscale_alarm_persistent_2", "startup_quality_warning",
    "active_noncan_sources",
}


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_policy_input() -> Path:
    if INPUT_PREFERRED.exists():
        return INPUT_PREFERRED
    matches = sorted(
        path for path in RESULTS.rglob(INPUT_FILENAME)
        if SOURCE_AUDIT_ROOT not in path.parents and OUT_ROOT not in path.parents
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


def locate_step30c_run() -> dict[str, Path]:
    summaries = sorted(SOURCE_AUDIT_ROOT.glob("run_*/source_robustness_summary.json"))
    if not summaries:
        raise FileNotFoundError(
            "No timestamped Step 30C result found below " + str(SOURCE_AUDIT_ROOT)
        )
    summary = summaries[-1]
    run_dir = summary.parent
    required = {
        "summary": summary,
        "aggregate": run_dir / "source_robustness_aggregate_metrics.csv",
        "checks": run_dir / "source_robustness_safety_checks.csv",
        "hypotheses": run_dir / "source_robustness_hypothesis_assessment.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Latest Step 30C run is incomplete: " + ", ".join(missing))
    return required


def split_sources(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item for item in str(value).split(";") if item} - {"sensor_control"}


def source_signals(group: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    noncan = group["active_noncan_sources"].map(split_sources)
    risk = {
        "can": group["multiscale_alarm_persistent_2"].astype(bool).to_numpy(),
        "gnss": noncan.map(lambda value: "gnss" in value).to_numpy(dtype=bool),
        "v2x": noncan.map(lambda value: "v2x" in value).to_numpy(dtype=bool),
        "identity": noncan.map(lambda value: "identity" in value).to_numpy(dtype=bool),
        "device_posture": noncan.map(lambda value: "device_posture" in value).to_numpy(dtype=bool),
    }
    instant = group["multiscale_alarm_instant"].astype(bool).to_numpy()
    warning = group["startup_quality_warning"].astype(bool).to_numpy()
    return risk, instant, warning


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
    if unknown:
        raise ValueError(f"Cannot parse Boolean values: {unknown}")
    return normalized.isin({"true", "1"})


def percentile_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    sampled = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [ALPHA / 2, 1.0 - ALPHA / 2])
    return float(low), float(high)


def signature_codes(columns: list[np.ndarray]) -> np.ndarray:
    output = np.zeros(len(columns[0]), dtype=np.uint16)
    for bit, values in enumerate(columns):
        output |= values.astype(np.uint16) << bit
    return output


def observability_run_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["seed", "source_file", "density_scenario"]
    for (seed, capture, density), raw in frame.groupby(keys, sort=True):
        group = raw.sort_values("simulation_time_s", kind="stable").reset_index(drop=True)
        truth = group["ground_truth_attack"].astype(bool).to_numpy()
        risk, instant, warning = source_signals(group)
        for target in SOURCES:
            base_visible = [risk[source] for source in SOURCES if source != target]
            visible_instant = np.zeros(len(group), dtype=bool) if target == "can" else instant
            interfaces = [
                (
                    "step30c_replay_interface",
                    False,
                    "exact Step 30C replay semantics; startup warning omitted from full-ALLOW calculation",
                ),
                (
                    "frozen_step25_action_interface",
                    True,
                    "Step 25 action semantics sensitivity; general startup warning prevents full ALLOW",
                ),
            ]
            for evidence_interface, include_warning, interpretation in interfaces:
                visible_columns = [*base_visible, visible_instant]
                if include_warning:
                    visible_columns.append(warning)
                codes = signature_codes(visible_columns)

                attack_codes = set(codes[truth].tolist())
                benign_codes = set(codes[~truth].tolist())
                shared_codes = attack_codes & benign_codes
                ambiguous = (
                    np.isin(codes, list(shared_codes))
                    if shared_codes else np.zeros(len(codes), dtype=bool)
                )
                silent = codes == 0

                attack_rows = int(np.sum(truth))
                benign_rows = int(np.sum(~truth))
                silent_attack = int(np.sum(truth & silent))
                silent_benign = int(np.sum((~truth) & silent))
                ambiguous_attack = int(np.sum(truth & ambiguous))
                ambiguous_benign = int(np.sum((~truth) & ambiguous))

                # Equal-cost row-wise Bayes error using only the selected
                # Boolean evidence interface after the target is false healthy.
                bayes_errors = 0
                for code in set(codes.tolist()):
                    mask = codes == code
                    bayes_errors += min(int(np.sum(mask & truth)), int(np.sum(mask & ~truth)))

                rows.append(
                    {
                        "analysis_partition": "development_observability_audit",
                        "unit_id": f"seed={seed}|capture={capture}|density={density}",
                        "seed": seed,
                        "source_capture": capture,
                        "density_scenario": density,
                        "target_source": target,
                        "evidence_interface": evidence_interface,
                        "startup_quality_warning_included": include_warning,
                        "rows": len(group),
                        "attack_rows": attack_rows,
                        "benign_rows": benign_rows,
                        "visible_boolean_inputs": len(visible_columns),
                        "shared_attack_benign_signatures": len(shared_codes),
                        "ambiguous_attack_rows": ambiguous_attack,
                        "ambiguous_attack_rate": safe_divide(ambiguous_attack, attack_rows),
                        "ambiguous_benign_rows": ambiguous_benign,
                        "ambiguous_benign_rate": safe_divide(ambiguous_benign, benign_rows),
                        "silent_attack_rows": silent_attack,
                        "silent_attack_rate": safe_divide(silent_attack, attack_rows),
                        "silent_benign_rows": silent_benign,
                        "silent_benign_rate": safe_divide(silent_benign, benign_rows),
                        "equal_cost_row_bayes_error_count": bayes_errors,
                        "equal_cost_row_bayes_error_rate": safe_divide(bayes_errors, len(group)),
                        "undetected_false_healthy_source_quality_signal_available": False,
                        "interpretation": interpretation,
                    }
                )
    return rows


def aggregate_observability(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    keys = ["target_source", "density_scenario", "evidence_interface"]
    for (source, density, evidence_interface), group in frame.groupby(keys, sort=True):
        silent = group["silent_attack_rate"].to_numpy(dtype=float)
        silent_low, silent_high = percentile_mean_ci(silent, rng)
        ambiguity = group["ambiguous_attack_rate"].to_numpy(dtype=float)
        ambiguity_low, ambiguity_high = percentile_mean_ci(ambiguity, rng)
        output.append(
            {
                "target_source": source,
                "density_scenario": density,
                "evidence_interface": evidence_interface,
                "startup_quality_warning_included": bool(
                    group["startup_quality_warning_included"].iloc[0]
                ),
                "independent_runs": len(group),
                "attack_rows": int(group["attack_rows"].sum()),
                "benign_rows": int(group["benign_rows"].sum()),
                "silent_attack_rows": int(group["silent_attack_rows"].sum()),
                "silent_attack_rate_pooled": safe_divide(
                    int(group["silent_attack_rows"].sum()), int(group["attack_rows"].sum())
                ),
                "silent_attack_rate_run_mean": float(group["silent_attack_rate"].mean()),
                "silent_attack_rate_ci95_low": silent_low,
                "silent_attack_rate_ci95_high": silent_high,
                "silent_benign_rows": int(group["silent_benign_rows"].sum()),
                "silent_benign_rate_pooled": safe_divide(
                    int(group["silent_benign_rows"].sum()), int(group["benign_rows"].sum())
                ),
                "silent_benign_rate_run_mean": float(group["silent_benign_rate"].mean()),
                "ambiguous_attack_rate_run_mean": float(group["ambiguous_attack_rate"].mean()),
                "ambiguous_attack_rate_ci95_low": ambiguity_low,
                "ambiguous_attack_rate_ci95_high": ambiguity_high,
                "ambiguous_benign_rate_run_mean": float(group["ambiguous_benign_rate"].mean()),
                "equal_cost_row_bayes_error_rate_mean": float(group["equal_cost_row_bayes_error_rate"].mean()),
                "fail_open_cost": "silent attack rows receive full ALLOW",
                "naive_fail_safe_cost": "silent benign rows are restricted",
                "ci_method": f"run_level_percentile_bootstrap_{BOOTSTRAP_REPLICATES}",
            }
        )
    return output


def verify_against_step30c(
    observability: list[dict[str, Any]], aggregate_30c: pd.DataFrame
) -> list[dict[str, Any]]:
    current = pd.DataFrame(observability)
    current = current[current["evidence_interface"] == "step30c_replay_interface"].copy()
    expected = aggregate_30c[
        aggregate_30c["scenario"] == "compromised_undetected_false_healthy"
    ][["target_source", "density_scenario", "unsafe_allow_count", "attack_rows"]].copy()
    merged = current.merge(
        expected, on=["target_source", "density_scenario"],
        how="outer", validate="one_to_one",
    )
    required_columns = [
        "target_source", "density_scenario", "silent_attack_rows",
        "unsafe_allow_count", "attack_rows_y",
    ]
    if merged[required_columns].isna().any().any():
        raise RuntimeError("Step 30C cross-check could not align all source/density conditions")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        passed = int(row["silent_attack_rows"]) == int(row["unsafe_allow_count"])
        rows.append(
            {
                "target_source": row["target_source"],
                "density_scenario": row["density_scenario"],
                "step30c_unsafe_allow_count": int(row["unsafe_allow_count"]),
                "step30c_attack_rows": int(row["attack_rows_y"]),
                "observability_silent_attack_count": int(row["silent_attack_rows"]),
                "count_match": passed,
                "interpretation": "exact reproduction" if passed else "mismatch",
            }
        )
    if not all(row["count_match"] for row in rows):
        raise RuntimeError("Observability audit did not exactly reproduce Step 30C unsafe-ALLOW counts")
    return rows


def build_claim_register(
    checks: pd.DataFrame, hypotheses: pd.DataFrame
) -> list[dict[str, Any]]:
    h4 = hypotheses[hypotheses["hypothesis"] == "H4"].iloc[0]
    detected = checks[checks["scenario"] == "compromised_detected_false_healthy"]
    conflict = checks[checks["scenario"] == "conflicting_high_risk"]
    undetected = checks[checks["scenario"] == "compromised_undetected_false_healthy"]
    detected_pass = boolean_series(detected["passed"])
    conflict_pass = boolean_series(conflict["passed"])
    undetected_pass = boolean_series(undetected["passed"])
    return [
        {
            "claim_id": "H4_original",
            "claim": "Loss of any one non-CAN context source reduces end-to-end F1 by no more than 0.10.",
            "development_status": str(h4["assessment"]),
            "conditions_passed": int(h4["tests"] - h4["failed_tests"]),
            "conditions_tested": int(h4["tests"]),
            "publication_permission": "not_supported_in_development",
            "confirmation_rule": "evaluate once in Step 30E and retain any rejection",
        },
        {
            "claim_id": "H5_detected_compromise",
            "claim": "A detected compromised source cannot force full ALLOW.",
            "development_status": "supported_conditionally" if detected_pass.all() else "not_supported",
            "conditions_passed": int(detected_pass.sum()),
            "conditions_tested": len(detected),
            "publication_permission": "conditional_on_observable_integrity_failure",
            "confirmation_rule": "report separately from undetected compromise",
        },
        {
            "claim_id": "H5_conflicting_high_risk",
            "claim": "Conflicting high-risk evidence cannot receive full ALLOW.",
            "development_status": "supported_conditionally" if conflict_pass.all() else "not_supported",
            "conditions_passed": int(conflict_pass.sum()),
            "conditions_tested": len(conflict),
            "publication_permission": "conditional_on_observable_conflict",
            "confirmation_rule": "evaluate once in Step 30E",
        },
        {
            "claim_id": "H5_undetected_false_healthy",
            "claim": "An undetected Byzantine false-healthy source cannot force full ALLOW.",
            "development_status": "not_supported_and_observability_limited",
            "conditions_passed": int(undetected_pass.sum()),
            "conditions_tested": len(undetected),
            "publication_permission": "prohibited_for_current_policy",
            "confirmation_rule": "retain as residual vulnerability; do not merge with detected compromise",
        },
        {
            "claim_id": "H5_universal",
            "claim": "A compromised source cannot independently force ALLOW under all modeled conditions.",
            "development_status": "not_supported",
            "conditions_passed": int(detected_pass.sum() + conflict_pass.sum() + undetected_pass.sum()),
            "conditions_tested": len(detected) + len(conflict) + len(undetected),
            "publication_permission": "prohibited_without_explicit_source_quality_observability",
            "confirmation_rule": "original H5 remains locked; report outcome without post-hoc relabelling",
        },
    ]


def architecture_requirements() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence = [
        ("authenticated_source_identity", "bind observations to an authenticated ECU or service identity"),
        ("heartbeat_and_freshness_age", "distinguish no anomaly from no current observation"),
        ("attestation_or_integrity_status", "make a detected compromise observable to the policy"),
        ("replay_counter_or_monotonicity", "detect replayed but apparently healthy evidence"),
        ("independent_corroboration", "avoid accepting a single false-healthy source as sufficient evidence"),
    ]
    for source in SOURCES:
        for requirement, rationale in evidence:
            rows.append(
                {
                    "source": source,
                    "required_quality_input": requirement,
                    "required_for_claim": "future detected-loss-or-compromise tolerance",
                    "present_in_frozen_step25_interface": False,
                    "policy_action_when_failed": "VERIFY_OR_RESTRICT; SAFE_FALLBACK when safety critical",
                    "rationale": rationale,
                    "validation_status": "not_implemented_not_claimed",
                }
            )
    return rows


def build_plot(step30c_aggregate: pd.DataFrame, observability: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (OUT / "plot_not_created.txt").write_text(
            f"Matplotlib unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        return False

    obs = pd.DataFrame(observability)
    missing = step30c_aggregate[
        (step30c_aggregate["scenario"] == "missing_full_session")
        & (step30c_aggregate["target_source"].isin(NONCAN_SOURCES))
    ]
    worst_h4 = (
        missing.sort_values("f1_loss_ci95_high", ascending=False)
        .groupby("target_source", as_index=False).first()
        .set_index("target_source").reindex(NONCAN_SOURCES)
    )
    exact_obs = obs[obs["evidence_interface"] == "step30c_replay_interface"]
    step25_obs = obs[obs["evidence_interface"] == "frozen_step25_action_interface"]
    worst_obs = (
        exact_obs.sort_values("silent_attack_rate_pooled", ascending=False)
        .groupby("target_source", as_index=False).first()
        .set_index("target_source").reindex(SOURCES)
    )
    worst_step25 = (
        step25_obs.sort_values("silent_attack_rate_pooled", ascending=False)
        .groupby("target_source", as_index=False).first()
        .set_index("target_source").reindex(SOURCES)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    x = np.arange(len(NONCAN_SOURCES))
    axes[0].bar(x, worst_h4["f1_loss_ci95_high"], color="#d62728")
    axes[0].axhline(H4_MARGIN, color="black", linestyle="--", label="H4 margin")
    axes[0].set_xticks(x, NONCAN_SOURCES, rotation=15)
    axes[0].set_ylabel("Worst-density upper 95% CI of F1 loss")
    axes[0].set_title("Original H4 development result")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    x2 = np.arange(len(SOURCES))
    width = 0.25
    axes[1].bar(
        x2 - width, worst_obs["silent_attack_rate_pooled"], width,
        label="Step 30C unsafe ALLOW", color="#d62728",
    )
    axes[1].bar(
        x2, worst_step25["silent_attack_rate_pooled"], width,
        label="Step 25 warning-adjusted unsafe ALLOW", color="#1f77b4",
    )
    axes[1].bar(
        x2 + width, worst_obs["silent_benign_rate_pooled"], width,
        label="Naive fail-safe: benign restriction", color="#ff7f0e",
    )
    axes[1].set_xticks(x2, SOURCES, rotation=15)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Worst-density pooled row rate")
    axes[1].set_title("Undetected false-healthy observability trade-off")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Step 30C2 source observability and publication claim boundary", fontsize=15)
    fig.savefig(OUT / "source_observability_claim_boundary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    policy_path = locate_policy_input()
    step30c_paths = locate_step30c_run()
    tracked_inputs = [policy_path, *step30c_paths.values()]
    before = {str(path): sha256(path) for path in tracked_inputs}

    frame = pd.read_csv(policy_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Step 25 decisions are missing columns: {sorted(missing)}")
    duplicate = frame.duplicated(
        ["seed", "source_file", "density_scenario", "simulation_time_s"], keep=False
    )
    if duplicate.any():
        raise ValueError("Duplicate ordered replay rows detected within an independent run")

    aggregate_30c = pd.read_csv(step30c_paths["aggregate"])
    checks_30c = pd.read_csv(step30c_paths["checks"])
    hypotheses_30c = pd.read_csv(step30c_paths["hypotheses"])
    if set(hypotheses_30c["hypothesis"]) != {"H4", "H5"}:
        raise ValueError("Step 30C hypothesis assessment does not contain exactly H4 and H5")

    rng = np.random.default_rng(RANDOM_SEED)
    run_rows = observability_run_metrics(frame)
    aggregate_rows = aggregate_observability(run_rows, rng)
    crosscheck_rows = verify_against_step30c(aggregate_rows, aggregate_30c)
    claim_rows = build_claim_register(checks_30c, hypotheses_30c)
    requirement_rows = architecture_requirements()
    plot_saved = build_plot(aggregate_30c, aggregate_rows)

    run_fields = [
        "analysis_partition", "unit_id", "seed", "source_capture", "density_scenario",
        "target_source", "evidence_interface", "startup_quality_warning_included",
        "rows", "attack_rows", "benign_rows", "visible_boolean_inputs",
        "shared_attack_benign_signatures", "ambiguous_attack_rows", "ambiguous_attack_rate",
        "ambiguous_benign_rows", "ambiguous_benign_rate", "silent_attack_rows",
        "silent_attack_rate", "silent_benign_rows", "silent_benign_rate",
        "equal_cost_row_bayes_error_count", "equal_cost_row_bayes_error_rate",
        "undetected_false_healthy_source_quality_signal_available", "interpretation",
    ]
    write_csv(OUT / "source_observability_run_metrics.csv", run_rows, run_fields)
    aggregate_fields = [
        "target_source", "density_scenario", "evidence_interface",
        "startup_quality_warning_included", "independent_runs", "attack_rows", "benign_rows",
        "silent_attack_rows", "silent_attack_rate_pooled", "silent_attack_rate_run_mean",
        "silent_attack_rate_ci95_low", "silent_attack_rate_ci95_high", "silent_benign_rows",
        "silent_benign_rate_pooled", "silent_benign_rate_run_mean",
        "ambiguous_attack_rate_run_mean", "ambiguous_attack_rate_ci95_low",
        "ambiguous_attack_rate_ci95_high", "ambiguous_benign_rate_run_mean",
        "equal_cost_row_bayes_error_rate_mean", "fail_open_cost", "naive_fail_safe_cost",
        "ci_method",
    ]
    write_csv(OUT / "source_observability_aggregate_metrics.csv", aggregate_rows, aggregate_fields)
    write_csv(
        OUT / "source_observability_step30c_crosscheck.csv", crosscheck_rows,
        ["target_source", "density_scenario", "step30c_unsafe_allow_count", "step30c_attack_rows",
         "observability_silent_attack_count", "count_match", "interpretation"],
    )
    write_csv(
        OUT / "publication_claim_boundary_register.csv", claim_rows,
        ["claim_id", "claim", "development_status", "conditions_passed", "conditions_tested",
         "publication_permission", "confirmation_rule"],
    )
    write_csv(
        OUT / "future_source_quality_requirements.csv", requirement_rows,
        ["source", "required_quality_input", "required_for_claim",
         "present_in_frozen_step25_interface", "policy_action_when_failed", "rationale",
         "validation_status"],
    )

    after = {str(path): sha256(path) for path in tracked_inputs}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("A tracked input changed during this read-only audit")

    h4 = next(row for row in claim_rows if row["claim_id"] == "H4_original")
    h5u = next(row for row in claim_rows if row["claim_id"] == "H5_universal")
    aggregate_frame = pd.DataFrame(aggregate_rows)
    exact_frame = aggregate_frame[
        aggregate_frame["evidence_interface"] == "step30c_replay_interface"
    ]
    step25_frame = aggregate_frame[
        aggregate_frame["evidence_interface"] == "frozen_step25_action_interface"
    ]
    exact_silent_attacks = int(exact_frame["silent_attack_rows"].sum())
    step25_silent_attacks = int(step25_frame["silent_attack_rows"].sum())
    summary = {
        "analysis_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_partition": "development_observability_and_claim_boundary",
        "existing_project_artifacts_changed": 0,
        "tracked_inputs_unchanged": unchanged,
        "step25_input": str(policy_path.relative_to(ROOT)),
        "step30c_run": str(step30c_paths["summary"].parent.relative_to(ROOT)),
        "independent_unit": "seed x external CAN capture x attack density",
        "window_level_pseudoreplication_used": False,
        "run_records": len(run_rows),
        "aggregate_records": len(aggregate_rows),
        "step30c_exact_crosschecks": len(crosscheck_rows),
        "all_step30c_crosschecks_passed": all(row["count_match"] for row in crosscheck_rows),
        "step30c_replay_interface_omits_startup_quality_warning": True,
        "frozen_step25_action_interface_sensitivity_reported": True,
        "step30c_interface_silent_attack_rows": exact_silent_attacks,
        "step25_warning_adjusted_silent_attack_rows": step25_silent_attacks,
        "startup_warning_prevented_full_allow_rows_in_sensitivity": (
            exact_silent_attacks - step25_silent_attacks
        ),
        "h4_original_development_status": h4["development_status"],
        "h5_universal_development_status": h5u["development_status"],
        "detected_compromise_claim_scope": "conditional_on_observable_integrity_failure",
        "undetected_false_healthy_claim_permitted": False,
        "source_quality_inputs_present_in_frozen_step25_interface": False,
        "post_hoc_tuning_performed": False,
        "confirmatory_claim_permitted": False,
        "step30e_required": True,
        "step31_permitted": False,
        "plot_saved": plot_saved,
        "input_sha256": before,
    }
    (OUT / "source_observability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        """# Step 30C2 source-observability and claim-boundary audit

This additive audit changed zero existing project artifacts and reproduced every
Step 30C undetected-false-healthy unsafe-ALLOW count exactly. It also reports a
separate sensitivity analysis using Step 25 action semantics, where the general
startup-quality warning prevents full ALLOW. This distinction is retained
because Step 30C omitted that warning from its replay-time full-ALLOW formula.

Both interfaces demonstrate a row-level observability boundary: attack and
benign rows can share the same visible signature after one source becomes
undetectably false healthy.

The result does not prove that all future architectures are unable to tolerate a
Byzantine source. It proves that the present policy cannot support that universal
claim without additional independently validated source-quality evidence.

H4 and the universal H5 remain unsupported in development. Detected compromise
and observable conflict are reported as conditional results. No threshold was
tuned, and Step 31 remains blocked pending untouched Step 30E confirmation and
final reproducibility review.
""",
        encoding="utf-8",
    )
    archive = Path(
        shutil.make_archive(
            str(OUT_ROOT / f"publication_source_observability_{RUN_ID}"),
            "zip", root_dir=OUT,
        )
    )

    h4_failed = int(hypotheses_30c.loc[hypotheses_30c["hypothesis"] == "H4", "failed_tests"].iloc[0])
    h5_failed = int(hypotheses_30c.loc[hypotheses_30c["hypothesis"] == "H5", "failed_tests"].iloc[0])
    print("=" * 82)
    print("Step 30C2 source-observability and claim-boundary audit completed successfully.")
    print("Existing project artifacts changed: 0")
    print(f"Independent replay units: {frame.groupby(['seed', 'source_file', 'density_scenario']).ngroups}")
    print(f"Observability run-level records: {len(run_rows):,}")
    print(f"Exact Step 30C unsafe-ALLOW cross-checks: {len(crosscheck_rows)}/{len(crosscheck_rows)} passed")
    print(f"Original H4 development failures retained: {h4_failed}")
    print(f"Original H5 development failures retained: {h5_failed}")
    print("Detected compromise/conflict claim: conditional on observable quality evidence")
    print("Undetected Byzantine false-healthy universal claim: NOT supported")
    print(f"Results directory: {OUT}")
    print(f"Results archive: {archive}")
    print("\nNext: send the terminal result, summary JSON, claim register, aggregate metrics, and plot.")
    print("Do not run Step 30E or Step 31 yet.")


if __name__ == "__main__":
    main()
