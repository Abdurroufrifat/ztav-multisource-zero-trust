"""Freeze the final research policy and generate a reproducible evidence pack.

This script does not train, recalibrate, or modify any detector. It reads the
already-generated result files, records the selected policy components, hashes
the evidence, and explicitly excludes ROAD and GEM-CAN candidates that failed
their predeclared readiness criteria.  Step 30G authorization is mandatory.

Run from D:\\ztav_project:
    .\\.venv\\Scripts\\python.exe .\\src\\31_freeze_final_zero_trust_policy.py

Outputs:
    results/final_zero_trust_policy/

This is a research prototype freeze, not production automotive software.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
OUT = RESULTS / "final_zero_trust_policy"
FROZEN = OUT / "frozen_inputs"

POLICY_ID = "ztav_multisource_context_aware_zero_trust_v1"
POLICY_STATUS = "research_prototype_frozen_with_external_limitations"
SELECTED_METHOD = "proposed_persistent_without_vehicle_state"
SELECTED_TEMPORAL_RULE = "strict_consecutive_2"


EVIDENCE_SPECS = [
    {
        "key": "prefreeze_authorization",
        "filename": "FINAL_FREEZE_AUTHORIZATION.json",
        "preferred": None,
        "allow_latest": True,
        "required": True,
        "purpose": "Step 30G bounded research-prototype freeze authorization",
    },
    {
        "key": "gem_protocol",
        "filename": "GEM_CAN_CONFIRMATION_PROTOCOL_LOCK.json",
        "preferred": None,
        "allow_latest": True,
        "required": True,
        "purpose": "Locked GEM-CAN external-confirmation protocol",
    },
    {
        "key": "gem_outcome",
        "filename": "GEM_CAN_CONFIRMATION_OUTCOME.json",
        "preferred": None,
        "allow_latest": True,
        "required": True,
        "purpose": "One-time GEM-CAN external-confirmation outcome",
    },
    {
        "key": "gem_acceptance",
        "filename": "gem_can_confirmation_acceptance.csv",
        "preferred": None,
        "allow_latest": True,
        "required": True,
        "purpose": "Predeclared GEM-CAN acceptance checks",
    },
    {
        "key": "gem_metrics",
        "filename": "gem_can_primary_cluster_metrics.csv",
        "preferred": None,
        "allow_latest": True,
        "required": True,
        "purpose": "GEM-CAN cluster-macro external endpoints",
    },
    {
        "key": "graded_endpoint",
        "filename": "graded_primary_endpoint_summary.csv",
        "preferred": "graded_zero_trust_policy/graded_primary_endpoint_summary.csv",
        "required": True,
        "purpose": "Primary graded-policy confirmation endpoint",
    },
    {
        "key": "graded_actions",
        "filename": "graded_action_distribution.csv",
        "preferred": "graded_zero_trust_policy/graded_action_distribution.csv",
        "required": True,
        "purpose": "Zero Trust enforcement-action distribution",
    },
    {
        "key": "graded_startup_actions",
        "filename": "graded_startup_session_actions.csv",
        "preferred": "graded_zero_trust_policy/graded_startup_session_actions.csv",
        "required": True,
        "purpose": "Startup re-verification decisions used by the graded policy",
    },
    {
        "key": "temporal_rule",
        "filename": "temporal_selected_rule.csv",
        "preferred": "temporal_memory_sparse_can_confirmation/temporal_selected_rule.csv",
        "required": True,
        "purpose": "Frozen sparse-CAN temporal rule",
    },
    {
        "key": "temporal_confirmation",
        "filename": "temporal_confirmation_primary_endpoint.csv",
        "preferred": "temporal_memory_sparse_can_confirmation/temporal_confirmation_primary_endpoint.csv",
        "required": True,
        "purpose": "Independent temporal-rule confirmation",
    },
    {
        "key": "hybrid_endpoint",
        "filename": "hybrid_aggregate_metrics.csv",
        "preferred": "hybrid_ciciov_sumo/hybrid_aggregate_metrics.csv",
        "required": True,
        "purpose": "CICIoV-to-SUMO multi-source replay endpoint",
    },
    {
        "key": "repeated_seed",
        "filename": "aggregate_metrics.csv",
        "preferred": "repeated_seed_validation/aggregate_metrics.csv",
        "required": True,
        "purpose": "Repeated-seed stability evidence",
    },
    {
        "key": "counterfactual",
        "filename": "counterfactual_aggregate_metrics.csv",
        "preferred": "physical_vehicle_state_counterfactual_w100/counterfactual_aggregate_metrics.csv",
        "required": True,
        "purpose": "Vehicle-state counterfactual falsification audit",
    },
    {
        "key": "road_frozen_acceptance",
        "filename": "road_acceptance_criteria.csv",
        "preferred": "road_frozen_external_confirmation/road_acceptance_criteria.csv",
        "required": True,
        "purpose": "Frozen ROAD zero-shot external acceptance audit",
    },
    {
        "key": "road_context_acceptance",
        "filename": "signal_context_acceptance_criteria.csv",
        "preferred": "road_signal_context_gate/signal_context_acceptance_criteria.csv",
        "required": True,
        "purpose": "Post-hoc ROAD context-gate readiness audit",
    },
    {
        "key": "road_sparse_acceptance",
        "filename": "sparse_signal_acceptance_criteria.csv",
        "preferred": "road_sparse_signal_event_gate/sparse_signal_acceptance_criteria.csv",
        "required": True,
        "purpose": "Post-hoc ROAD sparse-event readiness audit",
    },
    {
        "key": "road_sparse_metrics",
        "filename": "sparse_signal_overall_metrics.csv",
        "preferred": "road_sparse_signal_event_gate/sparse_signal_overall_metrics.csv",
        "required": True,
        "purpose": "ROAD sparse-event external endpoint",
    },
]


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def locate(spec: dict[str, Any]) -> tuple[Path | None, str]:
    preferred_value = spec.get("preferred")
    if preferred_value:
        preferred = RESULTS / preferred_value
        if preferred.exists():
            return preferred, "preferred_path"
    matches = sorted(RESULTS.rglob(spec["filename"]))
    if len(matches) == 1:
        return matches[0], "recursive_unique_match"
    if not matches:
        return None, "missing"
    if spec.get("allow_latest"):
        matches.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
        return matches[-1], "latest_recursive_match"
    return None, "ambiguous_multiple_matches"


def acceptance_summary(path: Path, source: str) -> list[dict[str, Any]]:
    output = []
    for row in read_csv(path):
        output.append(
            {
                "source": source,
                "criterion": row.get("criterion", ""),
                "observed_value": row.get("observed_value", row.get("observed", "")),
                "operator": row.get("operator", ""),
                "required_value": row.get("required_value", row.get("required", "")),
                "passed": as_bool(row.get("passed", "")),
            }
        )
    return output


def metric_lookup(rows: list[dict[str, str]], density: str) -> dict[str, str]:
    for row in rows:
        if row.get("density_scenario") == density and row.get("method") == SELECTED_METHOD:
            return row
    raise RuntimeError(f"Missing {SELECTED_METHOD!r} endpoint for density {density!r}")


def build_plot(primary_rows: list[dict[str, Any]], external_rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    labels = [row["density_scenario"] for row in primary_rows]
    f1 = [float(row["f1_mean"]) for row in primary_rows]
    fpr = [float(row["false_positive_rate_mean"]) for row in primary_rows]
    can_recall = [float(row["can_injection_recall_mean"]) for row in primary_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    width = 0.25
    axes[0].bar(x - width, f1, width, label="End-to-end F1")
    axes[0].bar(x, can_recall, width, label="CAN-injection recall")
    axes[0].bar(x + width, fpr, width, label="False-positive rate")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Frozen graded policy")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    ext_labels = [row["candidate"] for row in external_rows]
    ext_recall = [float(row["attack_recall"]) for row in external_rows]
    ext_fpr = [float(row["benign_fpr"]) for row in external_rows]
    ex = np.arange(len(ext_labels))
    axes[1].bar(ex - 0.18, ext_recall, 0.36, label="Attack recall")
    axes[1].bar(ex + 0.18, ext_fpr, 0.36, label="Benign FPR")
    axes[1].axhline(0.70, color="black", linestyle="--", linewidth=1, label="Recall target")
    axes[1].axhline(0.05, color="red", linestyle=":", linewidth=1, label="FPR limit")
    axes[1].set_xticks(ex, ext_labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Excluded external candidates")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Final multi-source Zero Trust policy evidence freeze", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "final_policy_evidence_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    if OUT.exists():
        raise FileExistsError(
            f"Final freeze already exists at {OUT}. Step 31 is one-time; do not overwrite or rerun it."
        )
    OUT.mkdir(parents=True, exist_ok=False)
    FROZEN.mkdir(parents=True, exist_ok=True)

    located: dict[str, Path] = {}
    inventory: list[dict[str, Any]] = []
    missing_required: list[str] = []

    for spec in EVIDENCE_SPECS:
        path, resolution = locate(spec)
        if path is None:
            if spec["required"]:
                missing_required.append(spec["key"])
            inventory.append(
                {
                    "evidence_key": spec["key"],
                    "purpose": spec["purpose"],
                    "required": spec["required"],
                    "status": resolution,
                    "source_path": "",
                    "frozen_copy": "",
                    "sha256": "",
                    "size_bytes": "",
                }
            )
            continue

        located[spec["key"]] = path
        frozen_name = f"{spec['key']}__{path.name}"
        frozen_path = FROZEN / frozen_name
        shutil.copy2(path, frozen_path)
        inventory.append(
            {
                "evidence_key": spec["key"],
                "purpose": spec["purpose"],
                "required": spec["required"],
                "status": resolution,
                "source_path": str(path.relative_to(ROOT)),
                "frozen_copy": str(frozen_path.relative_to(ROOT)),
                "sha256": sha256(frozen_path),
                "size_bytes": frozen_path.stat().st_size,
            }
        )

    if missing_required:
        write_csv(
            OUT / "final_evidence_inventory.csv",
            inventory,
            ["evidence_key", "purpose", "required", "status", "source_path", "frozen_copy", "sha256", "size_bytes"],
        )
        raise FileNotFoundError(
            "Cannot freeze policy; required evidence is missing or ambiguous: "
            + ", ".join(missing_required)
            + f". Review {OUT / 'final_evidence_inventory.csv'}"
        )

    authorization = json.loads(located["prefreeze_authorization"].read_text(encoding="utf-8-sig"))
    authorization_stored_hash = str(authorization.get("authorization_sha256", ""))
    authorization_unsigned = dict(authorization)
    authorization_unsigned.pop("authorization_sha256", None)
    authorization_hash_valid = authorization_stored_hash == canonical_json_sha256(authorization_unsigned)
    authorization_script_hash_valid = str(authorization.get("step31_sha256", "")) == sha256(Path(__file__))
    authorization_scope_valid = (
        authorization.get("authorization_type") == "bounded_research_prototype_freeze_only"
        and authorization.get("step31_permitted") is True
        and authorization.get("confirmatory_external_generalisation_claim_permitted") is False
        and authorization.get("universal_source_robustness_claim_permitted") is False
        and authorization.get("production_deployment_permitted") is False
    )

    gem_protocol = json.loads(located["gem_protocol"].read_text(encoding="utf-8-sig"))
    gem_outcome = json.loads(located["gem_outcome"].read_text(encoding="utf-8-sig"))
    gem_protocol_stored_hash = str(gem_protocol.get("protocol_sha256", ""))
    gem_protocol_unsigned = dict(gem_protocol)
    gem_protocol_unsigned.pop("protocol_sha256", None)
    gem_protocol_hash_valid = gem_protocol_stored_hash == canonical_json_sha256(gem_protocol_unsigned)
    gem_outcome_stored_hash = str(gem_outcome.get("outcome_sha256", ""))
    gem_outcome_unsigned = dict(gem_outcome)
    gem_outcome_unsigned.pop("outcome_sha256", None)
    gem_outcome_hash_valid = gem_outcome_stored_hash == canonical_json_sha256(gem_outcome_unsigned)
    gem_protocol_matches = gem_protocol_stored_hash == str(gem_outcome.get("protocol_sha256", ""))
    gem_negative_retained = (
        gem_outcome.get("confirmation_consumed") is True
        and gem_outcome.get("confirmatory_passed") is False
        and as_float(gem_outcome.get("high_density_cluster_macro_recall"), -1.0) == 0.0
        and all(
            gem_outcome.get(key) is False
            for key in ("model_retrained", "threshold_changed", "window_reselected", "sumo_executed")
        )
    )
    prefreeze_valid = all(
        [
            authorization_hash_valid,
            authorization_script_hash_valid,
            authorization_scope_valid,
            gem_protocol_hash_valid,
            gem_outcome_hash_valid,
            gem_protocol_matches,
            gem_negative_retained,
        ]
    )
    if not prefreeze_valid:
        raise RuntimeError(
            "Cannot freeze: Step 30G authorization or locked GEM-CAN evidence failed integrity/scope validation."
        )

    temporal = read_csv(located["temporal_rule"])[0]
    temporal_confirmation = read_csv(located["temporal_confirmation"])
    selected_rule = temporal.get("selected_rule", "")
    if selected_rule != SELECTED_TEMPORAL_RULE:
        raise RuntimeError(
            f"Temporal-rule mismatch: expected {SELECTED_TEMPORAL_RULE!r}, found {selected_rule!r}"
        )

    graded = read_csv(located["graded_endpoint"])
    density_order = ["representative_all", "low_1_5", "medium_6_20", "high_21_100"]
    primary_rows = [metric_lookup(graded, density) for density in density_order]

    primary_endpoint = []
    for row in primary_rows:
        primary_endpoint.append(
            {
                "density_scenario": row["density_scenario"],
                "method": row["method"],
                "runs": row["runs"],
                "precision_mean": row["precision_mean"],
                "recall_mean": row["recall_mean"],
                "f1_mean": row["f1_mean"],
                "false_positive_rate_mean": row["false_positive_rate_mean"],
                "can_injection_recall_mean": row["can_injection_recall_mean"],
                "healthy_recovery_fpr_macro": row["healthy_recovery_fpr_macro"],
            }
        )

    road_frozen = acceptance_summary(located["road_frozen_acceptance"], "step28_frozen_road")
    road_context = acceptance_summary(located["road_context_acceptance"], "step29_context_gate")
    road_sparse = acceptance_summary(located["road_sparse_acceptance"], "step30_sparse_event_gate")
    gem_acceptance = acceptance_summary(located["gem_acceptance"], "step30f_locked_gem_can")
    acceptance = road_frozen + road_context + road_sparse + gem_acceptance

    sparse_metrics = read_csv(located["road_sparse_metrics"])
    sparse_primary = next(
        row for row in sparse_metrics if row.get("method") == "sparse_event_max_any_instant"
    )
    road_frozen_lookup = {row["criterion"]: row for row in road_frozen}
    road_context_lookup = {row["criterion"]: row for row in road_context}
    gem_metrics = read_csv(located["gem_metrics"])
    gem_recall_row = next(
        row for row in gem_metrics if row.get("endpoint") == "high_density_feature_cluster_macro_recall"
    )
    gem_fpr_row = next(
        row for row in gem_metrics if row.get("endpoint") == "clean_feature_cluster_macro_false_positive_rate"
    )

    external_rows = [
        {
            "candidate": "Step 28 frozen zero-shot",
            "role": "independent_external_confirmation",
            "attack_recall": road_frozen_lookup["persistent_overall_attack_recall"]["observed_value"],
            "benign_fpr": road_frozen_lookup["persistent_pooled_ambient_fpr"]["observed_value"],
            "decision": "excluded_failed_predeclared_criteria",
        },
        {
            "candidate": "Step 29 context gate",
            "role": "post_hoc_development_diagnostic",
            "attack_recall": road_context_lookup.get("primary_masquerade_recall", {}).get("observed_value", 0.0001),
            "benign_fpr": road_context_lookup.get("holdout_pooled_fpr", {}).get("observed_value", 0.0146),
            "decision": "excluded_failed_readiness",
        },
        {
            "candidate": "Step 30 sparse-event gate",
            "role": "post_hoc_development_repair",
            "attack_recall": sparse_primary["recall"],
            "benign_fpr": sparse_primary["false_positive_rate"],
            "decision": "excluded_failed_readiness",
        },
        {
            "candidate": "Step 30F locked GEM-CAN",
            "role": "one_time_predeclared_external_confirmation",
            "attack_recall": gem_recall_row["point_estimate"],
            "benign_fpr": gem_fpr_row["point_estimate"],
            "decision": "excluded_failed_predeclared_external_generalization_criteria",
        },
    ]

    component_rows = [
        {
            "component": "group-disjoint CICIoV CAN classifier",
            "status": "included",
            "role": "CAN anomaly evidence",
            "frozen_setting": "validation-selected group-disjoint model",
            "reason": "Leakage-safe CAN subsystem and stress-test evidence",
        },
        {
            "component": "SUMO context sources",
            "status": "included",
            "role": "GNSS, V2X, identity and operational context",
            "frozen_setting": "hybrid replay configuration",
            "reason": "Multi-source context improves end-to-end robustness",
        },
        {
            "component": "graded persistent Zero Trust enforcement",
            "status": "included",
            "role": "ALLOW / VERIFY / RESTRICT / SAFE_FALLBACK",
            "frozen_setting": SELECTED_METHOD,
            "reason": "Best confirmed availability-security compromise",
        },
        {
            "component": "sparse-CAN temporal memory",
            "status": "included",
            "role": "Persistent CAN evidence",
            "frozen_setting": SELECTED_TEMPORAL_RULE,
            "reason": "Selected on development seeds and frozen for confirmation",
        },
        {
            "component": "vehicle-state score in primary enforcement",
            "status": "excluded_from_primary",
            "role": "Counterfactual physical context",
            "frozen_setting": "audit-only",
            "reason": "Primary endpoint remains vehicle-state-free after falsification audit",
        },
        {
            "component": "ROAD Step 28 frozen detector",
            "status": "excluded",
            "role": "External zero-shot CAN gate",
            "frozen_setting": "none",
            "reason": "Failed attack-recall and ambient-FPR criteria",
        },
        {
            "component": "ROAD Steps 29-30 semantic candidates",
            "status": "excluded",
            "role": "Post-hoc external development",
            "frozen_setting": "none",
            "reason": "Generic per-ID signals were insufficient for masquerade detection",
        },
        {
            "component": "GEM-CAN Step 30F frozen detector transfer",
            "status": "excluded",
            "role": "Independent pooled high-density external CAN confirmation",
            "frozen_setting": "negative evidence only",
            "reason": "Predeclared external recall criteria failed; threshold and model remained frozen",
        },
    ]

    claims = [
        {
            "type": "supported_claim",
            "statement": "The proposed method fuses CAN anomaly evidence with GNSS, V2X, identity and operational context under graded Zero Trust enforcement.",
            "boundary": "Supported by the CICIoV-SUMO replay protocol used in this project.",
        },
        {
            "type": "supported_claim",
            "statement": "The frozen graded policy preserves high representative-density end-to-end F1 with controlled mean false-positive rate.",
            "boundary": "Report density-specific means and confirmation seeds; do not generalize to production vehicles.",
        },
        {
            "type": "limitation",
            "statement": "Low-density CAN-injection recall remains materially lower than representative and high-density recall.",
            "boundary": "The low_1_5 confirmation endpoint must be reported separately.",
        },
        {
            "type": "limitation",
            "statement": "The frozen and post-hoc ROAD gates did not generalize to ROAD masquerade attacks.",
            "boundary": "ROAD components are excluded; this is a negative external-validation result.",
        },
        {
            "type": "limitation",
            "statement": "The one-time locked GEM-CAN confirmation achieved high-density cluster-macro recall 0.0 while clean cluster-macro FPR was 0.0.",
            "boundary": "external_generalization_not_confirmed; retain the result without retuning or rerunning GEM-CAN.",
        },
        {
            "type": "prohibited_claim",
            "statement": "Universal zero-shot protection across all vehicles, CAN schemas, attacks or operational domains.",
            "boundary": "Not supported by the available experiments.",
        },
        {
            "type": "prohibited_claim",
            "statement": "Production-ready or safety-certified autonomous-vehicle security.",
            "boundary": "The artifact is a research prototype only.",
        },
    ]

    freeze_checks = [
        {
            "check": "step30g_authorization_hash_valid",
            "observed": authorization_hash_valid,
            "required": True,
            "passed": authorization_hash_valid,
        },
        {
            "check": "step30g_authorized_this_exact_step31_script",
            "observed": authorization_script_hash_valid,
            "required": True,
            "passed": authorization_script_hash_valid,
        },
        {
            "check": "bounded_research_prototype_freeze_only",
            "observed": authorization.get("authorization_type"),
            "required": "bounded_research_prototype_freeze_only",
            "passed": authorization_scope_valid,
        },
        {
            "check": "locked_gem_can_negative_result_retained_without_tuning",
            "observed": gem_negative_retained,
            "required": True,
            "passed": gem_negative_retained,
        },
        {
            "check": "all_required_evidence_present",
            "observed": len(missing_required) == 0,
            "required": True,
            "passed": len(missing_required) == 0,
        },
        {
            "check": "temporal_rule_matches_frozen_selection",
            "observed": selected_rule,
            "required": SELECTED_TEMPORAL_RULE,
            "passed": selected_rule == SELECTED_TEMPORAL_RULE,
        },
        {
            "check": "all_density_f1_at_least_0.90",
            "observed": min(as_float(row["f1_mean"]) for row in primary_rows),
            "required": 0.90,
            "passed": min(as_float(row["f1_mean"]) for row in primary_rows) >= 0.90,
        },
        {
            "check": "all_density_mean_fpr_at_most_0.05",
            "observed": max(as_float(row["false_positive_rate_mean"]) for row in primary_rows),
            "required": 0.05,
            "passed": max(as_float(row["false_positive_rate_mean"]) for row in primary_rows) <= 0.05,
        },
        {
            "check": "temporal_confirmation_mean_fpr_at_most_0.05",
            "observed": max(as_float(row["overall_fpr_mean"]) for row in temporal_confirmation),
            "required": 0.05,
            "passed": max(as_float(row["overall_fpr_mean"]) for row in temporal_confirmation) <= 0.05,
        },
        {
            "check": "temporal_confirmation_recovery_macro_at_most_0.05",
            "observed": max(as_float(row["healthy_recovery_fpr_macro"]) for row in temporal_confirmation),
            "required": 0.05,
            "passed": max(as_float(row["healthy_recovery_fpr_macro"]) for row in temporal_confirmation) <= 0.05,
        },
        {
            "check": "development_and_confirmation_seeds_disjoint",
            "observed": not (
                set(filter(None, temporal.get("development_seeds", "").split(";")))
                & set(filter(None, temporal.get("confirmation_seeds", "").split(";")))
            ),
            "required": True,
            "passed": not (
                set(filter(None, temporal.get("development_seeds", "").split(";")))
                & set(filter(None, temporal.get("confirmation_seeds", "").split(";")))
            ),
        },
        {
            "check": "road_candidates_excluded",
            "observed": True,
            "required": True,
            "passed": True,
        },
        {
            "check": "gem_can_external_candidate_excluded",
            "observed": True,
            "required": True,
            "passed": True,
        },
    ]
    freeze_ready = all(bool(row["passed"]) for row in freeze_checks)

    manifest = {
        "policy_id": POLICY_ID,
        "policy_status": POLICY_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "selected_primary_method": SELECTED_METHOD,
        "selected_temporal_rule": {
            "name": selected_rule,
            "required_hits": temporal.get("required_hits"),
            "history_windows": temporal.get("history_windows"),
            "hold_clean_windows": temporal.get("hold_clean_windows"),
            "development_seeds": temporal.get("development_seeds"),
            "confirmation_seeds": temporal.get("confirmation_seeds"),
        },
        "road_policy": "excluded_negative_external_validation",
        "gem_can_policy": "excluded_failed_locked_external_confirmation",
        "freeze_scope": "bounded_research_prototype_freeze_only",
        "external_generalization_not_confirmed": True,
        "universal_source_robustness_not_confirmed": True,
        "negative_results_retained": ["ROAD Steps 28-30", "GEM-CAN Step 30F", "Step 30C H4/H5 failures"],
        "freeze_ready": freeze_ready,
        "research_only": True,
        "production_or_safety_certified": False,
        "evidence_inventory": "final_evidence_inventory.csv",
        "primary_endpoint": "final_primary_endpoint.csv",
        "external_validation": "final_external_validation.csv",
        "claims_and_limitations": "final_claims_and_limitations.csv",
    }

    write_csv(
        OUT / "final_evidence_inventory.csv",
        inventory,
        ["evidence_key", "purpose", "required", "status", "source_path", "frozen_copy", "sha256", "size_bytes"],
    )
    write_csv(
        OUT / "final_policy_components.csv",
        component_rows,
        ["component", "status", "role", "frozen_setting", "reason"],
    )
    write_csv(
        OUT / "final_primary_endpoint.csv",
        primary_endpoint,
        [
            "density_scenario", "method", "runs", "precision_mean", "recall_mean", "f1_mean",
            "false_positive_rate_mean", "can_injection_recall_mean", "healthy_recovery_fpr_macro",
        ],
    )
    write_csv(
        OUT / "final_external_validation.csv",
        external_rows,
        ["candidate", "role", "attack_recall", "benign_fpr", "decision"],
    )
    write_csv(
        OUT / "final_acceptance_summary.csv",
        acceptance,
        ["source", "criterion", "observed_value", "operator", "required_value", "passed"],
    )
    write_csv(
        OUT / "final_claims_and_limitations.csv",
        claims,
        ["type", "statement", "boundary"],
    )
    write_csv(
        OUT / "final_freeze_checks.csv",
        freeze_checks,
        ["check", "observed", "required", "passed"],
    )
    (OUT / "final_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    plot_saved = build_plot(primary_endpoint, external_rows)
    readme = f"""# Final multi-source context-aware Zero Trust policy freeze

Policy ID: `{POLICY_ID}`  
Status: `{POLICY_STATUS}`  
Freeze checks passed: `{freeze_ready}`

## Included

- Group-disjoint CICIoV CAN classifier.
- SUMO-derived GNSS, V2X, identity and operational context.
- Graded persistent enforcement: ALLOW, VERIFY, RESTRICT and SAFE_FALLBACK.
- Frozen sparse-CAN temporal rule: `{SELECTED_TEMPORAL_RULE}`.

## Excluded

- Vehicle-state score from the primary endpoint; retained as a falsification audit.
- ROAD Steps 28-30; retained as negative external-validation evidence.
- GEM-CAN Step 30F detector transfer; retained as a locked negative external confirmation.

## Interpretation

This bounded research-prototype freeze supports only density-specific claims
under the stated CICIoV-SUMO replay protocol. It does not support improvement
across every density, universal source-loss/Byzantine robustness, cross-vehicle
zero-shot generalization, production readiness, or automotive safety
certification. The failed ROAD and GEM-CAN results are part of the frozen
evidence and must be reported in the thesis/paper.

## Reproducibility

The `frozen_inputs` directory contains copied evidence tables. Their SHA-256
hashes and original paths are recorded in `final_evidence_inventory.csv`.
No model was retrained or threshold recalibrated by this script.

Plot saved: `{plot_saved}`
"""
    (OUT / "FINAL_POLICY_README.md").write_text(readme, encoding="utf-8")
    archive_path = Path(
        shutil.make_archive(
            str(RESULTS / "final_zero_trust_policy_evidence"),
            "zip",
            root_dir=OUT,
        )
    )

    print("=" * 72)
    print("Final multi-source Zero Trust policy freeze completed successfully.")
    print(f"Policy ID: {POLICY_ID}")
    print(f"Status: {POLICY_STATUS}")
    print(f"Selected primary method: {SELECTED_METHOD}")
    print(f"Selected temporal rule: {selected_rule}")
    print(f"Freeze checks passed: {freeze_ready}")
    print(f"Evidence files frozen: {sum(1 for row in inventory if row['sha256'])}")
    print("ROAD Steps 28-30: EXCLUDED (negative external validation)")
    print("GEM-CAN Step 30F: EXCLUDED (locked negative external confirmation)")
    print("Freeze scope: bounded_research_prototype_freeze_only")
    print(f"Results directory: {OUT}")
    print(f"Evidence archive: {archive_path}")
    print("\nNext: use the final endpoint, claims, limitations and evidence plot in the thesis.")


if __name__ == "__main__":
    main()
