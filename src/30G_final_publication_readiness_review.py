"""Final, read-only publication and bounded-freeze readiness review.

This stage reconciles the development, falsification, efficiency, ROAD, and
locked GEM-CAN evidence before Step 31.  A failed external confirmation is not
converted into a pass: it is retained as a claim boundary.  The only possible
authorization is a bounded research-prototype freeze for thesis/publication;
external generalisation, universal source robustness, production readiness,
and safety certification remain prohibited.

Run from D:\\ztav_project::

    .\\.venv\\Scripts\\python.exe .\\src\\30G_final_publication_readiness_review.py

The script does not train, score, recalibrate, simulate, or modify any existing
project artifact.  It writes one new timestamped results directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC = ROOT / "src"
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUT_ROOT = RESULTS / "final_publication_readiness_review"
OUT = OUT_ROOT / RUN_ID
STEP31 = SRC / "31_freeze_final_zero_trust_policy.py"


EVIDENCE = [
    ("30B_summary", "publication_statistical_summary.json", None, True),
    ("30B_comparisons", "publication_statistical_comparisons.csv", None, True),
    ("30B_hypotheses", "publication_hypothesis_assessment.csv", None, True),
    ("30C_summary", "source_robustness_summary.json", None, True),
    ("30C_hypotheses", "source_robustness_hypothesis_assessment.csv", None, True),
    ("30C_safety", "source_robustness_safety_checks.csv", None, True),
    ("30C2_summary", "source_observability_summary.json", None, True),
    ("30C2_claims", "publication_claim_boundary_register.csv", None, True),
    ("30D_manifest", "publication_efficiency_manifest.json", None, True),
    ("30D_runtime", "publication_runtime_metrics.csv", None, True),
    ("30D2_manifest", "publication_resource_completion_manifest.json", None, True),
    ("30D2_resources", "publication_long_run_resource_metrics.csv", None, True),
    ("ROAD_frozen", "road_acceptance_criteria.csv", "road_frozen_external_confirmation/road_acceptance_criteria.csv", True),
    ("ROAD_context", "signal_context_acceptance_criteria.csv", "road_signal_context_gate/signal_context_acceptance_criteria.csv", True),
    ("ROAD_sparse", "sparse_signal_acceptance_criteria.csv", "road_sparse_signal_event_gate/sparse_signal_acceptance_criteria.csv", True),
    ("30F_protocol", "GEM_CAN_CONFIRMATION_PROTOCOL_LOCK.json", None, True),
    ("30F_outcome", "GEM_CAN_CONFIRMATION_OUTCOME.json", None, True),
    ("30F_acceptance", "gem_can_confirmation_acceptance.csv", None, True),
    ("30F_metrics", "gem_can_primary_cluster_metrics.csv", None, True),
    ("30F_domain_shift", "gem_can_feature_domain_shift.csv", None, True),
    ("graded_endpoint", "graded_primary_endpoint_summary.csv", "graded_zero_trust_policy/graded_primary_endpoint_summary.csv", True),
    ("temporal_rule", "temporal_selected_rule.csv", "temporal_memory_sparse_can_confirmation/temporal_selected_rule.csv", True),
    ("temporal_confirmation", "temporal_confirmation_primary_endpoint.csv", "temporal_memory_sparse_can_confirmation/temporal_confirmation_primary_endpoint.csv", True),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def locate(filename: str, preferred: str | None) -> tuple[Path | None, int, str]:
    if preferred:
        path = RESULTS / preferred
        if path.exists():
            return path, 1, "preferred_path"
    matches = [path for path in RESULTS.rglob(filename) if "final_zero_trust_policy" not in path.parts]
    if not matches:
        return None, 0, "missing"
    matches.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return matches[-1], len(matches), "latest_recursive_match"


def add_check(rows: list[dict[str, Any]], check: str, observed: Any, required: Any, passed: bool) -> None:
    rows.append({"check": check, "observed": observed, "required": required, "passed": bool(passed)})


def build_plot(checks: list[dict[str, Any]], claims: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    passed = sum(bool(row["passed"]) for row in checks)
    failed = len(checks) - passed
    permissions = [row["publication_permission"] for row in claims]
    categories = ["supported", "bounded", "prohibited"]
    counts = [
        sum(value == "supported" for value in permissions),
        sum(value == "bounded" for value in permissions),
        sum(value == "prohibited" for value in permissions),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(["Passed", "Failed"], [passed, failed], color=["#2ca02c", "#d62728"])
    axes[0].set_title("Pre-freeze integrity checks")
    axes[0].set_ylabel("Checks")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(categories, counts, color=["#2ca02c", "#ff7f0e", "#d62728"])
    axes[1].set_title("Publication claim permissions")
    axes[1].set_ylabel("Claims")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Final publication-readiness and claim-boundary review", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "final_publication_readiness_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    if (RESULTS / "final_zero_trust_policy").exists():
        raise RuntimeError("Step 31 has already produced a freeze; do not create a second freeze.")
    if not STEP31.exists():
        raise FileNotFoundError(f"Missing Step 31 script: {STEP31}")
    OUT.mkdir(parents=True, exist_ok=False)

    located: dict[str, Path] = {}
    inventory: list[dict[str, Any]] = []
    for key, filename, preferred, required in EVIDENCE:
        path, count, resolution = locate(filename, preferred)
        located_path = ""
        file_hash = ""
        size = ""
        if path is not None:
            located[key] = path
            located_path = str(path.relative_to(ROOT))
            file_hash = sha256(path)
            size = path.stat().st_size
        inventory.append({
            "evidence_key": key,
            "filename": filename,
            "required": required,
            "status": "available" if path else "missing",
            "resolution": resolution,
            "candidate_count": count,
            "source_path": located_path,
            "sha256": file_hash,
            "size_bytes": size,
        })

    checks: list[dict[str, Any]] = []
    missing = [row["evidence_key"] for row in inventory if row["required"] and row["status"] != "available"]
    add_check(checks, "all_required_evidence_available", len(missing), 0, not missing)
    if missing:
        write_csv(OUT / "final_publication_evidence_registry.csv", inventory,
                  ["evidence_key", "filename", "required", "status", "resolution", "candidate_count", "source_path", "sha256", "size_bytes"])
        raise FileNotFoundError("Missing required publication evidence: " + ", ".join(missing))

    protocol = read_json(located["30F_protocol"])
    outcome = read_json(located["30F_outcome"])
    acceptance = read_csv(located["30F_acceptance"])
    statistics = read_json(located["30B_summary"])
    robustness = read_json(located["30C_summary"])
    observability = read_json(located["30C2_summary"])
    efficiency = read_json(located["30D_manifest"])
    resources = read_json(located["30D2_manifest"])

    stored_protocol_hash = str(protocol.get("protocol_sha256", ""))
    unsigned_protocol = dict(protocol)
    unsigned_protocol.pop("protocol_sha256", None)
    protocol_hash_valid = stored_protocol_hash == canonical_json_sha256(unsigned_protocol)
    stored_outcome_hash = str(outcome.get("outcome_sha256", ""))
    unsigned_outcome = dict(outcome)
    unsigned_outcome.pop("outcome_sha256", None)
    outcome_hash_valid = stored_outcome_hash == canonical_json_sha256(unsigned_outcome)
    add_check(checks, "step30f_protocol_hash_valid", protocol_hash_valid, True, protocol_hash_valid)
    add_check(checks, "step30f_outcome_hash_valid", outcome_hash_valid, True, outcome_hash_valid)
    same_protocol = stored_protocol_hash and stored_protocol_hash == str(outcome.get("protocol_sha256", ""))
    add_check(checks, "step30f_protocol_matches_outcome", same_protocol, True, bool(same_protocol))
    add_check(checks, "step30f_confirmation_consumed_once", outcome.get("confirmation_consumed"), True, outcome.get("confirmation_consumed") is True)
    no_tuning = all(outcome.get(key) is False for key in ("model_retrained", "threshold_changed", "window_reselected", "sumo_executed"))
    add_check(checks, "step30f_no_post_confirmation_tuning", no_tuning, True, no_tuning)
    acceptance_complete = len(acceptance) == 4 and sum(as_bool(row.get("passed")) for row in acceptance) == 2
    add_check(checks, "step30f_acceptance_table_retains_two_failures", acceptance_complete, True, acceptance_complete)
    negative_retained = outcome.get("confirmatory_passed") is False and float(outcome.get("high_density_cluster_macro_recall", -1)) == 0.0
    add_check(checks, "step30f_negative_external_result_retained", negative_retained, True, negative_retained)

    add_check(checks, "30B_used_independent_run_level_units", not statistics.get("window_level_pseudoreplication_used", True), True,
              not statistics.get("window_level_pseudoreplication_used", True))
    add_check(checks, "30C_failed_h4_retained", robustness.get("h4_development_assessment"), "development_not_supported",
              robustness.get("h4_development_assessment") == "development_not_supported")
    add_check(checks, "30C_failed_h5_retained", robustness.get("h5_development_assessment"), "not_supported_under_undetected_byzantine_source",
              robustness.get("h5_development_assessment") == "not_supported_under_undetected_byzantine_source")
    add_check(checks, "30C2_crosschecks_passed", observability.get("all_step30c_crosschecks_passed"), True,
              observability.get("all_step30c_crosschecks_passed") is True)
    add_check(checks, "30C2_undetected_false_healthy_claim_prohibited", observability.get("undetected_false_healthy_claim_permitted"), False,
              observability.get("undetected_false_healthy_claim_permitted") is False)
    add_check(checks, "30D_changed_no_existing_artifacts", efficiency.get("existing_project_artifacts_changed"), 0,
              efficiency.get("existing_project_artifacts_changed") == 0)
    add_check(checks, "30D2_changed_no_existing_artifacts", resources.get("existing_project_artifacts_changed"), 0,
              resources.get("existing_project_artifacts_changed") == 0)

    step31_text = STEP31.read_text(encoding="utf-8")
    integration_markers = [
        "FINAL_FREEZE_AUTHORIZATION.json",
        "GEM_CAN_CONFIRMATION_OUTCOME.json",
        "external_generalization_not_confirmed",
        "bounded_research_prototype_freeze_only",
    ]
    integration_ready = all(marker in step31_text for marker in integration_markers)
    add_check(checks, "step31_integrates_final_claim_boundaries", integration_ready, True, integration_ready)

    claims = [
        {
            "claim_id": "C1_multisource_improvement",
            "statement": "Multi-source context improves end-to-end detection over eligible single-source baselines.",
            "evidence_status": statistics.get("h1_development_assessment", "unknown"),
            "publication_permission": "bounded",
            "required_wording": "Report density-specific paired results; do not claim improvement across every density.",
        },
        {
            "claim_id": "C2_in_domain_operating_point",
            "statement": "The graded persistent policy met the declared in-domain F1 and mean-FPR criteria.",
            "evidence_status": statistics.get("h2_development_assessment", "unknown"),
            "publication_permission": "bounded",
            "required_wording": "Describe this as development/in-domain replay evidence, not independent deployment confirmation.",
        },
        {
            "claim_id": "C3_detected_source_compromise",
            "statement": "Detected source compromise or explicit high-risk conflict can trigger restrictive action.",
            "evidence_status": observability.get("detected_compromise_claim_scope", "unknown"),
            "publication_permission": "bounded",
            "required_wording": "Condition the claim on observable integrity or conflict evidence.",
        },
        {
            "claim_id": "C4_universal_source_independence",
            "statement": "No single false-healthy source can independently force ALLOW under all modeled conditions.",
            "evidence_status": robustness.get("h5_development_assessment", "unknown"),
            "publication_permission": "prohibited",
            "required_wording": "Report undetected Byzantine false-healthy evidence as a residual vulnerability.",
        },
        {
            "claim_id": "C5_external_generalisation",
            "statement": "The frozen CAN detector generalises to unseen vehicle CAN schemas.",
            "evidence_status": "not_supported_by_locked_gem_can_confirmation",
            "publication_permission": "prohibited",
            "required_wording": "Report GEM-CAN recall=0.0 and the measured domain shift as a negative external result.",
        },
        {
            "claim_id": "C6_efficiency",
            "statement": "The research implementation has measured inference cost on the stated research hardware.",
            "evidence_status": "supported_with_scope_limitations",
            "publication_permission": "supported",
            "required_wording": "State that raw parsing, acquisition, transport, SUMO runtime, and ECU deployment were excluded.",
        },
        {
            "claim_id": "C7_production_readiness",
            "statement": "The method is production-ready or safety-certified automotive software.",
            "evidence_status": "not_evaluated_and_not_permitted",
            "publication_permission": "prohibited",
            "required_wording": "Research prototype only; no production or safety-certification claim.",
        },
    ]

    hypotheses = [
        {"hypothesis": "H1", "final_status": statistics.get("h1_development_assessment"), "confirmatory": False,
         "publication_interpretation": "density-dependent development evidence"},
        {"hypothesis": "H2", "final_status": statistics.get("h2_development_assessment"), "confirmatory": False,
         "publication_interpretation": "in-domain development operating point only"},
        {"hypothesis": "H3", "final_status": statistics.get("h3_development_assessment"), "confirmatory": False,
         "publication_interpretation": "descriptive sparse-CAN improvement"},
        {"hypothesis": "H4", "final_status": robustness.get("h4_development_assessment"), "confirmatory": False,
         "publication_interpretation": "source-loss non-inferiority not supported"},
        {"hypothesis": "H5", "final_status": robustness.get("h5_development_assessment"), "confirmatory": False,
         "publication_interpretation": "universal Byzantine-source claim not supported"},
        {"hypothesis": "External GEM-CAN", "final_status": "predeclared_confirmation_failed", "confirmatory": True,
         "publication_interpretation": "valid negative external confirmation; no retuning or rerun"},
    ]

    all_checks_passed = all(bool(row["passed"]) for row in checks)
    authorization = {
        "review": "Step 30G final publication-readiness and claim-boundary review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_type": "bounded_research_prototype_freeze_only",
        "step31_permitted": all_checks_passed,
        "step31_sha256": sha256(STEP31),
        "negative_external_result_retained": negative_retained,
        "gem_can_protocol_sha256": stored_protocol_hash,
        "gem_can_outcome_sha256": stored_outcome_hash,
        "confirmatory_external_generalisation_claim_permitted": False,
        "universal_source_robustness_claim_permitted": False,
        "production_deployment_permitted": False,
        "automotive_safety_certification_claim_permitted": False,
        "publication_submission_scope": "bounded research-prototype evidence with all negative and failed hypotheses retained",
        "instruction": "Run Step 31 once only if step31_permitted is true. Do not tune or rerun GEM-CAN.",
    }
    authorization["authorization_sha256"] = canonical_json_sha256(authorization)

    write_csv(OUT / "final_publication_evidence_registry.csv", inventory,
              ["evidence_key", "filename", "required", "status", "resolution", "candidate_count", "source_path", "sha256", "size_bytes"])
    write_csv(OUT / "final_readiness_checks.csv", checks, ["check", "observed", "required", "passed"])
    write_csv(OUT / "final_publication_claim_boundaries.csv", claims,
              ["claim_id", "statement", "evidence_status", "publication_permission", "required_wording"])
    write_csv(OUT / "final_hypothesis_verdicts.csv", hypotheses,
              ["hypothesis", "final_status", "confirmatory", "publication_interpretation"])
    (OUT / "FINAL_FREEZE_AUTHORIZATION.json").write_text(json.dumps(authorization, indent=2), encoding="utf-8")
    plot_saved = build_plot(checks, claims)
    (OUT / "FINAL_PUBLICATION_READINESS_REVIEW.md").write_text(
        f"""# Final publication-readiness review

Authorization: `{authorization['authorization_type']}`  
Step 31 permitted: `{all_checks_passed}`  
External generalisation confirmed: `False`  
Universal source robustness confirmed: `False`  
Production/safety certification permitted: `False`

The one-time GEM-CAN confirmation is retained as a valid negative result
(cluster-macro recall 0.0; clean cluster-macro FPR 0.0). No model, threshold,
window definition, or simulator configuration was changed. The final freeze may
therefore describe only the evaluated research prototype and its explicit claim
boundaries. Failed H1 density conditions, failed H4/H5 tests, ROAD failures, and
the GEM-CAN transfer failure must all appear in the thesis/paper.

Summary plot saved: `{plot_saved}`
""",
        encoding="utf-8",
    )
    archive = Path(shutil.make_archive(str(OUT_ROOT / f"final_publication_readiness_review_{RUN_ID}"), "zip", root_dir=OUT))

    print("=" * 76)
    print("Step 30G final publication-readiness review completed successfully.")
    print("Existing project artifacts changed: 0")
    print(f"Required evidence files: {len(EVIDENCE)}")
    print(f"Readiness checks: {sum(row['passed'] for row in checks)}/{len(checks)} passed")
    print(f"Step 31 bounded research-prototype freeze permitted: {all_checks_passed}")
    print("External generalisation claim permitted: False")
    print("Universal source-robustness claim permitted: False")
    print("Production/safety-certification claim permitted: False")
    print(f"Results directory: {OUT}")
    print(f"Results archive: {archive}")
    if all_checks_passed:
        print("\nNext: run Step 31 once. Do not tune or rerun GEM-CAN.")
    else:
        print("\nDo not run Step 31; review final_readiness_checks.csv.")


if __name__ == "__main__":
    main()
