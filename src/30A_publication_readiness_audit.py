"""Read-only publication-readiness audit for the ZTAV research project.

The script never trains a model, changes a threshold, edits a prior result, or
modifies data. It reads the existing project and writes a new timestamped audit
under results/publication_readiness_audit/.

Run from D:\\ztav_project:
    .\\.venv\\Scripts\\python.exe .\\src\\30A_publication_readiness_audit.py

Do not run Step 31 until this audit and Steps 30B-30E have been reviewed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
AUDIT_ROOT = RESULTS / "publication_readiness_audit"
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUT = AUDIT_ROOT / RUN_ID

MAX_CSV_BYTES_FOR_ROW_COUNT = 10 * 1024 * 1024
MAX_HASH_BYTES = 512 * 1024 * 1024


REQUIREMENTS = [
    {
        "id": "R01",
        "category": "data_provenance",
        "requirement": "CICIoV schema and class-distribution audit",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/ciciov2024_file_summary.csv", "**/ciciov2024_class_distribution.csv"],
        "rationale": "Documents source files, schema consistency and class counts.",
    },
    {
        "id": "R02",
        "category": "leakage_control",
        "requirement": "Signature audit and group-disjoint split manifest",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/ciciov2024_signature_audit.csv", "**/ciciov2024_signature_split_manifest.csv"],
        "rationale": "Prevents identical CAN signatures crossing evaluation partitions.",
    },
    {
        "id": "R03",
        "category": "baseline_comparison",
        "requirement": "Conventional binary baseline results",
        "level": "required",
        "stage": "existing",
        "mode": "any",
        "patterns": ["**/binary_test_metrics.csv", "**/binary_validation_metrics.csv"],
        "rationale": "Provides Logistic Regression, Random Forest and XGBoost comparators.",
    },
    {
        "id": "R04",
        "category": "leakage_control",
        "requirement": "Group-disjoint CAN evaluation",
        "level": "required",
        "stage": "existing",
        "mode": "any",
        "patterns": ["**/group_disjoint*metrics*.csv", "**/group_disjoint*summary*.csv"],
        "rationale": "Tests the CAN model without repeated-signature leakage.",
    },
    {
        "id": "R05",
        "category": "sanity_testing",
        "requirement": "Shuffled-label sanity test",
        "level": "required",
        "stage": "existing",
        "mode": "any",
        "patterns": ["**/*shuffled*label*.csv", "**/*sanity*test*.csv", "**/group_disjoint*metrics*.csv"],
        "rationale": "Checks whether performance remains implausibly high under destroyed labels.",
    },
    {
        "id": "R06",
        "category": "stability",
        "requirement": "Repeated-seed validation and threshold sensitivity",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/per_seed_metrics.csv", "**/threshold_sensitivity.csv"],
        "rationale": "Quantifies seed variability and operating-point sensitivity.",
    },
    {
        "id": "R07",
        "category": "robustness",
        "requirement": "Attack-severity and subtle-attack sweep",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/severity_sweep_aggregate.csv", "**/detection_boundaries.csv"],
        "rationale": "Identifies minimum detectable severity and failure boundaries.",
    },
    {
        "id": "R08",
        "category": "multisource_evaluation",
        "requirement": "CICIoV plus SUMO hybrid replay endpoint",
        "level": "required",
        "stage": "existing",
        "mode": "any",
        "patterns": ["**/hybrid_aggregate_metrics.csv", "**/hybrid_per_seed_metrics.csv"],
        "rationale": "Evaluates the end-to-end multi-source context-aware policy.",
    },
    {
        "id": "R09",
        "category": "ablation",
        "requirement": "Feature-family and source-family ablation",
        "level": "required",
        "stage": "existing",
        "mode": "any",
        "patterns": ["**/ablation_cross_domain_summary.csv", "**/ablation_all_metrics.csv"],
        "rationale": "Shows which evidence families drive in-domain and cross-domain performance.",
    },
    {
        "id": "R10",
        "category": "domain_shift",
        "requirement": "External HCRL/Car-Hacking domain-shift evaluation",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/external_zero_shot_metrics.csv", "**/external_feature_domain_shift.csv"],
        "rationale": "Evaluates cross-dataset generalization and feature shift.",
    },
    {
        "id": "R11",
        "category": "availability",
        "requirement": "Session-normalized FPR and startup-poisoning audit",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/session_gate_fpr_sensitivity.csv", "**/bootstrap_poisoning_aggregate.csv"],
        "rationale": "Measures availability cost and baseline-poisoning behaviour.",
    },
    {
        "id": "R12",
        "category": "falsification",
        "requirement": "Vehicle-state counterfactual falsification",
        "level": "required",
        "stage": "existing",
        "mode": "any",
        "patterns": ["**/counterfactual_aggregate_metrics.csv", "**/counterfactual_phase_summary.csv"],
        "rationale": "Tests whether performance depends on reported rather than physical state.",
    },
    {
        "id": "R13",
        "category": "sparse_attack",
        "requirement": "Multiscale and temporal sparse-CAN confirmation",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/multiscale_density_recall.csv", "**/temporal_confirmation_primary_endpoint.csv"],
        "rationale": "Reports sparse-attack limits using a frozen temporal rule.",
    },
    {
        "id": "R14",
        "category": "graded_policy",
        "requirement": "Graded Zero Trust primary endpoint and action distribution",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": ["**/graded_primary_endpoint_summary.csv", "**/graded_action_distribution.csv"],
        "rationale": "Supports the final security-availability operating point.",
    },
    {
        "id": "R15",
        "category": "external_validation",
        "requirement": "ROAD frozen confirmation and documented post-hoc failures",
        "level": "required",
        "stage": "existing",
        "mode": "all",
        "patterns": [
            "**/road_acceptance_criteria.csv",
            "**/signal_context_acceptance_criteria.csv",
            "**/sparse_signal_acceptance_criteria.csv",
        ],
        "rationale": "Preserves negative external results and prevents selective reporting.",
    },
    {
        "id": "R16",
        "category": "statistical_inference",
        "requirement": "Run-level confidence intervals, paired tests, effect sizes and Holm correction",
        "level": "required",
        "stage": "30B",
        "mode": "all",
        "patterns": [
            "**/publication_run_level_metrics.csv",
            "**/publication_statistical_comparisons.csv",
            "**/publication_confidence_intervals.csv",
        ],
        "rationale": "Required to support inferential claims without window-level pseudoreplication.",
    },
    {
        "id": "R17",
        "category": "source_robustness",
        "requirement": "Missing, stale, compromised and conflicting-source experiment",
        "level": "required",
        "stage": "30C",
        "mode": "all",
        "patterns": [
            "**/source_robustness_run_metrics.csv",
            "**/source_robustness_safety_checks.csv",
        ],
        "rationale": "Tests the central claim that no single source is implicitly trusted.",
    },
    {
        "id": "R18",
        "category": "efficiency",
        "requirement": "Latency, throughput, CPU, memory and model-size evaluation",
        "level": "required",
        "stage": "30D",
        "mode": "all",
        "patterns": [
            "**/publication_runtime_metrics.csv",
            "**/publication_resource_summary.csv",
        ],
        "rationale": "Required for deployability discussion and honest hardware boundaries.",
    },
    {
        "id": "R19",
        "category": "final_confirmation",
        "requirement": "Untouched final confirmation of locked policy and baselines",
        "level": "required",
        "stage": "30E",
        "mode": "all",
        "patterns": [
            "**/final_confirmation_manifest.json",
            "**/final_confirmation_primary_endpoint.csv",
            "**/final_confirmation_acceptance.csv",
        ],
        "rationale": "Separates final confirmation from all inspected development evidence.",
    },
    {
        "id": "R20",
        "category": "reproducibility",
        "requirement": "Environment lock, data manifest and one-command replication guide",
        "level": "required",
        "stage": "pre_freeze",
        "mode": "all",
        "patterns": [
            "../requirements*.txt",
            "../README*.md",
            "**/final_confirmation_manifest.json",
        ],
        "rationale": "Allows examiners and reviewers to reproduce the declared experiment.",
    },
]


EXPERIMENT_REGISTRY = [
    {
        "step": "30B",
        "experiment": "Publication statistical confirmation",
        "status": "pending",
        "changes_existing_artifacts": False,
        "primary_output": "publication_statistical_comparisons.csv",
        "decision_rule": "Paired run-level inference with clustered 95% CI and Holm correction",
    },
    {
        "step": "30C",
        "experiment": "Missing and compromised source robustness",
        "status": "pending",
        "changes_existing_artifacts": False,
        "primary_output": "source_robustness_safety_checks.csv",
        "decision_rule": "No source independently forces ALLOW; report F1 loss and unsafe-allow rate",
    },
    {
        "step": "30D",
        "experiment": "Runtime and resource evaluation",
        "status": "pending",
        "changes_existing_artifacts": False,
        "primary_output": "publication_resource_summary.csv",
        "decision_rule": "Report median/p95/p99 latency, throughput, CPU, memory and model size",
    },
    {
        "step": "30E",
        "experiment": "Untouched final confirmation",
        "status": "pending",
        "changes_existing_artifacts": False,
        "primary_output": "final_confirmation_acceptance.csv",
        "decision_rule": "Evaluate locked H1-H5 once without post-hoc retuning",
    },
    {
        "step": "31",
        "experiment": "Final policy and evidence freeze",
        "status": "blocked_until_30B_30E_complete",
        "changes_existing_artifacts": False,
        "primary_output": "final_zero_trust_policy_evidence.zip",
        "decision_rule": "Run only after pre-freeze review",
    },
]


CLAIMS = [
    {
        "claim_id": "C1",
        "claim": "Multi-source context improves end-to-end detection over eligible single-source baselines.",
        "required_evidence": "30B paired effect size, 95% CI and adjusted p-value",
        "current_status": "pending_statistical_confirmation",
    },
    {
        "claim_id": "C2",
        "claim": "The graded persistent policy maintains mean FPR <= 0.05 and F1 >= 0.90 across declared densities.",
        "required_evidence": "30E untouched final primary endpoint",
        "current_status": "supported_in_development_pending_final_confirmation",
    },
    {
        "claim_id": "C3",
        "claim": "No individual context source is implicitly trusted or can independently force ALLOW.",
        "required_evidence": "30C compromised/conflicting-source safety checks",
        "current_status": "pending_source_robustness_experiment",
    },
    {
        "claim_id": "C4",
        "claim": "The method has measured computational cost on the research hardware.",
        "required_evidence": "30D latency, throughput, CPU and memory tables",
        "current_status": "pending_efficiency_experiment",
    },
    {
        "claim_id": "C5",
        "claim": "The method universally generalizes across vehicles and CAN schemas.",
        "required_evidence": "Not available; contradicted by ROAD external validation",
        "current_status": "prohibited_claim",
    },
    {
        "claim_id": "C6",
        "claim": "The prototype is production ready or safety certified.",
        "required_evidence": "Outside project scope",
        "current_status": "prohibited_claim",
    },
]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def file_hash(path: Path) -> str:
    if path.stat().st_size > MAX_HASH_BYTES:
        return "not_hashed_over_512MiB"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def matches_for_pattern(pattern: str) -> list[Path]:
    if pattern.startswith("../"):
        return sorted(ROOT.glob(pattern[3:]))
    return sorted(RESULTS.glob(pattern))


def evaluate_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
    match_groups = [matches_for_pattern(pattern) for pattern in requirement["patterns"]]
    if requirement["mode"] == "all":
        passed = all(bool(group) for group in match_groups)
    else:
        passed = any(bool(group) for group in match_groups)
    paths = []
    for group in match_groups:
        paths.extend(relative(path) for path in group)
    paths = sorted(set(paths))
    return {
        "requirement_id": requirement["id"],
        "category": requirement["category"],
        "requirement": requirement["requirement"],
        "level": requirement["level"],
        "expected_stage": requirement["stage"],
        "status": "available" if passed else "missing",
        "matched_files": ";".join(paths),
        "rationale": requirement["rationale"],
    }


def inventory_files(base: Path, kind: str, hash_files: bool) -> list[dict[str, Any]]:
    if not base.exists():
        return []
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            {
                "artifact_kind": kind,
                "path": relative(path),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": file_hash(path) if hash_files else "not_hashed_metadata_only",
            }
        )
    return rows


def metric_inventory() -> list[dict[str, Any]]:
    rows = []
    if not RESULTS.exists():
        return rows
    for path in sorted(RESULTS.rglob("*.csv")):
        if AUDIT_ROOT in path.parents:
            continue
        header: list[str] = []
        row_count: str | int = "not_counted_large_file"
        error = ""
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                if path.stat().st_size <= MAX_CSV_BYTES_FOR_ROW_COUNT:
                    row_count = sum(1 for _ in reader)
        except Exception as exc:  # audit must record, not conceal, unreadable files
            error = f"{type(exc).__name__}: {exc}"
            row_count = "unreadable"
        rows.append(
            {
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "row_count": row_count,
                "column_count": len(header),
                "columns": ";".join(header),
                "read_error": error,
            }
        )
    return rows


def data_inventory() -> list[dict[str, Any]]:
    rows = []
    if not DATA.exists():
        return rows
    for path in sorted(item for item in DATA.rglob("*") if item.is_file()):
        stat = path.stat()
        parts = [part.lower() for part in path.parts]
        if "processed" in parts:
            role = "processed"
        elif "external" in parts:
            role = "external"
        else:
            role = "raw_or_source"
        rows.append(
            {
                "path": relative(path),
                "dataset_role": role,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return rows


def build_plot(requirement_rows: list[dict[str, Any]]) -> bool:
    mpl_cache = OUT / ".matplotlib_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    counts = Counter(row["status"] for row in requirement_rows)
    stage_missing = Counter(
        row["expected_stage"] for row in requirement_rows if row["status"] == "missing"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(["Available", "Missing"], [counts["available"], counts["missing"]], color=["#2ca02c", "#d62728"])
    axes[0].set_ylabel("Requirements")
    axes[0].set_title("Publication evidence readiness")
    axes[0].grid(axis="y", alpha=0.25)

    labels = list(stage_missing)
    values = [stage_missing[label] for label in labels]
    axes[1].bar(labels, values, color="#ff7f0e")
    axes[1].set_ylabel("Missing requirements")
    axes[1].set_title("Remaining evidence by stage")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Master's thesis and SCI/IEEE publication-readiness audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "publication_readiness_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.rmtree(mpl_cache, ignore_errors=True)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)

    requirement_rows = [evaluate_requirement(item) for item in REQUIREMENTS]
    code_rows = inventory_files(SRC, "source_code", hash_files=True)
    model_rows = inventory_files(MODELS, "model", hash_files=True)
    result_rows = inventory_files(RESULTS, "result", hash_files=False)
    artifact_rows = code_rows + model_rows + result_rows
    dataset_rows = data_inventory()
    metrics = metric_inventory()

    required = [row for row in requirement_rows if row["level"] == "required"]
    available = [row for row in required if row["status"] == "available"]
    missing = [row for row in required if row["status"] == "missing"]
    pending_stages = sorted({row["expected_stage"] for row in missing})
    step31_present = (SRC / "31_freeze_final_zero_trust_policy.py").exists()
    step31_has_run = (RESULTS / "final_zero_trust_policy").exists()
    publication_ready = len(missing) == 0 and not step31_has_run

    summary = {
        "audit_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "audit_is_read_only_for_existing_artifacts": True,
        "required_requirements": len(required),
        "available_requirements": len(available),
        "missing_requirements": len(missing),
        "pending_stages": pending_stages,
        "publication_ready_for_final_freeze": publication_ready,
        "step31_script_present": step31_present,
        "step31_already_run": step31_has_run,
        "instruction": "Do not run Step 31 until Steps 30B-30E and pre-freeze reproducibility checks are complete.",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "inventory_counts": {
            "source_files": len(code_rows),
            "model_files": len(model_rows),
            "result_files": len(result_rows),
            "dataset_files": len(dataset_rows),
            "metric_csv_files": len(metrics),
        },
    }

    write_csv(
        OUT / "publication_readiness_checklist.csv",
        requirement_rows,
        ["requirement_id", "category", "requirement", "level", "expected_stage", "status", "matched_files", "rationale"],
    )
    write_csv(
        OUT / "missing_publication_requirements.csv",
        missing,
        ["requirement_id", "category", "requirement", "level", "expected_stage", "status", "matched_files", "rationale"],
    )
    write_csv(
        OUT / "experiment_evidence_inventory.csv",
        artifact_rows,
        ["artifact_kind", "path", "extension", "size_bytes", "modified_utc", "sha256"],
    )
    write_csv(
        OUT / "dataset_inventory.csv",
        dataset_rows,
        ["path", "dataset_role", "extension", "size_bytes", "modified_utc"],
    )
    write_csv(
        OUT / "available_metrics_inventory.csv",
        metrics,
        ["path", "size_bytes", "row_count", "column_count", "columns", "read_error"],
    )
    write_csv(
        OUT / "proposed_experiment_registry.csv",
        EXPERIMENT_REGISTRY,
        ["step", "experiment", "status", "changes_existing_artifacts", "primary_output", "decision_rule"],
    )
    write_csv(
        OUT / "publication_claims_register.csv",
        CLAIMS,
        ["claim_id", "claim", "required_evidence", "current_status"],
    )
    (OUT / "publication_readiness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_saved = build_plot(requirement_rows)
    readme = f"""# Publication-readiness audit: {RUN_ID}

This audit read the existing project and wrote only this new timestamped result
directory. It did not train, recalibrate, overwrite or delete anything.

## Decision

- Required requirements: {len(required)}
- Available: {len(available)}
- Missing: {len(missing)}
- Publication ready for Step 31: {publication_ready}
- Pending stages: {', '.join(pending_stages) if pending_stages else 'none'}

Step 31 present: {step31_present}  
Step 31 already run: {step31_has_run}

## Next action

Review `missing_publication_requirements.csv`. Complete Steps 30B, 30C, 30D
and 30E in order. Do not run Step 31 until the final pre-freeze review.

Summary plot saved: {plot_saved}
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    archive = Path(
        shutil.make_archive(
            str(AUDIT_ROOT / f"publication_readiness_audit_{RUN_ID}"),
            "zip",
            root_dir=OUT,
        )
    )

    print("=" * 76)
    print("Step 30A publication-readiness audit completed successfully.")
    print("Existing project artifacts changed: 0")
    print(f"Required publication requirements: {len(required)}")
    print(f"Available evidence requirements: {len(available)}")
    print(f"Missing evidence requirements: {len(missing)}")
    print(f"Pending stages: {', '.join(pending_stages) if pending_stages else 'none'}")
    print(f"Publication ready for Step 31: {publication_ready}")
    print(f"Audit directory: {OUT}")
    print(f"Audit archive: {archive}")
    print("\nNext: send the terminal result and missing_publication_requirements.csv.")
    print("Do not run Step 31 yet.")


if __name__ == "__main__":
    main()
