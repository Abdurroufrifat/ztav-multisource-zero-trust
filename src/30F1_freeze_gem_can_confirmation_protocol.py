#!/usr/bin/env python3
"""Step 30F1: freeze the GEM-CAN external-confirmation protocol.

This additive, non-consuming stage converts the successful Step 30F0 audit
into one immutable protocol before the frozen CAN model is allowed to score any
GEM-CAN window.  It hashes the external data, the exact selected window IDs,
the frozen CICIoV2024 model, validation-selected threshold, and training feature
schema.  It also predeclares estimands, uncertainty analysis, acceptance
criteria, and publication claim boundaries.

It does not load the model, calculate predictions, tune a threshold, run SUMO,
or permit Step 31.  The lock is single-use: if a lock already exists, this
script refuses to create a competing protocol.

Run from ``D:\\ztav_project`` only after Step 30F0 succeeds:

    .\\.venv\\Scripts\\python.exe .\\src\\30F1_freeze_gem_can_confirmation_protocol.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def infer_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "src" else here


ROOT = infer_root()
STEP30F0_RESULTS = Path("results/gem_can_schema_capacity_audit")
LOCK_RESULTS = Path("results/gem_can_confirmation_protocol_lock")
LOCK_FILENAME = "GEM_CAN_CONFIRMATION_PROTOCOL_LOCK.json"

MODEL_RELATIVE = Path(
    "models/group_disjoint_w100/group_disjoint_logistic_regression.joblib"
)
THRESHOLD_RELATIVE = Path(
    "results/group_disjoint_w100/group_disjoint_thresholds.json"
)
TRAINING_RELATIVE = Path(
    "data/processed/ciciov2024_windows_w100_group_disjoint_train.csv"
)
FEATURE_BUILDER_RELATIVE = Path("src/03_build_window_dataset.py")

WINDOW_SIZE = 100
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 30_031
CONFIDENCE_LEVEL = 0.95

# Confirmatory gates were fixed before any GEM-CAN model score was calculated.
HIGH_CLUSTER_RECALL_MINIMUM = 0.80
HIGH_CLUSTER_RECALL_CI_LOWER_MINIMUM = 0.60
BENIGN_CLUSTER_FPR_MAXIMUM = 0.05
BENIGN_CLUSTER_FPR_CI_UPPER_MAXIMUM = 0.10

EXPECTED_FEATURE_COUNT = 63
NON_MODEL_COLUMNS = {
    "source_file",
    "window_index",
    "start_row",
    "end_row",
    "split",
    "chronological_split",
    "feature_signature",
    "binary_target",
    "multiclass_target",
}
REQUIRED_WINDOW_COLUMNS = {
    "window_id",
    "capture",
    "window_index",
    "window_size",
    "density_band",
    "binary_target",
    "feature_window_sha256",
    "labelled_window_sha256",
}
REQUIRED_MANIFEST_COLUMNS = {"file", "path", "sha256"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze GEM-CAN confirmation design without scoring its windows."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def latest_step30f0_run(root: Path) -> Path:
    base = root / STEP30F0_RESULTS
    if not base.is_dir():
        raise FileNotFoundError(f"Step 30F0 results not found: {base}")
    candidates = sorted(
        path.parent
        for path in base.glob("run_*/gem_can_schema_capacity_summary.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No complete Step 30F0 run found below {base}")
    return candidates[-1]


def refuse_competing_lock(root: Path) -> None:
    base = root / LOCK_RESULTS
    existing = sorted(base.glob(f"run_*/{LOCK_FILENAME}")) if base.exists() else []
    if existing:
        raise RuntimeError(
            "A GEM-CAN protocol lock already exists. Do not create a competing "
            f"confirmatory protocol: {existing[-1]}"
        )


def resolve_manifest_path(value: str, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def verify_external_files(file_manifest: pd.DataFrame, root: Path) -> list[dict[str, Any]]:
    require_columns(file_manifest.columns, REQUIRED_MANIFEST_COLUMNS, "Step 30F0 file manifest")
    rows: list[dict[str, Any]] = []
    for record in file_manifest.to_dict("records"):
        path = resolve_manifest_path(str(record["path"]), root)
        if not path.is_file():
            raise FileNotFoundError(f"Locked GEM-CAN input is missing: {path}")
        observed = sha256(path)
        expected = str(record["sha256"])
        passed = observed == expected
        rows.append(
            {
                "check": "external_file_sha256",
                "item": str(record["file"]),
                "expected": expected,
                "observed": observed,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"GEM-CAN file changed after Step 30F0: {path}")
    return rows


def load_threshold(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "Logistic Regression" not in payload:
        raise KeyError(f"Logistic Regression threshold missing from {path}")
    threshold = float(payload["Logistic Regression"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid frozen threshold: {threshold}")
    return threshold


def feature_schema(training_path: Path) -> list[str]:
    columns = list(pd.read_csv(training_path, nrows=0).columns)
    features = [column for column in columns if column not in NON_MODEL_COLUMNS]
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} frozen model features, found "
            f"{len(features)} in {training_path}"
        )
    if len(set(features)) != len(features):
        raise ValueError("Frozen feature schema contains duplicate column names")
    expected_prefix = [
        "id_unique_count",
        "id_entropy",
        "dominant_id_fraction",
        "frame_unique_count",
        "frame_unique_fraction",
        "id_change_rate",
        "consecutive_frame_repeat_rate",
    ]
    if features[: len(expected_prefix)] != expected_prefix:
        raise ValueError("Frozen feature schema does not match the Step 06 CAN representation")
    return features


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    refuse_competing_lock(root)
    step30f0 = latest_step30f0_run(root)

    summary_path = step30f0 / "gem_can_schema_capacity_summary.json"
    window_path = step30f0 / "gem_can_natural_window_manifest.csv"
    density_path = step30f0 / "gem_can_natural_density_capacity.csv"
    family_path = step30f0 / "gem_can_attack_family_capacity.csv"
    verdict_path = step30f0 / "gem_can_capacity_verdict.csv"
    file_manifest_path = step30f0 / "gem_can_file_manifest.csv"
    required_step30f0 = [
        summary_path,
        window_path,
        density_path,
        family_path,
        verdict_path,
        file_manifest_path,
    ]
    for path in required_step30f0:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete Step 30F0 run; missing {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_summary_values = {
        "schema_valid": True,
        "pooled_high_density_external_confirmation_feasible": True,
        "confirmation_consumed": False,
        "model_loaded": False,
        "threshold_changed": False,
        "sumo_executed": False,
        "step31_permitted": False,
    }
    checks: list[dict[str, Any]] = []
    for key, expected in required_summary_values.items():
        observed = summary.get(key)
        passed = observed == expected
        checks.append(
            {
                "check": "step30f0_summary_value",
                "item": key,
                "expected": json.dumps(expected),
                "observed": json.dumps(observed),
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(
                f"Step 30F0 prerequisite failed: {key}={observed!r}, expected {expected!r}"
            )

    file_manifest = pd.read_csv(file_manifest_path)
    checks.extend(verify_external_files(file_manifest, root))
    windows = pd.read_csv(window_path)
    require_columns(windows.columns, REQUIRED_WINDOW_COLUMNS, "Step 30F0 window manifest")
    if windows["window_id"].duplicated().any():
        raise ValueError("Step 30F0 window IDs are not unique")
    if windows["window_size"].astype(int).ne(WINDOW_SIZE).any():
        raise ValueError("Step 30F0 contains a non-100-frame window")

    benign = windows[
        windows["capture"].eq("normal")
        & windows["density_band"].eq("benign")
        & windows["binary_target"].astype(int).eq(0)
    ].copy()
    high = windows[
        windows["capture"].eq("attack")
        & windows["density_band"].eq("high_21_100")
        & windows["binary_target"].astype(int).eq(1)
    ].copy()
    if len(benign) < 200 or benign["feature_window_sha256"].nunique() < 200:
        raise RuntimeError("Insufficient locked clean-capture capacity")
    if len(high) < 200 or high["feature_window_sha256"].nunique() < 30:
        raise RuntimeError("Insufficient locked high-density attack capacity")

    benign["confirmatory_role"] = "primary_clean_fpr"
    high["confirmatory_role"] = "primary_high_density_recall"
    locked = pd.concat([benign, high], ignore_index=True)
    locked = locked.sort_values(
        ["confirmatory_role", "capture", "window_index"], kind="stable"
    ).reset_index(drop=True)
    locked["protocol_order"] = range(len(locked))

    model_path = root / MODEL_RELATIVE
    threshold_path = root / THRESHOLD_RELATIVE
    training_path = root / TRAINING_RELATIVE
    feature_builder_path = root / FEATURE_BUILDER_RELATIVE
    for path in (model_path, threshold_path, training_path, feature_builder_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen Step 06 asset: {path}")
    frozen_threshold = load_threshold(threshold_path)
    features = feature_schema(training_path)

    asset_rows = [
        {
            "asset": "frozen_logistic_regression_model",
            "path": str(MODEL_RELATIVE),
            "sha256": sha256(model_path),
        },
        {
            "asset": "validation_selected_thresholds",
            "path": str(THRESHOLD_RELATIVE),
            "sha256": sha256(threshold_path),
        },
        {
            "asset": "frozen_training_feature_schema",
            "path": str(TRAINING_RELATIVE),
            "sha256": sha256(training_path),
        },
        {
            "asset": "frozen_window_feature_builder",
            "path": str(FEATURE_BUILDER_RELATIVE),
            "sha256": sha256(feature_builder_path),
        },
    ]
    assets = pd.DataFrame(asset_rows)

    lock: dict[str, Any] = {
        "protocol": "Step 30F GEM-CAN pooled high-density external confirmation",
        "locked_utc": utc_now(),
        "protocol_status": "locked_before_first_gem_can_model_score",
        "step30f0_run": safe_relative(step30f0, root),
        "step30f0_artifacts": {
            path.name: sha256(path) for path in required_step30f0
        },
        "external_input_files": {
            str(row["file"]): str(row["sha256"])
            for row in file_manifest.to_dict("records")
        },
        "frozen_assets": {
            row["asset"]: {"path": row["path"], "sha256": row["sha256"]}
            for row in asset_rows
        },
        "frozen_model": "group_disjoint_logistic_regression",
        "frozen_threshold": frozen_threshold,
        "feature_count": len(features),
        "feature_names": features,
        "feature_schema_sha256": digest_lines(features),
        "window_protocol": {
            "size": WINDOW_SIZE,
            "overlap": 0,
            "ordering": "stable timestamp order within each original capture",
            "duplicates": "retain observed windows; cluster inference by feature_window_sha256",
            "clean_primary_selection": "capture == normal and density_band == benign",
            "attack_primary_selection": "capture == attack and density_band == high_21_100",
            "attack_capture_benign_windows": "excluded from primary clean-FPR endpoint",
            "locked_window_count": len(locked),
            "clean_window_count": len(benign),
            "clean_unique_feature_clusters": benign["feature_window_sha256"].nunique(),
            "attack_window_count": len(high),
            "attack_unique_feature_clusters": high["feature_window_sha256"].nunique(),
            "locked_window_id_sha256": digest_lines(locked["window_id"]),
            "locked_feature_hash_sequence_sha256": digest_lines(
                locked["feature_window_sha256"]
            ),
        },
        "primary_estimands": {
            "clean_feature_cluster_macro_false_positive_rate": (
                "mean of per-feature-cluster false-positive indicators on the normal capture"
            ),
            "high_density_feature_cluster_macro_recall": (
                "mean of per-feature-cluster detection indicators on natural high-density attack windows"
            ),
        },
        "secondary_estimands": [
            "observed-window FPR and recall",
            "precision, F1, PR-AUC, ROC-AUC and confusion counts",
            "strict-consecutive-2 temporal policy recall and FPR",
            "descriptive mixed-family metrics without family-specific confirmation claims",
            "domain-shift statistics relative to frozen CICIoV2024 training features",
        ],
        "uncertainty": {
            "method": "nonparametric percentile bootstrap over unique feature-window clusters",
            "confidence_level": CONFIDENCE_LEVEL,
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "cluster_key": "feature_window_sha256",
            "individual_windows_are_not_assumed_independent": True,
        },
        "predeclared_acceptance_criteria": {
            "high_cluster_macro_recall_point_minimum": HIGH_CLUSTER_RECALL_MINIMUM,
            "high_cluster_macro_recall_ci_lower_minimum": HIGH_CLUSTER_RECALL_CI_LOWER_MINIMUM,
            "benign_cluster_macro_fpr_point_maximum": BENIGN_CLUSTER_FPR_MAXIMUM,
            "benign_cluster_macro_fpr_ci_upper_maximum": BENIGN_CLUSTER_FPR_CI_UPPER_MAXIMUM,
            "all_four_gates_required": True,
            "secondary_temporal_policy_cannot_rescue_failed_primary_detector": True,
        },
        "failure_policy": {
            "threshold_retuning_forbidden": True,
            "model_retraining_forbidden": True,
            "window_reselection_forbidden": True,
            "failed_result_retained": True,
            "failed_result_interpretation": "external domain-generalisation limitation",
        },
        "claim_boundaries": {
            "allowed_if_primary_passes": (
                "The frozen CICIoV2024 CAN detector generalised to GEM-CAN's pooled natural "
                "high-density attack endpoint while controlling clean-capture false alarms."
            ),
            "allowed_if_primary_fails": (
                "The frozen detector did not meet the predeclared GEM-CAN high-density external "
                "generalisation criteria; the result bounds the system's domain transfer."
            ),
            "always_forbidden": [
                "GEM-CAN confirmed low-density sparse-CAN performance",
                "GEM-CAN confirmed medium-density sparse-CAN performance",
                "GEM-CAN independently confirmed Brake, Steering, or DoS family performance",
                "GEM-CAN confirmed the complete multi-source policy under physical deployment",
            ],
        },
        "confirmation_consumed": False,
        "model_loaded": False,
        "predictions_calculated": False,
        "threshold_changed": False,
        "sumo_executed": False,
        "existing_project_artifacts_changed": 0,
        "step31_permitted": False,
    }
    lock["protocol_sha256"] = canonical_json_sha256(lock)

    out = root / LOCK_RESULTS / run_id()
    out.mkdir(parents=True, exist_ok=False)
    locked.to_csv(out / "gem_can_locked_evaluation_manifest.csv", index=False)
    assets.to_csv(out / "gem_can_locked_assets.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "gem_can_protocol_preflight_checks.csv", index=False)
    (out / "gem_can_locked_feature_columns.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    (out / LOCK_FILENAME).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    summary_out = {
        "protocol_sha256": lock["protocol_sha256"],
        "lock_file": LOCK_FILENAME,
        "locked_windows": len(locked),
        "clean_windows": len(benign),
        "clean_unique_feature_clusters": benign["feature_window_sha256"].nunique(),
        "high_density_attack_windows": len(high),
        "high_density_unique_feature_clusters": high[
            "feature_window_sha256"
        ].nunique(),
        "frozen_threshold": frozen_threshold,
        "confirmation_consumed": False,
        "step31_permitted": False,
    }
    (out / "gem_can_protocol_lock_summary.json").write_text(
        json.dumps(summary_out, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("Step 30F1 GEM-CAN confirmation protocol locked successfully.")
    print("Protocol locked before first GEM-CAN model score: True")
    print("Confirmation consumed: False")
    print("Model loaded: False")
    print("Predictions calculated: False")
    print("Threshold changed: False")
    print("SUMO executed: False")
    print("Existing project artifacts changed: 0")
    print(f"Frozen model: {MODEL_RELATIVE}")
    print(f"Frozen threshold: {frozen_threshold:.12g}")
    print(f"Frozen feature count: {len(features)}")
    print(f"Locked clean windows: {len(benign):,}")
    print(
        "Locked clean unique feature clusters: "
        f"{benign['feature_window_sha256'].nunique():,}"
    )
    print(f"Locked high-density attack windows: {len(high):,}")
    print(
        "Locked high-density unique feature clusters: "
        f"{high['feature_window_sha256'].nunique():,}"
    )
    print(f"Bootstrap: {BOOTSTRAP_REPLICATES:,} cluster replicates, seed={BOOTSTRAP_SEED}")
    print("\nPredeclared primary criteria:")
    print(
        f"  high cluster-macro recall >= {HIGH_CLUSTER_RECALL_MINIMUM:.2f}; "
        f"95% CI lower >= {HIGH_CLUSTER_RECALL_CI_LOWER_MINIMUM:.2f}"
    )
    print(
        f"  clean cluster-macro FPR <= {BENIGN_CLUSTER_FPR_MAXIMUM:.2f}; "
        f"95% CI upper <= {BENIGN_CLUSTER_FPR_CI_UPPER_MAXIMUM:.2f}"
    )
    print("Low/medium and attack-family-specific confirmation claims remain forbidden.")
    print("Do not run Step 31.")
    print(f"\nProtocol SHA-256: {lock['protocol_sha256']}")
    print(f"Results directory: {out}")
    print(f"Protocol lock: {out / LOCK_FILENAME}")
    print(f"Locked manifest: {out / 'gem_can_locked_evaluation_manifest.csv'}")
    print("\nNext: send the terminal result, protocol lock JSON, and locked manifest CSV.")


if __name__ == "__main__":
    main()
