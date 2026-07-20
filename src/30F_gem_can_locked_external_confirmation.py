#!/usr/bin/env python3
"""Step 30F: one-time locked GEM-CAN external confirmation.

This script may run only after Step 30F1 creates an immutable protocol lock. It
verifies every locked data/model/code hash, reconstructs the exact selected
100-frame windows, writes a consumption-start marker, and only then loads the
frozen group-disjoint CICIoV2024 logistic-regression model.

No training, calibration, threshold selection, window reselection, or SUMO
execution is allowed.  Both passing and failing outcomes are retained.  A
started or completed confirmation cannot be rerun by this script.

Run from ``D:\\ztav_project``:

    .\\.venv\\Scripts\\python.exe .\\src\\30F_gem_can_locked_external_confirmation.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ztav_matplotlib_cache")
)
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def infer_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "src" else here


ROOT = infer_root()
LOCK_RESULTS = Path("results/gem_can_confirmation_protocol_lock")
OUTCOME_RESULTS = Path("results/gem_can_frozen_external_confirmation")
LOCK_FILENAME = "GEM_CAN_CONFIRMATION_PROTOCOL_LOCK.json"
OUTCOME_FILENAME = "GEM_CAN_CONFIRMATION_OUTCOME.json"
STARTED_FILENAME = "GEM_CAN_CONFIRMATION_CONSUMPTION_STARTED.json"
WINDOW_SIZE = 100
CAN_COLUMNS = ["arbitration_id", "dlc"] + [f"data{i}" for i in range(8)]
REQUIRED_LOCKED_COLUMNS = {
    "window_id",
    "capture",
    "window_index",
    "binary_target",
    "density_band",
    "attack_families",
    "feature_window_sha256",
    "labelled_window_sha256",
    "confirmatory_role",
    "protocol_order",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the one-time locked GEM-CAN external confirmation."
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


def digest_text(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def latest_protocol_lock(root: Path) -> tuple[Path, Path]:
    base = root / LOCK_RESULTS
    locks = sorted(base.glob(f"run_*/{LOCK_FILENAME}")) if base.exists() else []
    if len(locks) != 1:
        raise RuntimeError(
            f"Expected exactly one GEM-CAN protocol lock, found {len(locks)} below {base}"
        )
    return locks[0].parent, locks[0]


def refuse_repeat(root: Path) -> None:
    base = root / OUTCOME_RESULTS
    if not base.exists():
        return
    started = sorted(base.glob(f"run_*/{STARTED_FILENAME}"))
    outcomes = sorted(base.glob(f"run_*/{OUTCOME_FILENAME}"))
    if started or outcomes:
        evidence = outcomes[-1] if outcomes else started[-1]
        raise RuntimeError(
            "GEM-CAN confirmation has already started or completed; a rerun would "
            f"invalidate the one-time protocol: {evidence}"
        )


def verify_protocol_hash(lock: dict[str, Any]) -> str:
    stored = str(lock.get("protocol_sha256", ""))
    unsigned = dict(lock)
    unsigned.pop("protocol_sha256", None)
    observed = canonical_json_sha256(unsigned)
    if not stored or stored != observed:
        raise RuntimeError("GEM-CAN protocol lock JSON hash is invalid")
    return stored


def resolve_path(value: str, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def verify_locked_assets(lock: dict[str, Any], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for asset, record in lock["frozen_assets"].items():
        path = resolve_path(str(record["path"]), root)
        if not path.is_file():
            raise FileNotFoundError(f"Locked asset missing: {path}")
        observed = sha256(path)
        if observed != str(record["sha256"]):
            raise RuntimeError(f"Locked asset changed: {asset} ({path})")
        paths[asset] = path
    return paths


def verify_step30f0_artifacts(lock: dict[str, Any], root: Path) -> Path:
    run = resolve_path(str(lock["step30f0_run"]), root)
    if not run.is_dir():
        raise FileNotFoundError(f"Locked Step 30F0 run missing: {run}")
    for filename, expected in lock["step30f0_artifacts"].items():
        path = run / filename
        if not path.is_file() or sha256(path) != str(expected):
            raise RuntimeError(f"Locked Step 30F0 artifact changed or missing: {path}")
    return run


def external_paths(lock: dict[str, Any], step30f0: Path, root: Path) -> dict[str, Path]:
    manifest = pd.read_csv(step30f0 / "gem_can_file_manifest.csv")
    require_columns(manifest.columns, {"file", "path", "sha256"}, "Step 30F0 file manifest")
    paths: dict[str, Path] = {}
    locked_hashes = lock["external_input_files"]
    for record in manifest.to_dict("records"):
        filename = str(record["file"])
        path = resolve_path(str(record["path"]), root)
        if filename not in locked_hashes:
            raise RuntimeError(f"External file absent from protocol lock: {filename}")
        observed = sha256(path) if path.is_file() else "missing"
        if observed != str(locked_hashes[filename]):
            raise RuntimeError(f"Locked external input changed or missing: {path}")
        paths[filename] = path
    if set(paths) != set(locked_hashes):
        raise RuntimeError("External file manifest does not exactly match protocol lock")
    return paths


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen feature builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_frame(path: Path, attack_capture: bool) -> tuple[pd.DataFrame, int]:
    source = pd.read_csv(path, dtype=str, keep_default_na=True)
    rename = {"tiTestaTp": "timestamp", "Attack_Type": "attack_type"}
    source = source.rename(columns=rename)
    required = {"timestamp", "arbitration_id", "dlc", "label", *[f"data{i}" for i in range(8)]}
    if attack_capture:
        required.add("attack_type")
    require_columns(source.columns, required, path.name)
    frame = source[list(required)].copy()
    frame["source_row_number"] = frame.index + 2
    for column in ["arbitration_id", "label", *[f"data{i}" for i in range(8)]]:
        frame[column] = frame[column].astype("string").str.strip().str.upper()
    if attack_capture:
        frame["attack_type"] = frame["attack_type"].astype("string").str.strip()
    else:
        frame["attack_type"] = "Normal"

    timestamp = pd.to_numeric(frame["timestamp"], errors="coerce")
    dlc = pd.to_numeric(frame["dlc"], errors="coerce")
    valid = timestamp.notna()
    valid &= frame["arbitration_id"].str.fullmatch(r"[0-9A-F]{1,8}", na=False)
    valid &= dlc.notna() & dlc.between(0, 8) & dlc.mod(1).eq(0)
    valid &= frame["label"].isin(["R", "T"])
    for index in range(8):
        required_byte = dlc.gt(index)
        valid &= (~required_byte) | frame[f"data{index}"].str.fullmatch(
            r"[0-9A-F]{1,2}", na=False
        )
    if attack_capture:
        valid &= frame["attack_type"].isin(
            ["Normal", "DoS", "Brake Tampering", "Steering Tampering"]
        )
        valid &= (
            frame["label"].eq("R") & frame["attack_type"].eq("Normal")
        ) | (frame["label"].eq("T") & frame["attack_type"].ne("Normal"))

    malformed = int((~valid).sum())
    clean = frame.loc[valid].copy()
    clean["timestamp"] = timestamp.loc[valid].astype(float)
    clean["dlc"] = dlc.loc[valid].astype(int)
    clean = clean.sort_values(["timestamp", "source_row_number"], kind="stable")
    return clean.reset_index(drop=True), malformed


def build_capture_windows(
    frame: pd.DataFrame, capture: str, feature_builder: ModuleType
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = len(frame) - len(frame) % WINDOW_SIZE
    frame = frame.iloc[:usable].copy()
    window_count = usable // WINDOW_SIZE
    raw = np.zeros((usable, 9), dtype=np.int64)
    raw[:, 0] = frame["arbitration_id"].map(lambda value: int(str(value), 16))
    dlc = frame["dlc"].to_numpy(dtype=np.int64)
    for index in range(8):
        values = frame[f"data{index}"].map(lambda value: int(str(value), 16))
        raw[:, index + 1] = np.where(index < dlc, values, 0)
    if np.any(raw < 0) or np.any(raw[:, 1:] > 255):
        raise ValueError("Canonical GEM-CAN values exceed CAN ranges")
    raw_windows = raw.reshape(window_count, WINDOW_SIZE, 9)
    features = feature_builder.extract_window_features(raw_windows, WINDOW_SIZE)

    metadata: list[dict[str, Any]] = []
    for index in range(window_count):
        window = frame.iloc[index * WINDOW_SIZE : (index + 1) * WINDOW_SIZE]
        attack_frames = int(window["label"].ne("R").sum())
        families = sorted(
            window.loc[window["attack_type"].ne("Normal"), "attack_type"].unique()
        )
        feature_lines = window[CAN_COLUMNS].astype(str).agg(",".join, axis=1)
        labelled_lines = window[
            CAN_COLUMNS + ["label", "attack_type"]
        ].astype(str).agg(",".join, axis=1)
        density = (
            "benign"
            if attack_frames == 0
            else "low_1_5"
            if attack_frames <= 5
            else "medium_6_20"
            if attack_frames <= 20
            else "high_21_100"
        )
        metadata.append(
            {
                "window_id": f"gem_{capture}_w100_{index:06d}",
                "capture": capture,
                "window_index": index,
                "binary_target": int(attack_frames > 0),
                "attack_frame_count": attack_frames,
                "density_band": density,
                "attack_families": "|".join(families) if families else "benign",
                "feature_window_sha256": digest_text(feature_lines),
                "labelled_window_sha256": digest_text(labelled_lines),
            }
        )
    metadata_frame = pd.DataFrame(metadata)
    features.insert(0, "window_id", metadata_frame["window_id"])
    return metadata_frame, features


def reconstruct_locked_features(
    external: dict[str, Path], locked: pd.DataFrame, feature_builder: ModuleType
) -> tuple[pd.DataFrame, dict[str, int]]:
    normal, normal_bad = canonical_frame(
        external["GEM_Normal_Driving_Raw.csv"], attack_capture=False
    )
    attack, attack_bad = canonical_frame(
        external["GEM_Attack_Scenario_Raw_Labeled.csv"], attack_capture=True
    )
    normal_meta, normal_features = build_capture_windows(normal, "normal", feature_builder)
    attack_meta, attack_features = build_capture_windows(attack, "attack", feature_builder)
    metadata = pd.concat([normal_meta, attack_meta], ignore_index=True)
    features = pd.concat([normal_features, attack_features], ignore_index=True)

    expected = locked.set_index("window_id")
    observed = metadata.set_index("window_id")
    missing = sorted(set(expected.index) - set(observed.index))
    if missing:
        raise RuntimeError(f"Cannot reconstruct {len(missing)} locked windows")
    for column in [
        "capture",
        "window_index",
        "binary_target",
        "density_band",
        "attack_families",
        "feature_window_sha256",
        "labelled_window_sha256",
    ]:
        left = expected[column].astype(str)
        right = observed.loc[expected.index, column].astype(str)
        if not left.equals(right):
            bad = left.index[left.ne(right)].tolist()[:5]
            raise RuntimeError(f"Locked window reconstruction mismatch in {column}: {bad}")

    selected_features = features.set_index("window_id").loc[expected.index].reset_index()
    selected_metadata = observed.loc[expected.index].reset_index()
    protocol_columns = locked[
        ["window_id", "confirmatory_role", "protocol_order"]
    ]
    selected = selected_metadata.merge(selected_features, on="window_id", validate="one_to_one")
    selected = selected.merge(protocol_columns, on="window_id", validate="one_to_one")
    selected = selected.sort_values("protocol_order", kind="stable").reset_index(drop=True)
    return selected, {"normal": normal_bad, "attack": attack_bad}


def predict_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Frozen model predict_proba output is invalid")
    output = probabilities[:, 1]
    if not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise ValueError("Frozen model produced invalid probabilities")
    return output


def cluster_interval(
    frame: pd.DataFrame, prediction_column: str, replicates: int, seed: int
) -> tuple[float, float, float, int]:
    cluster_values = (
        frame.groupby("feature_window_sha256")[prediction_column].mean().to_numpy(float)
    )
    if len(cluster_values) < 2:
        raise ValueError("At least two unique feature clusters are required")
    point = float(cluster_values.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    chunk = 500
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        indices = rng.integers(0, len(cluster_values), size=(count, len(cluster_values)))
        samples[start : start + count] = cluster_values[indices].mean(axis=1)
    alpha = (1.0 - 0.95) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return point, float(lower), float(upper), len(cluster_values)


def combined_window_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    truth = frame["binary_target"].to_numpy(dtype=np.uint8)
    prediction = frame[prediction_column].to_numpy(dtype=np.uint8)
    probability = frame["attack_probability"].to_numpy(dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "method": prediction_column,
        "windows": len(frame),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "false_positive_rate": safe_divide(int(fp), int(fp + tn)),
        "false_negative_rate": safe_divide(int(fn), int(fn + tp)),
        "pr_auc": average_precision_score(truth, probability),
        "roc_auc": roc_auc_score(truth, probability),
    }


def add_persistent_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output = output.sort_values(["capture", "window_index"], kind="stable")
    previous = output.groupby("capture")["frozen_prediction"].shift(1).fillna(0).astype(int)
    output["strict_consecutive_2_prediction"] = (
        output["frozen_prediction"].astype(int) & previous
    ).astype(np.uint8)
    return output.sort_values("protocol_order", kind="stable").reset_index(drop=True)


def save_plot(
    frame: pd.DataFrame,
    high_result: dict[str, float],
    clean_result: dict[str, float],
    criteria: dict[str, float],
    output: Path,
) -> None:
    clean = frame[frame["confirmatory_role"].eq("primary_clean_fpr")]
    high = frame[frame["confirmatory_role"].eq("primary_high_density_recall")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    bins = np.linspace(0, 1, 31)
    axes[0].hist(clean["attack_probability"], bins=bins, alpha=0.7, label="Clean")
    axes[0].hist(high["attack_probability"], bins=bins, alpha=0.7, label="High attack")
    axes[0].axvline(
        frame["frozen_threshold"].iloc[0], color="black", linestyle="dashed", label="Frozen threshold"
    )
    axes[0].set_title("Frozen-model score transfer")
    axes[0].set_xlabel("Attack probability")
    axes[0].set_ylabel("Windows")
    axes[0].legend()

    labels = ["High recall", "Clean FPR"]
    points = [high_result["point"], clean_result["point"]]
    lower = [
        high_result["point"] - high_result["ci_lower"],
        clean_result["point"] - clean_result["ci_lower"],
    ]
    upper = [
        high_result["ci_upper"] - high_result["point"],
        clean_result["ci_upper"] - clean_result["point"],
    ]
    axes[1].errorbar(labels, points, yerr=[lower, upper], fmt="o", capsize=6)
    axes[1].axhline(criteria["recall_point_min"], color="#54a24b", linestyle="dashed", label="Recall point minimum")
    axes[1].axhline(criteria["fpr_point_max"], color="#e45756", linestyle="dotted", label="FPR point maximum")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Predeclared cluster-macro endpoints")
    axes[1].set_ylabel("Rate with 95% cluster-bootstrap CI")
    axes[1].legend()
    fig.suptitle("Step 30F locked GEM-CAN external confirmation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    refuse_repeat(root)
    lock_dir, lock_path = latest_protocol_lock(root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol_sha256 = verify_protocol_hash(lock)
    if lock.get("confirmation_consumed") is not False:
        raise RuntimeError("Protocol lock is not in an unconsumed state")
    if lock.get("predictions_calculated") is not False:
        raise RuntimeError("Protocol lock indicates prior predictions")

    locked_manifest_path = lock_dir / "gem_can_locked_evaluation_manifest.csv"
    if not locked_manifest_path.is_file():
        raise FileNotFoundError(locked_manifest_path)
    locked = pd.read_csv(locked_manifest_path)
    require_columns(locked.columns, REQUIRED_LOCKED_COLUMNS, "locked evaluation manifest")
    locked = locked.sort_values("protocol_order", kind="stable").reset_index(drop=True)
    expected_ids_hash = lock["window_protocol"]["locked_window_id_sha256"]
    if digest_text(locked["window_id"]) != expected_ids_hash:
        raise RuntimeError("Locked evaluation manifest window sequence changed")
    expected_features_hash = lock["window_protocol"]["locked_feature_hash_sequence_sha256"]
    if digest_text(locked["feature_window_sha256"]) != expected_features_hash:
        raise RuntimeError("Locked evaluation manifest feature hashes changed")

    assets = verify_locked_assets(lock, root)
    step30f0 = verify_step30f0_artifacts(lock, root)
    external = external_paths(lock, step30f0, root)
    feature_builder = load_module(
        assets["frozen_window_feature_builder"], "ztav_step03_gem_confirmation"
    )
    reconstructed, malformed = reconstruct_locked_features(external, locked, feature_builder)
    feature_names = list(lock["feature_names"])
    require_columns(reconstructed.columns, set(feature_names), "reconstructed GEM-CAN features")
    if digest_text(feature_names) != lock["feature_schema_sha256"]:
        raise RuntimeError("Locked feature schema hash is invalid")
    values = reconstructed[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Reconstructed GEM-CAN model features are not finite")

    # All hashes and feature reconstructions have passed.  From this point the
    # independent confirmation is considered consumed even if scoring fails.
    out = root / OUTCOME_RESULTS / run_id()
    out.mkdir(parents=True, exist_ok=False)
    started = {
        "protocol_sha256": protocol_sha256,
        "started_utc": utc_now(),
        "confirmation_consumed": True,
        "reason": "all locked preflight checks passed; frozen model scoring began",
    }
    (out / STARTED_FILENAME).write_text(json.dumps(started, indent=2), encoding="utf-8")

    model = joblib.load(assets["frozen_logistic_regression_model"])
    n_features = getattr(model, "n_features_in_", len(feature_names))
    if int(n_features) != len(feature_names):
        raise ValueError(
            f"Frozen model expects {n_features} features; protocol locks {len(feature_names)}"
        )
    probabilities = predict_probabilities(model, values)
    threshold = float(lock["frozen_threshold"])
    predictions = (probabilities >= threshold).astype(np.uint8)
    scored = reconstructed.copy()
    scored["attack_probability"] = probabilities
    scored["frozen_threshold"] = threshold
    scored["frozen_prediction"] = predictions
    scored = add_persistent_prediction(scored)

    clean = scored[scored["confirmatory_role"].eq("primary_clean_fpr")]
    high = scored[scored["confirmatory_role"].eq("primary_high_density_recall")]
    uncertainty = lock["uncertainty"]
    replicates = int(uncertainty["replicates"])
    seed = int(uncertainty["random_seed"])
    clean_point, clean_lower, clean_upper, clean_clusters = cluster_interval(
        clean, "frozen_prediction", replicates, seed
    )
    high_point, high_lower, high_upper, high_clusters = cluster_interval(
        high, "frozen_prediction", replicates, seed + 1
    )
    primary = pd.DataFrame(
        [
            {
                "endpoint": "high_density_feature_cluster_macro_recall",
                "point_estimate": high_point,
                "ci_lower": high_lower,
                "ci_upper": high_upper,
                "confidence_level": uncertainty["confidence_level"],
                "unique_clusters": high_clusters,
                "observed_windows": len(high),
            },
            {
                "endpoint": "clean_feature_cluster_macro_false_positive_rate",
                "point_estimate": clean_point,
                "ci_lower": clean_lower,
                "ci_upper": clean_upper,
                "confidence_level": uncertainty["confidence_level"],
                "unique_clusters": clean_clusters,
                "observed_windows": len(clean),
            },
        ]
    )

    acceptance = lock["predeclared_acceptance_criteria"]
    acceptance_rows = [
        {
            "criterion": "high_cluster_macro_recall_point_minimum",
            "observed": high_point,
            "operator": ">=",
            "required": acceptance["high_cluster_macro_recall_point_minimum"],
            "passed": high_point >= acceptance["high_cluster_macro_recall_point_minimum"],
        },
        {
            "criterion": "high_cluster_macro_recall_ci_lower_minimum",
            "observed": high_lower,
            "operator": ">=",
            "required": acceptance["high_cluster_macro_recall_ci_lower_minimum"],
            "passed": high_lower >= acceptance["high_cluster_macro_recall_ci_lower_minimum"],
        },
        {
            "criterion": "benign_cluster_macro_fpr_point_maximum",
            "observed": clean_point,
            "operator": "<=",
            "required": acceptance["benign_cluster_macro_fpr_point_maximum"],
            "passed": clean_point <= acceptance["benign_cluster_macro_fpr_point_maximum"],
        },
        {
            "criterion": "benign_cluster_macro_fpr_ci_upper_maximum",
            "observed": clean_upper,
            "operator": "<=",
            "required": acceptance["benign_cluster_macro_fpr_ci_upper_maximum"],
            "passed": clean_upper <= acceptance["benign_cluster_macro_fpr_ci_upper_maximum"],
        },
    ]
    acceptance_frame = pd.DataFrame(acceptance_rows)
    passed = bool(acceptance_frame["passed"].all())

    secondary = pd.DataFrame(
        [
            combined_window_metrics(scored, "frozen_prediction"),
            combined_window_metrics(scored, "strict_consecutive_2_prediction"),
        ]
    )
    family_rows: list[dict[str, Any]] = []
    for family in ["DoS", "Brake Tampering", "Steering Tampering"]:
        selected = high[
            high["attack_families"].str.split("|").apply(lambda values: family in values)
        ]
        if len(selected):
            family_rows.append(
                {
                    "attack_family": family,
                    "mixed_windows_containing_family": len(selected),
                    "unique_feature_clusters": selected[
                        "feature_window_sha256"
                    ].nunique(),
                    "observed_window_recall": selected["frozen_prediction"].mean(),
                    "cluster_macro_recall": selected.groupby(
                        "feature_window_sha256"
                    )["frozen_prediction"].mean().mean(),
                    "confirmatory_status": "descriptive_only_mixed_family_windows",
                }
            )
    family_metrics = pd.DataFrame(family_rows)

    training = pd.read_csv(assets["frozen_training_feature_schema"], usecols=feature_names)
    training = training.apply(pd.to_numeric, errors="raise")
    external_features = scored[feature_names]
    train_mean = training.mean()
    train_std = training.std(ddof=0)
    external_mean = external_features.mean()
    shift = (external_mean - train_mean) / train_std.replace(0, np.nan)
    domain_shift = pd.DataFrame(
        {
            "feature": feature_names,
            "training_mean": train_mean.reindex(feature_names).to_numpy(),
            "training_std": train_std.reindex(feature_names).to_numpy(),
            "gem_can_mean": external_mean.reindex(feature_names).to_numpy(),
            "standardized_mean_shift": shift.reindex(feature_names).to_numpy(),
            "absolute_standardized_mean_shift": shift.abs().reindex(feature_names).to_numpy(),
        }
    ).sort_values("absolute_standardized_mean_shift", ascending=False, na_position="last")

    prediction_columns = [
        "window_id",
        "capture",
        "window_index",
        "confirmatory_role",
        "binary_target",
        "attack_frame_count",
        "density_band",
        "attack_families",
        "feature_window_sha256",
        "attack_probability",
        "frozen_threshold",
        "frozen_prediction",
        "strict_consecutive_2_prediction",
    ]
    scored[prediction_columns].to_csv(out / "gem_can_external_predictions.csv", index=False)
    primary.to_csv(out / "gem_can_primary_cluster_metrics.csv", index=False)
    acceptance_frame.to_csv(out / "gem_can_confirmation_acceptance.csv", index=False)
    secondary.to_csv(out / "gem_can_secondary_window_metrics.csv", index=False)
    family_metrics.to_csv(out / "gem_can_descriptive_family_metrics.csv", index=False)
    domain_shift.to_csv(out / "gem_can_feature_domain_shift.csv", index=False)
    pd.DataFrame(
        [
            {"capture": "normal", "malformed_rows_excluded": malformed["normal"]},
            {"capture": "attack", "malformed_rows_excluded": malformed["attack"]},
        ]
    ).to_csv(out / "gem_can_parser_audit.csv", index=False)

    high_result = {"point": high_point, "ci_lower": high_lower, "ci_upper": high_upper}
    clean_result = {"point": clean_point, "ci_lower": clean_lower, "ci_upper": clean_upper}
    plot_criteria = {
        "recall_point_min": float(acceptance["high_cluster_macro_recall_point_minimum"]),
        "fpr_point_max": float(acceptance["benign_cluster_macro_fpr_point_maximum"]),
    }
    save_plot(
        scored,
        high_result,
        clean_result,
        plot_criteria,
        out / "gem_can_confirmation_summary.png",
    )

    outcome = {
        "experiment": "Step 30F one-time locked GEM-CAN external confirmation",
        "completed_utc": utc_now(),
        "protocol_sha256": protocol_sha256,
        "confirmation_consumed": True,
        "confirmatory_passed": passed,
        "high_density_cluster_macro_recall": high_point,
        "high_density_cluster_macro_recall_ci": [high_lower, high_upper],
        "clean_cluster_macro_fpr": clean_point,
        "clean_cluster_macro_fpr_ci": [clean_lower, clean_upper],
        "locked_threshold": threshold,
        "model_retrained": False,
        "threshold_changed": False,
        "window_reselected": False,
        "sumo_executed": False,
        "claim_statement": (
            lock["claim_boundaries"]["allowed_if_primary_passes"]
            if passed
            else lock["claim_boundaries"]["allowed_if_primary_fails"]
        ),
        "forbidden_claims": lock["claim_boundaries"]["always_forbidden"],
        "step31_permitted": False,
        "next_stage": "rerun Step 30A publication-readiness audit before considering Step 31",
    }
    outcome["outcome_sha256"] = canonical_json_sha256(outcome)
    (out / OUTCOME_FILENAME).write_text(json.dumps(outcome, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("Step 30F locked GEM-CAN external confirmation completed successfully.")
    print(f"Protocol SHA-256: {protocol_sha256}")
    print("Confirmation consumed: True")
    print("Model retrained: False")
    print("Threshold changed: False")
    print("Window reselected: False")
    print("SUMO executed: False")
    print(f"Frozen threshold: {threshold:.12g}")
    print("\nPrimary cluster-aware endpoints:")
    print(
        f"  high-density recall={high_point:.4f}, 95% CI=[{high_lower:.4f}, {high_upper:.4f}], "
        f"clusters={high_clusters}, windows={len(high)}"
    )
    print(
        f"  clean FPR={clean_point:.4f}, 95% CI=[{clean_lower:.4f}, {clean_upper:.4f}], "
        f"clusters={clean_clusters}, windows={len(clean)}"
    )
    print("\nPredeclared acceptance checks:")
    print(acceptance_frame.to_string(index=False))
    print(f"\nConfirmatory criteria passed: {passed}")
    if not passed:
        print("The failed external result is retained; do not tune on GEM-CAN.")
    print("Low/medium and family-specific confirmation claims remain forbidden.")
    print("Do not run Step 31 yet.")
    print(f"\nResults directory: {out}")
    print(f"Outcome: {out / OUTCOME_FILENAME}")
    print(f"Primary metrics: {out / 'gem_can_primary_cluster_metrics.csv'}")
    print(f"Acceptance: {out / 'gem_can_confirmation_acceptance.csv'}")
    print("\nNext: send the terminal result and all primary/outcome evidence files.")


if __name__ == "__main__":
    main()
