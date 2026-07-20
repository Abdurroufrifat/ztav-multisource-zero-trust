#!/usr/bin/env python3
"""Step 30E: locked, leakage-safe final internal confirmation.

This additive script consumes the predeclared confirmation data exactly once.
It never trains a model, changes a threshold, overwrites a historical result,
or selects a rule using confirmation outcomes.  Before SUMO is started it
locks:

* new SUMO seeds 2032--2036;
* the Step 21/23/26 CAN, guard, and temporal-policy settings;
* an eligible strongest single-source baseline selected only from the original
  development seeds;
* H1--H5 decision criteria and all input/code hashes;
* source-specific HCRL parent windows disjoint from every Step 24/25 replay.

The HCRL corpus itself is not a new external dataset.  This is therefore a
strict internal confirmation using unseen HCRL windows and unseen SUMO seeds.
The negative frozen ROAD result remains the independent external evidence.

Run once from D:\\ztav_project after Steps 30A--30D2 and 30C2:

    .\\.venv\\Scripts\\python.exe .\\src\\30E_untouched_final_confirmation.py

If a technical interruption occurs, use the exact same script and:

    .\\.venv\\Scripts\\python.exe .\\src\\30E_untouched_final_confirmation.py --resume

Do not run Step 31 until the Step 30E outputs and a fresh Step 30A audit have
been reviewed.  This is research software, not production vehicle software.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "src" else here


ROOT = infer_root()
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT
RESULTS = ROOT / "results"
AUDIT_ROOT = RESULTS / "publication_untouched_confirmation"
REGISTRY = AUDIT_ROOT / "CONFIRMATION_LOCK.json"

CONFIRMATION_SEEDS = (2032, 2033, 2034, 2035, 2036)
SUMO_STEPS = 720
RANDOM_SEED = 314159
BOOTSTRAP_REPLICATES = 50_000
PERMUTATION_REPLICATES = 100_000
ALPHA = 0.05
FPR_LIMIT = 0.05
F1_MINIMUM = 0.90
H4_MARGIN = 0.10
EXPECTED_TEMPORAL_RULE = "strict_consecutive_2"
DEVELOPMENT_SEED_COUNT = 3

DENSITY_ORDER = (
    "representative_all",
    "low_1_5",
    "medium_6_20",
    "high_21_100",
)
ALLOCATION_ORDER = (
    "low_1_5",
    "medium_6_20",
    "high_21_100",
    "representative_all",
)
ATTACK_DENSITY_SCENARIOS = {
    "representative_all": (1, 100),
    "low_1_5": (1, 5),
    "medium_6_20": (6, 20),
    "high_21_100": (21, 100),
}
CAN_ATTACK_PHASES = {"can_injection", "combined_attack"}
HEALTHY_PHASES = {
    "healthy_baseline",
    "recovery_after_gps",
    "recovery_after_can",
    "recovery_after_v2x",
    "recovery_after_identity",
    "final_recovery",
}
METHODS = (
    "proposed_persistent_without_vehicle_state",
    "temporal_can_only",
    "frozen_w100_can_only",
    "context_only_without_vehicle_state",
)
BASELINE_CANDIDATES = (
    "temporal_can_only",
    "context_only_without_vehicle_state",
)

INPUT_PATHS = {
    "parents": RESULTS / "multiscale_sparse_can_gate" / "multiscale_parent_predictions.csv",
    "guard": RESULTS / "reference_anchored_startup_guard" / "reference_guard_clean_source_audit.csv",
    "development_decisions": RESULTS / "graded_zero_trust_policy" / "graded_policy_decisions.csv",
    "temporal_rule": RESULTS / "temporal_memory_sparse_can_confirmation" / "temporal_selected_rule.csv",
}

SCRIPT_NAMES = (
    "07_build_sumo_context_testbed.py",
    "08_run_sumo_attack_experiments.py",
    "19_integrate_guarded_can_multisource_policy.py",
    "21_multiscale_sparse_can_gate.py",
    "23_reference_anchored_startup_guard.py",
    "24_integrate_soft_guarded_multiscale_policy.py",
    "25_evaluate_graded_zero_trust_policy.py",
    "26_temporal_memory_sparse_can_confirmation.py",
    "30C_publication_source_robustness.py",
    "30C2_publication_observability_claim_audit.py",
)

REQUIRED_PARENT_COLUMNS = {
    "source_file",
    "parent_window_index",
    "micro_attack_frames",
    "w100_binary_target",
    "w100_attack_frame_count",
    "w100_alarm_instant",
    "w100_continuous_can_trust",
    "multiscale_alarm_instant",
    "multiscale_continuous_can_trust",
}
REQUIRED_DECISION_COLUMNS = {
    "seed",
    "source_file",
    "parent_window_index",
    "density_scenario",
    "simulation_time_s",
    "phase",
    "ground_truth_attack",
    "w100_alarm_persistent_2",
    "multiscale_alarm_instant",
    "active_noncan_sources",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked untouched final internal confirmation exactly once."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")


def locate_script(root: Path, name: str) -> Path:
    for path in (root / "src" / name, root / name):
        if path.exists():
            return path
    raise FileNotFoundError(f"Required script is missing: {name}")


def load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_columns(actual: Iterable[str], required: set[str], label: str) -> None:
    missing = required - set(actual)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def split_sources(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item for item in str(value).split(";") if item} - {"sensor_control"}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    truth: Sequence[int | bool], prediction: Sequence[int | bool]
) -> dict[str, float | int]:
    expected = np.asarray(truth, dtype=bool)
    predicted = np.asarray(prediction, dtype=bool)
    if len(expected) != len(predicted):
        raise ValueError("Truth and prediction lengths differ")
    tp = int(np.sum(expected & predicted))
    fp = int(np.sum(~expected & predicted))
    tn = int(np.sum(~expected & ~predicted))
    fn = int(np.sum(expected & ~predicted))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
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
        "accuracy": safe_divide(tp + tn, len(expected)),
    }


def input_files(root: Path) -> dict[str, Path]:
    files = {key: path for key, path in INPUT_PATHS.items()}
    files["step30e_script"] = Path(__file__).resolve()
    files["ztav_core"] = root / "ztav_phase0.py"
    for name in SCRIPT_NAMES:
        files[f"script__{name}"] = locate_script(root, name)
    missing = [key for key, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing locked inputs: " + ", ".join(missing))
    return files


def hash_inventory(files: dict[str, Path], root: Path) -> list[dict[str, Any]]:
    rows = []
    for key, path in sorted(files.items()):
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
        rows.append(
            {
                "input_key": key,
                "path": display,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def compare_hashes(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    old = {row["input_key"]: row for row in before}
    new = {row["input_key"]: row for row in after}
    keys = sorted(set(old) | set(new))
    return [
        {
            "input_key": key,
            "before_sha256": old.get(key, {}).get("sha256", "missing"),
            "after_sha256": new.get(key, {}).get("sha256", "missing"),
            "unchanged": old.get(key, {}).get("sha256") == new.get(key, {}).get("sha256"),
        }
        for key in keys
    ]


def selected_temporal_rule(path: Path) -> tuple[str, tuple[int, int, int]]:
    frame = pd.read_csv(path)
    required = {"selected_rule", "required_hits", "history_windows", "hold_clean_windows"}
    validate_columns(frame.columns, required, "Step 26 selection")
    if len(frame) != 1:
        raise ValueError("Step 26 selection must contain exactly one row")
    row = frame.iloc[0]
    name = str(row["selected_rule"])
    parameters = (
        int(row["required_hits"]),
        int(row["history_windows"]),
        int(row["hold_clean_windows"]),
    )
    if name != EXPECTED_TEMPORAL_RULE or parameters != (2, 2, 0):
        raise RuntimeError(
            f"Frozen temporal rule changed: found {name} {parameters}, "
            f"expected {EXPECTED_TEMPORAL_RULE} (2, 2, 0)"
        )
    return name, parameters


def add_method_predictions(
    raw: pd.DataFrame,
    temporal_parameters: tuple[int, int, int],
    step26: ModuleType,
) -> pd.DataFrame:
    validate_columns(raw.columns, REQUIRED_DECISION_COLUMNS, "policy decisions")
    parts = []
    keys = ["seed", "source_file", "density_scenario"]
    hits, history, hold = temporal_parameters
    for _, group in raw.groupby(keys, sort=True):
        group = group.sort_values("simulation_time_s", kind="stable").copy()
        temporal = step26.temporal_memory(
            group["multiscale_alarm_instant"].astype(bool).to_numpy(),
            hits,
            history,
            hold,
        )
        context = group["active_noncan_sources"].map(split_sources).map(bool).to_numpy()
        w100 = group["w100_alarm_persistent_2"].astype(bool).to_numpy()
        group["pred__temporal_can_only"] = temporal.astype(int)
        group["pred__frozen_w100_can_only"] = w100.astype(int)
        group["pred__context_only_without_vehicle_state"] = context.astype(int)
        group["pred__proposed_persistent_without_vehicle_state"] = (
            temporal | context
        ).astype(int)
        group["multiscale_alarm_persistent_2"] = temporal.astype(int)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    keys = ["seed", "source_file", "density_scenario"]
    for (seed, source, density), group in frame.groupby(keys, sort=True):
        truth = group["ground_truth_attack"].astype(int).tolist()
        for method in METHODS:
            run_rows.append(
                {
                    "seed": int(seed),
                    "source_file": str(source),
                    "density_scenario": str(density),
                    "method": method,
                    **binary_metrics(truth, group[f"pred__{method}"].astype(int).tolist()),
                }
            )
        for phase, phase_group in group.groupby("phase", sort=False):
            phase_truth = phase_group["ground_truth_attack"].astype(int).tolist()
            for method in METHODS:
                phase_rows.append(
                    {
                        "seed": int(seed),
                        "source_file": str(source),
                        "density_scenario": str(density),
                        "phase": str(phase),
                        "method": method,
                        **binary_metrics(
                            phase_truth,
                            phase_group[f"pred__{method}"].astype(int).tolist(),
                        ),
                    }
                )
    return pd.DataFrame(run_rows), pd.DataFrame(phase_rows)


def aggregate_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate")
    for (density, method), group in per_run.groupby(
        ["density_scenario", "method"], sort=True
    ):
        row: dict[str, Any] = {
            "density_scenario": density,
            "method": method,
            "independent_runs": len(group),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def select_development_baseline(
    development: pd.DataFrame,
    temporal_parameters: tuple[int, int, int],
    step26: ModuleType,
) -> tuple[str, list[dict[str, Any]]]:
    seeds = sorted(int(seed) for seed in development["seed"].unique())
    if len(seeds) < DEVELOPMENT_SEED_COUNT:
        raise ValueError(f"Expected at least three development seeds, found {seeds}")
    development_seeds = set(seeds[:DEVELOPMENT_SEED_COUNT])
    methods = add_method_predictions(
        development[development["seed"].astype(int).isin(development_seeds)],
        temporal_parameters,
        step26,
    )
    per_run, _ = metric_tables(methods)
    rows = []
    for method in BASELINE_CANDIDATES:
        group = per_run[per_run["method"] == method]
        mean_fpr = float(group["false_positive_rate"].mean())
        rows.append(
            {
                "method": method,
                "development_seeds": ";".join(map(str, sorted(development_seeds))),
                "independent_runs": len(group),
                "f1_mean": float(group["f1"].mean()),
                "false_positive_rate_mean": mean_fpr,
                "eligible_fpr_le_0_05": mean_fpr <= FPR_LIMIT,
            }
        )
    eligible = [row for row in rows if row["eligible_fpr_le_0_05"]]
    if not eligible:
        raise RuntimeError("No single-source baseline meets the locked development FPR limit")
    selected = sorted(
        eligible,
        key=lambda row: (-float(row["f1_mean"]), str(row["method"])),
    )[0]["method"]
    for row in rows:
        row["selected_before_confirmation"] = row["method"] == selected
    return str(selected), rows


def previous_seeds(root: Path, development: pd.DataFrame) -> set[int]:
    seeds = {int(value) for value in development["seed"].unique()}
    seed_dir = root / "data" / "processed" / "sumo_repeated_seeds"
    for path in seed_dir.glob("sumo_context_attacks_seed_*.csv"):
        try:
            seeds.add(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return seeds


def parent_capacity_audit(
    parents: pd.DataFrame,
    development: pd.DataFrame,
    sources: Sequence[str],
    benign_per_run: int,
    attack_per_run: int,
) -> list[dict[str, Any]]:
    used = {
        source: set(
            development.loc[
                development["source_file"] == source, "parent_window_index"
            ].astype(int)
        )
        for source in sources
    }
    rows = []
    for source in sources:
        source_frame = parents[parents["source_file"] == source]
        unseen = source_frame[~source_frame["parent_window_index"].astype(int).isin(used[source])]
        benign_available = int(
            unseen.loc[unseen["w100_binary_target"] == 0, "parent_window_index"].nunique()
        )
        benign_required = len(CONFIRMATION_SEEDS) * len(DENSITY_ORDER) * benign_per_run
        all_attack_available = int(
            unseen.loc[unseen["w100_binary_target"] == 1, "parent_window_index"].nunique()
        )
        all_attack_required = len(CONFIRMATION_SEEDS) * len(DENSITY_ORDER) * attack_per_run
        rows.append(
            {
                "source_file": source,
                "pool": "benign",
                "available_disjoint_windows": benign_available,
                "required_unique_windows": benign_required,
                "capacity_passed": benign_available >= benign_required,
            }
        )
        rows.append(
            {
                "source_file": source,
                "pool": "all_attack_densities",
                "available_disjoint_windows": all_attack_available,
                "required_unique_windows": all_attack_required,
                "capacity_passed": all_attack_available >= all_attack_required,
            }
        )
        for density in ("low_1_5", "medium_6_20", "high_21_100"):
            lower, upper = ATTACK_DENSITY_SCENARIOS[density]
            available = int(
                unseen.loc[
                    (unseen["w100_binary_target"] == 1)
                    & unseen["w100_attack_frame_count"].between(lower, upper),
                    "parent_window_index",
                ].nunique()
            )
            required = len(CONFIRMATION_SEEDS) * attack_per_run
            rows.append(
                {
                    "source_file": source,
                    "pool": density,
                    "available_disjoint_windows": available,
                    "required_unique_windows": required,
                    "capacity_passed": available >= required,
                }
            )
    return rows


def choose_unique(
    pool: pd.DataFrame,
    count: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    if len(pool) < count:
        raise RuntimeError(f"Insufficient disjoint windows for {label}: need {count}, have {len(pool)}")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(pool), size=count, replace=False)
    return pool.iloc[positions].reset_index(drop=True)


def build_disjoint_replay(
    sumo_rows: Sequence[dict[str, str]],
    parents: pd.DataFrame,
    source_file: str,
    density: str,
    seed: int,
    used_by_source: dict[str, set[int]],
    step19: ModuleType,
    step24: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = parents[parents["source_file"] == source_file].copy()
    source["parent_window_index"] = source["parent_window_index"].astype(int)
    available = source[~source["parent_window_index"].isin(used_by_source[source_file])]
    lower, upper = ATTACK_DENSITY_SCENARIOS[density]
    benign_pool = available[available["w100_binary_target"] == 0]
    attack_pool = available[
        (available["w100_binary_target"] == 1)
        & available["w100_attack_frame_count"].between(lower, upper)
    ]
    attack_count = sum(row["phase"] in CAN_ATTACK_PHASES for row in sumo_rows)
    benign_count = len(sumo_rows) - attack_count
    benign = choose_unique(
        benign_pool,
        benign_count,
        step19.stable_seed(seed, source_file, density, "step30e_disjoint_benign"),
        f"{source_file}/{density}/benign",
    )
    attack = choose_unique(
        attack_pool,
        attack_count,
        step19.stable_seed(seed, source_file, density, "step30e_disjoint_attack"),
        f"{source_file}/{density}/attack",
    )
    selected_ids = set(benign["parent_window_index"].astype(int)) | set(
        attack["parent_window_index"].astype(int)
    )
    if selected_ids & used_by_source[source_file]:
        raise RuntimeError("Disjoint allocation invariant failed")
    used_before = len(used_by_source[source_file])
    used_by_source[source_file].update(selected_ids)

    output: list[dict[str, Any]] = []
    benign_i = attack_i = 0
    for sumo_row in sumo_rows:
        if sumo_row["phase"] in CAN_ATTACK_PHASES:
            can_row = attack.iloc[attack_i]
            attack_i += 1
        else:
            can_row = benign.iloc[benign_i]
            benign_i += 1
        row: dict[str, Any] = dict(sumo_row)
        row.update(
            {
                "hcrl_source_file": source_file,
                "hcrl_parent_window_index": int(can_row["parent_window_index"]),
                "hcrl_binary_target": int(can_row["w100_binary_target"]),
                "hcrl_attack_frame_count": int(can_row["w100_attack_frame_count"]),
                "w100_alarm_instant_input": step24.as_bool(
                    can_row["w100_alarm_instant"]
                ),
                "w100_continuous_can_trust": float(can_row["w100_continuous_can_trust"]),
                "multiscale_alarm_instant_input": step24.as_bool(
                    can_row["multiscale_alarm_instant"]
                ),
                "multiscale_continuous_can_trust": float(
                    can_row["multiscale_continuous_can_trust"]
                ),
            }
        )
        output.append(row)
    return output, {
        "seed": seed,
        "source_file": source_file,
        "density_scenario": density,
        "replay_rows": len(output),
        "benign_windows": benign_count,
        "attack_windows": attack_count,
        "unique_confirmation_windows": len(selected_ids),
        "historical_and_prior_confirmation_windows_excluded": used_before,
        "sampling_with_replacement": False,
        "development_overlap_count": 0,
        "confirmation_cross_run_overlap_count": 0,
    }


def bootstrap_mean_ci(
    values: Sequence[float], rng: np.random.Generator
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return math.nan, math.nan
    means: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        batch = min(5_000, remaining)
        indices = rng.integers(0, len(data), size=(batch, len(data)))
        means.append(data[indices].mean(axis=1))
        remaining -= batch
    low, high = np.quantile(np.concatenate(means), [ALPHA / 2, 1 - ALPHA / 2])
    return float(low), float(high)


def paired_sign_flip_p(values: Sequence[float], rng: np.random.Generator) -> float:
    data = np.asarray(values, dtype=float)
    observed = float(np.mean(data))
    exceed = 0
    remaining = PERMUTATION_REPLICATES
    while remaining:
        batch = min(5_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(data)))
        exceed += int(np.sum((signs * data).mean(axis=1) >= observed))
        remaining -= batch
    return float((exceed + 1) / (PERMUTATION_REPLICATES + 1))


def primary_endpoint(
    per_run: pd.DataFrame,
    per_phase: pd.DataFrame,
) -> pd.DataFrame:
    aggregate = aggregate_metrics(per_run)
    proposed = aggregate[
        aggregate["method"] == "proposed_persistent_without_vehicle_state"
    ].copy()
    can = per_phase[
        (per_phase["method"] == "proposed_persistent_without_vehicle_state")
        & (per_phase["phase"] == "can_injection")
    ].groupby("density_scenario", as_index=False)["recall"].mean()
    can = can.rename(columns={"recall": "can_injection_recall_mean"})
    healthy = per_phase[
        (per_phase["method"] == "proposed_persistent_without_vehicle_state")
        & per_phase["phase"].isin(HEALTHY_PHASES)
    ].groupby("density_scenario", as_index=False)["false_positive_rate"].mean()
    healthy = healthy.rename(
        columns={"false_positive_rate": "healthy_recovery_fpr_macro"}
    )
    return proposed.merge(can, on="density_scenario").merge(
        healthy, on="density_scenario"
    )


def assess_hypotheses(
    per_run: pd.DataFrame,
    per_phase: pd.DataFrame,
    primary: pd.DataFrame,
    baseline: str,
    source_checks: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    proposed = per_run[
        per_run["method"] == "proposed_persistent_without_vehicle_state"
    ].sort_values(["seed", "source_file", "density_scenario"])
    comparator = per_run[per_run["method"] == baseline].sort_values(
        ["seed", "source_file", "density_scenario"]
    )
    diffs = proposed["f1"].to_numpy() - comparator["f1"].to_numpy()
    h1_low, h1_high = bootstrap_mean_ci(diffs, rng)
    h1_p = paired_sign_flip_p(diffs, rng)
    comparisons = [
        {
            "hypothesis": "H1",
            "comparison": f"proposed_minus_{baseline}",
            "independent_paired_runs": len(diffs),
            "paired_f1_difference_mean": float(np.mean(diffs)),
            "ci95_low": h1_low,
            "ci95_high": h1_high,
            "one_sided_sign_flip_p": h1_p,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
    ]

    h2_density = []
    for row in primary.itertuples(index=False):
        passed = row.f1_mean >= F1_MINIMUM and row.false_positive_rate_mean <= FPR_LIMIT
        h2_density.append(
            {
                "hypothesis": "H2",
                "density_scenario": row.density_scenario,
                "f1_mean": row.f1_mean,
                "false_positive_rate_mean": row.false_positive_rate_mean,
                "criterion": f"F1 >= {F1_MINIMUM:.2f} and FPR <= {FPR_LIMIT:.2f}",
                "passed": passed,
            }
        )

    low = per_phase[
        (per_phase["density_scenario"] == "low_1_5")
        & (per_phase["phase"] == "can_injection")
    ]
    temporal = low[low["method"] == "temporal_can_only"].sort_values(
        ["seed", "source_file"]
    )
    w100 = low[low["method"] == "frozen_w100_can_only"].sort_values(
        ["seed", "source_file"]
    )
    h3_diffs = temporal["recall"].to_numpy() - w100["recall"].to_numpy()
    h3_low, h3_high = bootstrap_mean_ci(h3_diffs, rng)
    temporal_fpr = float(
        per_run.loc[per_run["method"] == "temporal_can_only", "false_positive_rate"].mean()
    )

    h4_checks = source_checks[source_checks["hypothesis"] == "H4"]
    h5_checks = source_checks[source_checks["hypothesis"] == "H5"]
    h5_detected = h5_checks[
        h5_checks["scenario"].isin(
            ["compromised_detected_false_healthy", "conflicting_high_risk"]
        )
    ]
    h5_undetected = h5_checks[
        h5_checks["scenario"] == "compromised_undetected_false_healthy"
    ]

    hypotheses = [
        {
            "hypothesis": "H1",
            "assessment": "supported" if h1_low > 0 and h1_p < ALPHA else "not_supported",
            "criterion": "paired F1 difference lower 95% CI > 0 and one-sided p < 0.05",
            "observed": float(np.mean(diffs)),
            "confirmatory": True,
        },
        {
            "hypothesis": "H2",
            "assessment": "supported" if all(row["passed"] for row in h2_density) else "not_supported",
            "criterion": "every density has mean F1 >= 0.90 and mean FPR <= 0.05",
            "observed": f"{sum(row['passed'] for row in h2_density)}/{len(h2_density)} density checks passed",
            "confirmatory": True,
        },
        {
            "hypothesis": "H3",
            "assessment": "supported" if h3_low > 0 and temporal_fpr <= FPR_LIMIT else "not_supported",
            "criterion": "low-density recall gain lower 95% CI > 0 and temporal CAN FPR <= 0.05",
            "observed": float(np.mean(h3_diffs)),
            "confirmatory": True,
        },
        {
            "hypothesis": "H4",
            "assessment": "supported" if len(h4_checks) and bool(h4_checks["passed"].all()) else "not_supported",
            "criterion": "all source-loss upper 95% CIs <= 0.10",
            "observed": f"{int(h4_checks['passed'].sum())}/{len(h4_checks)} checks passed",
            "confirmatory": True,
        },
        {
            "hypothesis": "H5",
            "assessment": "supported" if len(h5_checks) and bool(h5_checks["passed"].all()) else "not_supported",
            "criterion": "zero unsafe/full-ALLOW rows for detected, undetected, and conflicting compromise",
            "observed": f"{int(h5_checks['passed'].sum())}/{len(h5_checks)} checks passed",
            "confirmatory": True,
        },
        {
            "hypothesis": "H5_detected_compromise_boundary",
            "assessment": "conditionally_supported" if len(h5_detected) and bool(h5_detected["passed"].all()) else "not_supported",
            "criterion": "detected compromise and conflict never receive unsafe full ALLOW",
            "observed": f"{int(h5_detected['passed'].sum())}/{len(h5_detected)} checks passed",
            "confirmatory": True,
        },
        {
            "hypothesis": "H5_undetected_byzantine_boundary",
            "assessment": "not_observable" if len(h5_undetected) and not bool(h5_undetected["passed"].all()) else "supported",
            "criterion": "explicitly separate unobservable false-healthy behavior from detected compromise",
            "observed": f"{int(h5_undetected['passed'].sum())}/{len(h5_undetected)} checks passed",
            "confirmatory": True,
        },
    ]
    comparisons.append(
        {
            "hypothesis": "H3",
            "comparison": "temporal_can_minus_frozen_w100_can_low_density_phase_recall",
            "independent_paired_runs": len(h3_diffs),
            "paired_f1_difference_mean": "",
            "paired_recall_difference_mean": float(np.mean(h3_diffs)),
            "ci95_low": h3_low,
            "ci95_high": h3_high,
            "one_sided_sign_flip_p": paired_sign_flip_p(h3_diffs, rng),
            "temporal_can_fpr_mean": temporal_fpr,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
    )
    return hypotheses, comparisons, h2_density


def make_plot(
    primary: pd.DataFrame,
    comparisons: Sequence[dict[str, Any]],
    hypotheses: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    primary = primary.set_index("density_scenario").reindex(DENSITY_ORDER)
    x = np.arange(len(DENSITY_ORDER))
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    axes[0].bar(x - 0.18, primary["f1_mean"], 0.36, label="F1")
    axes[0].bar(x + 0.18, primary["false_positive_rate_mean"], 0.36, label="FPR")
    axes[0].axhline(F1_MINIMUM, linestyle="--", color="green", linewidth=1)
    axes[0].axhline(FPR_LIMIT, linestyle=":", color="red", linewidth=1)
    axes[0].set_xticks(x, DENSITY_ORDER, rotation=18)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Locked primary endpoint")
    axes[0].legend()

    labels = [row["comparison"] for row in comparisons]
    effects = [
        float(row.get("paired_f1_difference_mean") or row.get("paired_recall_difference_mean"))
        for row in comparisons
    ]
    lows = [float(row["ci95_low"]) for row in comparisons]
    highs = [float(row["ci95_high"]) for row in comparisons]
    y = np.arange(len(labels))
    axes[1].errorbar(
        effects,
        y,
        xerr=[np.asarray(effects) - np.asarray(lows), np.asarray(highs) - np.asarray(effects)],
        fmt="o",
        capsize=4,
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, labels)
    axes[1].set_title("Paired confirmation effects")

    core = [row for row in hypotheses if row["hypothesis"] in {"H1", "H2", "H3", "H4", "H5"}]
    colors = ["tab:green" if row["assessment"] == "supported" else "tab:red" for row in core]
    axes[2].bar([row["hypothesis"] for row in core], [1] * len(core), color=colors)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_yticks([])
    axes[2].set_title("Predeclared hypothesis outcomes")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Step 30E untouched internal confirmation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    global ROOT, SRC, RESULTS, AUDIT_ROOT, REGISTRY, INPUT_PATHS
    ROOT = root
    SRC = root / "src" if (root / "src").is_dir() else root
    RESULTS = root / "results"
    AUDIT_ROOT = RESULTS / "publication_untouched_confirmation"
    REGISTRY = AUDIT_ROOT / "CONFIRMATION_LOCK.json"
    INPUT_PATHS = {
        "parents": RESULTS / "multiscale_sparse_can_gate" / "multiscale_parent_predictions.csv",
        "guard": RESULTS / "reference_anchored_startup_guard" / "reference_guard_clean_source_audit.csv",
        "development_decisions": RESULTS / "graded_zero_trust_policy" / "graded_policy_decisions.csv",
        "temporal_rule": RESULTS / "temporal_memory_sparse_can_confirmation" / "temporal_selected_rule.csv",
    }

    step07 = load_script(locate_script(root, "07_build_sumo_context_testbed.py"), "ztav_step30e_07")
    step08 = load_script(locate_script(root, "08_run_sumo_attack_experiments.py"), "ztav_step30e_08")
    step19 = load_script(locate_script(root, "19_integrate_guarded_can_multisource_policy.py"), "ztav_step30e_19")
    step24 = load_script(locate_script(root, "24_integrate_soft_guarded_multiscale_policy.py"), "ztav_step30e_24")
    step26 = load_script(locate_script(root, "26_temporal_memory_sparse_can_confirmation.py"), "ztav_step30e_26")
    step30c = load_script(locate_script(root, "30C_publication_source_robustness.py"), "ztav_step30e_30c")
    step30c2 = load_script(
        locate_script(root, "30C2_publication_observability_claim_audit.py"),
        "ztav_step30e_30c2",
    )

    files = input_files(root)
    parents = pd.read_csv(INPUT_PATHS["parents"])
    validate_columns(parents.columns, REQUIRED_PARENT_COLUMNS, "Step 21 parents")
    parents["parent_window_index"] = parents["parent_window_index"].astype(int)
    if parents.duplicated(["source_file", "parent_window_index"]).any():
        raise ValueError("Step 21 contains duplicate source/window identifiers")
    development = pd.read_csv(INPUT_PATHS["development_decisions"])
    validate_columns(development.columns, REQUIRED_DECISION_COLUMNS, "Step 25 decisions")
    temporal_name, temporal_parameters = selected_temporal_rule(INPUT_PATHS["temporal_rule"])
    baseline, baseline_rows = select_development_baseline(
        development, temporal_parameters, step26
    )

    overlap = set(CONFIRMATION_SEEDS) & previous_seeds(root, development)
    if overlap:
        raise RuntimeError(f"Confirmation seeds were previously used: {sorted(overlap)}")
    sources = sorted(
        str(source)
        for source in parents["source_file"].unique()
        if str(source) != "normal_run_data.txt"
    )
    guards = step24.load_guard(INPUT_PATHS["guard"])
    missing_guards = set(sources) - set(guards)
    if missing_guards:
        raise ValueError(f"Step 23 guard rows are missing for {sorted(missing_guards)}")

    expected = [step08.phase_for_step(index)[0] for index in range(SUMO_STEPS)]
    attack_per_run = sum(phase in CAN_ATTACK_PHASES for phase in expected)
    benign_per_run = SUMO_STEPS - attack_per_run
    capacity = parent_capacity_audit(
        parents, development, sources, benign_per_run, attack_per_run
    )
    if not all(row["capacity_passed"] for row in capacity):
        failed = [f"{row['source_file']}:{row['pool']}" for row in capacity if not row["capacity_passed"]]
        raise RuntimeError("Insufficient leakage-safe HCRL capacity: " + ", ".join(failed))

    hashes_before = hash_inventory(files, root)
    if args.resume:
        if not REGISTRY.exists():
            raise FileNotFoundError("No Step 30E lock exists; omit --resume for the first run")
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        out = root / registry["run_directory"]
        lock = json.loads((out / "confirmation_protocol_lock.json").read_text(encoding="utf-8"))
        locked_hashes = {row["input_key"]: row["sha256"] for row in lock["input_hashes"]}
        current_hashes = {row["input_key"]: row["sha256"] for row in hashes_before}
        if locked_hashes != current_hashes:
            raise RuntimeError("Resume refused: locked code or inputs changed")
        if (out / "final_confirmation_manifest.json").exists():
            raise RuntimeError("Step 30E already completed; confirmation may not be rerun")
    else:
        AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
        if REGISTRY.exists():
            raise RuntimeError(
                f"A Step 30E confirmation lock already exists at {REGISTRY}. "
                "Do not rerun; use --resume only after a technical interruption."
            )
        out = AUDIT_ROOT / run_id()
        out.mkdir(parents=False, exist_ok=False)
        lock = {
            "lock_status": "LOCKED_BEFORE_CONFIRMATION_DATA_GENERATION",
            "locked_utc": utc_now(),
            "run_directory": str(out.relative_to(root)),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "sumo_steps_per_seed": SUMO_STEPS,
            "hcrl_sampling": "unique source-specific parent windows disjoint from all Step 24/25 replay IDs and from all confirmation runs",
            "temporal_rule": temporal_name,
            "temporal_parameters": {
                "required_hits": temporal_parameters[0],
                "history_windows": temporal_parameters[1],
                "hold_clean_windows": temporal_parameters[2],
            },
            "strongest_single_source_baseline": baseline,
            "baseline_selection_partition": "first three historical development seeds only",
            "hypothesis_criteria": {
                "H1": "paired F1 difference lower 95% CI > 0 and one-sided sign-flip p < 0.05",
                "H2": "mean F1 >= 0.90 and mean FPR <= 0.05 in every density",
                "H3": "low-density CAN recall-gain lower 95% CI > 0 and temporal CAN FPR <= 0.05",
                "H4": "all source-loss paired F1-loss upper 95% CIs <= 0.10",
                "H5": "zero unsafe full-ALLOW under detected, undetected and conflicting compromise; detected and unobservable cases reported separately",
            },
            "external_validity": "internal held-out replay confirmation; HCRL corpus reused with disjoint windows; ROAD remains independent negative external evidence",
            "posthoc_tuning_permitted": False,
            "input_hashes": hashes_before,
        }
        write_json(out / "confirmation_protocol_lock.json", lock)
        write_json(
            REGISTRY,
            {
                "status": "confirmation_consumed_or_in_progress",
                "created_utc": utc_now(),
                "run_directory": str(out.relative_to(root)),
                "script_sha256": sha256(Path(__file__).resolve()),
                "instruction": "Never delete this lock to rerun confirmation.",
            },
        )
        write_csv(out / "input_hashes_before.csv", hashes_before)
        write_csv(out / "baseline_selection_locked.csv", baseline_rows)
        write_csv(out / "hcrl_disjoint_capacity_audit.csv", capacity)
    write_json(out / "run_status.json", {"status": "running", "updated_utc": utc_now()})

    confirmation_dir = out / "confirmation_inputs"
    sumo_manifest = []
    sumo_data: dict[int, list[dict[str, str]]] = {}
    for index, seed in enumerate(CONFIRMATION_SEEDS, start=1):
        seed_dir = confirmation_dir / f"seed_{seed}"
        output_file = seed_dir / f"sumo_context_attacks_seed_{seed}.csv"
        scenario_dir = seed_dir / "scenario"
        if output_file.exists():
            rows = step19.read_csv_rows(output_file)
            source = "resumed_existing_locked_seed_file"
        else:
            print(f"\n[{index}/{len(CONFIRMATION_SEEDS)}] Generating untouched SUMO seed {seed}")
            config = step07.build_scenario(
                scenario_dir, steps=SUMO_STEPS, seed=seed, rebuild=True
            )
            step08.run_experiment(
                root,
                step07,
                config,
                output_file,
                steps=SUMO_STEPS,
                seed=seed,
                use_gui=False,
            )
            rows = step19.read_csv_rows(output_file)
            source = "generated_after_protocol_lock"
        if len(rows) != SUMO_STEPS:
            raise RuntimeError(f"Seed {seed} produced {len(rows)} rows; expected {SUMO_STEPS}")
        step19.validate_columns(rows[0], step19.REQUIRED_SUMO_COLUMNS, output_file.name)
        sumo_data[seed] = rows
        sumo_manifest.append(
            {
                "seed": seed,
                "rows": len(rows),
                "source": source,
                "relative_path": str(output_file.relative_to(root)),
                "sha256": sha256(output_file),
                "can_attack_phase_rows": sum(row["phase"] in CAN_ATTACK_PHASES for row in rows),
            }
        )
    write_csv(out / "confirmation_sumo_manifest.csv", sumo_manifest)

    historical_used = {
        source: set(
            development.loc[
                development["source_file"] == source, "parent_window_index"
            ].astype(int)
        )
        for source in sources
    }
    used_by_source = {source: set(values) for source, values in historical_used.items()}
    decisions: list[dict[str, Any]] = []
    replay_audit: list[dict[str, Any]] = []
    total = len(CONFIRMATION_SEEDS) * len(sources) * len(DENSITY_ORDER)
    number = 0
    for seed in CONFIRMATION_SEEDS:
        for source in sources:
            for density in ALLOCATION_ORDER:
                number += 1
                print(f"[{number}/{total}] seed={seed}, source={source}, density={density}")
                replay, audit = build_disjoint_replay(
                    sumo_data[seed],
                    parents,
                    source,
                    density,
                    seed,
                    used_by_source,
                    step19,
                    step24,
                )
                evaluated = step24.evaluate_replay(replay, guards[source], step19)
                replay_audit.append(
                    {
                        **audit,
                        "startup_quality_warning": guards[source]["warning"],
                        "startup_guard_score": guards[source]["score"],
                        "startup_guard_threshold": guards[source]["threshold"],
                    }
                )
                for row in evaluated:
                    decisions.append(
                        {"seed": seed, "density_scenario": density, **row}
                    )
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(out / "final_confirmation_policy_decisions.csv", index=False)
    write_csv(out / "final_confirmation_replay_audit.csv", replay_audit)

    predictions = add_method_predictions(decision_frame, temporal_parameters, step26)
    per_run, per_phase = metric_tables(predictions)
    aggregate = aggregate_metrics(per_run)
    primary = primary_endpoint(per_run, per_phase)
    per_run.to_csv(out / "final_confirmation_per_run_metrics.csv", index=False)
    per_phase.to_csv(out / "final_confirmation_per_phase_metrics.csv", index=False)
    aggregate.to_csv(out / "final_confirmation_aggregate_metrics.csv", index=False)
    primary.to_csv(out / "final_confirmation_primary_endpoint.csv", index=False)

    robustness_input = predictions.copy()
    robustness_rows = step30c.run_metrics(robustness_input)
    for row in robustness_rows:
        row["analysis_partition"] = "untouched_confirmation"
    robustness_aggregate = step30c.aggregate_metrics(
        robustness_rows, np.random.default_rng(RANDOM_SEED)
    )
    source_checks = pd.DataFrame(step30c.safety_checks(robustness_aggregate))
    write_csv(out / "final_confirmation_source_robustness_run_metrics.csv", robustness_rows)
    write_csv(out / "final_confirmation_source_robustness_aggregate.csv", robustness_aggregate)
    source_checks.to_csv(out / "final_confirmation_source_safety_checks.csv", index=False)

    # Preserve both previously audited action interfaces.  The exact Step 30C
    # interface omits the general startup warning; the frozen Step 25 action
    # sensitivity includes it.  Neither interface invents a source-quality
    # signal for an undetected false-healthy source.
    observability_rows = step30c2.observability_run_metrics(robustness_input)
    for row in observability_rows:
        row["analysis_partition"] = "untouched_confirmation_observability"
    observability_aggregate = step30c2.aggregate_observability(
        observability_rows, np.random.default_rng(RANDOM_SEED)
    )
    write_csv(
        out / "final_confirmation_observability_run_metrics.csv",
        observability_rows,
    )
    write_csv(
        out / "final_confirmation_observability_aggregate.csv",
        observability_aggregate,
    )

    hypotheses, comparisons, h2_density = assess_hypotheses(
        per_run, per_phase, primary, baseline, source_checks
    )
    write_csv(out / "final_confirmation_statistical_comparisons.csv", comparisons)
    write_csv(out / "final_confirmation_hypothesis_assessment.csv", hypotheses)
    write_csv(out / "final_confirmation_h2_density_checks.csv", h2_density)

    hashes_after = hash_inventory(files, root)
    changes = compare_hashes(hashes_before, hashes_after)
    write_csv(out / "input_hashes_after.csv", hashes_after)
    write_csv(out / "existing_artifact_immutability_check.csv", changes)
    unchanged = all(row["unchanged"] for row in changes)
    if not unchanged:
        raise RuntimeError("One or more locked project inputs changed during Step 30E")

    core = {row["hypothesis"]: row["assessment"] for row in hypotheses}
    all_supported = all(core.get(name) == "supported" for name in ("H1", "H2", "H3", "H4", "H5"))
    h5_boundary = core.get("H5_undetected_byzantine_boundary") == "not_observable"
    observability_frame = pd.DataFrame(observability_aggregate)
    step30c_interface = observability_frame[
        observability_frame["evidence_interface"] == "step30c_replay_interface"
    ]
    step25_interface = observability_frame[
        observability_frame["evidence_interface"] == "frozen_step25_action_interface"
    ]
    step30c_silent_attack_rows = int(step30c_interface["silent_attack_rows"].sum())
    step25_silent_attack_rows = int(step25_interface["silent_attack_rows"].sum())
    step25_silent_benign_rows = int(step25_interface["silent_benign_rows"].sum())
    step25_attack_rows = int(step25_interface["attack_rows"].sum())
    step25_benign_rows = int(step25_interface["benign_rows"].sum())
    acceptance = [
        {
            "criterion": "confirmation_protocol_completed_once",
            "observed_value": True,
            "required_value": True,
            "passed": True,
            "interpretation": "The locked confirmation executed without outcome-driven retuning.",
        },
        {
            "criterion": "historical_inputs_unchanged",
            "observed_value": unchanged,
            "required_value": True,
            "passed": unchanged,
            "interpretation": "Existing project artifacts remained immutable.",
        },
        {
            "criterion": "hcrl_parent_windows_disjoint",
            "observed_value": max(row["development_overlap_count"] for row in replay_audit),
            "required_value": 0,
            "passed": all(row["development_overlap_count"] == 0 for row in replay_audit),
            "interpretation": "No Step 24/25 parent window was reused in confirmation.",
        },
        {
            "criterion": "all_predeclared_hypotheses_supported",
            "observed_value": all_supported,
            "required_value": True,
            "passed": all_supported,
            "interpretation": "A failed hypothesis is retained as a result; it is not a failed experiment.",
        },
        {
            "criterion": "unqualified_universal_byzantine_claim_permitted",
            "observed_value": not h5_boundary and core.get("H5") == "supported",
            "required_value": True,
            "passed": not h5_boundary and core.get("H5") == "supported",
            "interpretation": "If false, claims must be conditional on observable source quality.",
        },
        {
            "criterion": "bounded_research_evidence_freeze_permitted",
            "observed_value": unchanged,
            "required_value": True,
            "passed": unchanged,
            "interpretation": "Negative findings may be frozen for thesis/publication with explicit boundaries.",
        },
    ]
    write_csv(out / "final_confirmation_acceptance.csv", acceptance)

    manifest = {
        "experiment": "Step 30E untouched final internal confirmation",
        "completed_utc": utc_now(),
        "run_directory": str(out.relative_to(root)),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "independent_replay_units": int(
            per_run[["seed", "source_file", "density_scenario"]].drop_duplicates().shape[0]
        ),
        "decision_rows": len(predictions),
        "hcrl_sources": sources,
        "hcrl_windows_reused_from_development": 0,
        "hcrl_windows_reused_between_confirmation_runs": 0,
        "selected_temporal_rule": temporal_name,
        "strongest_single_source_baseline": baseline,
        "all_hypotheses_supported": all_supported,
        "bounded_research_freeze_permitted": unchanged,
        "unqualified_universal_claims_permitted": all_supported and not h5_boundary,
        "external_validity": "internal disjoint-window/new-seed confirmation; not a new external vehicle dataset",
        "road_external_result": "independent negative external evidence retained",
        "claim_boundary": "detected compromise/conflict claims are conditional on observable quality; unobservable false-healthy sources require attestation/freshness/corroboration",
        "step30c_interface_silent_attack_rows": step30c_silent_attack_rows,
        "step25_warning_adjusted_silent_attack_rows": step25_silent_attack_rows,
        "step25_warning_adjusted_silent_attack_rate": safe_divide(
            step25_silent_attack_rows, step25_attack_rows
        ),
        "step25_naive_fail_safe_benign_restriction_rate": safe_divide(
            step25_silent_benign_rows, step25_benign_rows
        ),
        "posthoc_retuning_performed": False,
        "existing_project_artifacts_changed": sum(not row["unchanged"] for row in changes),
    }
    write_json(out / "final_confirmation_manifest.json", manifest)
    summary = {
        **manifest,
        "hypotheses": hypotheses,
        "acceptance": acceptance,
    }
    write_json(out / "final_confirmation_summary.json", summary)
    make_plot(
        primary,
        comparisons,
        hypotheses,
        out / "final_confirmation_summary.png",
    )
    write_json(out / "run_status.json", {"status": "completed", "updated_utc": utc_now()})
    archive = shutil.make_archive(str(out), "zip", root_dir=out)

    print("\n" + "=" * 92)
    print("Step 30E untouched final internal confirmation completed successfully.")
    print(f"Existing project artifacts changed: {manifest['existing_project_artifacts_changed']}")
    print(f"New SUMO seeds: {list(CONFIRMATION_SEEDS)}")
    print(f"Independent replay units: {manifest['independent_replay_units']}")
    print(f"Disjoint HCRL-window overlap with development: 0")
    print(f"Locked strongest single-source baseline: {baseline}")
    print("\nConfirmatory hypotheses:")
    for row in hypotheses:
        if row["hypothesis"] in {"H1", "H2", "H3", "H4", "H5"}:
            print(f"  {row['hypothesis']}: {row['assessment']} ({row['observed']})")
    print(f"\nBounded research evidence freeze permitted: {unchanged}")
    print(f"Unqualified universal claims permitted: {manifest['unqualified_universal_claims_permitted']}")
    print(f"Results directory: {out}")
    print(f"Results archive: {archive}")
    print("\nNext: send the terminal output, final_confirmation_manifest.json,")
    print("final_confirmation_acceptance.csv, hypothesis assessment, primary endpoint, and plot.")
    print("Do not run Step 31 until a fresh Step 30A audit and the claim boundaries are reviewed.")


if __name__ == "__main__":
    main()
