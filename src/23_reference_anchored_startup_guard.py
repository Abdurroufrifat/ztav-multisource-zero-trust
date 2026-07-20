#!/usr/bin/env python3
"""Confirm a reference-anchored startup guard for the Step 21 micro-gate.

Step 22 showed that the two-hit multiscale detector remains accurate even when
its 50-window startup baseline is contaminated.  However, the original guard
rejected 20% of clean normal holdout sessions because a discrete score tied the
calibration threshold and the decision used ``>=``.  It also described only
within-startup consistency, so a fully poisoned but internally consistent
startup could evade it.

This confirmation stage corrects both problems:

* startup statistics are anchored to a healthy reference learned only from
  normal calibration sessions;
* only unusually *large* deviations are suspicious (one-sided score);
* a finite-sample conformal order statistic selects the threshold; and
* rejection is strict (score > threshold), so threshold ties are accepted.

Attack labels are used only to construct controlled poisoning trials and to
report sensitivity.  They are never used to fit the reference, score scales,
or threshold.

Run from D:\\ztav_project after Step 22:

    .\\.venv\\Scripts\\python.exe src\\23_reference_anchored_startup_guard.py

This is an exploratory research test, not production automotive software.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTAMINATION_COUNTS = (0, 1, 2, 5, 10, 20, 25, 30, 40, 50)
REPETITIONS = 5
TARGET_FPR = 0.05
EPSILON = 1e-12
FEATURE_NAMES = (
    "internal_dispersion_q75",
    "internal_half_gap_q75",
    "internal_range_q75",
    "reference_center_shift_q75",
    "reference_window_deviation_q90",
    "reference_window_deviation_max",
    "reference_outlier_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm a reference-anchored micro startup guard."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def locate_script(project_root: Path, name: str) -> Path:
    for candidate in (project_root / "src" / name, project_root / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find required script: {name}")


def load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    center = np.median(values, axis=axis)
    return 1.4826 * np.median(np.abs(values - center), axis=axis)


def startup_features(
    bootstrap: np.ndarray,
    floors: np.ndarray,
    reference_center: np.ndarray,
    reference_scale: np.ndarray,
    micro_threshold: float,
) -> np.ndarray:
    midpoint = len(bootstrap) // 2
    center = np.median(bootstrap, axis=0)
    local_scale = robust_scale(bootstrap, axis=0)
    half_gap = np.abs(
        np.median(bootstrap[:midpoint], axis=0)
        - np.median(bootstrap[midpoint:], axis=0)
    )
    value_range = np.ptp(bootstrap, axis=0)
    reference_deviations = np.abs(bootstrap - reference_center) / np.maximum(
        reference_scale, EPSILON
    )
    per_window = np.quantile(reference_deviations, 0.75, axis=1)
    center_shift = np.abs(center - reference_center) / np.maximum(
        reference_scale, EPSILON
    )
    return np.asarray(
        [
            np.quantile(local_scale / np.maximum(floors, EPSILON), 0.75),
            np.quantile(half_gap / np.maximum(floors, EPSILON), 0.75),
            np.quantile(value_range / np.maximum(floors, EPSILON), 0.75),
            np.quantile(center_shift, 0.75),
            np.quantile(per_window, 0.90),
            np.max(per_window),
            np.mean(per_window >= micro_threshold),
        ],
        dtype=float,
    )


def normal_sessions(
    normal: pd.DataFrame,
    step21: ModuleType,
) -> list[pd.DataFrame]:
    ordered = normal.sort_values("window_index").reset_index(drop=True)
    count = len(ordered) // step21.PSEUDO_SESSION_MICRO_WINDOWS
    if count < 20:
        raise ValueError("At least 20 complete normal pseudo-sessions are required")
    sessions = []
    for index in range(count):
        start = index * step21.PSEUDO_SESSION_MICRO_WINDOWS
        end = start + step21.PSEUDO_SESSION_MICRO_WINDOWS
        sessions.append(ordered.iloc[start:end].copy())
    return sessions


def finite_sample_threshold(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Split-conformal upper quantile with strict threshold comparison."""
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
    return float(np.sort(scores)[rank - 1]), rank


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def feature_row(
    session_id: str,
    partition: str,
    values: np.ndarray,
    score: float,
    threshold: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "session_id": session_id,
        "partition": partition,
        "guard_score": score,
        "guard_threshold": threshold,
        "guard_rejected": bool(score > threshold),
        "decision_rule": "guard_score > guard_threshold",
    }
    row.update({name: float(values[i]) for i, name in enumerate(FEATURE_NAMES)})
    return row


def plot_results(
    summary: pd.DataFrame,
    macro: pd.DataFrame,
    normal_audit: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for source, group in summary.groupby("source_file", sort=True):
        group = group.sort_values("contamination_fraction")
        axes[0].plot(
            group["contamination_fraction"],
            group["guard_rejection_rate"],
            marker="o",
            label=source,
        )
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Per-source poisoning detection", ylabel="Rejection rate")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        macro["contamination_fraction"],
        macro["guard_rejection_rate_macro"],
        marker="o",
        color="tab:red",
    )
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Macro poisoning detection", ylabel="Rejection rate")

    partitions = ["guard_calibration", "guard_holdout"]
    values = [
        normal_audit.loc[normal_audit["partition"] == part, "guard_rejected"].mean()
        for part in partitions
    ]
    axes[2].bar(partitions, values, color=["tab:blue", "tab:green"])
    axes[2].axhline(TARGET_FPR, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Normal-session false rejection", ylabel="Rejection rate")

    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_xlabel("Poisoned startup-window fraction")
    axes[1].set_xlabel("Poisoned startup-window fraction")
    axes[2].tick_params(axis="x", rotation=15)
    figure.suptitle("Reference-anchored startup guard confirmation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    step21 = load_script(
        locate_script(project_root, "21_multiscale_sparse_can_gate.py"),
        "ztav_step21_reference_guard",
    )
    cache_path = (
        project_root / "data" / "processed" / "car_hacking_windows_w20_structural.csv"
    )
    calibration_path = (
        project_root
        / "results"
        / "multiscale_sparse_can_gate"
        / "micro_calibration_summary.csv"
    )
    if not cache_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("Step 21 cache or calibration summary is missing")
    data = pd.read_csv(cache_path)
    normal = data[data["source_file"] == step21.NORMAL_SOURCE].copy()
    attacks = data[data["source_file"] != step21.NORMAL_SOURCE].copy()
    floors = step21.scale_floors(normal)
    micro_calibration = pd.read_csv(calibration_path)
    micro_threshold = float(micro_calibration.iloc[0]["micro_deviation_threshold"])

    sessions = normal_sessions(normal, step21)
    split = len(sessions) // 2
    calibration_sessions = sessions[:split]
    reference_values = pd.concat(calibration_sessions, ignore_index=True)[
        list(step21.STRUCTURAL_FEATURES)
    ].to_numpy(dtype=float)
    reference_center = np.median(reference_values, axis=0)
    reference_scale = np.maximum(robust_scale(reference_values, axis=0), floors)

    session_features = []
    for session in sessions:
        bootstrap = session.iloc[: step21.BOOTSTRAP_MICRO_WINDOWS][
            list(step21.STRUCTURAL_FEATURES)
        ].to_numpy(dtype=float)
        session_features.append(
            startup_features(
                bootstrap,
                floors,
                reference_center,
                reference_scale,
                micro_threshold,
            )
        )
    feature_matrix = np.vstack(session_features)
    calibration_matrix = feature_matrix[:split]
    guard_center = np.median(calibration_matrix, axis=0)
    guard_mad = robust_scale(calibration_matrix, axis=0)
    guard_std = calibration_matrix.std(axis=0)
    guard_scale = np.where(
        guard_mad > EPSILON,
        guard_mad,
        np.where(guard_std > EPSILON, guard_std, 1.0),
    )
    # One-sided: only increases above the healthy calibration center are risky.
    scores = np.max((feature_matrix - guard_center) / guard_scale, axis=1)
    threshold, conformal_rank = finite_sample_threshold(
        scores[:split], TARGET_FPR
    )
    normal_rows = []
    for index, values in enumerate(feature_matrix):
        partition = "guard_calibration" if index < split else "guard_holdout"
        normal_rows.append(
            feature_row(
                f"normal_micro_session_{index:03d}",
                partition,
                values,
                float(scores[index]),
                threshold,
            )
        )
    normal_audit = pd.DataFrame(normal_rows)

    clean_rows: list[dict[str, object]] = []
    poisoning_rows: list[dict[str, object]] = []
    total = attacks["source_file"].nunique() * len(CONTAMINATION_COUNTS) * args.repetitions
    run_number = 0
    for source_file, source in attacks.groupby("source_file", sort=True):
        source = source.sort_values("window_index").reset_index(drop=True)
        clean_frame = source.iloc[: step21.BOOTSTRAP_MICRO_WINDOWS]
        if int(clean_frame["attack_frame_count"].sum()) != 0:
            raise ValueError(f"Clean startup for {source_file} contains attack frames")
        clean = clean_frame[list(step21.STRUCTURAL_FEATURES)].to_numpy(dtype=float)
        attack_pool = source[source["binary_target"] == 1][
            list(step21.STRUCTURAL_FEATURES)
        ].to_numpy(dtype=float)
        if len(attack_pool) < step21.BOOTSTRAP_MICRO_WINDOWS:
            raise ValueError(f"Too few attack windows for {source_file}")
        clean_values = startup_features(
            clean, floors, reference_center, reference_scale, micro_threshold
        )
        clean_score = float(np.max((clean_values - guard_center) / guard_scale))
        clean_rows.append(
            {
                "source_file": source_file,
                **feature_row(
                    source_file,
                    "clean_attack_capture_startup",
                    clean_values,
                    clean_score,
                    threshold,
                ),
            }
        )
        for count in CONTAMINATION_COUNTS:
            for repetition in range(args.repetitions):
                run_number += 1
                print(
                    f"[{run_number}/{total}] source={source_file}, "
                    f"poisoned={count}/{step21.BOOTSTRAP_MICRO_WINDOWS}, "
                    f"rep={repetition + 1}"
                )
                seed = 42 + repetition + count * 1_000 + sum(source_file.encode("utf-8"))
                rng = np.random.default_rng(seed)
                contaminated = clean.copy()
                if count:
                    positions = rng.choice(
                        step21.BOOTSTRAP_MICRO_WINDOWS, size=count, replace=False
                    )
                    samples = rng.choice(len(attack_pool), size=count, replace=False)
                    contaminated[positions] = attack_pool[samples]
                values = startup_features(
                    contaminated,
                    floors,
                    reference_center,
                    reference_scale,
                    micro_threshold,
                )
                score = float(np.max((values - guard_center) / guard_scale))
                row: dict[str, object] = {
                    "source_file": source_file,
                    "contaminated_bootstrap_windows": count,
                    "contamination_fraction": count / step21.BOOTSTRAP_MICRO_WINDOWS,
                    "repetition": repetition,
                    "random_seed": seed,
                    "guard_score": score,
                    "guard_threshold": threshold,
                    "guard_rejected": bool(score > threshold),
                    "decision_rule": "guard_score > guard_threshold",
                }
                row.update(
                    {name: float(values[i]) for i, name in enumerate(FEATURE_NAMES)}
                )
                poisoning_rows.append(row)

    poisoning = pd.DataFrame(poisoning_rows)
    summary = (
        poisoning.groupby(
            ["source_file", "contaminated_bootstrap_windows", "contamination_fraction"],
            as_index=False,
            sort=True,
        )
        .agg(
            repetitions=("repetition", "size"),
            guard_rejection_rate=("guard_rejected", "mean"),
            guard_score_mean=("guard_score", "mean"),
            guard_score_min=("guard_score", "min"),
            guard_score_max=("guard_score", "max"),
        )
    )
    macro = (
        summary.groupby(
            ["contaminated_bootstrap_windows", "contamination_fraction"],
            as_index=False,
            sort=True,
        )
        .agg(
            sources=("source_file", "size"),
            guard_rejection_rate_macro=("guard_rejection_rate", "mean"),
        )
    )
    boundary_rows = []
    for source_file, group in summary.groupby("source_file", sort=True):
        group = group.sort_values("contaminated_bootstrap_windows")
        detected = group[
            (group["contaminated_bootstrap_windows"] > 0)
            & (group["guard_rejection_rate"] >= 0.90)
        ]
        boundary_rows.append(
            {
                "source_file": source_file,
                "first_count_guard_rejection_at_least_0_90": (
                    int(detected.iloc[0]["contaminated_bootstrap_windows"])
                    if len(detected)
                    else -1
                ),
                "clean_start_guard_rejection_rate": float(
                    group.loc[
                        group["contaminated_bootstrap_windows"] == 0,
                        "guard_rejection_rate",
                    ].iloc[0]
                ),
            }
        )
    boundaries = pd.DataFrame(boundary_rows)

    output_dir = project_root / "results" / "reference_anchored_startup_guard"
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_audit.to_csv(output_dir / "reference_guard_normal_audit.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(
        output_dir / "reference_guard_clean_source_audit.csv", index=False
    )
    poisoning.to_csv(output_dir / "reference_guard_poisoning_runs.csv", index=False)
    summary.to_csv(output_dir / "reference_guard_poisoning_summary.csv", index=False)
    macro.to_csv(output_dir / "reference_guard_macro_summary.csv", index=False)
    boundaries.to_csv(output_dir / "reference_guard_failure_boundaries.csv", index=False)

    reference_rows = []
    for index, name in enumerate(step21.STRUCTURAL_FEATURES):
        reference_rows.append(
            {
                "feature": name,
                "healthy_reference_center": reference_center[index],
                "healthy_reference_scale": reference_scale[index],
                "guard_feature": "",
                "guard_center": "",
                "guard_scale": "",
            }
        )
    for index, name in enumerate(FEATURE_NAMES):
        reference_rows.append(
            {
                "feature": "",
                "healthy_reference_center": "",
                "healthy_reference_scale": "",
                "guard_feature": name,
                "guard_center": guard_center[index],
                "guard_scale": guard_scale[index],
            }
        )
    pd.DataFrame(reference_rows).to_csv(
        output_dir / "reference_guard_parameters.csv", index=False
    )

    calibration_rejections = int(
        normal_audit.loc[
            normal_audit["partition"] == "guard_calibration", "guard_rejected"
        ].sum()
    )
    holdout = normal_audit[normal_audit["partition"] == "guard_holdout"]
    holdout_rejections = int(holdout["guard_rejected"].sum())
    holdout_rate = holdout_rejections / len(holdout)
    holdout_low, holdout_high = wilson_interval(holdout_rejections, len(holdout))
    clean_source_rejections = int(pd.DataFrame(clean_rows)["guard_rejected"].sum())
    threshold_rows = [
        {
            "target_fpr": TARGET_FPR,
            "normal_calibration_sessions": split,
            "normal_holdout_sessions": len(sessions) - split,
            "finite_sample_rank": conformal_rank,
            "guard_threshold": threshold,
            "decision_rule": "guard_score > guard_threshold",
            "calibration_rejections": calibration_rejections,
            "normal_holdout_rejections": holdout_rejections,
            "normal_holdout_rejection_rate": holdout_rate,
            "normal_holdout_wilson_95_low": holdout_low,
            "normal_holdout_wilson_95_high": holdout_high,
            "clean_attack_capture_startups": len(clean_rows),
            "clean_attack_capture_rejections": clean_source_rejections,
            "micro_threshold_frozen": micro_threshold,
        }
    ]
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "reference_guard_threshold_summary.csv", index=False
    )
    manifest = pd.DataFrame(
        [
            {"item": "experiment", "value": "reference-anchored startup guard"},
            {"item": "score_direction", "value": "one-sided upper deviation"},
            {"item": "threshold_method", "value": "finite-sample split conformal"},
            {"item": "decision_rule", "value": "strict greater-than"},
            {"item": "contamination_counts", "value": ";".join(map(str, CONTAMINATION_COUNTS))},
            {"item": "repetitions", "value": args.repetitions},
            {"item": "label_usage", "value": "controlled poisoning construction/evaluation only"},
            {"item": "external_validity", "value": "exploratory HCRL; independent data still required"},
        ]
    )
    manifest.to_csv(output_dir / "reference_guard_manifest.csv", index=False)
    plot_results(
        summary,
        macro,
        normal_audit,
        output_dir / "reference_guard_sensitivity.png",
    )

    print("\n" + "=" * 84)
    print("Reference-anchored startup guard confirmation completed successfully.")
    print(f"Normal sessions: calibration={split}, holdout={len(sessions) - split}")
    print(f"Finite-sample guard threshold: {threshold:.6f} (strict >)")
    print(
        f"Normal holdout rejection: {holdout_rejections}/{len(holdout)} "
        f"= {holdout_rate:.4f}; Wilson 95% CI [{holdout_low:.4f}, {holdout_high:.4f}]"
    )
    print(
        f"Clean attack-capture startup rejection: "
        f"{clean_source_rejections}/{len(clean_rows)}"
    )
    print("\nFirst poisoned-window count reaching >= 90% rejection:")
    print(boundaries.to_string(index=False))
    print("\nMacro guard sensitivity:")
    print(macro.to_string(index=False))
    print(f"\nResults directory: {output_dir}")
    print("\nNext: decide whether the corrected guard is ready for final policy integration.")


if __name__ == "__main__":
    main()
