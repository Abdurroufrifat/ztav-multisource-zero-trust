#!/usr/bin/env python3
"""Step 30F0: non-consuming GEM-CAN schema and capacity audit.

This additive preflight validates the independent GEM-CAN dataset before any
frozen-model external confirmation is allowed.  It deliberately does not load
a project model, train or calibrate anything, choose a threshold, run SUMO,
reserve a confirmation partition, or overwrite a historical result.

The audit:

* ignores macOS ``__MACOSX`` and ``._*`` metadata files;
* verifies the raw and attack-type-labelled attack CSVs are row-aligned;
* canonicalises the published attack timestamp typo ``tiTestaTp``;
* logs malformed rows instead of silently repairing them;
* retains repeated CAN frames (they can be legitimate traffic);
* builds non-overlapping, timestamp-ordered 100-frame windows separately for
  the normal and attack captures;
* measures conservative capacity using unique feature-window hashes; and
* separates high-density pooled feasibility from unsupported sparse and
  attack-family-specific claims.

This script always keeps Step 31 locked.  A failed sparse-capacity result is a
dataset limitation, not a reason to synthesize confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ztav_matplotlib_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def infer_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "src" else here


ROOT = infer_root()
DEFAULT_DATASET_ROOT = ROOT / "data" / "external" / "gem_can" / "extracted"
RESULTS_RELATIVE = Path("results/gem_can_schema_capacity_audit")

WINDOW_SIZE = 100
EXPECTED_FILES = {
    "GEM_Attack_Scenario_Raw.csv": "attack_raw",
    "GEM_Attack_Scenario_Raw_Labeled.csv": "attack_labeled",
    "GEM_Normal_Driving_Raw.csv": "normal_raw",
}
CAN_COLUMNS = ["arbitration_id", "dlc"] + [f"data{i}" for i in range(8)]
RAW_COLUMNS = ["timestamp"] + CAN_COLUMNS + ["label"]
ATTACK_COLUMNS = RAW_COLUMNS + ["attack_type"]
EXPECTED_LABELS = {"R", "T"}
EXPECTED_ATTACK_TYPES = {
    "Normal",
    "DoS",
    "Brake Tampering",
    "Steering Tampering",
}
MAX_MALFORMED_FRACTION = 0.0001  # 0.01%; logs the known trailing normal row.

# Predeclared screening requirements for deciding what kind of external
# confirmation the independent dataset can support.  These are evidence-design
# gates, not a post-hoc statistical power calculation.
DENSITY_REQUIREMENTS = {
    # (observed natural windows, unique feature-window clusters).  The later
    # confirmation must cluster its intervals by feature hash; it may not treat
    # repeated DoS sequences as independent observations.
    "benign": (200, 200),
    "low_1_5": (100, 30),
    "medium_6_20": (100, 30),
    "high_21_100": (200, 30),
}
FAMILY_EXCLUSIVE_REQUIREMENT = 30
DENSITY_ORDER = ["benign", "low_1_5", "medium_6_20", "high_21_100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GEM-CAN without consuming external confirmation evidence."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Directory recursively containing the three readable GEM-CAN CSVs.",
    )
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


def digest_text(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def discover_files(dataset_root: Path) -> dict[str, Path]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"GEM-CAN dataset directory not found: {dataset_root}. "
            "Extract GEM_CAN_Dataset.zip first."
        )
    found: dict[str, list[Path]] = {name: [] for name in EXPECTED_FILES}
    for path in dataset_root.rglob("*.csv"):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if path.name in found:
            found[path.name].append(path)
    missing = [name for name, paths in found.items() if not paths]
    duplicated = {name: paths for name, paths in found.items() if len(paths) > 1}
    if missing:
        raise FileNotFoundError(f"Missing required GEM-CAN files: {missing}")
    if duplicated:
        detail = "; ".join(
            f"{name}: {[str(path) for path in paths]}"
            for name, paths in duplicated.items()
        )
        raise RuntimeError(f"Multiple readable copies found; keep one dataset copy: {detail}")
    return {name: paths[0] for name, paths in found.items()}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def canonical_column_name(name: str) -> str:
    stripped = name.strip()
    if stripped in {"timestamp", "tiTestaTp"}:
        return "timestamp"
    if stripped == "Attack_Type":
        return "attack_type"
    return stripped


def require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalise_and_validate(
    source: pd.DataFrame, source_file: str, require_attack_type: bool
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    frame = source.copy()
    frame.columns = [canonical_column_name(str(value)) for value in frame.columns]
    required = set(ATTACK_COLUMNS if require_attack_type else RAW_COLUMNS)
    require_columns(frame.columns, required, source_file)
    frame = frame[list(ATTACK_COLUMNS if require_attack_type else RAW_COLUMNS)].copy()
    frame["source_row_number"] = frame.index + 2  # one-indexed CSV line incl. header

    text_columns = ["arbitration_id", "label"] + [f"data{i}" for i in range(8)]
    if require_attack_type:
        text_columns.append("attack_type")
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()
    frame["arbitration_id"] = frame["arbitration_id"].str.upper()
    for column in [f"data{i}" for i in range(8)]:
        frame[column] = frame[column].str.upper()
    frame["label"] = frame["label"].str.upper()

    timestamp_numeric = pd.to_numeric(frame["timestamp"], errors="coerce")
    dlc_numeric = pd.to_numeric(frame["dlc"], errors="coerce")
    timestamp_ok = timestamp_numeric.notna()
    id_ok = frame["arbitration_id"].str.fullmatch(r"[0-9A-F]{1,8}", na=False)
    dlc_ok = dlc_numeric.notna() & dlc_numeric.between(0, 8) & dlc_numeric.mod(1).eq(0)
    label_ok = frame["label"].isin(EXPECTED_LABELS)

    byte_ok = pd.Series(True, index=frame.index)
    for index in range(8):
        column = f"data{index}"
        syntactically_valid = frame[column].str.fullmatch(r"[0-9A-F]{1,2}", na=False)
        required_byte = dlc_numeric.gt(index)
        byte_ok &= (~required_byte) | syntactically_valid

    if require_attack_type:
        attack_type_ok = frame["attack_type"].isin(EXPECTED_ATTACK_TYPES)
        label_type_consistent = (
            frame["label"].eq("R") & frame["attack_type"].eq("Normal")
        ) | (frame["label"].eq("T") & ~frame["attack_type"].eq("Normal"))
    else:
        attack_type_ok = pd.Series(True, index=frame.index)
        label_type_consistent = pd.Series(True, index=frame.index)

    checks = {
        "timestamp_parseable": timestamp_ok,
        "arbitration_id_hex": id_ok,
        "dlc_integer_0_to_8": dlc_ok,
        "payload_matches_dlc": byte_ok,
        "label_recognised": label_ok,
        "attack_type_recognised": attack_type_ok,
        "label_attack_type_consistent": label_type_consistent,
    }
    valid = pd.Series(True, index=frame.index)
    for mask in checks.values():
        valid &= mask

    malformed_rows: list[dict[str, Any]] = []
    for idx in frame.index[~valid]:
        malformed_rows.append(
            {
                "source_file": source_file,
                "source_row_number": int(frame.at[idx, "source_row_number"]),
                "failure_reasons": ";".join(
                    name for name, mask in checks.items() if not bool(mask.at[idx])
                ),
                "raw_row_json": json.dumps(
                    {
                        key: (None if pd.isna(value) else str(value))
                        for key, value in source.loc[idx].to_dict().items()
                    },
                    sort_keys=True,
                ),
            }
        )

    check_rows: list[dict[str, Any]] = []
    for name, mask in checks.items():
        failed = int((~mask).sum())
        check_rows.append(
            {
                "source_file": source_file,
                "check": name,
                "rows_checked": len(frame),
                "failed_rows": failed,
                "passed": failed == 0,
            }
        )

    clean = frame.loc[valid].copy()
    clean["timestamp"] = timestamp_numeric.loc[valid].astype(float)
    clean["dlc"] = dlc_numeric.loc[valid].astype(int)
    if not require_attack_type:
        clean["attack_type"] = "Normal"
    return clean, pd.DataFrame(malformed_rows), check_rows


def density_band(attack_frames: int) -> str:
    if attack_frames == 0:
        return "benign"
    if attack_frames <= 5:
        return "low_1_5"
    if attack_frames <= 20:
        return "medium_6_20"
    return "high_21_100"


def build_windows(frame: pd.DataFrame, capture: str) -> pd.DataFrame:
    # Timestamp is the declared sequencing field.  Stable sorting preserves the
    # published order for equal timestamps while correcting recorded inversions.
    ordered = frame.sort_values(["timestamp", "source_row_number"], kind="stable")
    rows: list[dict[str, Any]] = []
    complete_windows = len(ordered) // WINDOW_SIZE
    for window_index in range(complete_windows):
        window = ordered.iloc[
            window_index * WINDOW_SIZE : (window_index + 1) * WINDOW_SIZE
        ]
        attack_frames = int(window["label"].ne("R").sum())
        families = sorted(
            value
            for value in window.loc[
                window["attack_type"].ne("Normal"), "attack_type"
            ].dropna().unique()
        )
        feature_lines = window[CAN_COLUMNS].astype(str).agg(",".join, axis=1)
        labelled_lines = window[
            CAN_COLUMNS + ["label", "attack_type"]
        ].astype(str).agg(",".join, axis=1)
        rows.append(
            {
                "window_id": f"gem_{capture}_w100_{window_index:06d}",
                "capture": capture,
                "window_index": window_index,
                "window_size": WINDOW_SIZE,
                "start_timestamp": float(window["timestamp"].iloc[0]),
                "end_timestamp": float(window["timestamp"].iloc[-1]),
                "first_source_row_number": int(window["source_row_number"].iloc[0]),
                "last_source_row_number": int(window["source_row_number"].iloc[-1]),
                "attack_frame_count": attack_frames,
                "attack_fraction": attack_frames / WINDOW_SIZE,
                "density_band": density_band(attack_frames),
                "binary_target": int(attack_frames > 0),
                "attack_families": "|".join(families) if families else "benign",
                "attack_family_count": len(families),
                "feature_window_sha256": digest_text(feature_lines),
                "labelled_window_sha256": digest_text(labelled_lines),
            }
        )
    return pd.DataFrame(rows)


def density_capacity(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for density in DENSITY_ORDER:
        selected = windows[windows["density_band"].eq(density)]
        natural = len(selected)
        unique_features = selected["feature_window_sha256"].nunique()
        natural_required, unique_required = DENSITY_REQUIREMENTS[density]
        rows.append(
            {
                "density_band": density,
                "natural_nonoverlapping_windows": natural,
                "unique_feature_windows": unique_features,
                "duplicate_feature_windows": natural - unique_features,
                "predeclared_natural_window_minimum": natural_required,
                "predeclared_unique_cluster_minimum": unique_required,
                "capacity_passed": (
                    natural >= natural_required and unique_features >= unique_required
                ),
                "requirement_role": (
                    "healthy external false-positive endpoint"
                    if density == "benign"
                    else "independent natural attack-density endpoint"
                ),
            }
        )
    return pd.DataFrame(rows)


def family_capacity(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in sorted(EXPECTED_ATTACK_TYPES - {"Normal"}):
        contains = windows["attack_families"].str.split("|").apply(
            lambda values: family in values
        )
        exclusive = contains & windows["attack_family_count"].eq(1)
        selected = windows[contains]
        exclusive_selected = windows[exclusive]
        rows.append(
            {
                "attack_family": family,
                "windows_containing_family": len(selected),
                "unique_windows_containing_family": selected[
                    "feature_window_sha256"
                ].nunique(),
                "exclusive_family_windows": len(exclusive_selected),
                "unique_exclusive_family_windows": exclusive_selected[
                    "feature_window_sha256"
                ].nunique(),
                "predeclared_unique_exclusive_minimum": FAMILY_EXCLUSIVE_REQUIREMENT,
                "family_specific_confirmation_passed": exclusive_selected[
                    "feature_window_sha256"
                ].nunique()
                >= FAMILY_EXCLUSIVE_REQUIREMENT,
            }
        )
    return pd.DataFrame(rows)


def save_plot(
    density: pd.DataFrame, families: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["#4c78a8", "#f58518", "#eeca3b", "#e45756"]
    bars = axes[0].bar(
        density["density_band"], density["unique_feature_windows"], color=colors
    )
    axes[0].set_title("Unique natural 100-frame windows")
    axes[0].set_ylabel("Unique feature-window hashes")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].bar_label(bars)
    for index, row in density.reset_index(drop=True).iterrows():
        axes[0].hlines(
            row["predeclared_unique_cluster_minimum"],
            index - 0.35,
            index + 0.35,
            colors="black",
            linestyles="dashed",
            linewidth=1,
        )

    x = range(len(families))
    axes[1].bar(
        [value - 0.18 for value in x],
        families["unique_windows_containing_family"],
        width=0.36,
        label="Contains family",
        color="#54a24b",
    )
    axes[1].bar(
        [value + 0.18 for value in x],
        families["unique_exclusive_family_windows"],
        width=0.36,
        label="Exclusive family",
        color="#b279a2",
    )
    axes[1].axhline(
        FAMILY_EXCLUSIVE_REQUIREMENT,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label="Exclusive minimum",
    )
    axes[1].set_xticks(list(x), families["attack_family"], rotation=20, ha="right")
    axes[1].set_title("Attack-family attribution capacity")
    axes[1].set_ylabel("Unique feature-window hashes")
    axes[1].legend()
    fig.suptitle("Step 30F0 GEM-CAN schema and natural-capacity audit", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else (root / "data" / "external" / "gem_can" / "extracted").resolve()
    )
    files = discover_files(dataset_root)

    source_frames = {name: read_csv(path) for name, path in files.items()}
    attack_raw = source_frames["GEM_Attack_Scenario_Raw.csv"]
    attack_labeled_source = source_frames["GEM_Attack_Scenario_Raw_Labeled.csv"]
    normal_source = source_frames["GEM_Normal_Driving_Raw.csv"]

    raw_columns = list(attack_raw.columns)
    labelled_common = attack_labeled_source[raw_columns]
    raw_labeled_aligned = attack_raw.fillna("<NA>").equals(
        labelled_common.fillna("<NA>")
    )
    if not raw_labeled_aligned:
        raise RuntimeError(
            "Attack raw and labelled CSVs are not row-aligned; stop before confirmation."
        )

    attack, attack_bad, attack_checks = normalise_and_validate(
        attack_labeled_source,
        "GEM_Attack_Scenario_Raw_Labeled.csv",
        require_attack_type=True,
    )
    normal, normal_bad, normal_checks = normalise_and_validate(
        normal_source,
        "GEM_Normal_Driving_Raw.csv",
        require_attack_type=False,
    )
    malformed = pd.concat([attack_bad, normal_bad], ignore_index=True)

    attack_malformed_fraction = len(attack_bad) / max(len(attack_labeled_source), 1)
    normal_malformed_fraction = len(normal_bad) / max(len(normal_source), 1)
    malformed_tolerance_passed = (
        attack_malformed_fraction <= MAX_MALFORMED_FRACTION
        and normal_malformed_fraction <= MAX_MALFORMED_FRACTION
    )

    attack_inversions = int((attack["timestamp"].diff() < 0).sum())
    normal_inversions = int((normal["timestamp"].diff() < 0).sum())
    attack_windows = build_windows(attack, "attack")
    normal_windows = build_windows(normal, "normal")
    windows = pd.concat([normal_windows, attack_windows], ignore_index=True)
    density = density_capacity(windows)
    families = family_capacity(windows)

    schema_valid = bool(
        raw_labeled_aligned
        and malformed_tolerance_passed
        and len(attack) > 0
        and len(normal) > 0
    )
    capacity_by_density = density.set_index("density_band")["capacity_passed"].to_dict()
    pooled_high_density_feasible = bool(
        schema_valid
        and capacity_by_density.get("benign", False)
        and capacity_by_density.get("high_21_100", False)
    )
    full_density_feasible = bool(
        schema_valid and all(capacity_by_density.get(value, False) for value in DENSITY_ORDER)
    )
    all_families_feasible = bool(
        schema_valid and families["family_specific_confirmation_passed"].all()
    )

    out = root / RESULTS_RELATIVE / run_id()
    out.mkdir(parents=True, exist_ok=False)

    file_manifest_rows: list[dict[str, Any]] = []
    for filename, role in EXPECTED_FILES.items():
        path = files[filename]
        frame = source_frames[filename]
        timestamp_column = "tiTestaTp" if "tiTestaTp" in frame.columns else "timestamp"
        file_manifest_rows.append(
            {
                "file": filename,
                "role": role,
                "path": safe_relative(path, root),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows_as_read": len(frame),
                "columns_json": json.dumps(list(frame.columns)),
                "timestamp_column_as_published": timestamp_column,
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_full_rows_retained": int(frame.duplicated().sum()),
            }
        )
    file_manifest = pd.DataFrame(file_manifest_rows)

    schema_checks = pd.DataFrame(attack_checks + normal_checks)
    schema_checks = pd.concat(
        [
            schema_checks,
            pd.DataFrame(
                [
                    {
                        "source_file": "attack raw versus labelled",
                        "check": "raw_columns_row_aligned",
                        "rows_checked": len(attack_raw),
                        "failed_rows": 0 if raw_labeled_aligned else len(attack_raw),
                        "passed": raw_labeled_aligned,
                    },
                    {
                        "source_file": "all canonical inputs",
                        "check": "malformed_fraction_at_or_below_0.01_percent",
                        "rows_checked": len(attack_labeled_source) + len(normal_source),
                        "failed_rows": len(malformed),
                        "passed": malformed_tolerance_passed,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    labels = pd.concat(
        [
            attack.assign(capture="attack"),
            normal.assign(capture="normal"),
        ],
        ignore_index=True,
    ).groupby(["capture", "label"], as_index=False).size().rename(columns={"size": "rows"})
    attack_types = (
        attack.groupby("attack_type", as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    verdict = pd.DataFrame(
        [
            {
                "schema_valid": schema_valid,
                "raw_labeled_alignment_passed": raw_labeled_aligned,
                "malformed_rows_logged": len(malformed),
                "malformed_tolerance_passed": malformed_tolerance_passed,
                "pooled_high_density_external_confirmation_feasible": pooled_high_density_feasible,
                "full_density_external_confirmation_feasible": full_density_feasible,
                "family_specific_external_confirmation_feasible": all_families_feasible,
                "confirmation_consumed": False,
                "model_loaded": False,
                "threshold_changed": False,
                "sumo_executed": False,
                "step31_permitted": False,
            }
        ]
    )

    file_manifest.to_csv(out / "gem_can_file_manifest.csv", index=False)
    schema_checks.to_csv(out / "gem_can_schema_checks.csv", index=False)
    malformed.to_csv(out / "gem_can_malformed_rows.csv", index=False)
    labels.to_csv(out / "gem_can_label_distribution.csv", index=False)
    attack_types.to_csv(out / "gem_can_attack_type_distribution.csv", index=False)
    windows.to_csv(out / "gem_can_natural_window_manifest.csv", index=False)
    density.to_csv(out / "gem_can_natural_density_capacity.csv", index=False)
    families.to_csv(out / "gem_can_attack_family_capacity.csv", index=False)
    verdict.to_csv(out / "gem_can_capacity_verdict.csv", index=False)
    save_plot(density, families, out / "gem_can_schema_capacity_summary.png")

    summary = {
        "experiment": "Step 30F0 non-consuming GEM-CAN schema and capacity audit",
        "completed_utc": utc_now(),
        "dataset_root": safe_relative(dataset_root, root),
        "window_protocol": {
            "window_size": WINDOW_SIZE,
            "overlap": 0,
            "capture_boundaries_preserved": True,
            "ordering": "stable ascending canonical timestamp within each capture",
            "tail_policy": "discard incomplete tail",
            "duplicate_can_frames": "retained",
            "capacity_unit": "unique feature-window SHA-256",
        },
        "canonicalisation": {
            "published_attack_timestamp_column": "tiTestaTp",
            "canonical_timestamp_column": "timestamp",
            "attack_timestamp_inversions_before_stable_sort": attack_inversions,
            "normal_timestamp_inversions_before_stable_sort": normal_inversions,
        },
        "rows": {
            "attack_read": len(attack_labeled_source),
            "attack_valid": len(attack),
            "normal_read": len(normal_source),
            "normal_valid": len(normal),
            "malformed_logged": len(malformed),
        },
        "tails_discarded": {
            "attack": len(attack) % WINDOW_SIZE,
            "normal": len(normal) % WINDOW_SIZE,
        },
        "schema_valid": schema_valid,
        "pooled_high_density_external_confirmation_feasible": pooled_high_density_feasible,
        "full_density_external_confirmation_feasible": full_density_feasible,
        "family_specific_external_confirmation_feasible": all_families_feasible,
        "decision": (
            "eligible only for a separately predeclared pooled high-density external confirmation"
            if pooled_high_density_feasible
            else "not eligible for an external confirmation run"
        ),
        "claim_boundary": (
            "GEM-CAN cannot confirm low/medium sparse-CAN performance unless natural "
            "low/medium capacity passes; synthetic thinning remains a stress test, not "
            "untouched confirmation. Family-specific claims require exclusive-family capacity."
        ),
        "confirmation_consumed": False,
        "model_loaded": False,
        "threshold_changed": False,
        "sumo_executed": False,
        "existing_project_artifacts_changed": 0,
        "step31_permitted": False,
    }
    (out / "gem_can_schema_capacity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("Step 30F0 GEM-CAN schema and capacity audit completed successfully.")
    print("Confirmation consumed: False")
    print("Model loaded: False")
    print("Threshold changed: False")
    print("SUMO executed: False")
    print("Existing project artifacts changed: 0")
    print(f"Schema valid: {schema_valid}")
    print(f"Raw/labeled attack rows aligned: {raw_labeled_aligned}")
    print(f"Malformed rows logged and excluded: {len(malformed)}")
    print(f"Attack timestamp inversions corrected by stable sort: {attack_inversions}")
    print("\nNatural non-overlapping 100-frame capacity:")
    print(
        density[
            [
                "density_band",
                "natural_nonoverlapping_windows",
                "unique_feature_windows",
                "predeclared_natural_window_minimum",
                "predeclared_unique_cluster_minimum",
                "capacity_passed",
            ]
        ].to_string(index=False)
    )
    print("\nAttack-family attribution capacity:")
    print(
        families[
            [
                "attack_family",
                "unique_windows_containing_family",
                "unique_exclusive_family_windows",
                "family_specific_confirmation_passed",
            ]
        ].to_string(index=False)
    )
    print(
        "\nPooled high-density external confirmation feasible: "
        f"{pooled_high_density_feasible}"
    )
    print(f"Full-density external confirmation feasible: {full_density_feasible}")
    print(f"Family-specific external confirmation feasible: {all_families_feasible}")
    if pooled_high_density_feasible and not full_density_feasible:
        print(
            "Decision: GEM-CAN may support a predeclared pooled high-density external "
            "confirmation only."
        )
        print("Low/medium sparse performance remains externally unconfirmed.")
    elif pooled_high_density_feasible:
        print("Decision: freeze a confirmation protocol before any model scoring.")
    else:
        print("Decision: do not score the frozen model on this dataset.")
    if not all_families_feasible:
        print("Do not claim independent attack-family confirmation from mixed windows.")
    print("Do not run Step 31.")
    print(f"\nResults directory: {out}")
    print(f"Capacity table: {out / 'gem_can_natural_density_capacity.csv'}")
    print(f"Verdict table: {out / 'gem_can_capacity_verdict.csv'}")
    print("\nNext: send the terminal result and the capacity, family, and verdict CSVs.")


if __name__ == "__main__":
    main()
