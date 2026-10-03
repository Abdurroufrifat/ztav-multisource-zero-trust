#!/usr/bin/env python3
"""Audit repeated CICIoV2024 CAN frames and create a leakage-safe split manifest.

Why this stage is necessary:
    The dataset contains many repeated CAN packets. A random row split can put
    identical feature vectors in both training and testing data. This script
    hashes ID + DATA_0...DATA_7 and assigns every identical feature vector to
    exactly one provisional split.

This script audits and proposes splits; it does not train a model or modify the
original dataset.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


FILES = (
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
)

FEATURES = (
    "ID",
    "DATA_0",
    "DATA_1",
    "DATA_2",
    "DATA_3",
    "DATA_4",
    "DATA_5",
    "DATA_6",
    "DATA_7",
)

TARGETS = ("label", "category", "specific_class")


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = (script_path.parent.parent, script_path.parent, Path.cwd())
    for candidate in candidates:
        if (candidate / "data" / "ciciov2024" / "decimal").is_dir():
            return candidate
    print("ERROR: Could not find data/ciciov2024/decimal.")
    sys.exit(1)


def assign_hash_split(signature: pd.Series) -> pd.Categorical:
    """Deterministic 70/15/15 split at unique-feature-vector level."""

    bucket = (signature.to_numpy(dtype=np.uint64) % np.uint64(100)).astype(np.uint8)
    values = np.where(bucket < 70, "train", np.where(bucket < 85, "validation", "test"))
    return pd.Categorical(values, categories=("train", "validation", "test"), ordered=True)


def main() -> None:
    root = find_project_root()
    data_dir = root / "data" / "ciciov2024" / "decimal"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [name for name in FILES if not (data_dir / name).is_file()]
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        sys.exit(1)

    print("CICIoV2024 Duplicate/Leakage Audit")
    print("Hashing ID and DATA_0...DATA_7. Original files remain unchanged.\n")

    audit_parts: list[pd.DataFrame] = []

    for index, file_name in enumerate(FILES, start=1):
        path = data_dir / file_name
        print(f"[{index}/{len(FILES)}] Reading and hashing {file_name} ...")

        frame = pd.read_csv(path, usecols=list(FEATURES + TARGETS), low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()

        # Normalize target text only; never alter the nine input features.
        for column in TARGETS:
            frame[column] = frame[column].astype("string").str.strip()
        frame["label"] = frame["label"].str.upper()

        signature = pd.util.hash_pandas_object(
            frame[list(FEATURES)], index=False, categorize=True
        ).astype("uint64")

        part = frame[list(TARGETS)].copy()
        part.insert(0, "signature", signature.to_numpy())
        part["source_file"] = file_name
        audit_parts.append(part)

        print(
            f"    rows={len(part):,}, "
            f"unique feature vectors={part['signature'].nunique():,}"
        )
        del frame, part, signature

    data = pd.concat(audit_parts, ignore_index=True)
    del audit_parts

    data["split"] = assign_hash_split(data["signature"])
    total_rows = len(data)
    total_unique = int(data["signature"].nunique())

    # A conflicting signature has the same nine feature values but different
    # labels. A static packet classifier cannot resolve such a case; it needs
    # timing, sequence, physical, or other context.
    binary_nunique = data.groupby("signature", observed=True)["label"].nunique()
    multiclass_nunique = data.groupby("signature", observed=True)["specific_class"].nunique()
    binary_conflicts = binary_nunique[binary_nunique > 1].index
    multiclass_conflicts = multiclass_nunique[multiclass_nunique > 1].index

    binary_conflict_rows = int(data["signature"].isin(binary_conflicts).sum())
    multiclass_conflict_rows = int(data["signature"].isin(multiclass_conflicts).sum())

    class_rows = (
        data.groupby("specific_class", observed=True)
        .agg(rows=("signature", "size"), unique_feature_vectors=("signature", "nunique"))
        .reset_index()
    )
    class_rows["repeated_rows"] = class_rows["rows"] - class_rows["unique_feature_vectors"]
    class_rows["repetition_percentage"] = (
        class_rows["repeated_rows"] / class_rows["rows"] * 100
    ).round(4)
    class_rows["dataset_percentage"] = (class_rows["rows"] / total_rows * 100).round(4)

    total_record = pd.DataFrame(
        [
            {
                "specific_class": "__TOTAL__",
                "rows": total_rows,
                "unique_feature_vectors": total_unique,
                "repeated_rows": total_rows - total_unique,
                "repetition_percentage": round((total_rows - total_unique) / total_rows * 100, 4),
                "dataset_percentage": 100.0,
            }
        ]
    )
    audit_summary = pd.concat([class_rows, total_record], ignore_index=True)
    # Use pandas' nullable integer dtype. Initializing these columns with empty
    # strings makes newer pandas versions infer a strict string dtype and then
    # reject the integer counts assigned to the __TOTAL__ row.
    for column in (
        "binary_conflicting_signatures_total",
        "rows_in_binary_conflicts_total",
        "multiclass_conflicting_signatures_total",
        "rows_in_multiclass_conflicts_total",
    ):
        audit_summary[column] = pd.Series(pd.NA, index=audit_summary.index, dtype="Int64")
    total_mask = audit_summary["specific_class"].eq("__TOTAL__")
    audit_summary.loc[total_mask, "binary_conflicting_signatures_total"] = len(binary_conflicts)
    audit_summary.loc[total_mask, "rows_in_binary_conflicts_total"] = binary_conflict_rows
    audit_summary.loc[total_mask, "multiclass_conflicting_signatures_total"] = len(multiclass_conflicts)
    audit_summary.loc[total_mask, "rows_in_multiclass_conflicts_total"] = multiclass_conflict_rows

    split_distribution = (
        data.groupby(["split", "specific_class"], observed=True)
        .agg(rows=("signature", "size"), unique_feature_vectors=("signature", "nunique"))
        .reset_index()
    )
    class_totals = split_distribution.groupby("specific_class", observed=True)["rows"].transform("sum")
    split_distribution["percentage_of_class"] = (
        split_distribution["rows"] / class_totals * 100
    ).round(4)

    # Only one row per unique vector is needed for the compact split manifest.
    manifest = (
        data[["signature", "split"]]
        .drop_duplicates("signature")
        .sort_values("signature")
        .reset_index(drop=True)
    )

    # Confirm that no identical input vector appears in multiple partitions.
    split_leakage_groups = int(
        data.groupby("signature", observed=True)["split"].nunique().gt(1).sum()
    )

    audit_path = results_dir / "ciciov2024_signature_audit.csv"
    split_path = results_dir / "ciciov2024_hash_split_distribution.csv"
    manifest_path = results_dir / "ciciov2024_signature_split_manifest.csv"
    audit_summary.to_csv(audit_path, index=False)
    split_distribution.to_csv(split_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    expected_classes = set(data["specific_class"].dropna().unique())
    missing_by_split: dict[str, list[str]] = {}
    for split_name in ("train", "validation", "test"):
        present = set(data.loc[data["split"] == split_name, "specific_class"].dropna().unique())
        missing_by_split[split_name] = sorted(expected_classes - present)

    print("\n" + "=" * 76)
    print(f"Total rows:                          {total_rows:,}")
    print(f"Unique ID+payload vectors:           {total_unique:,}")
    print(f"Repeated-vector percentage:          {(total_rows-total_unique)/total_rows*100:.4f}%")
    print(f"Binary-label conflicting signatures: {len(binary_conflicts):,}")
    print(f"Rows in binary conflicts:             {binary_conflict_rows:,}")
    print(f"Multiclass conflicting signatures:    {len(multiclass_conflicts):,}")
    print(f"Rows in multiclass conflicts:         {multiclass_conflict_rows:,}")
    print(f"Signatures crossing splits:           {split_leakage_groups:,} (must be 0)")

    for split_name, missing_classes in missing_by_split.items():
        if missing_classes:
            print(f"WARNING: {split_name} is missing classes: {missing_classes}")

    print("\nProvisional split distribution:")
    print(split_distribution.to_string(index=False))
    print("\nSaved:")
    print(f"  {audit_path}")
    print(f"  {split_path}")
    print(f"  {manifest_path}")
    print("\nDo not train the classifier yet. We must review this audit first.")


if __name__ == "__main__":
    main()
