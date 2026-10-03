#!/usr/bin/env python3
"""
Expected dataset directory:
    D:\\ztav_project\\data\\ciciov2024\\decimal

Outputs:
    results/ciciov2024_file_summary.csv
    results/ciciov2024_class_distribution.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


EXPECTED_FILES = (
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
)

EXPECTED_FEATURES = (
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

TARGET_COLUMNS = ("label", "category", "specific_class")


def find_project_root() -> Path:
    """Support running the script from either the project root or src folder."""

    script_path = Path(__file__).resolve()
    candidates = (script_path.parent.parent, script_path.parent, Path.cwd())
    for candidate in candidates:
        if (candidate / "data" / "ciciov2024" / "decimal").is_dir():
            return candidate

    print("ERROR: Could not find data/ciciov2024/decimal.")
    print("Expected project structure:")
    print("  ztav_project/data/ciciov2024/decimal/")
    print("  ztav_project/src/01_inspect_ciciov2024.py")
    sys.exit(1)


def normalized_value_counts(series: pd.Series) -> pd.DataFrame:
    """Return count and percentage, including missing target values."""

    display = series.astype("string").fillna("<MISSING>")
    counts = display.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "class_value": counts.index.astype(str),
            "count": counts.values,
            "percentage": (counts.values / len(series) * 100).round(4),
        }
    )


def main() -> None:
    project_root = find_project_root()
    data_dir = project_root / "data" / "ciciov2024" / "decimal"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [name for name in EXPECTED_FILES if not (data_dir / name).is_file()]
    if missing_files:
        print("ERROR: The following expected files are missing:")
        for name in missing_files:
            print(f"  - {name}")
        sys.exit(1)

    print("CICIoV2024 Decimal Dataset Inspection")
    print(f"Dataset: {data_dir}")
    print("Original CSV files will only be read; they will not be modified.\n")

    file_summaries: list[dict[str, object]] = []
    distribution_frames: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None

    for index, file_name in enumerate(EXPECTED_FILES, start=1):
        file_path = data_dir / file_name
        print(f"[{index}/{len(EXPECTED_FILES)}] Reading {file_name} ...")

        data = pd.read_csv(file_path, low_memory=False)
        data.columns = data.columns.astype(str).str.strip()

        if reference_columns is None:
            reference_columns = data.columns.tolist()
        schema_matches_first_file = data.columns.tolist() == reference_columns

        required_columns = set(EXPECTED_FEATURES + TARGET_COLUMNS)
        missing_columns = sorted(required_columns - set(data.columns))
        unexpected_columns = sorted(set(data.columns) - required_columns)

        missing_cells = int(data.isna().sum().sum())
        duplicate_rows = int(data.duplicated().sum())
        memory_mb = data.memory_usage(deep=True).sum() / (1024**2)

        file_summaries.append(
            {
                "file": file_name,
                "rows": len(data),
                "columns": len(data.columns),
                "missing_cells": missing_cells,
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": round(duplicate_rows / max(len(data), 1) * 100, 4),
                "memory_mb": round(memory_mb, 2),
                "schema_matches_first_file": schema_matches_first_file,
                "missing_required_columns": ";".join(missing_columns),
                "unexpected_columns": ";".join(unexpected_columns),
            }
        )

        print(f"    Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
        print(f"    Missing cells: {missing_cells:,}")
        print(f"    Duplicate rows: {duplicate_rows:,}")
        print(f"    Memory used while loaded: {memory_mb:.2f} MB")

        if missing_columns:
            print(f"    WARNING - missing columns: {missing_columns}")
        if unexpected_columns:
            print(f"    Note - additional columns: {unexpected_columns}")
        if not schema_matches_first_file:
            print("    WARNING - schema differs from the first CSV file")

        for target in TARGET_COLUMNS:
            if target not in data.columns:
                continue
            distribution = normalized_value_counts(data[target])
            distribution.insert(0, "target_column", target)
            distribution.insert(0, "file", file_name)
            distribution_frames.append(distribution)

            values = ", ".join(
                f"{row.class_value}={int(row.count):,}"
                for row in distribution.itertuples(index=False)
            )
            print(f"    {target}: {values}")

        print("    First two records:")
        print(data.head(2).to_string(index=False))
        print()

        # Release each file before loading the next one.
        del data

    summary = pd.DataFrame(file_summaries)
    summary_path = results_dir / "ciciov2024_file_summary.csv"
    summary.to_csv(summary_path, index=False)

    if distribution_frames:
        class_distribution = pd.concat(distribution_frames, ignore_index=True)
    else:
        class_distribution = pd.DataFrame(
            columns=("file", "target_column", "class_value", "count", "percentage")
        )
    distribution_path = results_dir / "ciciov2024_class_distribution.csv"
    class_distribution.to_csv(distribution_path, index=False)

    print("=" * 72)
    print("Inspection completed successfully.")
    print(f"File summary:       {summary_path}")
    print(f"Class distribution: {distribution_path}")
    print("\nImportant: do not remove repeated CAN packets automatically.")
    print("We will decide how to handle repetitions after examining these results.")


if __name__ == "__main__":
    main()
