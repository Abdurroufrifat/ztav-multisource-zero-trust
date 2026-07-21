#!/usr/bin/env python3
"""Build and evaluate a group-disjoint CICIoV2024 binary IDS split.

Every identical 100-message engineered feature signature is assigned wholly to
one partition. Each class receives at least one unique signature in train,
validation, and test. This removes the exact cross-partition overlap detected
in the chronological baseline.

"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]


RANDOM_STATE = 42
WINDOW_SIZE = 100
SHUFFLE_SEEDS = (101, 202, 303, 404, 505)
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

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


def find_project_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent.parent, script.parent, Path.cwd()):
        expected = candidate / "data" / "processed" / "ciciov2024_windows_w100_all.csv"
        if expected.is_file():
            return candidate
    print("ERROR: Could not find data/processed/ciciov2024_windows_w100_all.csv")
    sys.exit(1)


def choose_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    prediction = (probability >= threshold).astype(np.uint8)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    row = {
        "threshold": threshold,
        "samples": len(y_true),
        "benign_samples": int((y_true == 0).sum()),
        "attack_samples": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "mcc": matthews_corrcoef(y_true, prediction),
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return row, matrix


def make_model_builders(scale_pos_weight: float) -> dict[str, Callable[[], Any]]:
    builders: dict[str, Callable[[], Any]] = {
        "Logistic Regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if XGBClassifier is not None:
        builders["XGBoost"] = lambda: XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    return builders


def allocate_signature_groups(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate complete signatures per class with at least one in each split."""

    multiclass_counts = (
        data.groupby("feature_signature", observed=True)["multiclass_target"]
        .nunique()
    )
    shared_signatures = set(
        int(value) for value in multiclass_counts[multiclass_counts > 1].index
    )

    # A shared signature cannot be assigned independently for each class. Keep
    # it entirely in training, where it can contribute evidence without leaking
    # an identical engineered vector into validation or test.
    signature_to_split: dict[int, str] = {
        signature: "train" for signature in shared_signatures
    }
    if shared_signatures:
        print(
            f"Found {len(shared_signatures)} signature(s) shared by multiple attack "
            "classes; assigning them wholly to training."
        )

    for class_name, subset in data.groupby("multiclass_target", observed=True):
        groups = (
            subset.groupby("feature_signature", observed=True)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(["rows", "feature_signature"], ascending=[False, True])
            .reset_index(drop=True)
        )
        if len(groups) < 3:
            raise ValueError(
                f"Class {class_name} has only {len(groups)} unique engineered signatures; "
                "at least 3 are required"
            )

        class_total = int(groups["rows"].sum())
        targets = {name: class_total * TARGET_RATIOS[name] for name in SPLITS}
        assigned_rows = {name: 0 for name in SPLITS}
        assigned_groups = {name: 0 for name in SPLITS}

        # Count globally preassigned shared signatures for this class.
        unassigned_indices: list[int] = []
        for index in groups.index:
            # Scalar .at access preserves uint64 exactly. Iterating mixed-type
            # rows can coerce a 64-bit hash to float and lose low-order bits.
            signature = int(groups.at[index, "feature_signature"])
            row_count = int(groups.at[index, "rows"])
            if signature in signature_to_split:
                split_name = signature_to_split[signature]
                assigned_rows[split_name] += row_count
                assigned_groups[split_name] += 1
            else:
                unassigned_indices.append(index)

        # Ensure every class has at least one complete signature in every split.
        # Shared signatures already cover training; otherwise the largest
        # unassigned group seeds it. Validation and test receive the next groups.
        for split_name in SPLITS:
            if assigned_groups[split_name] > 0:
                continue
            if not unassigned_indices:
                raise ValueError(
                    f"Class {class_name} cannot be represented in all partitions "
                    "without splitting an identical feature signature"
                )
            index = unassigned_indices.pop(0)
            signature = int(groups.at[index, "feature_signature"])
            row_count = int(groups.at[index, "rows"])
            signature_to_split[signature] = split_name
            assigned_rows[split_name] += row_count
            assigned_groups[split_name] += 1

        # Assign remaining groups to the partition with the largest normalized
        # deficit relative to its 70/15/15 class-specific target.
        for index in unassigned_indices:
            signature = int(groups.at[index, "feature_signature"])
            row_count = int(groups.at[index, "rows"])
            deficits = {
                name: (targets[name] - assigned_rows[name]) / max(targets[name], 1.0)
                for name in SPLITS
            }
            split_name = max(SPLITS, key=lambda name: (deficits[name], -SPLITS.index(name)))
            signature_to_split[signature] = split_name
            assigned_rows[split_name] += row_count
            assigned_groups[split_name] += 1

    allocated = data.copy()
    allocated["split"] = allocated["feature_signature"].map(signature_to_split)
    if allocated["split"].isna().any():
        raise RuntimeError("Some feature signatures were not allocated")
    allocated["split"] = pd.Categorical(
        allocated["split"], categories=SPLITS, ordered=True
    )

    # Build the summary from the final global allocation so shared signatures
    # are counted consistently for every class in which they occur.
    allocation = (
        allocated.groupby(["multiclass_target", "split"], observed=True)
        .agg(
            rows=("feature_signature", "size"),
            unique_feature_signatures=("feature_signature", "nunique"),
        )
        .reset_index()
    )
    class_totals = allocation.groupby("multiclass_target", observed=True)["rows"].transform("sum")
    allocation["percentage_of_class"] = allocation["rows"] / class_totals * 100
    allocation["target_percentage"] = allocation["split"].map(
        {name: TARGET_RATIOS[name] * 100 for name in SPLITS}
    ).astype(float)
    allocation["multiclass_shared_signatures_assigned_to_train"] = len(
        shared_signatures
    )
    return allocated, allocation


def save_feature_importance(
    model_name: str,
    model: Any,
    feature_names: list[str],
    results_dir: Path,
) -> None:
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    else:
        return
    table = pd.DataFrame({"feature": feature_names, "importance": importance})
    table = table.sort_values("importance", ascending=False)
    safe_name = model_name.lower().replace(" ", "_")
    table.to_csv(
        results_dir / f"group_disjoint_feature_importance_{safe_name}.csv",
        index=False,
    )


def plot_confusion_matrices(matrices: dict[str, np.ndarray], output: Path) -> None:
    columns = 2
    rows = int(np.ceil(len(matrices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 4.4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (name, matrix) in zip(axes_array, matrices.items()):
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks([0, 1], ["Benign", "Attack"])
        axis.set_yticks([0, 1], ["Benign", "Attack"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes_array[len(matrices) :]:
        axis.axis("off")
    figure.suptitle("Group-Disjoint Binary IDS — Test Confusion Matrices", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def train_and_evaluate(
    data: pd.DataFrame,
    feature_names: list[str],
    results_dir: Path,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partitions = {name: data[data["split"] == name] for name in SPLITS}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, frame in partitions.items():
        x = frame[feature_names].to_numpy(dtype=np.float64)
        y = frame["binary_target"].to_numpy(dtype=np.uint8)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"{name} does not contain both binary classes")
        arrays[name] = (x, y)

    x_train, y_train = arrays["train"]
    x_validation, y_validation = arrays["validation"]
    x_test, y_test = arrays["test"]
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    builders = make_model_builders(negative / positive)

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    print("\nTraining on the group-disjoint partitions ...")
    for index, (model_name, builder) in enumerate(builders.items(), start=1):
        print(f"  [{index}/{len(builders)}] {model_name}")
        model = builder()
        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start
        validation_probability = model.predict_proba(x_validation)[:, 1]
        threshold = choose_f1_threshold(y_validation, validation_probability)
        thresholds[model_name] = threshold

        validation_row, _ = evaluate(
            y_validation, validation_probability, threshold
        )
        validation_row.update({"model": model_name, "train_seconds": train_seconds})
        validation_rows.append(validation_row)

        test_probability = model.predict_proba(x_test)[:, 1]
        test_row, matrix = evaluate(y_test, test_probability, threshold)
        test_row.update({"model": model_name, "train_seconds": train_seconds})
        test_rows.append(test_row)
        matrices[model_name] = matrix

        safe_name = model_name.lower().replace(" ", "_")
        joblib.dump(model, models_dir / f"group_disjoint_{safe_name}.joblib")
        save_feature_importance(model_name, model, feature_names, results_dir)
        print(
            f"      threshold={threshold:.6f}, validation F1={validation_row['f1']:.4f}, "
            f"test F1={test_row['f1']:.4f}, test PR-AUC={test_row['pr_auc']:.4f}"
        )

    validation_metrics = pd.DataFrame(validation_rows).sort_values("f1", ascending=False)
    test_metrics = pd.DataFrame(test_rows).sort_values("f1", ascending=False)
    validation_metrics.to_csv(
        results_dir / "group_disjoint_validation_metrics.csv", index=False
    )
    test_metrics.to_csv(results_dir / "group_disjoint_test_metrics.csv", index=False)
    with (results_dir / "group_disjoint_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    plot_confusion_matrices(
        matrices, results_dir / "group_disjoint_confusion_matrices.png"
    )
    return validation_metrics, test_metrics


def repeated_shuffle_sanity(
    data: pd.DataFrame,
    feature_names: list[str],
    results_dir: Path,
) -> pd.DataFrame:
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    x_train = train[feature_names].to_numpy(dtype=np.float64)
    y_train = train["binary_target"].to_numpy(dtype=np.uint8)
    x_test = test[feature_names].to_numpy(dtype=np.float64)
    y_test = test["binary_target"].to_numpy(dtype=np.uint8)
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())

    rows: list[dict[str, Any]] = []
    print("\nRunning five shuffled-label repetitions ...")
    for seed in SHUFFLE_SEEDS:
        shuffled = np.random.default_rng(seed).permutation(y_train)
        for model_name, builder in make_model_builders(negative / positive).items():
            model = builder()
            model.fit(x_train, shuffled)
            probability = model.predict_proba(x_test)[:, 1]
            row, _ = evaluate(y_test, probability, 0.5)
            row.update({"seed": seed, "model": model_name})
            rows.append(row)

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "group_disjoint_shuffled_label_runs.csv", index=False)
    summary = (
        runs.groupby("model", observed=True)[
            ["balanced_accuracy", "roc_auc", "pr_auc", "f1"]
        ]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(
        results_dir / "group_disjoint_shuffled_label_summary.csv", index=False
    )
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    root = find_project_root()
    input_path = (
        root / "data" / "processed" / f"ciciov2024_windows_w{WINDOW_SIZE}_all.csv"
    )
    processed_dir = root / "data" / "processed"
    results_dir = root / "results" / "group_disjoint_w100"
    models_dir = root / "models" / "group_disjoint_w100"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    data = pd.read_csv(input_path)
    feature_names = [column for column in data.columns if column not in NON_MODEL_COLUMNS]
    if not feature_names:
        raise ValueError("No model features found")
    if data[feature_names].isna().any().any():
        raise ValueError("Model features contain missing values")

    data = data.rename(columns={"split": "chronological_split"})
    data["feature_signature"] = pd.util.hash_pandas_object(
        data[feature_names], index=False, categorize=True
    ).astype("uint64")
    allocated, split_summary = allocate_signature_groups(data)

    leakage_groups = int(
        allocated.groupby("feature_signature", observed=True)["split"]
        .nunique()
        .gt(1)
        .sum()
    )
    if leakage_groups != 0:
        raise RuntimeError(f"Group-disjoint allocation failed: {leakage_groups} leaking signatures")

    missing_classes: dict[str, list[str]] = {}
    expected_classes = set(allocated["multiclass_target"].unique())
    for split_name in SPLITS:
        present = set(
            allocated.loc[allocated["split"] == split_name, "multiclass_target"].unique()
        )
        missing_classes[split_name] = sorted(expected_classes - present)
        if missing_classes[split_name]:
            raise RuntimeError(
                f"{split_name} is missing classes: {missing_classes[split_name]}"
            )

    split_summary.to_csv(
        results_dir / "group_disjoint_split_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "total_windows": len(allocated),
                "unique_feature_signatures": allocated["feature_signature"].nunique(),
                "signatures_crossing_partitions": leakage_groups,
                "all_classes_present_in_all_partitions": True,
            }
        ]
    ).to_csv(results_dir / "group_disjoint_overlap_audit.csv", index=False)

    prefix = f"ciciov2024_windows_w{WINDOW_SIZE}_group_disjoint"
    allocated.to_csv(processed_dir / f"{prefix}_all.csv", index=False)
    for split_name in SPLITS:
        allocated.loc[allocated["split"] == split_name].to_csv(
            processed_dir / f"{prefix}_{split_name}.csv", index=False
        )

    print("\nGroup-disjoint split distribution:")
    print(split_summary.to_string(index=False))
    print(f"\nFeature signatures crossing partitions: {leakage_groups} (must be 0)")

    validation_metrics, test_metrics = train_and_evaluate(
        allocated, feature_names, results_dir, models_dir
    )
    shuffle_summary = repeated_shuffle_sanity(
        allocated, feature_names, results_dir
    )

    print("\n" + "=" * 100)
    print("Group-disjoint validation metrics:")
    print(
        validation_metrics[
            ["model", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate"]
        ].to_string(index=False)
    )
    print("\nGroup-disjoint test metrics:")
    print(
        test_metrics[
            ["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate", "false_negative_rate"]
        ].to_string(index=False)
    )
    print("\nShuffled-label summary:")
    print(shuffle_summary.to_string(index=False))
    print(f"\nResults directory: {results_dir}")
    print(f"Models directory:  {models_dir}")
    print("Group-disjoint evaluation completed successfully.")


if __name__ == "__main__":
    main()
