# ============================================
# Script: 08_ml_parameter_tuning_evidence.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Demonstrate AI/ML model parameter tuning for apprenticeship evidence
#   - Tune KNN k, SVM kernel, SVM C, and Decision Tree max_depth
#   - Support either built-in demo dataset or user-provided CSV training data
#   - Support configurable label column, feature columns, train/test split, and cross-validation
#   - Save tuning results to Reports folder
#   - Save log to Logs folder beside this script
#   - Create Reports and Logs folders if they do not exist
#
# Technology:
#   - Python
#   - pandas
#   - NumPy
#   - scikit-learn
#
# AI/ML Methods Demonstrated:
#   - K-Nearest Neighbours classifier
#   - Support Vector Machine classifier
#   - Decision Tree classifier
#   - Hyperparameter tuning
#   - Cross-validation
#   - Train/test evaluation
#   - Classification metrics
#
# Expected Folder Structure:
#   Project2
#   |
#   |-- Scripts
#   |   |-- 08_ml_parameter_tuning_evidence.py
#   |   |-- Logs
#   |       |-- 08_ml_parameter_tuning_evidence.log
#   |
#   |-- Data
#   |   |-- optional_training_data.csv
#   |
#   |-- Reports
#       |-- 08_ml_parameter_tuning_results.csv
#       |-- 08_ml_parameter_tuning_best_models.csv
#       |-- 08_ml_parameter_tuning_summary.txt
#
# Usage Examples:
#   python 08_ml_parameter_tuning_evidence.py
#
#   python 08_ml_parameter_tuning_evidence.py --dataset iris --cv-folds 5
#
#   python 08_ml_parameter_tuning_evidence.py --data-csv ..\Data\training_data.csv --label-column target
#
#   python 08_ml_parameter_tuning_evidence.py --knn-k 3,5,7 --svm-kernels linear,rbf --svm-c 0.1,1,10 --tree-depths 2,3,5,none
#
# Notes:
#   - It demonstrates supervised classification using labelled data.
#   - It does not modify source data.
#   - When no CSV file is provided, the built-in Iris dataset is used.
# ============================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import logging
import sys
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# ----------------------------------------------------------------------
# Project configuration
# ----------------------------------------------------------------------

SCRIPT_NAME = "08_ml_parameter_tuning_evidence.py"
PROJECT_NAME = "AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "Reports"
LOGS_DIR = SCRIPT_DIR / "Logs"

LOG_FILE = LOGS_DIR / "08_ml_parameter_tuning_evidence.log"
RESULTS_FILE = REPORTS_DIR / "08_ml_parameter_tuning_results.csv"
BEST_MODELS_FILE = REPORTS_DIR / "08_ml_parameter_tuning_best_models.csv"
SUMMARY_FILE = REPORTS_DIR / "08_ml_parameter_tuning_summary.txt"

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def setup_logging(enable_logging: bool = True) -> logging.Logger:
    """Configure console logging and optional log file output."""
    logger = logging.getLogger("ml_parameter_tuning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if enable_logging:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_int_list(value: str, allow_none: bool = False) -> List[Optional[int]]:
    """Parse comma-separated integer values from command line."""
    result: List[Optional[int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if allow_none and item in {"none", "null"}:
            result.append(None)
        else:
            result.append(int(item))
    return result


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated float values from command line."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    """Parse comma-separated string values from command line."""
    return [item.strip() for item in value.split(",") if item.strip()]


def load_training_data(
    dataset: str,
    data_csv: Optional[str],
    label_column: Optional[str],
    feature_columns: Optional[str],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.Series, List[str], str]:
    """Load source training data and labels from built-in dataset or CSV."""

    if data_csv:
        csv_path = Path(data_csv)
        if not csv_path.is_absolute():
            csv_path = (SCRIPT_DIR / csv_path).resolve()

        logger.info("Loading user-provided CSV dataset: %s", csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data = pd.read_csv(csv_path)

        if label_column is None:
            raise ValueError("When using --data-csv, you must provide --label-column.")

        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV columns: {list(data.columns)}")

        if feature_columns:
            selected_features = parse_str_list(feature_columns)
            missing_features = [col for col in selected_features if col not in data.columns]
            if missing_features:
                raise ValueError(f"Feature column(s) not found in CSV: {missing_features}")
        else:
            selected_features = [col for col in data.columns if col != label_column]

        X = data[selected_features].copy()
        y = data[label_column].copy()
        dataset_name = csv_path.name

    else:
        logger.info("Loading built-in dataset: %s", dataset)

        if dataset == "iris":
            bunch = load_iris(as_frame=True)
        elif dataset == "wine":
            bunch = load_wine(as_frame=True)
        elif dataset == "breast_cancer":
            bunch = load_breast_cancer(as_frame=True)
        else:
            raise ValueError("Unsupported built-in dataset. Choose iris, wine, or breast_cancer.")

        X = bunch.data.copy()
        y = bunch.target.copy()
        selected_features = list(X.columns)
        dataset_name = dataset

    # Keep only numeric feature columns for this simple evidence script.
    numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
    dropped_columns = [col for col in X.columns if col not in numeric_columns]

    if dropped_columns:
        logger.warning("Non-numeric feature columns dropped: %s", dropped_columns)

    X = X[numeric_columns]

    if X.empty:
        raise ValueError("No numeric feature columns available for model training.")

    # Encode text labels if required.
    if not pd.api.types.is_numeric_dtype(y):
        logger.info("Encoding non-numeric labels using LabelEncoder.")
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y), name=label_column or "target")
        logger.info("Label classes: %s", list(encoder.classes_))

    # Fill missing feature values with median values for a simple repeatable workflow.
    if X.isna().any().any():
        logger.warning("Missing feature values found. Filling missing values using column medians.")
        X = X.fillna(X.median(numeric_only=True))

    if pd.Series(y).isna().any():
        raise ValueError("Target labels contain missing values. Please clean the label column before training.")

    logger.info("Dataset loaded: %s", dataset_name)
    logger.info("Samples: %s", X.shape[0])
    logger.info("Features: %s", X.shape[1])
    logger.info("Classes: %s", sorted(pd.Series(y).unique().tolist()))

    return X, pd.Series(y), numeric_columns, dataset_name


def build_models(
    knn_k_values: List[Optional[int]],
    svm_kernels: List[str],
    svm_c_values: List[float],
    tree_depths: List[Optional[int]]
) -> List[Dict[str, Any]]:
    """Build list of model configurations for manual hyperparameter tuning."""

    model_configs: List[Dict[str, Any]] = []

    for k in knn_k_values:
        if k is None or k <= 0:
            continue
        model_configs.append({
            "model_family": "KNN",
            "model_name": f"KNN_k_{k}",
            "parameters": {"k": k},
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=k))
            ])
        })

    for kernel in svm_kernels:
        for c_value in svm_c_values:
            model_configs.append({
                "model_family": "SVM",
                "model_name": f"SVM_kernel_{kernel}_C_{c_value}",
                "parameters": {"kernel": kernel, "C": c_value},
                "estimator": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel=kernel, C=c_value, random_state=RANDOM_STATE))
                ])
            })

    for depth in tree_depths:
        depth_label = "None" if depth is None else str(depth)
        model_configs.append({
            "model_family": "DecisionTree",
            "model_name": f"DecisionTree_depth_{depth_label}",
            "parameters": {"max_depth": depth},
            "estimator": DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        })

    return model_configs


def evaluate_models(
    model_configs: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    cv_shuffle: bool,
    scoring: List[str],
    test_size: float,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run cross-validation and final hold-out test evaluation for each model configuration."""

    # Stratified split helps preserve class proportions for classification.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info("Training samples: %s", X_train.shape[0])
    logger.info("Test samples: %s", X_test.shape[0])

    if cv_shuffle:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=False)

    results = []
    reports = []
    best_model_name = ""
    best_test_accuracy = -1.0

    for config in model_configs:
        model_name = config["model_name"]
        model_family = config["model_family"]
        parameters = config["parameters"]
        estimator = config["estimator"]

        logger.info("Evaluating model: %s", model_name)

        cv_result = cross_validate(
            estimator=estimator,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=None
        )

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        test_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row: Dict[str, Any] = {
            "model_family": model_family,
            "model_name": model_name,
            "parameters": str(parameters),
            "cv_folds": cv_folds,
            "cv_shuffle": cv_shuffle,
            "test_size": test_size,
            "test_accuracy": test_accuracy,
            "test_precision_weighted": test_precision,
            "test_recall_weighted": test_recall,
            "test_f1_weighted": test_f1,
            "fit_time_mean": float(np.mean(cv_result["fit_time"])),
            "score_time_mean": float(np.mean(cv_result["score_time"])),
        }

        for score_name in scoring:
            test_key = f"test_{score_name}"
            train_key = f"train_{score_name}"

            row[f"cv_test_{score_name}_mean"] = float(np.mean(cv_result[test_key]))
            row[f"cv_test_{score_name}_std"] = float(np.std(cv_result[test_key]))
            row[f"cv_train_{score_name}_mean"] = float(np.mean(cv_result[train_key]))
            row[f"cv_train_{score_name}_std"] = float(np.std(cv_result[train_key]))

        results.append(row)

        report_text = classification_report(y_test, y_pred, zero_division=0)
        matrix_text = str(confusion_matrix(y_test, y_pred))
        reports.append(
            f"\nModel: {model_name}\n"
            f"Parameters: {parameters}\n"
            f"Test accuracy: {test_accuracy:.4f}\n"
            f"Classification report:\n{report_text}\n"
            f"Confusion matrix:\n{matrix_text}\n"
        )

        logger.info("Model completed: %s | Test accuracy: %.4f | Test weighted F1: %.4f", model_name, test_accuracy, test_f1)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_name = model_name

    results_df = pd.DataFrame(results)
    best_df = (
        results_df.sort_values(by=["test_accuracy", "test_f1_weighted"], ascending=False)
        .groupby("model_family", as_index=False)
        .head(1)
        .sort_values(by="model_family")
    )

    full_report_text = "\n".join(reports)
    logger.info("Best overall model by hold-out test accuracy: %s", best_model_name)

    return results_df, best_df, full_report_text


def write_summary(
    args: argparse.Namespace,
    dataset_name: str,
    feature_columns: List[str],
    results_df: pd.DataFrame,
    best_df: pd.DataFrame,
    classification_reports: str,
    logger: logging.Logger
) -> None:
    """Write a plain-text summary suitable for apprenticeship evidence."""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "AI/ML Parameter Tuning Evidence Summary",
        "=======================================",
        f"Generated: {generated}",
        f"Script: {SCRIPT_NAME}",
        f"Project: {PROJECT_NAME}",
        f"Dataset: {dataset_name}",
        f"Number of features: {len(feature_columns)}",
        f"Feature columns: {', '.join(feature_columns)}",
        "",
        "Purpose:",
        "This run demonstrates supervised machine learning model training, hyperparameter tuning, cross-validation, and hold-out test evaluation.",
        "The evidence includes KNN, SVM, and Decision Tree classifiers with configurable parameters.",
        "",
        "Tuning parameters used:",
        f"KNN k values: {args.knn_k}",
        f"SVM kernels: {args.svm_kernels}",
        f"SVM C values: {args.svm_c}",
        f"Decision Tree depths: {args.tree_depths}",
        f"Cross-validation folds: {args.cv_folds}",
        f"Cross-validation shuffle: {args.cv_shuffle}",
        f"Scoring metrics: {args.scoring}",
        f"Hold-out test size: {args.test_size}",
        "",
        "Best model per model family:",
        best_df.to_string(index=False),
        "",
        "All model results sorted by test accuracy:",
        results_df.sort_values(by=["test_accuracy", "test_f1_weighted"], ascending=False).to_string(index=False),
        "",
        "Detailed classification reports:",
        classification_reports,
        "",
        "Suggested evidence wording:",
        "I configured and compared several supervised machine learning classifiers using controlled hyperparameter settings. I used cross-validation to evaluate model performance across training splits and a separate hold-out test set to assess final generalisation performance. I recorded the selected parameters, scoring metrics, results, and logs to support repeatability and evidence-based model selection.",
    ]

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary report saved to: %s", SUMMARY_FILE)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune KNN, SVM, and Decision Tree parameters for AI/ML apprenticeship evidence."
    )

    parser.add_argument("--dataset", default="iris", choices=["iris", "wine", "breast_cancer"], help="Built-in dataset to use when --data-csv is not provided.")
    parser.add_argument("--data-csv", default=None, help="Optional path to CSV training data.")
    parser.add_argument("--label-column", default=None, help="Target/label column name when using --data-csv.")
    parser.add_argument("--feature-columns", default=None, help="Optional comma-separated feature column names when using --data-csv.")

    parser.add_argument("--knn-k", default="3,5,7,9", help="Comma-separated KNN k values.")
    parser.add_argument("--svm-kernels", default="linear,rbf,poly", help="Comma-separated SVM kernel values.")
    parser.add_argument("--svm-c", default="0.1,1,10", help="Comma-separated SVM C values.")
    parser.add_argument("--tree-depths", default="2,3,5,none", help="Comma-separated Decision Tree max_depth values. Use none for unlimited depth.")

    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--cv-shuffle", action="store_true", help="Shuffle samples before creating cross-validation folds.")
    parser.add_argument("--scoring", default="accuracy,f1_weighted", help="Comma-separated scikit-learn scoring metrics.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Hold-out test size fraction.")

    parser.add_argument("--no-log", action="store_true", help="Disable log file output. Console output still appears.")

    args = parser.parse_args()

    enable_logging = not args.no_log
    logger = setup_logging(enable_logging=enable_logging)

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("============================================")
        logger.info("Script started.")
        logger.info("Script name: %s", SCRIPT_NAME)
        logger.info("Project: %s", PROJECT_NAME)
        logger.info("Script folder: %s", SCRIPT_DIR)
        logger.info("Project root: %s", PROJECT_ROOT)
        logger.info("Reports folder: %s", REPORTS_DIR)

        if enable_logging:
            logger.info("Logs folder: %s", LOGS_DIR)
            logger.info("Log file: %s", LOG_FILE)
        else:
            logger.warning("File logging disabled by --no-log option.")

        if args.cv_folds < 2:
            raise ValueError("--cv-folds must be 2 or greater.")

        if not 0.0 < args.test_size < 1.0:
            raise ValueError("--test-size must be between 0 and 1.")

        X, y, feature_columns, dataset_name = load_training_data(
            dataset=args.dataset,
            data_csv=args.data_csv,
            label_column=args.label_column,
            feature_columns=args.feature_columns,
            logger=logger
        )

        knn_k_values = parse_int_list(args.knn_k)
        svm_kernels = parse_str_list(args.svm_kernels)
        svm_c_values = parse_float_list(args.svm_c)
        tree_depths = parse_int_list(args.tree_depths, allow_none=True)
        scoring = parse_str_list(args.scoring)

        logger.info("KNN k values: %s", knn_k_values)
        logger.info("SVM kernels: %s", svm_kernels)
        logger.info("SVM C values: %s", svm_c_values)
        logger.info("Decision Tree depths: %s", tree_depths)
        logger.info("Cross-validation folds: %s", args.cv_folds)
        logger.info("Cross-validation shuffle: %s", args.cv_shuffle)
        logger.info("Scoring metrics: %s", scoring)
        logger.info("Hold-out test size: %s", args.test_size)

        model_configs = build_models(
            knn_k_values=knn_k_values,
            svm_kernels=svm_kernels,
            svm_c_values=svm_c_values,
            tree_depths=tree_depths
        )

        logger.info("Total model configurations to evaluate: %s", len(model_configs))

        results_df, best_df, classification_reports = evaluate_models(
            model_configs=model_configs,
            X=X,
            y=y,
            cv_folds=args.cv_folds,
            cv_shuffle=args.cv_shuffle,
            scoring=scoring,
            test_size=args.test_size,
            logger=logger
        )

        results_df.sort_values(by=["test_accuracy", "test_f1_weighted"], ascending=False).to_csv(RESULTS_FILE, index=False)
        best_df.to_csv(BEST_MODELS_FILE, index=False)

        logger.info("Full tuning results saved to: %s", RESULTS_FILE)
        logger.info("Best model results saved to: %s", BEST_MODELS_FILE)

        write_summary(
            args=args,
            dataset_name=dataset_name,
            feature_columns=feature_columns,
            results_df=results_df,
            best_df=best_df,
            classification_reports=classification_reports,
            logger=logger
        )

        logger.info("Script completed successfully.")
        logger.info("============================================")
        return 0

    except Exception as error:
        logger.error("Script failed: %s", error)
        logger.info("============================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
