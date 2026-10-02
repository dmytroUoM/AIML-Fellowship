# ============================================
# Script: import_history_to_mlflow.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
#
# Purpose:
#   Backfill MLflow with runs from training history JSON files that were
#   already produced by aiml_08_train_segmentation_demo.py (i.e. you
#   trained before adding MLflow logging, and want the existing results
#   to show up in the MLflow dashboard without retraining).
#
#   For each results/*_history.json file, creates one MLflow run:
#     - params:  lr, batch_size, device, n_params (whatever is present)
#     - metrics: train_loss, val_loss, val_iou logged per epoch (as steps)
#     - artifact: the original history JSON file
#     - artifact: the matching *_model.pt file, if found next to it
#
# Usage:
#   python import_history_to_mlflow.py
#   python import_history_to_mlflow.py --results-dir results --experiment TinyUNet-Segmentation
#
# Then launch the dashboard:
#   mlflow ui --backend-store-uri file:./mlruns --port 5000
#   -> open http://127.0.0.1:5000
# ============================================

import argparse
import json
from pathlib import Path

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="results",
                     help="Folder containing *_history.json files (default: results).")
parser.add_argument("--tracking-uri", type=str, default="file:./mlruns",
                     help="MLflow tracking store location (default: file:./mlruns).")
parser.add_argument("--experiment", type=str, default="TinyUNet-Segmentation",
                     help="MLflow experiment name to log runs under.")
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)

# Params we might find at the top level of a history.json; anything else
# in the dict is skipped when logging metrics (epoch/train_loss/val_loss/val_iou
# are handled separately as per-epoch metrics).
KNOWN_PARAM_KEYS = ["lr", "batch_size", "device", "n_params", "total_seconds"]
METRIC_KEYS = ["train_loss", "val_loss", "val_iou", "epoch_seconds"]


def import_one(history_path: Path, mlflow_client_experiment: str):
    with open(history_path, "r") as f:
        history = json.load(f)

    run_name = history_path.name.replace("_history.json", "")
    epochs = history.get("epoch", [])

    if not epochs:
        print(f"  SKIP {history_path.name}: no 'epoch' list found, not a recognised history file.")
        return

    with mlflow.start_run(run_name=run_name):
        # Log scalar hyperparameters / summary values present in the file
        params = {k: history[k] for k in KNOWN_PARAM_KEYS if k in history and k != "total_seconds"}
        if params:
            mlflow.log_params(params)

        # Log per-epoch metrics using the recorded epoch number as the step,
        # so the MLflow UI reconstructs the original training curves.
        for i, epoch in enumerate(epochs):
            metrics_this_epoch = {}
            for key in METRIC_KEYS:
                values = history.get(key)
                if values and i < len(values):
                    metrics_this_epoch[key] = values[i]
            if metrics_this_epoch:
                mlflow.log_metrics(metrics_this_epoch, step=int(epoch))

        if "total_seconds" in history:
            mlflow.log_metric("total_seconds", history["total_seconds"])

        # Attach the original history file as an artifact
        mlflow.log_artifact(str(history_path))

        # Attach the matching model weights file, if it exists alongside
        model_path = history_path.parent / f"{run_name}_model.pt"
        if model_path.is_file():
            mlflow.log_artifact(str(model_path))
        else:
            print(f"  NOTE: no matching model file found for {run_name} "
                  f"(looked for {model_path.name}); logged metrics/params only.")

    print(f"  Imported: {history_path.name} -> MLflow run '{run_name}'")


def main():
    if not RESULTS_DIR.is_dir():
        print(f"Results folder not found: {RESULTS_DIR.resolve()}")
        return

    history_files = sorted(RESULTS_DIR.glob("*_history.json"))
    if not history_files:
        print(f"No *_history.json files found in {RESULTS_DIR.resolve()}")
        return

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Experiment:   {args.experiment}")
    print(f"Found {len(history_files)} history file(s) in {RESULTS_DIR.resolve()}")

    for history_path in history_files:
        import_one(history_path, args.experiment)

    print("\nDone. Launch the dashboard with:")
    print(f"  mlflow ui --backend-store-uri {args.tracking_uri} --port 5000")
    print("  -> then open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
