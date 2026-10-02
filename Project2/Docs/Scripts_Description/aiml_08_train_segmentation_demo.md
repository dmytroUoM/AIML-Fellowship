# Script Description: aiml_08_train_segmentation_demo.py

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `aiml_08_train_segmentation_demo.py` |
| **Language** | Python 3 |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Model Training / Semantic Segmentation / CPU vs GPU Benchmarking |
| **Platform** | Cross-platform (Windows / Linux / macOS); PyTorch (CPU or CUDA), OpenCV |
| **Execution Type** | Standalone CLI script; supports command-line arguments for hyperparameters, regularisation, and device selection |

---

## Categories

1. **Deep Learning Model Training**
   Trains a compact U-Net-style convolutional network end to end for binary segmentation, including data loading, the training/validation loop, loss curves, and IoU evaluation.

2. **Model Regularisation**
   Applies dropout and early stopping so the training run and the saved checkpoint are demonstrably resistant to overfitting on the small demo dataset.

3. **Hardware Benchmarking**
   Records per-epoch and total wall-clock timing so identical runs on CPU and GPU (`aiml_08_submit_gpu_training.sh`) can be compared directly.

4. **Experiment Tracking & Reproducibility**
   Exposes all hyperparameters as CLI arguments, seeds all random number generators, and persists a structured JSON history plus the trained model weights for later analysis.

---

## Script Purpose

`aiml_08_train_segmentation_demo.py` is a minimal, real, working training loop for a small segmentation model, built to demonstrate the training process end to end rather than to produce a production-grade model. It is intentionally small — a tiny dataset, a tiny model, and few epochs — so it runs in seconds to minutes on a laptop CPU, while still exercising the full training/validation/checkpointing workflow used in the wider AIML pipeline.

### Key Features & Execution Flow

* **Tiny U-Net Architecture (`TinyUNet`):**
  * A three-level encoder/decoder network with skip connections, built from repeated `Conv2d` + `ReLU` blocks, `MaxPool2d` downsampling, and `ConvTranspose2d` upsampling.
  * Deliberately small (a few hundred thousand parameters) so it trains quickly on CPU for teaching purposes.
* **Dropout Regularisation:**
  * `nn.Dropout2d` is optionally appended to every encoder/decoder block, controlled by `--dropout` (probability, default `0.0` = disabled).
  * `Dropout2d` zeroes entire feature-map channels rather than individual pixels, which is the standard way to regularise convolutional layers.
* **Early Stopping:**
  * After every epoch, validation loss is compared against the best value seen so far.
  * If `val_loss` fails to improve by at least `--min-delta` for `--patience` consecutive epochs, training stops before reaching `--epochs`.
  * The best-performing model weights (lowest `val_loss`) are checkpointed in memory during training and restored before the final model is saved, so the exported model is never an overfit final epoch.
  * Setting `--patience 0` disables early stopping entirely (the script always runs the full `--epochs`, matching the script's original behaviour).
* **CPU / GPU Device Handling:**
  * `--device cuda` is requested explicitly; if no GPU is available, the script falls back to CPU with a warning rather than failing.
* **Reproducibility:**
  * `--seed` seeds both `torch.manual_seed` and `numpy.random.seed` so runs are repeatable.
* **Metrics & Timing:**
  * Tracks per-epoch training loss, validation loss, validation IoU (`iou_score`), and epoch duration; reports total training time and average seconds/epoch at the end.

### Step-by-Step Process Workflow

1. **Argument Parsing & Seeding:**
   * Parses hyperparameters (`--epochs`, `--lr`, `--batch-size`, `--dropout`, `--patience`, `--min-delta`, `--val-fraction`, `--device`, `--images-dir`, `--masks-dir`, `--run-name`, `--seed`).
   * Seeds PyTorch and NumPy random number generators.

2. **Dataset Loading & Splitting:**
   * Loads matched image/mask PNG pairs from `--images-dir` / `--masks-dir` via `SegmentationDataset`, resizing images to 224x224 and binarising masks at a 127 threshold.
   * Splits files into train/validation subsets according to `--val-fraction`.

3. **Model & Optimiser Setup:**
   * Instantiates `TinyUNet(dropout=args.dropout)` on the selected device.
   * Uses the Adam optimiser (`--lr`) and `BCEWithLogitsLoss`.

4. **Training / Validation Loop:**
   * For each epoch: runs a full training pass (forward, loss, backward, optimiser step), then a validation pass under `torch.no_grad()` computing validation loss and IoU.
   * Checks the early-stopping condition; if triggered, breaks out of the loop and reports the best epoch found.

5. **Best-Model Restoration & Export:**
   * Restores the in-memory best checkpoint (by `val_loss`) into the model before saving.
   * Writes `results/{run_name}_history.json` (per-epoch metrics plus run metadata: device, hyperparameters, regularisation settings, whether early stopping triggered, and the best epoch/loss) and `results/{run_name}_model.pt` (model `state_dict`).

---

## Business / Process Purpose

Demonstrating a correctly regularised, benchmarked training loop is a core learning outcome for this stage of the AIML Fellowship project, independent of the small scale of the demo dataset.

`aiml_08_train_segmentation_demo.py` supports this objective by:
1. **Demonstrating Standard Regularisation Practice:** Applying dropout and early stopping in a visible, configurable way so their effect on train/validation curves can be inspected and discussed, rather than assumed.
2. **Enabling Fair Hardware Comparison:** Keeping the training script identical between CPU and GPU runs (only `--device` differs), so timing differences are attributable to hardware alone.
3. **Producing Auditable, Reproducible Artefacts:** Persisting a structured JSON history and model checkpoint per run, so training behaviour can be reviewed and compared after the fact.

---

## Command-Line Arguments & Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epochs` | `int` | `15` | Maximum number of training epochs (may end earlier if early stopping triggers). |
| `--lr` | `float` | `1e-3` | Learning rate for the Adam optimiser. |
| `--batch-size` | `int` | `4` | Mini-batch size for both training and validation loaders. |
| `--val-fraction` | `float` | `0.2` | Fraction of the dataset held out for validation. |
| `--device` | `str` | `"cpu"` | Compute device: `cpu` or `cuda`. Falls back to `cpu` with a warning if CUDA is requested but unavailable. |
| `--images-dir` | `str` | `"images"` | Directory containing input PNG images. |
| `--masks-dir` | `str` | `"masks"` | Directory containing corresponding PNG ground-truth masks. |
| `--run-name` | `str` | `"run"` | Identifier used for output filenames under `results/`. |
| `--seed` | `int` | `0` | Random seed for PyTorch and NumPy. |
| `--dropout` | `float` | `0.0` | Dropout probability applied inside each encoder/decoder block (`0.0` disables dropout). |
| `--patience` | `int` | `5` | Early stopping patience: epochs without a val_loss improvement (beyond `--min-delta`) before training stops. `0` disables early stopping. |
| `--min-delta` | `float` | `1e-4` | Minimum decrease in val_loss counted as an improvement for early stopping purposes. |

---

## Input & Output Structure

```text
Project_Root/
├── Images/
│   └── 04_Dataset/
│       └── train/
│           ├── images/                 # INPUT: --images-dir
│           │   ├── frame_0001.png
│           │   └── ...
│           └── masks/                  # INPUT: --masks-dir
│               ├── frame_0001.png
│               └── ...
└── Scripts/
    ├── aiml_08_train_segmentation_demo.py   # Execution script
    └── results/                             # OUTPUT
        ├── {run_name}_history.json          # Per-epoch metrics + run metadata
        └── {run_name}_model.pt              # Trained model state_dict (best checkpoint)
```

---

## Technical Dependencies

| Package | Purpose |
| :--- | :--- |
| `torch` | Model definition, training loop, optimiser, checkpoint save/load. |
| `opencv-python` (`cv2`) | Image and mask loading, resizing. |
| `numpy` | Array operations and random seeding. |
| `pathlib` / `json` / `time` / `argparse` | Standard library path handling, history serialisation, timing, and CLI parsing. |

---

## Execution Examples

### Basic CPU Run (dropout and early stopping enabled, defaults)
```bash
python aiml_08_train_segmentation_demo.py --images-dir ../Images/04_Dataset/train/images --masks-dir ../Images/04_Dataset/train/masks --device cpu
```

### Explicit Regularisation Settings, Named Run
```bash
python aiml_08_train_segmentation_demo.py --epochs 15 --lr 1e-3 --batch-size 4 --dropout 0.2 --patience 5 --min-delta 1e-4 --device cpu --run-name lr1e-3_bs4_CPU
```

### GPU Run (matching `aiml_08_submit_gpu_training.sh`)
```bash
python aiml_08_train_segmentation_demo.py --epochs 15 --lr 1e-3 --batch-size 4 --dropout 0.2 --patience 5 --device cuda --run-name lr1e-3_bs4_GPU
```

### Disabling Early Stopping (train the full epoch budget)
```bash
python aiml_08_train_segmentation_demo.py --patience 0 --device cpu
```
