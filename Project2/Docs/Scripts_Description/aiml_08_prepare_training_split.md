# Script Description: aiml_08_prepare_training_split.py

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `aiml_08_prepare_training_split.py` |
| **Language** | Python 3 |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Dataset Splitting / Machine Learning Data Management / Pipeline Automation |
| **Platform** | Cross-platform (Windows / Linux / macOS), Python Standard Library (`shutil`, `random`, `pathlib`) + OpenCV |
| **Execution Type** | Standalone CLI script; supports command-line arguments and standard PowerShell integration (`--no-log` switch) |

---

## Categories

1. **Dataset Splitting & Data Partitioning**  
   Partitions cleaned image sequences or synthetic background datasets into deterministic train/validation sets using random seed sampling.

2. **Ground-Truth Mask Association & Quality Control**  
   Pairs each frame with either a single shared ground-truth protected region mask or individual per-image masks, preserving exact pixel-level spatial alignment across splits.

3. **Automation & Pipeline Integration**  
   Provides a structured dataset directory layout ready for direct consumption by downstream segmentation and volumetric neural network training scripts.

4. **Logging & Process Auditing**  
   Generates timestamped log files (`INFO`, `SUCCESS`, `WARNING`, `ERROR`) capturing file transfer counts, split parameters, and error handling.

---

## Script Purpose

The `aiml_08_prepare_training_split.py` script automates the creation of training and validation datasets for segmentation and super-resolution models within the mixing tank reconstruction pipeline. It randomly selects a fixed user-defined count of frames for training and routes all remaining frames to the validation set, ensuring paired copies of images and corresponding ground-truth masks are structured in standard dataset directory trees.

### Key Features & Mask Management Modes

* **Dual Mask Ingestion Modes:**
  * **Single Shared Mask Mode (`--mask-source`):** Reuses a static protected region mask (e.g., `Images_Masks_lama\_protected_region_mask.png`) across all real frames. Useful when plot circles remain geometrically stationary across an experimental sequence.
  * **Per-Image Mask Mode (`--masks-folder`):** Dynamically pairs individual frame masks matched by filename (e.g., outputs generated from `aiml_08_generate_synthetic_backgrounds.py`).
  * **Mutually Exclusive Validation:** Enforces strict command-line argument validation so that only one mask ingestion mode is active during execution.
* **Deterministic Random Splitting:** Uses a configurable seed (`--seed`) to ensure dataset partitions are completely reproducible across execution environments.
* **Metadata-Preserving File Transfer:** Employs `shutil.copy2` to preserve original file timestamps and metadata during dataset assembly.
* **Graceful Boundary Handling:** Automatically catches cases where requested train counts exceed available frames, issuing warnings and allocating all frames to the training set.

### Step-by-Step Process Workflow

1. **Path Setup & Argument Parsing:**
   * Maps relative paths to project root folders (`Images`, `Logs`, `Scripts`).
   * Validates target directories and mask argument configurations (`--mask-source` vs. `--masks-folder`).
   * Initializes stream logging and optional log file output (`Logsiml_08_prepare_training_split.log`).

2. **Ground-Truth Mask & Input Validation:**
   * Verifies the existence of input image directories (`Images_Cleaned_lama` by default).
   * Validates single mask readability via OpenCV in single-mask mode, or verifies the existence of the source masks subfolder in per-image mode.

3. **Image Discovery & Reproducible Splitting:**
   * Discovers all PNG frames within the target input directory.
   * Shuffles the file list using Python's `random.Random(seed)`.
   * Partitions files into `train` ($N = 	ext{train\_count}$) and `val` (remainder) file lists.

4. **Directory Structure Creation & File Distribution:**
   * Creates output directories: `Images_Dataset	rain\images`, `Images_Dataset	rain\masks`, `Images_Datasetal\images`, `Images_Datasetal\masks`.
   * Copies images and paired masks into respective target folders using `shutil.copy2`.

5. **Execution Diagnostics & Reporting:**
   * Records total files transferred, failure counts, target dataset locations, and execution status upon completion.

---

## Business / Process Purpose

In machine learning pipelines for industrial automation and tomography, preparing structured, leakage-free dataset splits with perfectly aligned ground-truth annotations is critical for model generalization and verification.

`aiml_08_prepare_training_split.py` supports this objective by:
1. **Preventing Data Contamination:** Ensuring clear separation between training and validation samples under reproducible random seed partitioning.
2. **Standardizing Machine Learning Ingestion:** Formatting output datasets into standardized `images/` and `masks/` directory pairs compatible with common deep learning frameworks (PyTorch DataLoader, TensorFlow Dataset API).
3. **Streamlining Pipeline Interoperability:** Providing a single script capable of preparing splits for both real experimental frames and synthetically augmented background datasets.

---

## Command-Line Arguments & Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--no-log` | Flag | `False` | Disables writing log messages to file (mirrors the `-NoLog` switch in project PowerShell scripts). |
| `--train-count` | `int` | `50` | Fixed number of frames randomly selected for the training set. Remaining frames form the validation set. |
| `--seed` | `int` | `42` | Seed value for reproducible random dataset shuffling and splitting. |
| `--input-images-folder` | `str` | `"03_Cleaned_lama"` | Subfolder relative to `Images\` containing cleaned source frames. |
| `--mask-source` | `str` | `"03_Masks_lama/_protected_region_mask.png"` | Path relative to `Images\` to a single shared ground-truth mask file. Mutually exclusive with `--masks-folder`. |
| `--masks-folder` | `str` | `None` | Subfolder relative to `Images\` containing individual per-image masks matched by filename. Mutually exclusive with `--mask-source`. |
| `--output-folder` | `str` | `"04_Dataset"` | Subfolder relative to `Images\` where train/val dataset splits are exported. |

---

## Input & Output Directory Structure

The script operates within the standardized project folder layout:

```text
Project_Root/
├── Images/
│   ├── 03_Cleaned_lama/                             # INPUT: Source cleaned frames
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── 03_Masks_lama/                               # INPUT: Shared protected region mask (Single Mask Mode)
│   │   └── _protected_region_mask.png
│   └── 04_Dataset/                                  # OUTPUT: Partitioned training dataset
│       ├── train/
│       │   ├── images/                              # Training set images
│       │   │   ├── frame_0001.png
│       │   │   └── ...
│       │   └── masks/                               # Paired training set masks
│       │       ├── frame_0001.png
│       │       └── ...
│       └── val/
│           ├── images/                              # Validation set images
│           │   ├── frame_0051.png
│           │   └── ...
│           └── masks/                               # Paired validation set masks
│               ├── frame_0051.png
│               └── ...
└── Scripts/
    ├── aiml_08_prepare_training_split.py           # Execution script
    └── Logs/                                        # OUTPUT: Log files
        └── aiml_08_prepare_training_split.log
```

---

## Technical Dependencies

| Package | Purpose |
| :--- | :--- |
| `shutil` / `pathlib` | Standard library utilities for file copying (`copy2`), path resolution, and directory creation. |
| `random` | Standard library module for pseudo-random number generation and sequence shuffling. |
| `opencv-python` (`cv2`) | Validates image readability for source mask files. |

---

## Execution Examples

### Basic Execution (Default 50 Train Frames, Shared Mask Mode)
```bash
python aiml_08_prepare_training_split.py
```

### Custom Split Count and Seed Initialization
```bash
python aiml_08_prepare_training_split.py --train-count 80 --seed 101
```

### Partitioning Synthetic Dataset with Per-Image Mask Mode
```bash
python aiml_08_prepare_training_split.py --input-images-folder "05_Synthetic_Backgrounds/images" --masks-folder "05_Synthetic_Backgrounds/masks" --output-folder "05_Synthetic_Dataset" --train-count 200
```