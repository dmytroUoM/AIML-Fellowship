# Script Description: aiml_09_run_trained_model_transparent.py

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `aiml_09_run_trained_model_transparent.py` |
| **Language** | Python 3 |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Machine Learning Inference / Computer Vision Segmentation / Image Compositing |
| **Platform** | Cross-platform (Windows / Linux / macOS), PyTorch + OpenCV |
| **Execution Type** | Standalone CLI script; supports command-line arguments and standard PowerShell integration (`--no-log` switch) |

---

## Categories

1. **Neural Network Inference & Segmentation**  
   Executes forward-pass predictions using a trained deep learning model (`TinyUNet`) to generate binary foreground/background masks.

2. **Image Processing & Transparency Compositing**  
   Converts standard RGB/BGR frame sequences into RGBA format with transparent backgrounds based on predicted segmentation masks.

3. **Model Evaluation & Quantitative Scoring**  
   Supports optional Intersection over Union ($\text{IoU}$) scoring against ground-truth validation masks to measure segmentation accuracy.

4. **Logging & Process Auditing**  
   Generates timestamped log files (`INFO`, `SUCCESS`, `WARNING`, `ERROR`) capturing per-frame inference time, mean $\text{IoU}$ metrics, and file status.

---

## Script Purpose

The `aiml_09_run_trained_model_transparent.py` script applies a trained `TinyUNet` segmentation model to raw frame sequences from the mixing tank dataset[cite: 9]. Rather than relying on classical rule-based background removal, it uses neural network predictions to generate binary masks, composite frames with an alpha channel for transparency, and optionally validate output quality against ground-truth masks[cite: 9].

### Key Features & Execution Workflow

* **Network Architecture Symmetry:** Re-instantiates the identical `TinyUNet` encoder-decoder architecture used during training (`aiml_08_train_segmentation_demo.py`), ensuring state dictionary compatibility[cite: 9].
* **Dynamic Tensor Resizing & Thresholding:**
  * Resizes input images to the model's target resolution ($224 \times 224$), normalizes color channels to $[0, 1]$, and executes forward pass inference[cite: 9].
  * Converts sigmoid output probabilities into binary masks via configurable probability thresholds (default: $0.5$)[cite: 9].
  * Resizes predicted binary masks back to original frame dimensions using nearest-neighbor interpolation (`cv2.INTER_NEAREST`) to prevent boundary blur[cite: 9].
* **RGBA Transparency Compositing:** Splits input BGR channels and appends the predicted 8-bit binary mask as the 4th alpha channel to save transparent PNG images (`cv2.merge([b, g, r, pred_mask])`)[cite: 9].
* **Optional Validation Scoring ($\text{IoU}$):** Calculates pixel-wise Intersection over Union ($\text{IoU}$) per frame when a ground-truth masks directory is supplied, logging individual and mean scores across the dataset[cite: 9].
* **Hardware Acceleration Auto-Selection:** Automatically selects GPU acceleration (`cuda`) when available while supporting manual device overrides (`cpu` vs. `cuda`)[cite: 9].

### Step-by-Step Process Workflow

1. **Path Setup & Argument Parsing:**
   * Maps relative paths to project root folders (`Images`, `Logs`, `results`)[cite: 9].
   * Configures input options (`--model-path`, `--input-folder`, `--output-folder`, `--ground-truth-masks-folder`, `--threshold`, `--device`, `--no-log`)[cite: 9].

2. **Model Weight Ingestion:**
   * Validates target paths and loads state dictionary weights into the `TinyUNet` model[cite: 9].
   * Sets the PyTorch model to evaluation mode (`model.eval()`) and disables gradient tracking (`torch.no_grad()`)[cite: 9].

3. **Frame Processing & Inference Loop:**
   * Discovers target PNG images in the input directory[cite: 9].
   * Generates single-channel predicted masks for each frame[cite: 9].
   * Saves predicted binary masks to `Images/06_Inference_Output/masks/`[cite: 9].
   * Merges input color channels with binary masks and exports RGBA transparent images to `Images/06_Inference_Output/images/`[cite: 9].

4. **Accuracy Evaluation & Log Export:**
   * Computes $\text{IoU}$ against matching ground-truth mask files if provided[cite: 9].
   * Records execution timing statistics (total elapsed time, average seconds per frame) and logs dataset summary metrics[cite: 9].

---

## Business / Process Purpose

In industrial tomography and mixing tank process analysis, isolating fluid features or experimental boundaries from background structure is essential for accurate super-resolution and 3D volumetric reconstruction[cite: 9].

`aiml_09_run_trained_model_transparent.py` supports this objective by:
1. **Automating Machine Learning Background Removal:** Replacing manual thresholding with trained neural network inference for robust foreground extraction[cite: 9].
2. **Standardizing Downstream Pipelines:** Producing transparent RGBA frames and isolated binary masks required by downstream volumetric modeling and rendering tools[cite: 9].
3. **Quantifying Model Performance:** Providing benchmark metrics ($\text{IoU}$) on unseen validation frames to verify segmentation accuracy prior to full-scale deployment[cite: 9].

---

## Command-Line Arguments & Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--no-log` | Flag | `False` | Disables log file output (mirrors `-NoLog` switch in PowerShell scripts)[cite: 9]. |
| `--model-path` | `str` | `"results/run_model.pt"` | Path to trained model weights relative to script folder[cite: 9]. |
| `--input-folder` | `str` | `"Images/02_Frames"` | Input directory relative to project root containing target frames[cite: 9]. |
| `--output-folder` | `str` | `"Images/06_Inference_Output"` | Target directory relative to project root for output RGBA images and masks[cite: 9]. |
| `--ground-truth-masks-folder` | `str` | `None` | Optional directory relative to project root containing ground-truth masks for $\text{IoU}$ evaluation[cite: 9]. |
| `--threshold` | `float` | `0.5` | Probability threshold for binary pixel classification[cite: 9]. |
| `--device` | `str` | `None` | Compute hardware selection (`"cpu"` or `"cuda"`). Defaults to CUDA if available[cite: 9]. |

---

## Input & Output Directory Structure

```text
Project_Root/
├── Images/
│   ├── 02_Frames/                                   # INPUT: Source image frames
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── 04_Dataset/val/masks/                        # OPTIONAL INPUT: Validation ground truth
│   │   ├── frame_0001.png
│   │   └── ...
│   └── 06_Inference_Output/                         # OUTPUT: Inference results
│       ├── images/                                  # Transparent RGBA images
│       │   ├── frame_0001.png
│       │   └── ...
│       └── masks/                                   # Predicted binary masks (grayscale)
│           ├── frame_0001.png
│           └── ...
├── Scripts/
│   ├── aiml_09_run_trained_model_transparent.py    # Execution script
│   └── Logs/                                        # OUTPUT: Log files
│       └── aiml_09_run_trained_model_transparent.log
└── results/
    └── run_model.pt                                 # INPUT: Trained PyTorch weights