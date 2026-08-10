# AIML-Fellowship Project 2

This repository contains the workflow and scripts for Project 2, focused on processing AVI video files from tomographic (mixing tank) experiments. The project involves extracting frames, cleaning and enhancing images, optionally training a segmentation model, and reconstructing the tomographic experiment visually using an HTML-based 3D tool.

---

## Project Overview

The goal of this project is to process tomographic video data exported as AVI files, remove overlay artifacts, clean and enhance the frames, and reconstruct the tomographic experiment visually. The pipeline supports two parallel background-removal paths — a classical/AI inpainting path (LaMa) and a trainable segmentation-model path — before converging on the same final visualisation stage.

---

## Repository Structure

- `Bin/` - Executable files and binaries (`ffmpeg.exe`, `ffprobe.exe`, `big-lama.pt`, etc.).
- `Docs/` - Documentation, folder trees, and helper tools.
- `Reports/` - Metadata, hashes, and project reports.
- `Scripts/` - Core PowerShell and Python scripts for processing (includes `Logs/`).
- `Images/` - All intermediate and final image outputs, in numbered subfolders.
- `results/` - Trained model weights (e.g. `run_model.pt`).
- `16_cylinder_reconstruction.html` - Current visualisation and reconstruction tool for tomographic data (supersedes `15_cylinder_reconstruction.html`).

---

## Step-by-Step Workflow

### Step 0: Clean and Prepare Folders
- **Script:** `Scripts/00_CleanFolders.ps1`
- **Description:** Safely clears the `Images` and `Reports` output folders (allow-listed by name) before a new pipeline run. Supports `-WhatIfMode` for a dry-run preview.

### Step 1: Generate AVI File Hash
- **Script:** `Scripts/01_get_avi_hash.ps1`
- **Description:** Computes a SHA256 hash of `Video\active.avi` and saves it to `Reports\01_avi_hash.txt` for integrity/provenance tracking.

### Step 2: Extract Video Metadata (FFprobe)
- **Script:** `Scripts/02_get_metadata_ffprobe.ps1`
- **Description:** Uses `ffprobe.exe` to extract container/codec/stream metadata to `Reports\02_avi_metadata_ffprobe.json`.

### Step 3: Extract AVI Metadata (PowerShell / Shell.Application)
- **Script:** `Scripts/03_get_metadata_powershell.ps1`
- **Description:** Extracts Windows Explorer-style metadata via the `Shell.Application` COM object to `Reports\03_active-avi_metadata_powershell.txt`, as a supplement to the ffprobe report.

### Step 4: Extract Raw Frames from AVI
- **Script:** `Scripts/04_extract_origin_frames.ps1`
- **Description:** Uses `ffmpeg.exe` to decode every frame of `active.avi` into PNGs (`frame_%04d.png`) in `Images\01_Origin`.

### Step 5: Crop Frames to Region of Interest (ROI)
- **Script:** `Scripts/05_extract_crop_frames_from_avi.ps1`
- **Description:** Crops the extracted frames to the region of interest, producing `Images\02_Frames`.

### Step 6: Remove Overlay Artifacts (LaMa Inpainting)
- **Script:** `Scripts/aiml_06_artifacts_removal_lama.py`
- **Description:** Detects text/numbering overlays of any colour outside the protected plot circles and removes them using the LaMa deep-learning inpainting model. Outputs cleaned frames to `Images\03_Cleaned_lama` and debug masks to `Images\03_Masks_lama`.

### Step 7: Make Background Transparent
- **Script:** `Scripts/07_make_transparent_background.py`
- **Description:** Converts white/near-white backgrounds in the cleaned frames to transparency (with edge-fringe erosion), producing `Images\04_Transparent`.

### Step 8: Round Edges (Cosmetic)
- **Script:** `Scripts/08_round_edges.py`
- **Description:** Replaces jagged blob edges with smooth, anti-aliased circles for presentation purposes (nearest-neighbour inpainting of the gap pixels). Output is **not** intended for further analysis. Produces `Images\05_Final_Rounded`.

### Step 8 (ML branch): Prepare a Segmentation Training Set
These scripts form an alternative/parallel branch that trains a background-removal segmentation model instead of relying purely on rule-based transparency:

- `Scripts/aiml_08_generate_synthetic_backgrounds.py` — Composites cleaned frames onto 10 procedural synthetic background types to build a robust augmented training set (`Images\05_Synthetic_Backgrounds`).
- `Scripts/aiml_08_prepare_training_split.py` — Splits cleaned/synthetic frames into reproducible train/val sets with paired masks (`Images\04_Dataset`).
- `Scripts/aiml_08_submit_gpu_training.sh` — SLURM batch script to train the `TinyUNet` segmentation model (`aiml_08_train_segmentation_demo.py`) on a GPU cluster.
- `Scripts/aiml_09_run_trained_model_transparent.py` — Runs the trained `TinyUNet` model to predict masks and produce transparent RGBA frames (`Images\06_Inference_Output`), with optional IoU scoring against ground truth.

### Step 9: Enhance Frames Using Real-ESRGAN
- **Script:** `Scripts/09_Final_improved_real-esrgan.py`
- **Description:** Applies the pretrained Real-ESRGAN (RRDBNet) model to upscale/sharpen RGB content while handling the alpha channel separately to avoid transparency artifacts. Produces `Images\06_Final_Upscaled`.

### Step 10: Smooth Gradients (Cosmetic)
- **Script:** `Scripts/10_smooth_gradients.py`
- **Description:** Applies cubic-spline resampling to reduce blocky colour transitions inside the data region only, without touching background/transparent areas. Produces `Images\06_Smoothed_Gradients`. Visualisation-only; does not improve measurement accuracy.

### Step 11: Reconstruct Tomographic Experiment (3D Viewer)
- **File:** `16_cylinder_reconstruction.html` (supersedes `15_cylinder_reconstruction.html`)
- **Description:** Interactive Three.js-based viewer that stacks 8 processed slices into a transparent cylindrical tank shell for 3D review. v16 corrects a slice 7/8 ordering issue present in v15 and adds institutional branding graphics; v15 should be considered deprecated.
- **Usage:** Open in Chrome, Firefox, or Edge. Frame folders can be loaded dynamically through the interface rather than requiring hardcoded paths.

---

## Installation Instructions

1. **Python Environment:**
   - Install Python 3.8+.
   - Create a virtual environment:
     ```
     python -m venv .venv-tomo
     ```
   - Activate the environment:
     - Windows: `.venv-tomo\Scripts\activate.ps1`
     - Linux/macOS: `source .venv-tomo/bin/activate`
   - Install required packages:
     ```
     pip install -r requirements.txt
     ```

2. **PowerShell Scripts:**
   - Run PowerShell scripts in order from Step 0 to Step 5.
   - Ensure you have execution permissions:
     ```
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```

3. **Python Scripts:**
   - Run Python scripts from Step 6 onward in the activated virtual environment (`aiml_06_...`, `07_...`, `08_...`, `aiml_08_...`/`aiml_09_...` for the ML branch, `09_...`, `10_...`).

4. **GPU Training (optional ML branch):**
   - Requires access to a SLURM-managed HPC cluster with CUDA-capable GPUs.
   - Submit with: `sbatch Scripts/aiml_08_submit_gpu_training.sh`

5. **Visualization:**
   - Open `16_cylinder_reconstruction.html` in a modern web browser.

---

## Rollout Instructions

### Clone the Repository

```bash
git clone https://github.com/dmytroUoM/AIML-Fellowship.git
cd AIML-Fellowship
git checkout active-avi
```
