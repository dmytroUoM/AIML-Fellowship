# Script Description: 06_artifacts_removal.py

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `06_artifacts_removal.py` (on-disk file name; script header refers to it as `06_actifacts_removal.py`) |
| **Language** | Python 3 |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Image Cleaning / Classical Computer Vision / Automation |
| **Platform** | Cross-platform (Windows / Linux / macOS), OpenCV, NumPy |
| **Execution Type** | Standalone CLI script; supports command-line arguments and a `--no-log` switch |

> **Note on naming:** This is the legacy/classical counterpart to `aiml_06_artifacts_removal_lama.py`. It uses simple colour-threshold masking and OpenCV inpainting rather than an AI inpainting model. It is invoked as step `06` in the "Legacy visual pipeline" branch of `run_pipeline.ps1`.

---

## Categories

1. **Colour-Threshold Masking**
   Detects magenta-coloured numbering overlays in each frame using fixed RGB channel thresholds.

2. **Classical Inpainting**
   Removes the detected artifacts using OpenCV's Telea or Navier-Stokes inpainting algorithms, filling the masked region from surrounding pixel data.

3. **Batch Image Processing & Logging**
   Processes an entire folder of PNG frames, saving cleaned frames and debug masks, with timestamped console/file logging (`INFO`, `WARNING`, `ERROR`, `SUCCESS`).

---

## Script Purpose

`06_artifacts_removal.py` removes magenta numbering artifacts that were burned into the cropped video frames (e.g. frame counters or plot annotations) using classical colour-thresholding and inpainting, rather than a learned model. It is the fast, dependency-light alternative to `aiml_06_artifacts_removal_lama.py`, used in the legacy visual pipeline (steps 06–10) that produces a classically upscaled result via Real-ESRGAN.

### Key Features & Execution Flow

* **Magenta Mask Detection (`build_magenta_mask`):**
  * Flags pixels where red and blue channels exceed fixed thresholds (`r_thresh=180`, `b_thresh=180`) while green stays below a ceiling (`g_thresh=200`), isolating the magenta overlay colour.
* **Mask Dilation:**
  * Dilates the detected mask by a small elliptical kernel (`dilate_px=2`) to ensure full coverage of anti-aliased artifact edges before inpainting.
* **Inpainting (`remove_numbering`):**
  * Fills the masked region using `cv2.inpaint` with the Telea algorithm by default (`inpaint_radius=3`), reconstructing plausible pixel values from the surrounding image.
* **Debug Mask Export:**
  * Saves the dilated mask used for each frame to `Images\03_Masks`, so masking accuracy can be visually reviewed independently of the inpainted result.
* **Logging (mirrors the project's `.ps1` `Write-Log` convention):**
  * Timestamped, leveled log messages written to console and, unless `--no-log` is passed, to `Logs\06_actifacts_removal.log`.
* **Per-File Error Isolation:**
  * Wraps each frame's processing in its own `try/except`, so one corrupt or unreadable file is logged and skipped rather than aborting the whole batch.

### Step-by-Step Process Workflow

1. **Argument Parsing & Logging Setup:**
   * Parses `--no-log`; configures a logger writing to stdout and, if logging is enabled, to `Logs\06_actifacts_removal.log`.

2. **Path Validation:**
   * Confirms the input folder `Images\02_Frames` exists; creates output folders `Images\03_Cleaned` and `Images\03_Masks` if missing.

3. **Frame Discovery:**
   * Collects and sorts all `.png` files in the input folder.

4. **Per-Frame Cleaning:**
   * For each frame: builds the magenta mask, dilates it, saves the debug mask, inpaints the artifact region, and writes the cleaned frame to `Images\03_Cleaned`.
   * Logs success/failure per file and tallies processed vs. failed counts.

5. **Completion Reporting:**
   * Logs the total number of cleaned frames and mask files saved, any failure count, and an overall success/failure exit code (`0` on success, `1` on unhandled error).

---

## Business / Process Purpose

Frame-burned annotations (magenta numbering) would otherwise propagate visible artifacts into every downstream stage of the pipeline — transparency masking, edge rounding, and super-resolution upscaling.

`06_artifacts_removal.py` supports the project by:
1. **Providing a Lightweight Cleaning Path:** Offering a fast, dependency-light (OpenCV-only) alternative to the LaMa-based AI inpainting script, suitable when GPU/PyTorch resources are unavailable or unnecessary.
2. **Preserving Debuggability:** Exporting the exact binary mask used for inpainting alongside the cleaned frame, so masking parameters (thresholds, dilation) can be tuned and verified visually.
3. **Maintaining Pipeline Consistency:** Following the same input/output folder conventions and logging style as the rest of the legacy visual pipeline (`07_make_transparent_background.py`, `08_round_edges.py`, etc.), so it can be run standalone or orchestrated via `run_pipeline.ps1`.

---

## Command-Line Arguments & Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--no-log` | Flag | `False` | Disables writing log messages to file (console logging still occurs). |

Additional behaviour is controlled by constants inside the script rather than CLI flags: mask thresholds (`r_thresh=180`, `b_thresh=180`, `g_thresh=200`), dilation radius (`dilate_px=2`), inpainting radius (`inpaint_radius=3`), and inpainting method (`method='telea'`, alternative `'ns'`).

---

## Input & Output Directory Structure

```text
Project_Root/
├── Images/
│   ├── 02_Frames/                # INPUT: cropped frames with magenta numbering
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── 03_Cleaned/                # OUTPUT: inpainted, artifact-free frames
│   │   ├── frame_0001.png
│   │   └── ...
│   └── 03_Masks/                  # OUTPUT: debug masks showing detected artifact regions
│       ├── frame_0001_mask.png
│       └── ...
└── Scripts/
    ├── 06_artifacts_removal.py    # Execution script
    └── Logs/                      # OUTPUT: log file (unless --no-log)
        └── 06_actifacts_removal.log
```

---

## Technical Dependencies

| Package | Purpose |
| :--- | :--- |
| `opencv-python` (`cv2`) | Image I/O, colour-channel splitting, morphological dilation, and inpainting. |
| `numpy` | Mask array construction and boolean channel comparisons. |
| `logging` / `pathlib` / `argparse` / `os` / `sys` | Standard library logging, path handling, and CLI parsing. |

---

## Execution Examples

### Basic Execution (logging enabled)
```bash
python 06_artifacts_removal.py
```

### Run Without Writing a Log File
```bash
python 06_artifacts_removal.py --no-log
```
