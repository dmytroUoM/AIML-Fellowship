# Script Description: aiml_06_artifacts_removal_lama.py

## Overview

| Field | Value |
|---------|---------|
| **Script Name** | `aiml_06_artifacts_removal_lama.py` |
| **Language** | Python |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Image Cleaning / AI-Assisted Preprocessing / Automation |
| **Platform** | Cross-platform (Python), external dependency on PyTorch + LaMa inpainting model |
| **Execution Type** | Standalone, run manually or scheduled; supports command-line arguments and a `--no-log` switch |

---

## Categories

1. **AI/ML-Assisted Image Cleanup**  
   Removes text and numbering artefacts from plot frames using a deep learning inpainting model (LaMa).

2. **Data Preparation / Quality Control**  
   Produces clean frames suitable as input for downstream super-resolution and volumetric reconstruction models.

3. **Automation / Scripting**  
   Fully automated and parameterised (white threshold, circle detection, dilation, device), requiring no interactive input.

4. **Logging & Reporting**  
   Writes a structured, timestamped log file and per-frame debug masks, with validation and error handling included.

---

## Script Purpose

The script removes unwanted overlay artifacts (numbers, labels, values, and stray text of any color) from circular plot frames using a deep learning model to reconstruct the affected image regions.

### Process Workflow

1. Resolves the script directory and derives the project root one level above.
2. Scans the input folder `Images\02_Frames` for PNG frames.
3. Samples a subset of frames to automatically detect plot circles (regions of interest) by identifying pixels that remain consistently colored across frames.
4. Retains large circular regions while filtering out small text-sized objects.
5. Caches the detected protected region mask to disk for reuse.
6. Detects any non-white pixels outside the protected plot circles as artifacts, regardless of colour.
7. Loads the LaMa inpainting model from a local checkpoint or downloads it automatically if missing.
8. Uses the model to remove detected artefacts on a frame-by-frame basis.
9. Saves:
   - Cleaned frames to `Images\03_Cleaned_lama`
   - Debug masks to `Images\03_Masks_lama`
10. Logs all processing activity (`INFO`, `SUCCESS`, `WARNING`, `ERROR`) to:

    ```text
    Logs\06_artifacts_removal_lama.log
    ```

    unless disabled with `--no-log`.

11. Continues processing even if individual frames fail.
12. Generates a final processing summary containing:
    - Total frames processed
    - Frames with artefact detections
    - Failed frames
    - Processing times

---

## Business / Process Purpose

This script supports **Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks** by preparing imaging data for downstream