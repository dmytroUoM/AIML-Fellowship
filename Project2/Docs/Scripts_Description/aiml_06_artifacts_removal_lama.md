# Script Description: aiml_06_artifacts_removal_lama.py

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `aiml_06_artifacts_removal_lama.py` |
| **Language** | Python 3 |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Image Cleaning / AI-Assisted Preprocessing / Automation |
| **Platform** | Cross-platform (Windows / Linux / macOS), PyTorch + LaMa Inpainting Model |
| **Execution Type** | Standalone CLI script; supports command-line arguments and standard PowerShell integration (`--no-log` switch) |

---

## Categories

1. **AI/ML-Assisted Image Cleanup**  
   Removes text, labels, plot indices, and numbering artifacts of any color from mixing tank tomogram/plot frames using the Large Mask Inpainting (LaMa) deep learning model.

2. **Data Preparation & Quality Control**  
   Produces highly clean, artifact-free image sequences suitable as baseline inputs for downstream machine learning algorithms, super-resolution models, and 3D volumetric reconstruction.

3. **Automation & Pipeline Integration**  
   Fully parameterized execution (configurable white threshold, sample size, connected component area, morphological opening kernel, and hardware device acceleration), requiring no manual intervention.

4. **Logging & Diagnostic Reporting**  
   Generates timestamped log output (`INFO`, `SUCCESS`, `WARNING`, `ERROR`), exports per-frame diagnostic artifact masks, and caches protected spatial region masks for auditability and rapid re-runs.

---

## Script Purpose

The `aiml_06_artifacts_removal_lama.py` script automatically identifies and removes non-data overlay artifacts—such as metadata text, P-labels, color-coded numbering, numerical scales, and debug overlays—from experimental plot frames. It uses a robust geometric protection strategy combined with deep-learning-based inpainting (LaMa) to preserve the actual mixing tank plot data while completely restoring background areas degraded by visual artifacts.

### Key Features & Detection Strategy

* **Color-Agnostic Artifact Detection:** Identifies any pixel differing significantly from the white background (`--white-thresh`), ensuring artifacts of any color (black, magenta, green, blue, etc.) are flagged.
* **True Shape Protection Mask:** Samples frames across the dataset to vote on static plot regions. It isolates large data blobs via connected component analysis and retains their exact pixel boundaries (preserving jagged, discretized tomogram edges rather than forcing an artificial smooth circle fitting).
* **Morphological Separation:** Utilizes morphological opening (`--open-kernel-px`) prior to labeling components to sever single-pixel or anti-aliased bridges between actual data regions and nearby text labels.
* **Deep Learning Inpainting:** Runs LaMa (`big-lama.pt`) over dilated artifact masks (`--artifact-dilate-px`) to reconstruct clean, natural background textures.
* **Automatic Dimension Alignment:** Dynamically crops LaMa output images back to match exact input frame dimensions, correcting for internal 8-pixel padding behavior in LaMa.
* **Local Checkpoint & Cache Strategy:** Prefers local weights at `Bin\big-lama.pt` (mirroring project binary conventions), falling back to automatic PyTorch Hub downloading if not found locally.

### Step-by-Step Process Workflow

1. **Path Initialization & Setup:**
   * Resolves the script execution path and maps relative input/output paths for images, logs, masks, and model binaries.
   * Configures stream logging and optional file logging under `Logs\aiml_06_artifacts_removal_lama.log`.

2. **Protected Region Mask Generation (or Loading):**
   * Checks for an existing cached protected mask (`_protected_region_mask.png`).
   * If absent or if `--rebuild-protected-mask` is set, samples $N$ frames evenly (`--sample-size`).
   * Computes a pixel-wise non-white presence map based on the voting threshold (`--protect-vote-fraction`).
   * Applies morphological opening to disconnect fine text bridges.
   * Evaluates connected component areas, retaining blobs with area $\ge$ `--min-circle-area` as valid plot regions and discarding small text elements.
   * Applies optional dilation/erosion (`--mask-pad-px`) and caches the binary mask.

3. **Model Initialization:**
   * Checks for hardware acceleration (`cuda` GPU or `cpu`).
   * Sets `LAMA_MODEL` path to `Bin\big-lama.pt` if present; otherwise, initiates auto-downloading via PyTorch Hub.
   * Instantiates `SimpleLama` in memory.

4. **Per-Frame Processing & Inpainting:**
   * For each PNG frame in `Images\02_Frames`:
     * Generates a non-white pixel mask.
     * Computes the raw artifact mask: `Artifacts = NonWhite AND NOT(ProtectedRegion)`.
     * Dilates the artifact mask (`--artifact-dilate-px`) to cover fringe anti-aliasing.
     * Writes the binary debug mask to `Images\03_Masks_lama\<frame_name>_mask.png`.
     * If artifacts exist, passes image and mask to LaMa, crops the output back to original dimensions, and writes the output image to `Images\03_Cleaned_lama\<frame_name>.png`.
     * If no artifacts are found, writes the original frame directly.

5. **Summary & Diagnostics:**
   * Records execution timing per frame and outputs a final statistics summary (total frames processed, detection counts, failure counts, and overall runtime).

---

## Business / Process Purpose

In industrial automation and chemical engineering applications—specifically mixing tank hydrodynamics and volumetric flow analysis—sensor outputs and reconstructed tomographic slice datasets are frequently generated with embedded graphical overlays (text annotations, spatial coordinates, time stamps, and sensor channel identifiers).

Directly passing frames with non-data annotations into super-resolution neural networks or volumetric 3D reconstruction algorithms introduces severe spatial artifacts, visual noise, and false high-frequency gradients. 

`aiml_06_artifacts_removal_lama.py` addresses this critical data-preprocessing challenge by:
1. **Ensuring High Model Accuracy:** Preventing downstream AI algorithms from treating text/numerical overlays as physical fluid features or concentration gradients.
2. **Automating Data Preparation:** Eliminating hundreds of hours of manual image editing or dynamic ROI masking across large experimental runs.
3. **Maintaining Reproducibility:** Providing a deterministic, parameter-driven cleaning workflow logged at every step to support academic and industrial audit requirements.

---

## Command-Line Arguments & Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--no-log` | Flag | `False` | Disables writing log messages to file (mirrors the `-NoLog` switch in project PowerShell scripts). |
| `--white-thresh` | `int` | `245` | Channel brightness threshold (0–255) above which a pixel is considered white background. |
| `--sample-size` | `int` | `20` | Number of evenly spaced frames sampled across the input set to construct the protected mask. |
| `--protect-vote-fraction` | `float` | `0.8` | Minimum fraction of sampled frames a pixel must be non-white to be considered protected plot area. |
| `--min-circle-area` | `int` | `5000` | Minimum connected-component pixel area required to classify a region as data rather than text/annotations. |
| `--open-kernel-px` | `int` | `1` | Morphological opening kernel radius (1 = $3 \times 3$ kernel) to sever thin anti-aliased bridges between data and text. |
| `--mask-pad-px` | `int` | `0` | Radius to dilate (positive) or erode (negative) the protected region mask for boundary safety margins. |
| `--artifact-dilate-px` | `int` | `3` | Dilation radius applied to detected artifact masks before passing them to the inpainting model. |
| `--rebuild-protected-mask` | Flag | `False` | Forces recomputation of the protected region mask, ignoring any cached `_protected_region_mask.png`. |
| `--device` | `str` | `Auto` | Explicitly targets execution on `cpu` or `cuda`. Auto-detects CUDA if unspecified. |

---

## Input & Output Directory Structure

The script relies on the standard relative folder structure within the repository context:

```text
Project_Root/
├── Bin/
│   └── big-lama.pt                      # Local LaMa model checkpoint (optional / preferred)
├── Images/
│   ├── 02_Frames/                       # INPUT: Raw extracted PNG plot frames
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── 03_Cleaned_lama/                 # OUTPUT: Cleaned, artifact-free frames
│   │   ├── frame_0001.png
│   │   └── ...
│   └── 03_Masks_lama/                   # OUTPUT: Debug masks & cached region mask
│       ├── _protected_region_mask.png   # Cached binary protected plot mask
│       ├── frame_0001_mask.png          # Per-frame detected artifact mask
│       └── ...
└── Scripts/
    ├── aiml_06_artifacts_removal_lama.py# Execution script
    └── Logs/                            # OUTPUT: Log files
        └── aiml_06_artifacts_removal_lama.log
```

---

## Technical Dependencies

| Package | Purpose |
| :--- | :--- |
| `opencv-python` (`cv2`) | Image loading, non-white masking, morphology, component labeling, mask dilation. |
| `numpy` | Array operations, pixel voting matrices, fast matrix math. |
| `Pillow` (`PIL`) | Image conversion format required for `simple-lama-inpainting`. |
| `torch` / `torchvision` | PyTorch runtime framework for running the deep learning model (CPU or CUDA). |
| `simple-lama-inpainting` | LaMa neural network architecture wrapper for image inpainting. |

---

## Execution Examples

### Basic Execution (Default Parameters)
```bash
python aiml_06_artifacts_removal_lama.py
```

### GPU Execution with Rebuilt Mask and Custom Dilation
```bash
python aiml_06_artifacts_removal_lama.py --device cuda --rebuild-protected-mask --artifact-dilate-px 4
```

### Headless Execution without File Logging
```bash
python aiml_06_artifacts_removal_lama.py --no-log --white-thresh 240 --min-circle-area 4000
```