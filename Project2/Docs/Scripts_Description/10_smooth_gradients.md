## Script Description: 10_smooth_gradients.py

### Overview

| Field | Value |
|---------|---------|
| **Script Name** | 10_smooth_gradients.py |
| **Language** | Python |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Image Enhancement / Gradient Smoothing / Visual Post-Processing |
| **Platform** | Cross-platform (Python) |
| **Execution Type** | Standalone Python script supporting command-line arguments and optional logging |

### Categories

- **Gradient Smoothing** — Reduces visible blockiness within tomographic images caused by coarse rendering grids.
- **Image Interpolation** — Uses cubic-spline resampling rather than blur-based techniques.
- **Visualisation Enhancement** — Improves image appearance for reports, presentations, and publication.
- **Automated Batch Processing** — Processes multiple PNG files automatically.
- **Logging & Reporting** — Generates structured logs capturing processing parameters and runtime information.

### Script Purpose

This script smooths blocky internal colour transitions within tomographic image regions while preserving the original image background.

The script:

- Resolves project folders automatically using relative paths.
- Scans the source image folder for PNG files.
- Identifies the valid data region using either:
  - Alpha transparency masks (RGBA images), or
  - Non-white pixel detection (RGB images).
- Applies cubic-spline interpolation by:
  - Downsampling the image,
  - Reconstructing it back to the original size,
  - Interpolating colour transitions between existing grid points.
- Restricts smoothing to valid data regions only.
- Preserves transparent or white background regions without modification.
- Saves processed images to the configured output folder.
- Records processing activity in a structured log file.
- Supports configurable smoothing aggressiveness through the downsample factor.

### Business / Process Purpose

This script provides a visual enhancement stage within:

**Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**

It addresses a common issue observed in rendered resistance-tomography images:

- The original data contains coarse measurement cells.
- Rendering methods can produce visible block boundaries.
- These block boundaries may reduce visual quality when images are used in presentations or reports.

The script improves presentation quality by:

- Producing smoother colour transitions.
- Preserving peak signal intensity.
- Maintaining original background transparency.
- Reducing visual artefacts associated with low-resolution rendering.
- Creating more publication-ready imagery.

### Scientific and Technical Basis

The script applies:

**Cubic-Spline Resampling**

Rather than applying a simple blur filter, the script:

1. Reduces image resolution.
2. Reconstructs the image using cubic interpolation.
3. Creates intermediate colour values between neighbouring measurement cells.

Compared with Gaussian blur:

- Preserves peak intensity values more effectively.
- Maintains clearer feature boundaries.
- Produces smoother gradients without excessive softening.
- Better represents transitions across the underlying measurement grid.

This process is an image-processing operation and does not create new scientific measurements.

### Important Limitation

This script operates on already-rendered PNG images.

It does **not** modify:

- Original measurement data.
- Numerical tomography results.
- Source reconstruction algorithms.

The documentation within the script notes that a more mathematically correct solution would be modifying the original rendering stage (for example, changing plotting methods to use interpolated shading directly).

Therefore:

- This script improves visual appearance.
- This script does not improve underlying measurement accuracy.
- This script should be considered a visualisation enhancement step.

### Pipeline Position

The script forms part of the final visualisation pipeline:

```text
08_round_edges.py
    ↓
Images\05_Final_Rounded

09_Final_improved_real-esrgan.py
    ↓
Images\06_Final_Upscaled

10_smooth_gradients.py
    ↓
Images\06_Smoothed_Gradients
```

### Configurable Parameters

| Parameter | Purpose |
|------------|------------|
| `--source-folder` | Input image folder |
| `--output-folder` | Output image folder |
| `--downsample-factor` | Controls smoothing strength |
| `--alpha-cutoff` | Alpha threshold used for RGBA images |
| `--white-thresh` | Background detection threshold for RGB images |
| `--no-log` | Disable file logging |

### Expected Folder Structure

```text
Project2\
│
├── Images\
│   ├── 05_Final_Rounded\
│   └── 06_Smoothed_Gradients\
│
└── Scripts\
    ├── 10_smooth_gradients.py
    └── Logs\
```

### Logging the Result

The script maintains a processing record for traceability and verification.

- **Log file location**: `<ScriptDir>\Logs\10_smooth_gradients.log`
- **Log format**:

```text
[yyyy-MM-dd HH:mm:ss] [LEVEL] Message
```

- **Log levels used**:
  - INFO
  - SUCCESS
  - WARNING
  - ERROR

- **Log content captured**:
  - Script start and completion
  - Input and output paths
  - Downsample factor used
  - Alpha cutoff value
  - White threshold value
  - Number of images discovered
  - Per-image processing results
  - Processing failures
  - Runtime statistics
  - Final batch summary

- **Output artifact**:
  - `Images\06_Smoothed_Gradients\*.png`

- **Console output**:
  - All log messages are echoed directly to the console.

### Technology Assessment

| Aspect | Assessment |
|----------|----------|
| **Complexity** | Moderate. Combines image masking, interpolation, batch processing, argument handling, and logging. |
| **Dependencies** | OpenCV (cv2), NumPy, argparse, pathlib, logging. |
| **Error Handling** | Moderate. Includes image validation, file-access checking, logging, and exception reporting. |
| **Portability** | Cross-platform Python solution with no platform-specific dependencies. |
| **Idempotency** | Yes. Re-running produces repeatable results using identical parameters. |
| **Security Considerations** | Operates entirely on local files with no network communication. |
| **Performance** | Lightweight to moderate. Processing speed depends primarily on image resolution and batch size. |
| **Maintainability** | High. Parameterised design and clear separation of smoothing and masking operations improve maintainability. |

### Approval Considerations

- Source images are never modified.
- Processing occurs entirely on local systems.
- No external network communication takes place.
- No credentials or sensitive authentication data are used.
- Output images remain visually derived from original data.
- Interpolated colour values are estimated visual transitions and should not be interpreted as new measurements.
- The script is intended for image presentation and visualisation purposes.
- Original scientific data should remain the authoritative source for analysis and reporting.

### Suggested Classification

**Classification: Low Risk / Scientific Visualisation Enhancement Utility**

- **Risk Level:** Low

- **Primary Risk:**
  - Users may incorrectly assume smoothed gradients represent additional measured data rather than interpolated visual transitions.

- **Change Category:**
  - Standard engineering image-processing utility.

- **Data Sensitivity:**
  - Input images originate from scientific tomography datasets associated with mixing-tank reconstruction experiments.

- **Recommended Handling:**
  - Suitable for routine use in reporting, presentations, publications, and visual review activities.
  - Retain original images as the authoritative scientific record.
  - Clearly identify outputs as visually enhanced representations derived from measured data.