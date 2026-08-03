# Script Documentation: 07_make_transparent_background_py_documentation.txt

## Overview

| Field | Value |
|---------|---------|
| **Script Name** | `07_make_transparent_background.py` |
| **Language** | Python |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Image Processing / Preprocessing / Automation |
| **Platform** | Cross-platform (Python) |
| **Execution Type** | Standalone, run manually or scheduled; supports command-line arguments and a `--no-log` switch |

---

## Categories

1. **Image Processing**
   - Converts white and near-white backgrounds into transparent pixels.
   - Preserves image foreground content while removing unwanted backgrounds.

2. **Data Preparation**
   - Produces transparent PNG images suitable for downstream processing, visualization, and machine learning workflows.

3. **Automation / Scripting**
   - Fully automated batch processing of all PNG files within the source directory.

4. **Logging & Reporting**
   - Generates timestamped logs and execution summaries for traceability and quality assurance.

---

## Script Purpose

The script removes white and near-white backgrounds from cleaned PNG images and converts them into transparent PNG files.

### Processing Workflow

1. Resolves the script directory and project root dynamically.
2. Verifies that the source folder exists.
3. Creates the output folder automatically if it does not already exist.
4. Scans the source folder for supported image files (`.png`).
5. Loads each image and converts it to RGBA format.
6. Identifies white and near-white pixels using configurable threshold values.
7. Applies transparency based on pixel whiteness:
   - Pure white pixels become fully transparent.
   - Near-white pixels gradually fade to transparency.
8. Applies an alpha mask erosion filter to remove white edge fringes and artefacts.
9. Saves the processed image as a PNG file with transparency preserved.
10. Logs processing details to a timestamped log file.
11. Continues processing even if an individual image fails.
12. Produces a final execution summary including processing statistics and timing information.

---

## Main Features

### Transparent Background Generation

- Converts white backgrounds into transparent pixels.
- Gradually fades near-white pixels for smoother edges.
- Preserves image detail and foreground objects.

### Edge Fringe Removal

- Applies an alpha mask erosion filter.
- Reduces residual white outlines around foreground objects.
- Produces cleaner transparent images.

### Automated Batch Processing

- Processes all PNG files automatically.
- No interactive input required.
- Suitable for large image datasets.

### Structured Logging

- Timestamped log entries.
- Multiple log levels:
  - `INFO`
  - `SUCCESS`
  - `WARNING`
  - `ERROR`
- Console output mirrors log file entries.

### Error Handling

- Individual image failures do not halt execution.
- Processing continues with remaining images.
- Final report includes failure counts.

---

## Expected Folder Structure

```text
Project2
├── Images
│   ├── 03_Cleaned
│   └── 04_Transparent
└── Scripts
    ├── 07_make_transparent_background.py
    └── Logs
        └── 07_make_transparent_background.log
```

---

## Input and Output Locations

### Source Folder

```text
<ProjectRoot>\Images\03_Cleaned
```

Contains cleaned PNG images with white backgrounds.

### Output Folder

```text
<ProjectRoot>\Images\04_Transparent
```

Contains processed PNG images with transparent backgrounds.

### Log File

```text
<ScriptDir>\Logs\07_make_transparent_background.log
```

Created automatically when logging is enabled.

---

## Configuration Parameters

### Transparency Detection

```python
THRESHOLD_FULL = 235
```

Pixels equal to or above this value become fully transparent.

```python
THRESHOLD_START = 180
```

Pixels between `THRESHOLD_START` and `THRESHOLD_FULL` gradually fade to transparency.

### Edge Cleanup

```python
ERODE_SIZE = 3
```

Controls the size of the alpha mask erosion filter used to remove edge artefacts.

### Supported Image Types

```python
VALID_EXTENSIONS = {".png"}
```

---

## Usage

### Standard Execution

```powershell
(.venv-tomo) py 07_make_transparent_background.py
```

### Disable File Logging

```powershell
(.venv-tomo) py 07_make_transparent_background.py --no-log
```

Console output remains enabled when file logging is disabled.

---

## Logging the Result

### Log File Format

```text
[yyyy-MM-dd HH:mm:ss] [LEVEL] Message
```

### Log Levels Used

- `INFO`
- `SUCCESS`
- `WARNING`
- `ERROR`

### Logged Information

The script records:

- Script startup and completion events.
- Resolved directories and paths.
- Processing parameters.
- Number of images discovered.
- Individual image processing results.
- Processing times.
- File creation events.
- Errors and exceptions.
- Final execution summary.

### Final Summary Information

The script reports:

- Images found
- Images processed
- Images skipped
- Images failed
- Output location
- Total execution time
- Average processing time per image

---

## Business / Process Purpose

This script supports **Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks** by preparing image datasets for subsequent processing stages.

### Key Benefits

#### 1. Clean Visual Outputs

Removes white backgrounds and prepares images for visualization and presentation.

#### 2. Improved Dataset Consistency

Ensures all processed images have uniform transparency handling.

#### 3. Improved Downstream Processing

Transparent backgrounds can reduce unwanted image artefacts during:

- Computer vision workflows
- Image segmentation
- Machine learning preprocessing
- Scientific visualisation

#### 4. Reduced Manual Effort

Eliminates the need for manual background removal across large image collections.

#### 5. Quality Assurance

Logging provides a traceable audit trail of all image-processing operations.

---

## Technology Assessment

### Complexity

**Low to Moderate**

The script combines:

- Image processing
- Alpha channel manipulation
- Batch file management
- Logging and reporting
- Structured exception handling

### Dependencies

Required packages:

```text
numpy
pillow
```

Installation:

```powershell
pip install numpy pillow
```

### Error Handling

**Strong**

Features include:

- Source folder validation
- Per-image exception handling
- Batch continuation after failures
- Final execution reporting

### Portability

**High**

- Pure Python implementation
- Cross-platform support
- Uses platform-independent path handling via `pathlib`

### Idempotency

**Yes**

Re-running the script:

- Overwrites existing output PNG files.
- Appends new log entries.
- Does not modify source images.

### Security Considerations

- Operates entirely on local files.
- No credential storage.
- No network communication.
- No elevated permissions required.

### Performance

**High**

- Lightweight pixel operations.
- Fast processing for typical image sizes.
- Suitable for large image batches.

### Maintainability

**High**

- Clearly structured configuration section.
- Reusable logging framework.
- Parameter-driven processing logic.
- Easily adjustable transparency thresholds.

---

## Approval Considerations

### Least Privilege

Requires:

- Read access to:

```text
Images\03_Cleaned
```

- Write access to:

```text
Images\04_Transparent
Scripts\Logs
```

No administrator rights required.

### No External Dependencies

Processing occurs entirely on the local machine.

### No Destructive Operations

The script:

- Reads source images only.
- Never deletes files.
- Never modifies original images.

### Auditability

Structured logging supports:

- QA reviews
- Process validation
- Project documentation
- Reproducibility requirements

### File Overwrite Behaviour

Output files are overwritten if a file with the same name already exists.

Approvers should determine whether output versioning is required.

---

## Suggested Classification

### Classification

**Low Risk / Standard Automation Utility**

### Risk Level

**Low**

The script:

- Operates locally.
- Uses no network connectivity