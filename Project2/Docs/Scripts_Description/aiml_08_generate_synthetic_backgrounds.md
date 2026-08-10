# Script Description: aiml_08_generate_synthetic_backgrounds.py

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `aiml_08_generate_synthetic_backgrounds.py` |
| **Language** | Python 3 |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Data Augmentation / Synthetic Dataset Generation / Machine Learning Preprocessing |
| **Platform** | Cross-platform (Windows / Linux / macOS), NumPy + OpenCV |
| **Execution Type** | Standalone CLI script; supports command-line arguments and standard PowerShell integration (`--no-log` switch) |

---

## Categories

1. **Data Augmentation & Synthetic Background Generation**  
   Generates diverse, realistic post-scan background variations (e.g., solid tones, noise, gradients, vignettes, textures, scan lines, dark backgrounds) behind extracted tomogram plot data to improve downstream deep learning robustness.

2. **Dataset Preparation & Training Preparation**  
   Produces paired synthetic training images and exact ground-truth masks in legacy-compatible 8-bit RGB PNG format, eliminating reliance on alpha channels or complex 16-bit color profiles.

3. **Automation & Scalability**  
   Dynamically scans the input folder at runtime to discover clean frames and automatically scales dataset output without hardcoding frame counts.

4. **Logging & Quality Assurance**  
   Provides timestamped log outputs (`INFO`, `SUCCESS`, `WARNING`, `ERROR`), frame-level validation checks, and reproducible pseudo-random generation using fixed seed initialization.

---

## Script Purpose

The `aiml_08_generate_synthetic_backgrounds.py` script automates synthetic background augmentation for mixing tank experimental frames. It composites foreground tomogram plot regions onto procedurally generated background environments while completely preserving real physical data inside the circular region of interest.

### Key Features & Synthetic Generators

* **Dynamic Dataset Scaling:** Automatically enumerates frames in the input directory and scales output volume according to the requested number of variants per image ($N_{	ext{total}} = N_{	ext{frames}} 	imes N_{	ext{variants}}$).
* **Reproducible Generation:** Uses NumPy's PCG64 random number generator with a configurable seed (`--seed`) for deterministic output generation across execution environments.
* **Legacy Tool Compatibility:** Exports all augmented images as standard 3-channel 8-bit BGR PNG files (without alpha channels) for seamless ingestion into older image processing tools and ML training pipelines.
* **10 Procedural Background Types:**
  * `solid_light`: Random bright uniform background (channel intensity 220–255).
  * `solid_gray`: Medium gray uniform background (intensity 160–220).
  * `warm_tint`: Warm-colored BGR background tint.
  * `noisy_white`: White background blended with normal Gaussian noise ($\sigma \in [5, 20]$).
  * `gradient`: Axis-aligned linear gradient with randomized endpoint intensities.
  * `vignette`: Radial distance-decay background with randomized edge attenuation strength.
  * `speckle_texture`: Subtracted random noise texture over light background values.
  * `scanlines`: Periodic horizontal lines simulating sensor scanning or screen artifacts.
  * `dark`: Low-intensity dark background (intensity 20–70).
  * `colored_bed`: Fully randomized RGB color background.

### Step-by-Step Process Workflow

1. **Path Initialization & Setup:**
   * Maps project paths relative to the script location for input images, mask source, output subfolders, and log files.
   * Initializes logging streams to console and optional log file (`Logs\aiml_08_generate_synthetic_backgrounds.log`).

2. **Input & Mask Validation:**
   * Verifies existence of the source image directory (`Images\03_Cleaned_lama` by default).
   * Validates and loads the single ground-truth protected region mask (`Images\03_Masks_lama\_protected_region_mask.png`) generated during artifact removal.

3. **Directory Discovery & Randomization:**
   * Enumerates all `.png` files present in the input subfolder.
   * Calculates total target synthetic output images ($N_{	ext{frames}} 	imes 	ext{variants\_per\_image}$).
   * Initializes target output directories (`Images\03_Synthetic_Backgrounds\images` and `Images\03_Synthetic_Backgrounds\masks`).
   * Configures the pseudo-random number generator seed (`--seed`).

4. **Synthetic Compositing & Mask Export:**
   * For each source frame:
     * Validates frame pixel dimensions against the ground-truth mask.
     * Iterates $N$ times to select background types at random.
     * Generates the synthetic background array using procedural NumPy/OpenCV routines.
     * Blends original frame data and synthetic background using normalized mask weighting:  
       $$	ext{Output} = 	ext{Frame} 	imes 	ext{Mask} + 	ext{Background} 	imes (1 - 	ext{Mask})$$
     * Writes out the composite 8-bit BGR PNG image to the `images` directory and a copy of the ground-truth mask to the `masks` directory using matching filename conventions (`<frame>_<bgtype>_<variant>.png`).

5. **Execution Summary & Logging:**
   * Logs periodic progress every 25 source frames.
   * Reports final generation counts, failure tallies, target folder locations, and status code upon completion.

---

## Business / Process Purpose

In AI-driven industrial process monitoring and volumetric flow reconstruction, deep learning models trained exclusively on homogeneous white or static laboratory backgrounds frequently suffer from over-fitting and poor generalization when deployed to real-world operating environments with variable lighting, sensor drift, or camera gain fluctuations.

`aiml_08_generate_synthetic_backgrounds.py` addresses this challenge by:
1. **Enhancing Model Robustness:** Exposing segmentation and volumetric models to high-variance background conditions during training, forcing networks to isolate true tomographic fluid features.
2. **Automating Data Augmentation:** Multiplying dataset size without requiring additional experimental physical runs or expensive tomographic data collection.
3. **Ensuring Pipeline Interoperability:** Exporting standardized paired dataset image-mask pairs ready for direct ingestion into model training frameworks (PyTorch, TensorFlow, OpenCV).

---

## Command-Line Arguments & Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--no-log` | Flag | `False` | Disables writing log messages to file (mirrors the `-NoLog` switch in project PowerShell scripts). |
| `--input-images-folder` | `str` | `"03_Cleaned_lama"` | Subfolder relative to `Images\` containing source cleaned frames. |
| `--mask-source` | `str` | `"03_Masks_lama/_protected_region_mask.png"` | Path relative to `Images\` to the ground-truth circular protected mask file. |
| `--output-folder` | `str` | `"05_Synthetic_Backgrounds"` | Subfolder relative to `Images\` where augmented dataset outputs are generated. |
| `--variants-per-image` | `int` | `3` | Number of synthetic background variants generated for each input frame. |
| `--seed` | `int` | `42` | Seed value for random background procedural generation and variant selection. |

---

## Input & Output Directory Structure

The script operates within the standardized project folder layout:

```text
Project_Root/
├── Images/
│   ├── 03_Cleaned_lama/                             # INPUT: Real cleaned frames
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── 03_Masks_lama/                               # INPUT: Ground-truth protected mask
│   │   └── _protected_region_mask.png
│   └── 05_Synthetic_Backgrounds/                    # OUTPUT: Synthetic dataset root
│       ├── images/                                  # Composite synthetic images
│       │   ├── frame_0001_vignette_00.png
│       │   ├── frame_0001_noisy_white_01.png
│       │   └── ...
│       └── masks/                                   # Paired ground-truth masks
│           ├── frame_0001_vignette_00.png
│           ├── frame_0001_noisy_white_01.png
│           └── ...
└── Scripts/
    ├── aiml_08_generate_synthetic_backgrounds.py   # Execution script
    └── Logs/                                        # OUTPUT: Log files
        └── aiml_08_generate_synthetic_backgrounds.log
```

---

## Technical Dependencies

| Package | Purpose |
| :--- | :--- |
| `opencv-python` (`cv2`) | Image loading, matrix blending, color conversion, image writing. |
| `numpy` | Procedural noise/gradient array generation, mathematical masking, random number generation. |

---

## Execution Examples

### Basic Execution (Default 3 Variants per Image)
```bash
python aiml_08_generate_synthetic_backgrounds.py
```

### High-Volume Augmentation (5 Variants per Image with Specific Seed)
```bash
python aiml_08_generate_synthetic_backgrounds.py --variants-per-image 5 --seed 123
```

### Execution without File Logging on Custom Subfolder
```bash
python aiml_08_generate_synthetic_backgrounds.py --no-log --input-images-folder "03_Cleaned_custom" --output-folder "05_Synthetic_Custom"
```