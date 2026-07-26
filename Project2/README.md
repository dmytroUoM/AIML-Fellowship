# AIML-Fellowship Project 2

This repository contains the workflow and scripts for Project 2, focused on processing AVI video files from tomographic experiments. The project involves extracting frames, processing images, enhancing them, and reconstructing tomographic data for scientific analysis.

---

## Project Overview

The goal of this project is to process tomographic video data exported as AVI files, clean and enhance the frames, and finally reconstruct the tomographic experiment visually using an HTML-based tool. This pipeline ensures high-quality data preparation for further scientific research.

---

## Repository Structure

- `Bin/` - Executable files and binaries.
- `Docs/` - Documentation, folder trees, and helper tools.
- `Reports/` - Metadata, hashes, and project reports.
- `Scripts/` - Core PowerShell and Python scripts for processing.
- `09_cylinder_reconstruction.html` - Visualization and reconstruction tool for tomographic data.

---

## Step-by-Step Workflow

### Step 0: Clean and Prepare Folders
- **Script:** `Scripts/00_CleanFolders.ps1`
- **Description:** Prepares the directory structure by cleaning and setting up necessary folders for the workflow.

### Step 1: Generate AVI File Hash
- **Script:** `Scripts/01_get_avi_hash.ps1`
- **Description:** Creates a hash for the AVI file to ensure data integrity and track file versions.

### Step 2: Extract Video Metadata (FFprobe)
- **Script:** `Scripts/02_get_metadata_ffprob.ps1`
- **Description:** Uses FFprobe to extract detailed metadata from the AVI video file.

### Step 3: Extract AVI Metadata (PowerShell)
- **Script:** `Scripts/03_get_avi_metadata_powershell.ps1`
- **Description:** Extracts additional metadata using PowerShell commands for further analysis.

### Step 4: Extract Raw Frames from AVI
- **Script:** `Scripts/04_extract_frames_from_avi.ps1`
- **Description:** Extracts all raw frames from the AVI video for processing.

### Step 5: Crop Frames to Region of Interest (ROI)
- **Script:** `Scripts/05_extract_crop_frames_from_avi.ps1`
- **Description:** Crops the extracted frames to focus on the region of interest, removing unnecessary parts of the image.

### Step 6: Remove Numbering and Make Background Transparent
- **Script:** `Scripts/06_remove_numbering_bulk.py` (or `06_make_transparent_background.py` depending on repo)
- **Description:** Removes artifacts such as orange numbering from frames and optionally makes the background transparent to improve image quality.

### Step 7: Round Edges of Frames
- **Script:** `Scripts/07_round_edges.py`
- **Description:** Applies edge rounding to the frames to prepare them for tomographic reconstruction.

### Step 8: Enhance Frames Using Real-ESRGAN
- **Script:** `Scripts/08_Final_improved_real-esrgan.py`
- **Description:** Enhances the resolution and quality of frames using the Real-ESRGAN model for clearer scientific images.

### Step 9: Reconstruct Tomographic Experiment
- **File:** `09_cylinder_reconstruction.html`
- **Description:** Open this HTML file in a web browser to visualize and reconstruct the tomographic experiment using the processed frames.
- **Usage:** Ensure all processed images are in the expected directory. Open the file in Chrome, Firefox, or Edge for interactive 3D reconstruction.

---

## Installation Instructions

1. **Python Environment:**
   - Install Python 3.8+.
   - Create a virtual environment:
     ```
     python -m venv .venv-tomo
     ```
   - Activate the environment:
     - Windows: `.venv-tomo\Scripts\activate`
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
   - Run Python scripts from Step 6 to Step 8 in the activated virtual environment.

4. **Visualization:**
   - Open `09_cylinder_reconstruction.html` in a modern web browser.

---

## Rollout Instructions

### Clone the Repository

```bash
git clone https://github.com/dmytroUoM/AIML-Fellowship.git
cd AIML-Fellowship
git checkout active-avi
