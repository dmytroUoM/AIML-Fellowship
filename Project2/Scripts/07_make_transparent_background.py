# ============================================
# Script: 07_make_transparent_background.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction
#           for Mixing Tanks
#
# Purpose:
#   Batch-remove white / near-white backgrounds from cleaned PNG images
#   and save the results with transparency enabled.
#
# Description:
#   This script reads PNG images from the cleaned image folder, converts
#   white or near-white pixels into transparent pixels, applies a small
#   alpha erosion filter to reduce white edge fringes, and saves the
#   output images as PNG files with an alpha channel.
#
# Category:
#   Image Processing / Preprocessing / Automation
#
# Language:
#   Python
#
# Main Features:
#   - Processes all PNG images in the source folder
#   - Converts white / near-white backgrounds to transparency
#   - Applies alpha mask erosion to reduce edge artefacts
#   - Creates output folder automatically if missing
#   - Adds structured timestamped logging
#   - Supports disabling file logging with --no-log
#   - Continues processing if an individual image fails
#   - Reports processed, skipped, and failed image counts
#
# Expected Folder Structure:
#   Project2
#   ├── Images
#   │   ├── 03_Cleaned
#   │   └── 04_Transparent
#   └── Scripts
#       ├── 07_make_transparent_background.py
#       └── Logs
#           └── 07_make_transparent_background.log
#
# Source Folder:
#   <ProjectRoot>/Images/03_Cleaned
#
# Output Folder:
#   <ProjectRoot>/Images/04_Transparent
#
# Log File:
#   <ScriptDir>/Logs/07_make_transparent_background.log
#
# Usage:
#   (.venv-tomo) py 07_make_transparent_background.py
#
# Optional Usage:
#   (.venv-tomo) py 07_make_transparent_background.py --no-log
#
#
# Notes:
#   - Output images are always saved as PNG to preserve transparency.
#   - Source images are read only and are not modified.
#   - Existing output files with the same name will be overwritten.
#
# Author:
#   Dmytro Denisiuc
# ============================================

from pathlib import Path
from datetime import datetime
import argparse
import sys
import time

import numpy as np
from PIL import Image, ImageFilter


# ----------------------------------------------------------------------
# Script Paths
# ----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SOURCE_DIR = PROJECT_ROOT / "Images" / "03_Cleaned"
OUTPUT_DIR = PROJECT_ROOT / "Images" / "04_Transparent"
LOG_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOG_DIR / "07_make_transparent_background.log"


# ----------------------------------------------------------------------
# Image Processing Config
# ----------------------------------------------------------------------

THRESHOLD_FULL = 235     # Pixels whiter than this become fully transparent
THRESHOLD_START = 180    # Pixels start fading to transparent below this value
ERODE_SIZE = 3           # Min-filter size used to clean edge fringe

VALID_EXTENSIONS = {".png"}


# ----------------------------------------------------------------------
# Logging Function
# ----------------------------------------------------------------------

LOGGING_ENABLED = True


def log(message: str, level: str = "INFO") -> None:
    """
    Write a timestamped log message to the console and, if enabled,
    to the log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"

    print(entry)

    if LOGGING_ENABLED:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as log_file:
                log_file.write(entry + "\n")
        except Exception as exc:
            print(f"[{timestamp}] [WARNING] Failed to write to log file: {exc}")


# ----------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Batch-remove white backgrounds from PNG images and save with transparency."
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing log entries to the log file. Console output remains enabled."
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Image Processing Function
# ----------------------------------------------------------------------

def make_transparent(img: Image.Image) -> Image.Image:
    """
    Return a copy of the image with white / near-white background made transparent.
    """
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float64)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Whiteness is based on the lowest RGB channel value.
    # This helps detect pixels that are close to white across all channels.
    whiteness = np.minimum(np.minimum(r, g), b)

    alpha = np.ones_like(whiteness) * 255

    # Fully transparent for very white pixels
    alpha[whiteness >= THRESHOLD_FULL] = 0

    # Gradual fade for near-white pixels
    fade_mask = (whiteness >= THRESHOLD_START) & (whiteness < THRESHOLD_FULL)
    alpha[fade_mask] = 255 * (
        1 - (whiteness[fade_mask] - THRESHOLD_START) / (THRESHOLD_FULL - THRESHOLD_START)
    )

    arr[..., 3] = alpha

    out = Image.fromarray(arr.astype(np.uint8), "RGBA")

    # Erode alpha mask slightly to remove stray white-ish fringe pixels
    alpha_img = out.split()[3].filter(ImageFilter.MinFilter(ERODE_SIZE))
    out.putalpha(alpha_img)

    return out


# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

def main() -> int:
    """
    Main execution function.
    """
    global LOGGING_ENABLED

    args = parse_arguments()
    LOGGING_ENABLED = not args.no_log

    start_time = time.time()

    processed_count = 0
    failed_count = 0
    skipped_count = 0

    log("============================================")
    log("Script started: 07_make_transparent_background.py")
    log("Project: Project 2 - AIML-Driven Super-Resolution and Volumetric Reconstruction")
    log("Purpose: Remove white backgrounds and create transparent PNG images")
    log("============================================")

    if not LOGGING_ENABLED:
        log("File logging disabled by --no-log switch", "WARNING")

    log(f"Script directory : {SCRIPT_DIR}")
    log(f"Project root     : {PROJECT_ROOT}")
    log(f"Source folder    : {SOURCE_DIR}")
    log(f"Output folder    : {OUTPUT_DIR}")
    log(f"Log file         : {LOG_FILE if LOGGING_ENABLED else 'Disabled'}")

    log("Image processing parameters:")
    log(f"  THRESHOLD_FULL  = {THRESHOLD_FULL}")
    log(f"  THRESHOLD_START = {THRESHOLD_START}")
    log(f"  ERODE_SIZE      = {ERODE_SIZE}")
    log(f"  VALID_EXTENSIONS = {', '.join(sorted(VALID_EXTENSIONS))}")

    try:
        if not SOURCE_DIR.exists():
            raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log("Output folder checked / created successfully", "SUCCESS")

        image_files = sorted(
            file_path
            for file_path in SOURCE_DIR.iterdir()
            if file_path.suffix.lower() in VALID_EXTENSIONS
        )

        if not image_files:
            log(f"No PNG images found in source folder: {SOURCE_DIR}", "WARNING")
            return 0

        log(f"Found {len(image_files)} image(s) to process")

        for index, file_path in enumerate(image_files, start=1):
            frame_start_time = time.time()

            try:
                log(f"Processing image {index}/{len(image_files)}: {file_path.name}")

                with Image.open(file_path) as img:
                    result = make_transparent(img)

                out_path = OUTPUT_DIR / f"{file_path.stem}.png"
                result.save(out_path)

                frame_elapsed = time.time() - frame_start_time
                processed_count += 1

                log(
                    f"Processed successfully: {file_path.name} -> {out_path.name} "
                    f"({frame_elapsed:.2f} sec)",
                    "SUCCESS"
                )

            except Exception as frame_error:
                failed_count += 1
                log(
                    f"Failed to process image: {file_path.name}. Error: {frame_error}",
                    "ERROR"
                )
                continue

        total_time = time.time() - start_time
        average_time = total_time / processed_count if processed_count > 0 else 0

        log("============================================")
        log("Processing summary")
        log(f"Images found      : {len(image_files)}")
        log(f"Images processed  : {processed_count}")
        log(f"Images skipped    : {skipped_count}")
        log(f"Images failed     : {failed_count}")
        log(f"Output folder     : {OUTPUT_DIR}")
        log(f"Total time        : {total_time:.2f} sec")
        log(f"Average time/img  : {average_time:.2f} sec")
        log("Script completed")
        log("============================================", "SUCCESS")

        if failed_count > 0:
            return 1

        return 0

    except Exception as error:
        log(f"Script failed: {error}", "ERROR")
        return 1


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())