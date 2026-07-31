# ============================================
# Script: 06_actifacts_removal.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Remove magenta numbering artifacts from cropped frames using inpainting
#   - Save cleaned frames to Images\03_Cleaned folder
#   - Save debug masks to Images\03_Masks folder
#   - Save log to Logs folder beside script
#   - Create output folders if they do not exist
# ============================================

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 script)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Remove magenta numbering artifacts from frames via inpainting."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "06_actifacts_removal.log"

INPUT_FOLDER = SCRIPT_DIR / ".." / "Images" / "02_Frames"
OUTPUT_FOLDER = SCRIPT_DIR / ".." / "Images" / "03_Cleaned"
MASK_FOLDER = SCRIPT_DIR / ".." / "Images" / "03_Masks"

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("actifacts_removal")
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if ENABLE_LOGGING:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def log(message: str, level: str = "INFO") -> None:
    """Write a log message at the given level (INFO, WARNING, ERROR, SUCCESS).

    SUCCESS is not a real logging level, so it is mapped to INFO but keeps
    the "SUCCESS" tag in the message, matching the .ps1 script's style.
    """
    level = level.upper()
    if level == "SUCCESS":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)


# ----------------------------------------------------
# Image processing functions
# ----------------------------------------------------
def build_magenta_mask(img_bgr: np.ndarray, r_thresh: int = 180, b_thresh: int = 180, g_thresh: int = 200) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    return (((r >= r_thresh) & (b >= b_thresh) & (g <= g_thresh)).astype(np.uint8) * 255)


def remove_numbering(input_path: str, output_path: str, dilate_px: int = 2, inpaint_radius: int = 3, method: str = 'telea') -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {input_path}')

    mask = build_magenta_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Save mask for debugging
    mask_file = MASK_FOLDER / f"{Path(input_path).stem}_mask.png"
    cv2.imwrite(str(mask_file), mask)

    algo = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
    cleaned = cv2.inpaint(img, mask, inpaint_radius, algo)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cleaned)


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main() -> int:
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 06_actifacts_removal.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Input folder: {INPUT_FOLDER.resolve()}")
        log(f"Output folder: {OUTPUT_FOLDER.resolve()}")
        log(f"Mask folder: {MASK_FOLDER.resolve()}")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        # Check input folder exists
        if not INPUT_FOLDER.is_dir():
            raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER.resolve()}")

        # Create output folders if they do not exist
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        MASK_FOLDER.mkdir(parents=True, exist_ok=True)

        extensions = {'.png'}
        files = sorted(
            f for f in INPUT_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        log(f"Found {len(files)} PNG file(s) to process.")

        processed = 0
        failed = 0

        for infile in files:
            outfile = OUTPUT_FOLDER / infile.name
            try:
                remove_numbering(str(infile), str(outfile))
                processed += 1
                log(f"Processed: {infile.name}")
            except Exception as file_err:
                failed += 1
                log(f"Failed to process {infile.name}: {file_err}", "ERROR")

        log(f"Cleaned frames saved: {processed} file(s) to {OUTPUT_FOLDER.resolve()}", "SUCCESS")
        log(f"Mask files saved: {processed} file(s) to {MASK_FOLDER.resolve()}", "SUCCESS")

        if failed > 0:
            log(f"{failed} file(s) failed to process.", "WARNING")

        log("Script completed successfully.", "SUCCESS")
        log("============================================")
        return 0

    except Exception as e:
        log(f"Script failed: {e}", "ERROR")
        log("============================================")
        return 1


if __name__ == '__main__':
    sys.exit(main())