# ============================================
# Script: 07_prepare_training_split.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Take the artifact-cleaned frames from Images\03_Cleaned_lama
#   - Pair each frame with the ground-truth background/circle mask (the
#     cached protected region mask from Images\03_Masks_lama - the same
#     mask applies to every frame, since the plot circles sit at fixed
#     positions across the whole dataset)
#   - Randomly select a fixed count of frames for TRAINING
#   - Put everything else into the REST (validation) set
#   - Copy images + matching masks into new subfolders under Images, ready
#     to hand straight to a training script (e.g. train_segmentation_demo.py)
#   - Save log to Logs folder beside script
#   - Create output folders if they do not exist
#
# Output layout:
#   Images\04_Dataset\train\images\*.png
#   Images\04_Dataset\train\masks\*.png
#   Images\04_Dataset\val\images\*.png
#   Images\04_Dataset\val\masks\*.png
# ============================================

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import cv2

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Split cleaned frames into a fixed-count training set and a "
                "'rest' (validation) set, paired with the ground-truth background mask."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--train-count", type=int, default=50,
    help="Fixed number of frames to randomly select for the training set (default: 50). "
         "Everything else goes into the 'rest' (val) set."
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed, so the same split can be reproduced (default: 42)."
)
parser.add_argument(
    "--input-images-folder", type=str, default="03_Cleaned_lama",
    help="Subfolder under Images\\ containing the cleaned frames to split (default: 03_Cleaned_lama)."
)
parser.add_argument(
    "--mask-source", type=str, default=None,
    help="Path (relative to Images\\) to a SINGLE ground-truth mask file, reused for every "
         "frame - use this for real cleaned frames sharing one fixed circle mask "
         "(e.g. 03_Masks_lama/_protected_region_mask.png). Mutually exclusive with --masks-folder."
)
parser.add_argument(
    "--masks-folder", type=str, default=None,
    help="Subfolder under Images\\ containing ONE mask PER image, matched by filename - use "
         "this for output from 08_generate_synthetic_backgrounds.py, which already writes a "
         "per-image mask (e.g. 05_Synthetic_Backgrounds/masks). Mutually exclusive with --mask-source."
)
parser.add_argument(
    "--output-folder", type=str, default="04_Dataset",
    help="Subfolder under Images\\ to write the train/val split into (default: 04_Dataset)."
)
args = parser.parse_args()

if args.mask_source and args.masks_folder:
    parser.error("--mask-source and --masks-folder are mutually exclusive; use only one.")
if not args.mask_source and not args.masks_folder:
    # Preserve original default behavior: one shared mask for all real frames.
    args.mask_source = "03_Masks_lama/_protected_region_mask.png"

ENABLE_LOGGING = not args.no_log

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "07_prepare_training_split.log"

IMAGES_ROOT = SCRIPT_DIR / ".." / "Images"
INPUT_FOLDER = IMAGES_ROOT / args.input_images_folder
MASK_SOURCE_FILE = IMAGES_ROOT / args.mask_source if args.mask_source else None
MASKS_FOLDER = IMAGES_ROOT / args.masks_folder if args.masks_folder else None
OUTPUT_ROOT = IMAGES_ROOT / args.output_folder

TRAIN_IMAGES_DIR = OUTPUT_ROOT / "train" / "images"
TRAIN_MASKS_DIR = OUTPUT_ROOT / "train" / "masks"
VAL_IMAGES_DIR = OUTPUT_ROOT / "val" / "images"
VAL_MASKS_DIR = OUTPUT_ROOT / "val" / "masks"

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("prepare_training_split")
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
    """Write a log message at the given level (INFO, WARNING, ERROR, SUCCESS)."""
    level = level.upper()
    if level == "SUCCESS":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)


def main() -> int:
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 07_prepare_training_split.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Input images folder: {INPUT_FOLDER.resolve()}")
        if MASK_SOURCE_FILE:
            log(f"Ground-truth mask mode: single shared file -> {MASK_SOURCE_FILE.resolve()}")
        else:
            log(f"Ground-truth mask mode: per-image folder -> {MASKS_FOLDER.resolve()}")
        log(f"Output dataset folder: {OUTPUT_ROOT.resolve()}")
        log(f"Requested training count: {args.train_count}")
        log(f"Random seed: {args.seed}")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        # Check input folder exists
        if not INPUT_FOLDER.is_dir():
            raise FileNotFoundError(f"Input images folder not found: {INPUT_FOLDER.resolve()}")

        # Validate whichever mask mode is active
        if MASK_SOURCE_FILE:
            if not MASK_SOURCE_FILE.is_file():
                raise FileNotFoundError(
                    f"Ground-truth mask not found: {MASK_SOURCE_FILE.resolve()}. "
                    f"Run 06_artifacts_removal_lama.py first so this mask gets created."
                )
            mask_check = cv2.imread(str(MASK_SOURCE_FILE), cv2.IMREAD_GRAYSCALE)
            if mask_check is None:
                raise RuntimeError(f"Ground-truth mask could not be read as an image: {MASK_SOURCE_FILE.resolve()}")
        else:
            if not MASKS_FOLDER.is_dir():
                raise FileNotFoundError(
                    f"Masks folder not found: {MASKS_FOLDER.resolve()}. "
                    f"Run 08_generate_synthetic_backgrounds.py first so per-image masks get created."
                )

        # Collect all cleaned frames
        extensions = {'.png'}
        files = sorted(
            f for f in INPUT_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total_files = len(files)
        log(f"Found {total_files} cleaned frame(s) in {INPUT_FOLDER.resolve()}.")

        if total_files == 0:
            log("No frames found, nothing to split.", "WARNING")
            log("Script completed successfully.", "SUCCESS")
            log("============================================")
            return 0

        train_count = args.train_count
        if train_count >= total_files:
            log(f"Requested train count ({train_count}) >= total available frames ({total_files}). "
                f"All frames will go to TRAIN, and the 'rest' (val) set will be empty.", "WARNING")
            train_count = total_files

        # Random, reproducible split
        rng = random.Random(args.seed)
        shuffled = files.copy()
        rng.shuffle(shuffled)
        train_files = sorted(shuffled[:train_count])
        val_files = sorted(shuffled[train_count:])

        log(f"Split: {len(train_files)} frame(s) -> train, {len(val_files)} frame(s) -> rest (val).")

        # Create output folders
        for d in [TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        def copy_set(file_list, images_dir: Path, masks_dir: Path, set_name: str):
            copied = 0
            failed = 0
            for f in file_list:
                try:
                    shutil.copy2(f, images_dir / f.name)
                    if MASK_SOURCE_FILE:
                        # Ground-truth mask is the same for every frame (fixed circle
                        # geometry), so it's copied once per frame under a matching name.
                        shutil.copy2(MASK_SOURCE_FILE, masks_dir / f.name)
                    else:
                        # Per-image mask, matched by filename (from 08_generate_synthetic_backgrounds.py)
                        source_mask = MASKS_FOLDER / f.name
                        if not source_mask.is_file():
                            raise FileNotFoundError(f"No matching mask found for {f.name} in {MASKS_FOLDER}")
                        shutil.copy2(source_mask, masks_dir / f.name)
                    copied += 1
                except Exception as file_err:
                    failed += 1
                    log(f"Failed to copy {f.name} into {set_name} set: {file_err}", "ERROR")
            return copied, failed

        train_copied, train_failed = copy_set(train_files, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, "train")
        val_copied, val_failed = copy_set(val_files, VAL_IMAGES_DIR, VAL_MASKS_DIR, "val")

        log(f"Train set written: {train_copied} image/mask pair(s) to {TRAIN_IMAGES_DIR.resolve()} "
            f"and {TRAIN_MASKS_DIR.resolve()}", "SUCCESS")
        log(f"Val (rest) set written: {val_copied} image/mask pair(s) to {VAL_IMAGES_DIR.resolve()} "
            f"and {VAL_MASKS_DIR.resolve()}", "SUCCESS")

        if train_failed or val_failed:
            log(f"{train_failed + val_failed} file(s) failed to copy.", "WARNING")

        log("Script completed successfully.", "SUCCESS")
        log("============================================")
        return 0

    except Exception as e:
        log(f"Script failed: {e}", "ERROR")
        log("============================================")
        return 1


if __name__ == '__main__':
    sys.exit(main())