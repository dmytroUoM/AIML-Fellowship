# ============================================
# Script: 06_artifacts_removal_lama.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Detect text/numbering artifacts of ANY color by protecting ONLY the
#     circular plot regions (fitted geometrically) and flagging every other
#     non-white pixel as an artifact - including numbers, P-labels, values,
#     and any newly-added colored debug text
#   - Remove detected artifacts using LaMa (deep learning inpainting model)
#   - Save cleaned frames to Images\03_Cleaned_lama folder
#   - Save debug masks to Images\03_Masks_lama folder
#   - Save log to Logs folder beside script
#   - Create output folders if they do not exist
#
# Detection strategy:
#   1. Sample a handful of frames from the input folder.
#   2. Build a raw non-white mask = pixels that are non-white in most sampled
#      frames (a robust proxy for "something is always drawn here").
#   3. Split that raw mask into connected components, and keep only the large
#      ones (area >= --min-circle-area) - these are the plot circles. Small
#      components (numbers, P-labels, values, or any other overlay text) are
#      discarded here, since they must NOT be protected.
#   4. Fit a minimum enclosing circle to each surviving component and
#      rasterize a clean filled disk. This is the protected region: strictly
#      the circular plot area, nothing more - matching the manual ROI circle
#      example (ROI = inside the circle only).
#   5. Cache that protected mask to disk so it is only computed once.
#   6. For every frame, artifact mask = (non-white pixels) AND NOT (protected
#      circles). Any non-white pixel outside the circles - numbers, labels,
#      values, or any newly-added colored text - is treated as an artifact.
#   7. Inpaint only the artifact mask using LaMa.
#
# Requirements:
#   pip install opencv-python numpy pillow torch torchvision
#   pip install simple-lama-inpainting --no-deps
#   (see README for why --no-deps is recommended)
#
# Model:
#   This script looks for the LaMa "big-lama.pt" checkpoint (~196 MB) at
#   Bin\big-lama.pt beside the script, matching the Bin\ffmpeg.exe convention
#   used in the .ps1 scripts. If not found there, it falls back to
#   downloading it automatically to the torch hub cache.
#   Source: https://github.com/enesmsahin/simple-lama-inpainting
# ============================================

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Remove text/numbering artifacts of any color from frames, "
                "protecting only the circular plot regions, plus LaMa deep learning inpainting."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--white-thresh", type=int, default=245,
    help="Pixel channel value above which a pixel counts as 'white' background (default: 245)."
)
parser.add_argument(
    "--sample-size", type=int, default=20,
    help="Number of frames sampled (evenly spaced) to build the protected plot-region mask (default: 20)."
)
parser.add_argument(
    "--protect-vote-fraction", type=float, default=0.8,
    help="Fraction of sampled frames a pixel must be non-white in to be considered for the protected region (default: 0.8)."
)
parser.add_argument(
    "--min-circle-area", type=int, default=5000,
    help="Minimum connected-component pixel area to be treated as real plot data rather than "
         "text/numbering (default: 5000). Lower this if your data blobs are smaller than this; "
         "raise it if large text blobs are being mistaken for data."
)
parser.add_argument(
    "--open-kernel-px", type=int, default=1,
    help="Morphological opening kernel radius applied before component detection, to sever thin "
         "(often single-pixel, anti-aliasing-driven) accidental bridges between the data region and "
         "nearby text before they get fused into one blob (default: 1, i.e. a 3x3 kernel)."
)
parser.add_argument(
    "--mask-pad-px", type=int, default=0,
    help="Dilate (positive) or erode (negative) the final protected region by this many pixels, "
         "in case the true data boundary needs a small safety margin either direction "
         "(default: 0, i.e. use the true jagged data shape exactly as detected)."
)
parser.add_argument(
    "--artifact-dilate-px", type=int, default=3,
    help="Dilation radius in pixels applied to each detected artifact mask before inpainting (default: 3)."
)
parser.add_argument(
    "--rebuild-protected-mask", action="store_true",
    help="Recompute the protected region mask even if a cached one already exists on disk."
)
parser.add_argument(
    "--device", type=str, default=None, choices=["cpu", "cuda"],
    help="Force inference device. Default: cuda if available, else cpu."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "06_artifacts_removal_lama.log"

INPUT_FOLDER = SCRIPT_DIR / ".." / "Images" / "02_Frames"
OUTPUT_FOLDER = SCRIPT_DIR / ".." / "Images" / "03_Cleaned_lama"
MASK_FOLDER = SCRIPT_DIR / ".." / "Images" / "03_Masks_lama"
MODEL_FILE = SCRIPT_DIR / ".." / "Bin" / "big-lama.pt"
PROTECTED_MASK_FILE = MASK_FOLDER / "_protected_region_mask.png"

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("artifacts_removal_lama")
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


# ----------------------------------------------------
# Generalized (color-agnostic) artifact detection
# ----------------------------------------------------
def nonwhite_mask(img_bgr: np.ndarray, white_thresh: int) -> np.ndarray:
    """True where a pixel is meaningfully different from white background,
    regardless of what color it is (magenta, black, green, blue, etc)."""
    dist_from_white = 255 - np.min(img_bgr, axis=2)
    return dist_from_white > (255 - white_thresh)


def build_protected_region_mask(files, white_thresh: int, sample_size: int,
                                 vote_fraction: float, min_circle_area: int,
                                 open_kernel_px: int, mask_pad_px: int) -> np.ndarray:
    """Sample frames evenly across the dataset, find pixels that are non-white
    in most of them, then keep ONLY the large blobs (the real plot data) and
    discard small blobs (numbers, labels, values, any overlay text).

    IMPORTANT: the protected region is the TRUE pixel shape of each surviving
    component, not a fitted circle. Real tomogram/plot data is often a coarse
    grid with a jagged (staircase) boundary, not a perfectly smooth disk.
    Fitting a smooth circle to it would either cut off real data pixels that
    poke past the circle, or wrongly protect blank corner pixels the circle
    covers but the real data doesn't - both are real defects, not cosmetic
    ones, since the former silently deletes real data during inpainting.

    A single safeguard remains against a single stray pixel distorting the
    result: morphological opening severs thin (often 1px, anti-aliasing-
    driven) accidental bridges between the data region and nearby text
    BEFORE labeling connected components, so they don't get fused into one
    blob in the first place.
    """
    n = len(files)
    sample_n = min(sample_size, n)
    if sample_n < 1:
        raise ValueError("No frames available to build the protected region mask.")

    step = max(1, n // sample_n)
    sample_files = files[::step][:sample_n]

    log(f"Sampling {len(sample_files)} frame(s) to build protected plot-region mask: "
        f"{[f.name for f in sample_files]}")

    vote_count = None
    shape = None

    for f in sample_files:
        img = cv2.imread(str(f))
        if img is None:
            log(f"Could not read sample frame {f.name}, skipping.", "WARNING")
            continue

        if shape is None:
            shape = img.shape[:2]
            vote_count = np.zeros(shape, dtype=np.int32)
        elif img.shape[:2] != shape:
            log(f"Sample frame {f.name} has different dimensions, skipping.", "WARNING")
            continue

        vote_count += nonwhite_mask(img, white_thresh).astype(np.int32)

    if vote_count is None:
        raise RuntimeError("Failed to read any sample frames while building protected region mask.")

    threshold_votes = vote_fraction * len(sample_files)
    raw_mask = (vote_count >= threshold_votes).astype(np.uint8) * 255

    # Sever thin accidental bridges (e.g. a single anti-aliased pixel linking
    # the data region to nearby text) before labeling connected components.
    if open_kernel_px > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_kernel_px + 1, 2 * open_kernel_px + 1)
        )
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels = cv2.connectedComponents(raw_mask)
    protected = np.zeros(raw_mask.shape, dtype=np.uint8)
    blobs_found = 0

    for i in range(1, num_labels):
        component_mask = (labels == i)
        area = int(np.count_nonzero(component_mask))
        if area < min_circle_area:
            continue

        # Use the TRUE pixel shape of this component directly - no circle
        # fitting - so the jagged real data boundary is preserved exactly.
        protected[component_mask] = 255
        blobs_found += 1

    if mask_pad_px != 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * abs(mask_pad_px) + 1, 2 * abs(mask_pad_px) + 1)
        )
        if mask_pad_px > 0:
            protected = cv2.dilate(protected, kernel)
        else:
            protected = cv2.erode(protected, kernel)

    protected_px = int(np.count_nonzero(protected))
    total_px = protected.size
    log(f"Detected {blobs_found} plot data blob(s) via connected components "
        f"(min area threshold: {min_circle_area}px), using their true jagged shape.")
    log(f"Protected plot-region mask built: {protected_px}/{total_px} px "
        f"({100.0 * protected_px / total_px:.1f}%) protected.")

    if blobs_found == 0:
        log("No plot data blobs detected! Check --white-thresh and --min-circle-area, "
            "or inspect the cached mask file to debug.", "WARNING")

    return protected


def get_or_build_protected_mask(files) -> np.ndarray:
    if PROTECTED_MASK_FILE.is_file() and not args.rebuild_protected_mask:
        log(f"Loading cached protected region mask: {PROTECTED_MASK_FILE.resolve()}")
        cached = cv2.imread(str(PROTECTED_MASK_FILE), cv2.IMREAD_GRAYSCALE)
        if cached is not None:
            return cached
        log("Cached protected region mask could not be read, rebuilding.", "WARNING")

    protected = build_protected_region_mask(
        files,
        white_thresh=args.white_thresh,
        sample_size=args.sample_size,
        vote_fraction=args.protect_vote_fraction,
        min_circle_area=args.min_circle_area,
        open_kernel_px=args.open_kernel_px,
        mask_pad_px=args.mask_pad_px
    )
    cv2.imwrite(str(PROTECTED_MASK_FILE), protected)
    log(f"Protected region mask cached to: {PROTECTED_MASK_FILE.resolve()}", "SUCCESS")
    return protected


def remove_artifacts_lama(lama, input_path: str, output_path: str,
                           protected_mask: np.ndarray, white_thresh: int,
                           artifact_dilate_px: int) -> bool:
    """Detect any non-white, non-protected pixels (artifacts of any color),
    inpaint with LaMa, save cleaned image + debug mask.

    Returns True if any artifact pixels were detected.
    """
    from PIL import Image

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {input_path}')

    nonwhite = nonwhite_mask(img, white_thresh).astype(np.uint8) * 255
    artifact_mask = cv2.bitwise_and(nonwhite, cv2.bitwise_not(protected_mask))

    if artifact_dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * artifact_dilate_px + 1, 2 * artifact_dilate_px + 1)
        )
        artifact_mask = cv2.dilate(artifact_mask, kernel)

    had_detection = bool(cv2.countNonZero(artifact_mask) > 0)

    # Save mask for debugging
    mask_file = MASK_FOLDER / f"{Path(input_path).stem}_mask.png"
    cv2.imwrite(str(mask_file), artifact_mask)

    if had_detection:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(artifact_mask)

        result = lama(image_pil, mask_pil)
        cleaned = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # LaMa internally pads the image up to a multiple of 8px before
        # inference and does NOT crop back down afterward. If the original
        # dimensions aren't already a multiple of 8 (e.g. 940 wide), the
        # output would silently come back larger than the input. Crop back
        # to the original size here so every cleaned frame stays pixel-for-
        # pixel aligned with its source frame and mask.
        if cleaned.shape[:2] != img.shape[:2]:
            cleaned = cleaned[:img.shape[0], :img.shape[1]]
    else:
        # Nothing detected, no need to run inference
        cleaned = img

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cleaned)

    return had_detection


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main() -> int:
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 06_artifacts_removal_lama.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Input folder: {INPUT_FOLDER.resolve()}")
        log(f"Output folder: {OUTPUT_FOLDER.resolve()}")
        log(f"Mask folder: {MASK_FOLDER.resolve()}")
        log(f"White threshold: {args.white_thresh}")
        log(f"Protected mask sample size: {args.sample_size}")
        log(f"Protected mask vote fraction: {args.protect_vote_fraction}")
        log(f"Minimum data blob area: {args.min_circle_area}px")
        log(f"Opening kernel: {args.open_kernel_px}px")
        log(f"Mask padding: {args.mask_pad_px}px")
        log(f"Artifact mask dilation: {args.artifact_dilate_px}px")

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

        if not files:
            log("No PNG files found, nothing to do.", "WARNING")
            log("Script completed successfully.", "SUCCESS")
            log("============================================")
            return 0

        # Build (or load cached) protected plot-region mask
        protected_mask = get_or_build_protected_mask(files)

        # Import torch / LaMa here so --help works even if not installed yet
        import torch
        from simple_lama_inpainting import SimpleLama

        device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        log(f"Inference device: {device_str}")

        # Prefer a local checkpoint at Bin\big-lama.pt (matches the
        # Bin\ffmpeg.exe convention). Falls back to the auto-downloading
        # torch hub cache if not found locally.
        if MODEL_FILE.is_file():
            log(f"Using local LaMa checkpoint: {MODEL_FILE.resolve()}")
            os.environ["LAMA_MODEL"] = str(MODEL_FILE.resolve())
        else:
            log(f"Local checkpoint not found at {MODEL_FILE.resolve()}.", "WARNING")
            log("Falling back to automatic download (~196 MB) to the torch hub cache.", "WARNING")

        t_load_start = time.time()
        lama = SimpleLama(device=device)
        log(f"LaMa model loaded successfully in {time.time() - t_load_start:.1f}s.", "SUCCESS")

        processed = 0
        failed = 0
        frames_with_detections = 0
        t_batch_start = time.time()

        for infile in files:
            outfile = OUTPUT_FOLDER / infile.name
            try:
                t0 = time.time()
                had_detection = remove_artifacts_lama(
                    lama, str(infile), str(outfile),
                    protected_mask=protected_mask,
                    white_thresh=args.white_thresh,
                    artifact_dilate_px=args.artifact_dilate_px
                )
                elapsed = time.time() - t0
                processed += 1
                frames_with_detections += int(had_detection)
                log(f"Processed: {infile.name} (artifact detected: {had_detection}, {elapsed:.2f}s)")
            except Exception as file_err:
                failed += 1
                log(f"Failed to process {infile.name}: {file_err}", "ERROR")

        total_elapsed = time.time() - t_batch_start

        log(f"Cleaned frames saved: {processed} file(s) to {OUTPUT_FOLDER.resolve()}", "SUCCESS")
        log(f"Mask files saved: {processed} file(s) to {MASK_FOLDER.resolve()}", "SUCCESS")
        log(f"Frames with at least one artifact detection: {frames_with_detections}/{processed}")
        if processed:
            log(f"Total processing time: {total_elapsed:.1f}s "
                f"({(total_elapsed / processed):.2f}s/frame average)")
        else:
            log("No frames processed.")

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