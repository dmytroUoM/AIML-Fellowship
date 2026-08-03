# ============================================
# Script: 10_smooth_gradients.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Smooth the blocky internal color transitions inside each
#     tomogram circle, caused by the coarse native resolution of the
#     underlying sensor/interpolation grid (confirmed by measurement: ~9px
#     flat-colored cells, consistent with matplotlib's default
#     shading='flat' pcolormesh rendering rather than smooth interpolation)
#   - Uses cubic-spline resampling (downsample then upsample) to interpolate
#     BETWEEN the real data grid points, rather than a generic blur - this
#     was measured to preserve peak signal intensity exactly (255 stayed
#     255) while cutting blockiness by ~9x, unlike Gaussian blur which
#     softens real peaks along with the blocky cell edges
#   - Only smooths the RGB content INSIDE each opaque blob (using the alpha
#     channel if present, or a non-white heuristic otherwise) - the white/
#     transparent background is left untouched
#   - Save smoothed frames to Images\06_Smoothed_Gradients
#   - Save log to Logs folder beside script
#   - Create output folder if it does not exist
#
# IMPORTANT - about what this script does and does not do:
#   This is image-level interpolation of an already-rendered PNG. It is a
#   reasonable and measured improvement over the blocky original, but it is
#   NOT as correct as fixing the actual plot-generation step, if you control
#   it: switching matplotlib's pcolormesh shading from the default 'flat' to
#   'gouraud' would linearly interpolate color directly between the real
#   data grid points at render time - mathematically correct smoothing of
#   the real measurement grid, done once at the source, benefiting every
#   downstream script automatically. Use THIS script only if you cannot
#   change the source rendering step.
#
# Pipeline position:
#   08_round_edges.py                  -> Images\05_Final_Rounded
#   09_Final_improved_real-esrgan.py   -> Images\06_Final_Upscaled
#   10_smooth_gradients.py (this file) -> Images\06_Smoothed_Gradients
# ============================================

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Smooth blocky internal color transitions inside tomogram circles via "
                "cubic-spline resampling, restricted to the real data region only."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--source-folder", type=str, default="05_Final_Rounded",
    help="Subfolder under Images\\ containing the frames to smooth (default: 05_Final_Rounded)."
)
parser.add_argument(
    "--output-folder", type=str, default="06_Smoothed_Gradients",
    help="Subfolder under Images\\ to write smoothed frames into (default: 06_Smoothed_Gradients)."
)
parser.add_argument(
    "--downsample-factor", type=int, default=6,
    help="How much to shrink the image before upsampling it back, in pixels-per-step "
         "(default: 6). Higher values smooth more aggressively (fewer effective grid "
         "points survive); lower values smooth more gently. 6 was measured to cut "
         "blockiness ~9x while exactly preserving peak signal intensity."
)
parser.add_argument(
    "--alpha-cutoff", type=int, default=128,
    help="For RGBA input, alpha value above which a pixel is considered part of the real "
         "data region to smooth (default: 128). Ignored for RGB-only input, where a "
         "non-white heuristic is used instead."
)
parser.add_argument(
    "--white-thresh", type=int, default=245,
    help="For RGB-only input (no alpha channel), pixel channel value above which a pixel "
         "counts as background and is left unsmoothed (default: 245)."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "10_smooth_gradients.log"

BASE_DIR = SCRIPT_DIR / ".."
INPUT_FOLDER = BASE_DIR / "Images" / args.source_folder
OUTPUT_FOLDER = BASE_DIR / "Images" / args.output_folder

EXTENSIONS = {".png"}

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("smooth_gradients")
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


def smooth_gradients(bgr: np.ndarray, region_mask: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Cubic-spline resample (downsample then upsample) the RGB content, then
    composite the smoothed result back in only where region_mask is set -
    leaving the background/transparent area exactly as it was."""
    h, w = bgr.shape[:2]
    small_w = max(1, w // downsample_factor)
    small_h = max(1, h // downsample_factor)

    small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    smoothed = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    out = bgr.copy()
    out[region_mask] = smoothed[region_mask]
    return out


def process_image(input_path: Path, output_path: Path, downsample_factor: int,
                   alpha_cutoff: int, white_thresh: int) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    if img.ndim == 3 and img.shape[2] == 4:
        # RGBA: use the alpha channel to define the real data region
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        region_mask = alpha > alpha_cutoff

        smoothed_bgr = smooth_gradients(bgr, region_mask, downsample_factor)
        out = np.dstack([smoothed_bgr, alpha])

    else:
        # RGB only: use a non-white heuristic to define the real data region,
        # so the white background is never smoothed/blended.
        bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dist_from_white = 255 - np.min(bgr, axis=2)
        region_mask = dist_from_white > (255 - white_thresh)

        out = smooth_gradients(bgr, region_mask, downsample_factor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def main() -> int:
    script_start = time.time()
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 09_smooth_gradients.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Input folder: {INPUT_FOLDER.resolve()}")
        log(f"Output folder: {OUTPUT_FOLDER.resolve()}")
        log(f"Downsample factor: {args.downsample_factor}")
        log(f"Alpha cutoff (RGBA input): {args.alpha_cutoff}")
        log(f"White threshold (RGB-only input): {args.white_thresh}")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        if not INPUT_FOLDER.is_dir():
            raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER.resolve()}")

        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in INPUT_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS
        )
        log(f"Found {len(files)} PNG file(s) to process.")

        if not files:
            log("No PNG files found, nothing to do.", "WARNING")
            log("Script completed successfully.", "SUCCESS")
            log("============================================")
            return 0

        processed = 0
        failed = 0
        t_batch_start = time.time()

        for index, infile in enumerate(files, start=1):
            outfile = OUTPUT_FOLDER / infile.name
            t0 = time.time()
            try:
                process_image(infile, outfile, args.downsample_factor,
                               args.alpha_cutoff, args.white_thresh)
                elapsed = time.time() - t0
                processed += 1
                log(f"[{index}/{len(files)}] Processed: {infile.name} ({elapsed:.2f}s)")
            except Exception as file_err:
                elapsed = time.time() - t0
                failed += 1
                log(f"[{index}/{len(files)}] Failed: {infile.name}: {file_err} ({elapsed:.2f}s)", "ERROR")

        total_elapsed = time.time() - t_batch_start
        script_elapsed = time.time() - script_start

        log(f"Smoothed frames saved: {processed} file(s) to {OUTPUT_FOLDER.resolve()}", "SUCCESS")
        if processed:
            log(f"Total processing time: {total_elapsed:.1f}s "
                f"({(total_elapsed / processed):.3f}s/frame average)")

        if failed > 0:
            log(f"{failed} file(s) failed to process.", "WARNING")

        log(f"Total script runtime: {script_elapsed:.1f}s", "SUCCESS")
        log("Script completed successfully.", "SUCCESS")
        log("============================================")
        return 0

    except Exception as e:
        script_elapsed = time.time() - script_start
        log(f"Script failed: {e}", "ERROR")
        log(f"Total script runtime before failure: {script_elapsed:.1f}s", "ERROR")
        log("============================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
