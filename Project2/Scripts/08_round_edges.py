# ============================================
# Script: 08_round_edges.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Turn jagged, staircase-edged blobs into perfect, anti-aliased circles
#   - For each opaque blob found in an image, fit a true circle (based on the
#     blob's bounding box), fill in any gap between the original jagged edge
#     and the new perfect circle using the nearest neighbouring pixel's
#     color, and render the final circle edge with smooth anti-aliasing
#     (via supersampling)
#   - Save rounded-edge frames to Images\05_Final_Rounded
#   - Save log to Logs folder beside script
#   - Create output folder if it does not exist
#
# Notes:
#   - This is a purely cosmetic/visual smoothing step: it invents pixel
#     colors in the gap between the true jagged data edge and the fitted
#     circle, using nearest-neighbour inpainting. It does NOT increase the
#     real resolution or accuracy of the underlying data - genuine
#     resolution improvement belongs to the super-resolution model
#     downstream, not this script. Use this only where a visually smooth
#     circular edge is wanted for display/publication purposes, not for
#     any output that will be used for further quantitative analysis or
#     model training.
#   - Requires transparency (RGBA PNG) on the input images, since opaque
#     blobs are detected via the alpha channel (pixels with alpha above
#     ALPHA_CUTOFF are treated as part of the data blob).
#
# Source folder : Images\04_Transparent
# Output folder : Images\05_Final_Rounded (created automatically if missing)
# ============================================

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Replace jagged opaque blob edges with perfect, anti-aliased circles."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--alpha-cutoff", type=int, default=128,
    help="Alpha value above which a pixel is considered 'opaque' / part of a data blob (default: 128)."
)
parser.add_argument(
    "--supersample", type=int, default=4,
    help="Supersampling factor used for smooth, anti-aliased circle edges (default: 4)."
)
parser.add_argument(
    "--source-folder", type=str, default="04_Transparent",
    help="Subfolder under Images\\ containing the transparent-background frames to round "
         "(default: 04_Transparent)."
)
parser.add_argument(
    "--output-folder", type=str, default="05_Final_Rounded",
    help="Subfolder under Images\\ to write the rounded-edge frames into "
         "(default: 05_Final_Rounded)."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log

ALPHA_CUTOFF = args.alpha_cutoff
SUPERSAMPLE = args.supersample
VALID_EXTENSIONS = {".png"}  # transparency requires PNG

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "08_round_edges.log"

SOURCE_DIR = SCRIPT_DIR / ".." / "Images" / args.source_folder
OUTPUT_DIR = SCRIPT_DIR / ".." / "Images" / args.output_folder

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("round_edges")
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


def make_perfect_circles(img: Image.Image, alpha_cutoff=ALPHA_CUTOFF,
                          supersample=SUPERSAMPLE) -> Image.Image:
    """Replace each jagged opaque blob with a perfect circle, inpainting any
    newly-exposed pixels using the nearest original neighbour's color."""
    arr = np.array(img.convert("RGBA")).astype(np.float64)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    H, W = a.shape
    orig_mask = a > alpha_cutoff

    labeled, n = ndimage.label(orig_mask)
    new_mask = np.zeros((H, W), dtype=bool)

    # Supersampled canvas so the circle edge can be anti-aliased smoothly
    alpha_super = np.zeros((H * supersample, W * supersample), dtype=np.float64)
    yy, xx = np.mgrid[0:H * supersample, 0:W * supersample]

    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue

        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        # Radius from the blob's bounding box (average of half-width/half-height)
        radius = ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4.0

        Y, X = np.ogrid[0:H, 0:W]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        new_mask |= dist <= radius

        cx_s, cy_s, r_s = cx * supersample, cy * supersample, radius * supersample
        dist_s = np.sqrt((xx - cx_s - supersample / 2) ** 2 + (yy - cy_s - supersample / 2) ** 2)
        alpha_super = np.where(dist_s <= r_s, 255.0, alpha_super)

    # Downsample the supersampled mask for a smooth, anti-aliased alpha edge
    alpha_final = alpha_super.reshape(H, supersample, W, supersample).mean(axis=(1, 3))

    # Inpaint pixels that are inside the new circle but weren't in the original
    # blob, using the color of the nearest originally-opaque pixel
    need_fill = new_mask & ~orig_mask
    if need_fill.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~orig_mask, return_indices=True)
        r[need_fill] = r[iy[need_fill], ix[need_fill]]
        g[need_fill] = g[iy[need_fill], ix[need_fill]]
        b[need_fill] = b[iy[need_fill], ix[need_fill]]

    out_rgb = np.dstack([r, g, b]).astype(np.uint8)
    out = Image.fromarray(out_rgb, "RGB").convert("RGBA")
    out.putalpha(Image.fromarray(alpha_final.astype(np.uint8), "L"))
    return out, n


def main() -> int:
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 08_round_edges.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Source folder: {SOURCE_DIR.resolve()}")
        log(f"Output folder: {OUTPUT_DIR.resolve()}")
        log(f"Alpha cutoff: {ALPHA_CUTOFF}")
        log(f"Supersample factor: {SUPERSAMPLE}")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        if not SOURCE_DIR.is_dir():
            raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR.resolve()}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in VALID_EXTENSIONS
        )

        log(f"Found {len(image_files)} PNG file(s) in {SOURCE_DIR.resolve()}.")

        if not image_files:
            log("No PNG images found, nothing to do.", "WARNING")
            log("Script completed successfully.", "SUCCESS")
            log("============================================")
            return 0

        processed = 0
        failed = 0
        total_blobs = 0

        for f in image_files:
            try:
                img = Image.open(f)
                result, n_blobs = make_perfect_circles(img)
                out_path = OUTPUT_DIR / f.name
                result.save(out_path)
                processed += 1
                total_blobs += n_blobs
                log(f"Processed: {f.name} ({n_blobs} blob(s) rounded) -> {out_path}")
            except Exception as file_err:
                failed += 1
                log(f"Failed to process {f.name}: {file_err}", "ERROR")

        log(f"Rounded-edge frames saved: {processed} file(s) to {OUTPUT_DIR.resolve()}", "SUCCESS")
        log(f"Total blobs rounded across all frames: {total_blobs}")

        if failed > 0:
            log(f"{failed} file(s) failed to process.", "WARNING")

        log("Script completed successfully.", "SUCCESS")
        log("============================================")
        return 0

    except Exception as e:
        log(f"Script failed: {e}", "ERROR")
        log("============================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())