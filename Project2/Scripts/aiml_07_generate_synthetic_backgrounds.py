# ============================================
# Script: aiml_08_generate_synthetic_backgrounds.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Take EVERY real, artifact-cleaned frame found in Images\03_Cleaned_lama
#     (whatever the count happens to be for a given experiment - this script
#     does not assume or hardcode a specific number of frames)
#   - For each frame, generate one or more synthetic "varied background"
#     versions (noise, gradient, vignette, texture, scan lines, different
#     colors/brightness), simulating plausible post-scan background variation
#   - Reuse the existing ground-truth circle mask (fixed geometry across all
#     frames in a given experiment) to keep the real tomogram data intact
#     and only replace the background
#   - Save outputs as plain, legacy-safe 8-bit RGB PNG (no alpha channel, no
#     16-bit depth, no unusual color profile) so results remain readable by
#     older/legacy image tooling, not just modern Python libraries
#   - Save matching ground-truth masks alongside, ready for training
#   - Save log to Logs folder beside script
#   - Create output folders if they do not exist
#
# Output layout:
#   Images\05_Synthetic_Backgrounds\images\<frame>_<bgtype>_<variant>.png
#   Images\05_Synthetic_Backgrounds\masks\<frame>_<bgtype>_<variant>.png
#
# Scaling:
#   Total output images = (number of real frames found) x (--variants-per-image)
#   e.g. 129 real frames x 3 variants = 387 synthetic training images.
#   The next experiment might have a different frame count; this script
#   reads the folder at run time and adapts automatically.
# ============================================

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate varied synthetic backgrounds for every real cleaned frame found, "
                "scaling automatically to however many frames are present."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--input-images-folder", type=str, default="03_Cleaned_lama",
    help="Subfolder under Images\\ containing the real cleaned frames (default: 03_Cleaned_lama)."
)
parser.add_argument(
    "--mask-source", type=str, default="03_Masks_lama/_protected_region_mask.png",
    help="Path (relative to Images\\) to the ground-truth circle mask, reused for every frame "
         "(default: 03_Masks_lama/_protected_region_mask.png)."
)
parser.add_argument(
    "--output-folder", type=str, default="05_Synthetic_Backgrounds",
    help="Subfolder under Images\\ to write synthetic-background images and masks into "
         "(default: 05_Synthetic_Backgrounds)."
)
parser.add_argument(
    "--variants-per-image", type=int, default=3,
    help="Number of different synthetic-background versions to generate per real frame "
         "(default: 3). Total output count = frames found x this value."
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed, so the same synthetic backgrounds can be reproduced (default: 42)."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "aiml_08_generate_synthetic_backgrounds.log"

IMAGES_ROOT = SCRIPT_DIR / ".." / "Images"
INPUT_FOLDER = IMAGES_ROOT / args.input_images_folder
MASK_SOURCE_FILE = IMAGES_ROOT / args.mask_source
OUTPUT_ROOT = IMAGES_ROOT / args.output_folder
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_MASKS_DIR = OUTPUT_ROOT / "masks"

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("generate_synthetic_backgrounds")
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
# Synthetic background generators
# Each returns an (h, w, 3) uint8 BGR image. Kept simple and dependency-free
# (just numpy/opencv) so this script has no unusual runtime requirements.
# ----------------------------------------------------
BACKGROUND_TYPES = [
    "solid_light", "solid_gray", "warm_tint", "noisy_white",
    "gradient", "vignette", "speckle_texture", "scanlines",
    "dark", "colored_bed",
]


def make_background(bg_type: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    if bg_type == "solid_light":
        c = rng.integers(220, 256, 3)
        return np.full((h, w, 3), c, dtype=np.uint8)

    if bg_type == "solid_gray":
        v = rng.integers(160, 220)
        return np.full((h, w, 3), v, dtype=np.uint8)

    if bg_type == "warm_tint":
        c = (rng.integers(200, 235), rng.integers(215, 245), rng.integers(230, 255))  # BGR
        return np.full((h, w, 3), c, dtype=np.uint8)

    if bg_type == "noisy_white":
        base = np.full((h, w, 3), 255, dtype=np.float32)
        noise = rng.normal(0, rng.uniform(5, 20), (h, w, 3))
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    if bg_type == "gradient":
        a, b = rng.uniform(150, 255), rng.uniform(150, 255)
        horizontal = rng.random() < 0.5
        ramp = np.linspace(a, b, w if horizontal else h).astype(np.uint8)
        grad = np.tile(ramp, (h, 1)) if horizontal else np.tile(ramp.reshape(-1, 1), (1, w))
        return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    if bg_type == "vignette":
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        strength = rng.uniform(40, 100)
        vig = 255 - (dist / dist.max()) * strength
        return cv2.cvtColor(vig.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if bg_type == "speckle_texture":
        base_val = rng.integers(220, 245)
        base = np.full((h, w, 3), base_val, dtype=np.uint8)
        speckle = (rng.random((h, w)) * rng.uniform(20, 50)).astype(np.uint8)
        return cv2.subtract(base, cv2.cvtColor(speckle, cv2.COLOR_GRAY2BGR))

    if bg_type == "scanlines":
        base_val = rng.integers(235, 250)
        line_val = base_val - rng.integers(15, 35)
        lines = np.full((h, w, 3), base_val, dtype=np.uint8)
        spacing = rng.integers(3, 6)
        lines[::spacing, :, :] = line_val
        return lines

    if bg_type == "dark":
        v = rng.integers(20, 70)
        return np.full((h, w, 3), v, dtype=np.uint8)

    if bg_type == "colored_bed":
        c = rng.integers(0, 255, 3)
        return np.full((h, w, 3), c, dtype=np.uint8)

    raise ValueError(f"Unknown background type: {bg_type}")


def composite_onto_background(frame: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Keep 'frame' pixels wherever mask is foreground (255), replace the rest
    with 'background'. Output stays plain 8-bit 3-channel BGR - no alpha
    channel - for maximum compatibility with older/legacy image tooling."""
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = frame.astype(np.float32) * mask3 + background.astype(np.float32) * (1 - mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 08_generate_synthetic_backgrounds.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Input images folder: {INPUT_FOLDER.resolve()}")
        log(f"Ground-truth mask source: {MASK_SOURCE_FILE.resolve()}")
        log(f"Output folder: {OUTPUT_ROOT.resolve()}")
        log(f"Variants per image: {args.variants_per_image}")
        log(f"Random seed: {args.seed}")
        log(f"Output format: plain 8-bit 3-channel PNG (no alpha), for legacy tool compatibility.")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        if not INPUT_FOLDER.is_dir():
            raise FileNotFoundError(f"Input images folder not found: {INPUT_FOLDER.resolve()}")

        if not MASK_SOURCE_FILE.is_file():
            raise FileNotFoundError(
                f"Ground-truth mask not found: {MASK_SOURCE_FILE.resolve()}. "
                f"Run 06_artifacts_removal_lama.py first so this mask gets created."
            )

        mask = cv2.imread(str(MASK_SOURCE_FILE), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Ground-truth mask could not be read: {MASK_SOURCE_FILE.resolve()}")

        # This does NOT assume any specific number of frames - it reads
        # whatever is actually present, so it scales automatically whether
        # this experiment has 129 frames or a different experiment has 40 or 900.
        extensions = {'.png'}
        files = sorted(
            f for f in INPUT_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total_files = len(files)
        log(f"Found {total_files} real cleaned frame(s) in {INPUT_FOLDER.resolve()}.")

        if total_files == 0:
            log("No frames found, nothing to generate.", "WARNING")
            log("Script completed successfully.", "SUCCESS")
            log("============================================")
            return 0

        expected_total = total_files * args.variants_per_image
        log(f"Will generate {args.variants_per_image} variant(s) per frame "
            f"-> {expected_total} total synthetic-background image(s).")

        OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(args.seed)

        generated = 0
        failed = 0

        for frame_idx, frame_file in enumerate(files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                failed += 1
                log(f"Could not read frame: {frame_file.name}", "ERROR")
                continue

            if frame.shape[:2] != mask.shape[:2]:
                failed += 1
                log(f"Frame {frame_file.name} size {frame.shape[:2]} does not match "
                    f"mask size {mask.shape[:2]}, skipping.", "ERROR")
                continue

            for v in range(args.variants_per_image):
                bg_type = BACKGROUND_TYPES[rng.integers(0, len(BACKGROUND_TYPES))]
                try:
                    background = make_background(bg_type, frame.shape[0], frame.shape[1], rng)
                    composite = composite_onto_background(frame, mask, background)

                    out_name = f"{frame_file.stem}_{bg_type}_{v:02d}.png"
                    # Plain 8-bit 3-channel PNG write - explicit, no alpha, no
                    # compression params that would require newer PNG decoders.
                    cv2.imwrite(str(OUTPUT_IMAGES_DIR / out_name), composite)
                    cv2.imwrite(str(OUTPUT_MASKS_DIR / out_name), mask)
                    generated += 1
                except Exception as variant_err:
                    failed += 1
                    log(f"Failed to generate variant {v} ({bg_type}) for {frame_file.name}: "
                        f"{variant_err}", "ERROR")

            if frame_idx % 25 == 0:
                log(f"Progress: {frame_idx + 1}/{total_files} source frames processed.")

        log(f"Synthetic images generated: {generated}/{expected_total} "
            f"to {OUTPUT_IMAGES_DIR.resolve()}", "SUCCESS")
        log(f"Matching masks written to {OUTPUT_MASKS_DIR.resolve()}", "SUCCESS")

        if failed > 0:
            log(f"{failed} generation failure(s) encountered.", "WARNING")

        log("Script completed successfully.", "SUCCESS")
        log("============================================")
        return 0

    except Exception as e:
        log(f"Script failed: {e}", "ERROR")
        log("============================================")
        return 1


if __name__ == '__main__':
    sys.exit(main())
