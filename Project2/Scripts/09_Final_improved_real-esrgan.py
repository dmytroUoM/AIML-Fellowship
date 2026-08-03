# ============================================
# Script: 09_Final_improved_real-esrgan.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - AI enhancement / super-resolution for scientific resistance-tomography
#     frames, using the pretrained Real-ESRGAN (RRDBNet) model
#   - Upscale the RGB content of each frame with Real-ESRGAN
#   - Resize the alpha (transparency) channel separately using standard image
#     interpolation - NOT through the AI model - so transparent backgrounds
#     stay cleanly transparent (see "Transparency note" below)
#   - Save upscaled frames to Images\06_Final_Upscaled
#   - Save log to Logs folder beside script, including per-file and total
#     script runtime
#   - Create output folder if it does not exist
#
# Default folders (overridable via --source-folder / --output-folder):
#   Input  : Images\05_Final_Rounded\*.png
#   Output : Images\06_Final_Upscaled\*.png
#
# Expected structure:
#   Project2\
#   |-- Images\
#   |   |-- 05_Final_Rounded\      (default input)
#   |   `-- 06_Final_Upscaled\     (default output)
#   `-- Scripts\
#       |-- 09_Final_improved_real-esrgan.py
#       `-- RealESRGAN_x4plus.pth  (model file, see "Model file required" below)
#
# Run from the Scripts folder:
#   py 09_Final_improved_real-esrgan.py
#
# CPU install:
#   py -m pip install torch torchvision realesrgan basicsr facexlib gfpgan pillow numpy opencv-python
#
# NVIDIA GPU install example:
#   py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   py -m pip install realesrgan basicsr facexlib gfpgan pillow numpy opencv-python
#
# Model file required:
#   RealESRGAN_x4plus.pth - place in the same folder as this script
#   (Project2\Scripts\RealESRGAN_x4plus.pth), or point elsewhere with
#   --model-path.
#
# Scientific-image note:
#   Real-ESRGAN can improve visual clarity, but it may also hallucinate
#   details. For scientific reporting, keep the original Cleaned frames as
#   the source of truth. Use these upscaled outputs for visualisation/
#   presentation only, unless separately validated against ground truth.
#
# Transparency note:
#   The alpha (transparency) channel is NOT run through the AI model. Alpha
#   is a binary/soft mask, not photographic content, and feeding it through
#   a network trained to hallucinate texture and sharpen edges would
#   introduce noise, edge bleed, and non-zero "haze" in areas that should be
#   fully transparent. Instead, only the RGB channels are upscaled with
#   Real-ESRGAN, and the alpha channel is resized separately with standard
#   image interpolation, matched exactly to the AI-upscaled RGB output size.
#
# AI model parameters (all exposed as CLI args - see --help, or the
# argparse definitions below for full descriptions):
#   --device-mode          auto / cuda / cpu - which device runs the model
#   --model-name            model architecture/weights file name (without .pth)
#   --model-path            explicit path to the .pth file, overrides the
#                            default Scripts\<model-name>.pth location
#   --model-scale            native upscale factor the model was trained for
#                            (Real-ESRGAN x4plus = 4; changing this without a
#                            matching model file will produce wrong results)
#   --outscale                actual output scale to resize to after inference
#                            (can differ from --model-scale; e.g. run a x4
#                            model but output x2 for a gentler result)
#   --tile-cpu / --tile-gpu   tile size for tiled inference (reduces memory
#                            use, at a small speed cost); use 0 to disable
#                            tiling entirely (GPU only, if memory allows)
#   --tile-pad                 padding (px) added around each tile to avoid
#                            seam artifacts at tile borders
#   --pre-pad                  padding (px) added around the whole image
#                            before inference
#   --use-half-on-gpu / --no-half-on-gpu   half-precision (fp16) inference on
#                            CUDA for speed/memory; automatically disabled on CPU
#   --alpha-interpolation    lanczos4 (smooth, anti-aliased) or nearest (hard
#                            edges, no new in-between alpha values)
# ============================================

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts, plus
# every AI model parameter that was previously a hardcoded module constant)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="AI super-resolution (Real-ESRGAN) for scientific tomography frames, "
                "with alpha transparency handled separately from the AI model."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--source-folder", type=str, default="05_Final_Rounded",
    help="Subfolder under Images\\ containing the frames to upscale (default: 05_Final_Rounded)."
)
parser.add_argument(
    "--output-folder", type=str, default="06_Final_Upscaled",
    help="Subfolder under Images\\ to write upscaled frames into (default: 06_Final_Upscaled)."
)
parser.add_argument(
    "--device-mode", type=str, default="auto", choices=["auto", "cuda", "cpu"],
    help="'auto' uses the GPU if available, otherwise CPU. 'cuda' forces GPU and fails if "
         "unavailable. 'cpu' forces CPU (default: auto)."
)
parser.add_argument(
    "--model-name", type=str, default="RealESRGAN_x4plus",
    help="Model architecture/weights file name, without the .pth extension "
         "(default: RealESRGAN_x4plus)."
)
parser.add_argument(
    "--model-path", type=str, default=None,
    help="Explicit path to the .pth model file. Default: <ScriptDir>\\<model-name>.pth."
)
parser.add_argument(
    "--model-scale", type=int, default=4,
    help="Native upscale factor the model architecture was trained for (default: 4, matching "
         "RealESRGAN_x4plus). Only change this if using a different model file trained at a "
         "different scale - mismatching this with the actual .pth file will produce wrong results."
)
parser.add_argument(
    "--outscale", type=float, default=2,
    help="Actual output scale applied after inference (default: 2). For scientific tomography "
         "frames, 2x is often visually safer than the model's native 4x, since it improves "
         "edges without making the image excessively large. Set to match --model-scale for "
         "full-scale output."
)
parser.add_argument(
    "--tile-cpu", type=int, default=128,
    help="Tile size (px) used for tiled inference on CPU, to reduce memory usage (default: 128). "
         "Larger tiles = fewer tiles = more memory per tile."
)
parser.add_argument(
    "--tile-gpu", type=int, default=0,
    help="Tile size (px) used for tiled inference on GPU (default: 0, i.e. no tiling). Set to "
         "256 or 512 if you hit GPU out-of-memory errors."
)
parser.add_argument(
    "--tile-pad", type=int, default=10,
    help="Padding (px) added around each tile to avoid visible seam artifacts at tile borders (default: 10)."
)
parser.add_argument(
    "--pre-pad", type=int, default=0,
    help="Padding (px) added around the whole image before inference (default: 0)."
)
parser.add_argument(
    "--use-half-on-gpu", dest="use_half_on_gpu", action="store_true", default=True,
    help="Use half-precision (fp16) inference on CUDA for speed/memory (default: enabled). "
         "Automatically disabled on CPU regardless of this setting."
)
parser.add_argument(
    "--no-half-on-gpu", dest="use_half_on_gpu", action="store_false",
    help="Disable half-precision inference on CUDA (use full fp32 precision instead)."
)
parser.add_argument(
    "--alpha-interpolation", type=str, default="lanczos4", choices=["lanczos4", "nearest"],
    help="'lanczos4' gives smooth, anti-aliased alpha edges (default, recommended for most "
         "cases). 'nearest' gives hard edges with no new in-between alpha values - use this if "
         "you need a strictly binary transparent/opaque mask preserved exactly."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log
ALPHA_INTERPOLATION = cv2.INTER_LANCZOS4 if args.alpha_interpolation == "lanczos4" else cv2.INTER_NEAREST
EXTENSIONS = {".png"}  # Process PNG only, as requested.

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "09_Final_improved_real-esrgan.log"

INPUT_FOLDER = BASE_DIR / "Images" / args.source_folder
OUTPUT_FOLDER = BASE_DIR / "Images" / args.output_folder
MODEL_PATH = Path(args.model_path) if args.model_path else SCRIPT_DIR / f"{args.model_name}.pth"

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("real_esrgan_upscale")
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
# DEVICE SELECTION
# ----------------------------------------------------
def select_device() -> torch.device:
    if args.device_mode.lower() == "cpu":
        return torch.device("cpu")

    if args.device_mode.lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device-mode is set to 'cuda', but CUDA is not available.\n"
                "Either install CUDA-enabled PyTorch or use --device-mode cpu / auto."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------
# REAL-ESRGAN LOADING
# ----------------------------------------------------
def load_realesrgan_model(device: torch.device):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise ImportError(
            "Missing Real-ESRGAN dependencies. Install with:\n\n"
            "CPU:\n"
            "  py -m pip install torch torchvision realesrgan basicsr facexlib gfpgan pillow numpy opencv-python\n\n"
            "NVIDIA GPU example:\n"
            "  py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  py -m pip install realesrgan basicsr facexlib gfpgan pillow numpy opencv-python\n"
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n  {MODEL_PATH}\n\n"
            f"Please place {args.model_name}.pth in:\n  {SCRIPT_DIR}\n"
            f"or pass --model-path pointing at the correct file."
        )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=args.model_scale,
    )

    using_gpu = device.type == "cuda"
    tile = args.tile_gpu if using_gpu else args.tile_cpu
    half = bool(using_gpu and args.use_half_on_gpu)

    log(f"Model file: {MODEL_PATH.resolve()}")
    log(f"Device: {device}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU name: {torch.cuda.get_device_name(0)}")
    log(f"Tile size: {tile}")
    log(f"Half precision: {half}")
    log(f"Model scale: {args.model_scale}, output scale: {args.outscale}")

    upsampler = RealESRGANer(
        scale=args.model_scale,
        model_path=str(MODEL_PATH),
        model=model,
        tile=tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=half,
        device=device,
    )

    return upsampler


# ----------------------------------------------------
# IMAGE PROCESSING
# ----------------------------------------------------
def upscale_rgb(upsampler, rgb_np: np.ndarray) -> np.ndarray:
    """
    Run RGB pixels through Real-ESRGAN.

    RealESRGANer.enhance() expects BGR channel order (it is written to be fed
    directly from cv2.imread output). Since PIL gives us RGB, we convert to
    BGR before enhancing and back to RGB afterwards, otherwise the red and
    blue channels come out swapped in the saved PNG.
    """
    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    bgr_upscaled, _ = upsampler.enhance(bgr_np, outscale=args.outscale)
    rgb_upscaled = cv2.cvtColor(bgr_upscaled, cv2.COLOR_BGR2RGB)
    return rgb_upscaled


def improve_image(upsampler, input_path: Path, output_path: Path) -> None:
    # Load image without forcing RGB, so we can detect transparency.
    image = Image.open(input_path)

    # ---------------------------------------------------------
    # PNG WITH TRANSPARENCY (RGBA)
    # ---------------------------------------------------------
    if image.mode == "RGBA":

        # Split RGB and Alpha
        rgb = image.convert("RGB")
        alpha = image.getchannel("A")

        rgb_np = np.array(rgb)
        alpha_np = np.array(alpha)

        # Upscale RGB with the AI model only. Alpha is intentionally NOT
        # passed through the network: see "Transparency note" in the header.
        rgb_upscaled = upscale_rgb(upsampler, rgb_np)

        # Resize alpha with plain interpolation, matched exactly to the
        # AI-upscaled RGB dimensions, so the two channels stay aligned and
        # fully transparent regions (0) and fully opaque regions (255) are
        # preserved rather than distorted.
        target_w = rgb_upscaled.shape[1]
        target_h = rgb_upscaled.shape[0]
        alpha_upscaled = cv2.resize(
            alpha_np,
            (target_w, target_h),
            interpolation=ALPHA_INTERPOLATION,
        )

        result = Image.fromarray(rgb_upscaled).convert("RGBA")
        alpha_image = Image.fromarray(alpha_upscaled.astype(np.uint8))
        result.putalpha(alpha_image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, format="PNG")

    # ---------------------------------------------------------
    # RGB IMAGE (NO TRANSPARENCY)
    # ---------------------------------------------------------
    else:

        rgb_np = np.array(image.convert("RGB"))
        rgb_upscaled = upscale_rgb(upsampler, rgb_np)

        result = Image.fromarray(rgb_upscaled)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, format="PNG")


# ----------------------------------------------------
# MAIN BATCH PROCESS
# ----------------------------------------------------
def main() -> int:
    script_start = time.time()
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 09_Final_improved_real-esrgan.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Input folder: {INPUT_FOLDER.resolve()}")
        log(f"Output folder: {OUTPUT_FOLDER.resolve()}")
        log(f"Device mode: {args.device_mode}")
        log(f"Alpha interpolation: {args.alpha_interpolation}")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        if not INPUT_FOLDER.exists():
            raise FileNotFoundError(
                f"Input folder does not exist: {INPUT_FOLDER.resolve()}\n"
                f"Expected cleaned PNG files in --source-folder (default: 05_Final_Rounded)."
            )

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

        device = select_device()
        t_load_start = time.time()
        upsampler = load_realesrgan_model(device)
        log(f"Model loaded successfully in {time.time() - t_load_start:.1f}s.", "SUCCESS")

        processed = 0
        failed = 0
        t_batch_start = time.time()

        for index, infile in enumerate(files, start=1):
            outfile = OUTPUT_FOLDER / infile.name
            t0 = time.time()
            try:
                improve_image(upsampler, infile, outfile)
                elapsed = time.time() - t0
                processed += 1
                log(f"[{index}/{len(files)}] Processed: {infile.name} ({elapsed:.2f}s)")
            except Exception as exc:
                elapsed = time.time() - t0
                failed += 1
                log(f"[{index}/{len(files)}] Failed: {infile.name}: {exc} ({elapsed:.2f}s)", "ERROR")

        total_elapsed = time.time() - t_batch_start
        script_elapsed = time.time() - script_start

        log(f"Upscaled frames saved: {processed} file(s) to {OUTPUT_FOLDER.resolve()}", "SUCCESS")
        if processed:
            log(f"Total inference time: {total_elapsed:.1f}s "
                f"({(total_elapsed / processed):.2f}s/frame average)")

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
        log(traceback.format_exc(), "ERROR")
        log("============================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())