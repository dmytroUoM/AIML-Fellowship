# ============================================
# Script: 07_Real-ESRGAN_Inference_Output.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - AI enhancement / super-resolution for scientific resistance-tomography
#     frames, using the pretrained Real-ESRGAN (RRDBNet) model
#   - Upscale the RGB content of each frame with Real-ESRGAN
#   - Resize the alpha (transparency) channel separately using standard image
#     interpolation - NOT through the AI model - so transparent backgrounds
#     stay cleanly transparent (see "Transparency note" below)
#   - Save upscaled frames to Images\07_Real-ESRGAN_Inference_Output
#   - Save log to Logs folder beside script, including per-file and total
#     script runtime
#   - Create output folder if it does not exist
#
# Default folders (overridable via --source-folder / --output-folder):
#   Input  : Images\06_LaMa_Inference_Output\*.png
#   Output : Images\07_Real-ESRGAN_Inference_Output\*.png
#
# Expected structure:
#   Project2\
#   |-- Images\
#   |   |-- 06_LaMa_Inference_Output\      (default input)
#   |   `-- 07_Real-ESRGAN_Inference_Output\     (default output)
#   `-- Scripts\
#       |-- 07_Real-ESRGAN_Inference_Output.py
#       `-- RealESRGAN_x4plus.pth  (model file, see "Model file required" below)
#
# Run locally (from the Scripts folder or the project root):
#   py 07_Real-ESRGAN_Inference_Output.py
#
# Run in Docker (mount the project folder at /workspace):
#   docker run --rm --gpus all -v "$PWD":/workspace -w /workspace my-image \
#       python Scripts/07_Real-ESRGAN_Inference_Output.py --project-root /workspace
#   (drop --gpus all for CPU-only)
#
# Path resolution (first match wins) - same rules as 06_Run_LaMa_Inference.py:
#   project root : --project-root > $PROJECT_ROOT > auto-detect (a folder that
#                  contains "Images": /workspace when in Docker, the current
#                  directory, the script folder, its parent)
#   model file   : --model-path > $REALESRGAN_MODEL_PATH > RealESRGAN_x4plus.pth
#                  (or --model-name) in Models/ or the root of: the project
#                  root, script folder, its parent, current directory,
#                  /workspace, /app, plus /models
#   log folder   : --log-dir > $LOG_DIR > <project root>/Logs
#
# CPU install:
#   py -m pip install torch torchvision realesrgan basicsr facexlib gfpgan pillow numpy opencv-python
#
# NVIDIA GPU install example:
#   py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   py -m pip install realesrgan basicsr facexlib gfpgan pillow numpy opencv-python
#
# Model file required:
#   RealESRGAN_x4plus.pth - place it in a Models folder under the project root
#   (or beside this script), mount it into the container, or point elsewhere
#   with --model-path / $REALESRGAN_MODEL_PATH.
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
#   --model-path            explicit path to the .pth file (or a folder that
#                            contains it), overrides the automatic search
#   --project-root / --log-dir   override the auto-detected project root and
#                            log folder (useful in Docker)
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
import os
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
    "--project-root", type=str, default=None,
    help="Folder containing Images/ (default: $PROJECT_ROOT, else auto-detected: /workspace in "
         "Docker, otherwise the current directory)."
)
parser.add_argument(
    "--log-dir", type=str, default=None,
    help="Log folder (default: $LOG_DIR, else <project root>/Logs)."
)
parser.add_argument(
    "--source-folder", type=str, default="06_LaMa_Inference_Output",
    help="Subfolder under Images\\ containing the frames to upscale (default: 06_LaMa_Inference_Output)."
)
parser.add_argument(
    "--output-folder", type=str, default="07_Real-ESRGAN_Inference_Output",
    help="Subfolder under Images\\ to write upscaled frames into (default: 07_Real-ESRGAN_Inference_Output)."
)
parser.add_argument(
    "--device-mode", type=str, default="auto", choices=["auto", "cuda", "cpu"],
    help="'auto' uses the GPU if available, otherwise CPU. 'cuda' forces GPU and fails if "
         "unavailable. 'cpu' forces CPU (default: auto)."
)
parser.add_argument(
    "--model-name", type=str, default="RealESRGAN_x4plus",
    help="Model weights file name, without the .pth extension "
         "(default: RealESRGAN_x4plus). Searched for in Models/ and the usual locations."
)
parser.add_argument(
    "--model-path", type=str, default=None,
    help="Explicit path to the .pth model file (or a folder containing it). "
         "Default: $REALESRGAN_MODEL_PATH, else search for <model-name>.pth in the usual locations."
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
SCRIPT_DIR = Path(__file__).resolve().parent


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"


def unique_paths(paths):
    seen, result = set(), []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_base_dir(cli_value, script_dir: Path) -> Path:
    """Locate the project root (the folder that holds Images/)."""
    explicit = cli_value or os.environ.get("PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [Path.cwd(), script_dir, script_dir.parent]
    if running_in_docker():
        candidates.insert(0, Path("/workspace"))
    for candidate in unique_paths(candidates):
        if (candidate / "Images").is_dir():
            return candidate
    return Path.cwd()


def resolve_model_path(cli_value, model_name: str, base_dir: Path, script_dir: Path) -> Path:
    """Find the Real-ESRGAN .pth file, searching locations that work locally and in Docker."""
    searched = []
    file_name = Path(model_name).name
    if not file_name.lower().endswith(".pth"):
        file_name += ".pth"

    explicit = cli_value or os.environ.get("REALESRGAN_MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        options = [path] if path.is_absolute() else [base_dir / path, Path.cwd() / path, script_dir / path]
        options += [option / file_name for option in list(options)]  # folder given instead of file
    else:
        options = []
        roots = [base_dir, script_dir, script_dir.parent, Path.cwd(), Path("/workspace"), Path("/app")]
        for root in unique_paths(roots):
            for folder in ("Models", "models", ""):
                options.append(root / folder / file_name)
        options.append(Path("/models") / file_name)

    for option in unique_paths(options):
        searched.append(option)
        if option.is_file():
            return option

    listing = "\n  ".join(str(p) for p in searched)
    raise FileNotFoundError(
        f"Model file {file_name} not found. Searched:\n  {listing}\n"
        "Place it in one of these locations, mount it into the container "
        "(e.g. -v /host/Models:/workspace/Models), or pass --model-path / "
        "set REALESRGAN_MODEL_PATH."
    )


BASE_DIR = resolve_base_dir(args.project_root, SCRIPT_DIR)
LOGS_DIR = Path(args.log_dir or os.environ.get("LOG_DIR") or BASE_DIR / "Logs")
LOG_FILE = LOGS_DIR / "07_Real-ESRGAN_Inference_Output.log"

INPUT_FOLDER = BASE_DIR / "Images" / args.source_folder
OUTPUT_FOLDER = BASE_DIR / "Images" / args.output_folder
MODEL_PATH = None  # resolved lazily in load_realesrgan_model()

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
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # e.g. read-only or unmounted volume in a container: keep going.
        logger.warning("File logging disabled (cannot write %s): %s", LOG_FILE, exc)
        ENABLE_LOGGING = False


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
def _patch_torchvision_for_basicsr() -> None:
    """basicsr imports torchvision.transforms.functional_tensor, which newer
    torchvision releases removed. Provide a tiny alias so fresh Docker images
    (which install the latest torchvision) do not fail on import."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        try:
            import types
            from torchvision.transforms import functional as tv_functional
            shim = types.ModuleType("torchvision.transforms.functional_tensor")
            shim.rgb_to_grayscale = tv_functional.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = shim
        except Exception:
            pass


def load_realesrgan_model(device: torch.device):
    global MODEL_PATH
    _patch_torchvision_for_basicsr()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise ImportError(
            "Missing Real-ESRGAN dependencies. Install with:\n\n"
            "CPU:\n"
            "  pip install torch torchvision realesrgan basicsr facexlib gfpgan pillow numpy opencv-python-headless\n\n"
            "NVIDIA GPU example:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install realesrgan basicsr facexlib gfpgan pillow numpy opencv-python-headless\n"
            "(On Windows use 'py -m pip' instead of 'pip'.)\n"
        ) from exc

    MODEL_PATH = resolve_model_path(args.model_path, args.model_name, BASE_DIR, SCRIPT_DIR)

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
        log("Script name: 07_Real-ESRGAN_Inference_Output.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Project root: {BASE_DIR}")
        log(f"Running in Docker: {running_in_docker()}")
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
                f"Expected PNG files in Images/{args.source_folder} (change with --source-folder)."
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