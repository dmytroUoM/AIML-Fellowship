"""
07_improve_scientific_realesrgan.py

AI enhancement / super-resolution for scientific resistance-tomography frames.

Input PNG files:
    Project2/Frames/Cleaned/*.png

Output PNG files:
    Project2/Frames/Improved_Real-ESRGAN/*.png

Expected structure:
    Project2/
    ├── Frames/
    │   ├── Cleaned/
    │   └── Improved_Real-ESRGAN/
    └── Scripts/
        ├── 07_improve_scientific_realesrgan.py
        └── RealESRGAN_x4plus.pth

Run from Scripts folder:
    py 07_improve_scientific_realesrgan.py

CPU install:
    py -m pip install torch torchvision realesrgan basicsr facexlib gfpgan pillow numpy opencv-python

NVIDIA GPU install example:
    py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    py -m pip install realesrgan basicsr facexlib gfpgan pillow numpy opencv-python

Model file required:
    RealESRGAN_x4plus.pth

Place the model file in the same folder as this script:
    Project2/Scripts/RealESRGAN_x4plus.pth

Scientific-image note:
    Real-ESRGAN can improve visual clarity, but it may also hallucinate details.
    For scientific reporting, keep the original Cleaned frames as the source of truth.
    Use Improved_Real-ESRGAN outputs for visualisation/presentation only unless validated.
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from PIL import Image


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FOLDER = BASE_DIR / "Frames" / "Cleaned"
OUTPUT_FOLDER = BASE_DIR / "Frames" / "Improved_Real-ESRGAN"


# -----------------------------------------------------------------------------
# CPU / GPU SETTINGS
# -----------------------------------------------------------------------------
# Use one script for both CPU and NVIDIA GPU.
# Options:
#   "auto"  -> use NVIDIA GPU if available, otherwise CPU
#   "cuda"  -> force NVIDIA GPU; fails if CUDA is not available
#   "cpu"   -> force CPU
DEVICE_MODE = "auto"

# If you want to force GPU, change to:
# DEVICE_MODE = "cuda"

# If you want to force CPU, change to:
# DEVICE_MODE = "cpu"


# -----------------------------------------------------------------------------
# MODEL SETTINGS
# -----------------------------------------------------------------------------
MODEL_NAME = "RealESRGAN_x4plus"
MODEL_SCALE = 4

# Output scale.
# For scientific tomographer frames, x2 is often safer visually than x4 because
# it improves edges without making the image excessively large.
# Set to 4 if you want full 4x output.
OUTSCALE = 2

# Tiling reduces memory usage.
# CPU recommendation: 128 or 256
# GPU recommendation: 0 for no tiling, or 256/512 if GPU memory is limited
TILE_CPU = 128
TILE_GPU = 0
TILE_PAD = 10
PRE_PAD = 0

# Use half precision on CUDA for speed/memory.
# Leave enabled. The script automatically disables it on CPU.
USE_HALF_ON_GPU = True

# Process PNG only, as requested.
EXTENSIONS = {".png"}


# -----------------------------------------------------------------------------
# DEVICE SELECTION
# -----------------------------------------------------------------------------
def select_device() -> torch.device:
    if DEVICE_MODE.lower() == "cpu":
        return torch.device("cpu")

    if DEVICE_MODE.lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE_MODE is set to 'cuda', but CUDA is not available.\n"
                "Either install CUDA-enabled PyTorch or set DEVICE_MODE = 'cpu' / 'auto'."
            )
        return torch.device("cuda")

    if DEVICE_MODE.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise ValueError("DEVICE_MODE must be 'auto', 'cuda', or 'cpu'.")


# -----------------------------------------------------------------------------
# REAL-ESRGAN LOADING
# -----------------------------------------------------------------------------
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

    model_path = SCRIPT_DIR / f"{MODEL_NAME}.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n  {model_path}\n\n"
            f"Please place {MODEL_NAME}.pth in:\n  {SCRIPT_DIR}\n"
        )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=MODEL_SCALE,
    )

    using_gpu = device.type == "cuda"
    tile = TILE_GPU if using_gpu else TILE_CPU
    half = bool(using_gpu and USE_HALF_ON_GPU)

    print("Model file :", model_path)
    print("Device     :", device)
    print("CUDA ready :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name   :", torch.cuda.get_device_name(0))
    print("Tile size  :", tile)
    print("Half mode  :", half)
    print("Outscale   :", OUTSCALE)

    upsampler = RealESRGANer(
        scale=MODEL_SCALE,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=TILE_PAD,
        pre_pad=PRE_PAD,
        half=half,
        device=device,
    )

    return upsampler


# -----------------------------------------------------------------------------
# IMAGE PROCESSING
# -----------------------------------------------------------------------------
def improve_image(upsampler, input_path: Path, output_path: Path) -> None:
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)

    improved_np, _ = upsampler.enhance(image_np, outscale=OUTSCALE)
    improved = Image.fromarray(improved_np)

    # Keep PNG output for lossless scientific-frame visualisation.
    improved.save(output_path, format="PNG")


# -----------------------------------------------------------------------------
# MAIN BATCH PROCESS
# -----------------------------------------------------------------------------
def main() -> None:
    print("Input Folder :", INPUT_FOLDER.resolve())
    print("Output Folder:", OUTPUT_FOLDER.resolve())

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder does not exist:\n  {INPUT_FOLDER}\n\n"
            "Expected cleaned PNG files in:\n  Project2/Frames/Cleaned"
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in INPUT_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    if not files:
        print("No PNG files found in Frames/Cleaned.")
        return

    device = select_device()
    upsampler = load_realesrgan_model(device)

    print(f"Found {len(files)} PNG file(s). Starting AI improvement...")

    failed = []
    for index, infile in enumerate(files, start=1):
        outfile = OUTPUT_FOLDER / infile.name
        print(f"[{index}/{len(files)}] {infile.name}")

        try:
            improve_image(upsampler, infile, outfile)
        except Exception as exc:
            failed.append((infile.name, str(exc)))
            print(f"FAILED: {infile.name} -> {exc}")

    print("\nCompleted.")
    print("Saved to:", OUTPUT_FOLDER.resolve())

    if failed:
        print("\nFiles that failed:")
        for name, error in failed:
            print(f"- {name}: {error}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERROR:")
        traceback.print_exc()
        sys.exit(1)
