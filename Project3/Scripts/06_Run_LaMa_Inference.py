# ============================================
# Script: 06_Run_LaMa_Inference.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction
#           for Mixing Tanks
#
# Purpose:
#   Run three image-processing stages in sequence:
#     1. Remove magenta numbering artefacts using LaMa (big-lama.pt)
#        inpainting, applied to a colour-threshold mask.
#     2. Remove white / near-white backgrounds and create RGBA PNGs.
#     3. Replace jagged opaque blob edges with anti-aliased circles.
#
# Default pipeline folders:
#   Images/02_Frames       -> input
#   Images/04_Cleaned      -> stage 1 output
#   Images/04_Masks        -> stage 1 debug masks
#   Images/05_Transparent  -> stage 2 output
#   Images/06_LaMa_Inference_Output -> stage 3 output
#
# Model:
#   big-lama.pt is the TorchScript export of the LaMa "Big" model
#   (Suvorov et al., 2022, "Resolution-robust Large Mask Inpainting with
#   Fourier Convolutions"). Place it at <ProjectRoot>/Models/big-lama.pt or
#   supply a different location with --model-path.
#
# Requirements:
#   pip install torch opencv-python numpy pillow scipy
#
# Usage:
#   py 06_Run_LaMa_Inference.py
#   py 06_Run_LaMa_Inference.py --no-log
#   py 06_Run_LaMa_Inference.py --stage 1
#   py 06_Run_LaMa_Inference.py --stage 2
#   py 06_Run_LaMa_Inference.py --stage 3
#   py 06_Run_LaMa_Inference.py --stage all
#   py 06_Run_LaMa_Inference.py --model-path D:\Models\big-lama.pt --device cpu
#
# Docker:
#   The script no longer hard-codes any OS-specific paths. Locations are
#   resolved in this order (first match wins):
#     project root : --project-root > $PROJECT_ROOT > auto-detect (a folder
#                    containing "Images": /workspace when in Docker, the
#                    current directory, the script folder, its parent)
#     model        : --model-path > $LAMA_MODEL_PATH > Models/big-lama.pt under
#                    the project root, script folder, current directory,
#                    /workspace, /app, /models
#     log folder   : --log-dir > $LOG_DIR > <project root>/Logs
#   Example:
#     docker run --rm -v "$PWD":/workspace -w /workspace my-image \
#         python Scripts/06_Run_LaMa_Inference.py --project-root /workspace
#
# Notes:
#   Stage 1 uses a learned generative model. Inpainted pixels are synthesised
#   rather than measured; only the masked region is replaced, and all other
#   pixels are preserved exactly as in the source frame.
#   Stage 3 is cosmetic. It creates pixels around fitted circles and should
#   not be used for quantitative analysis or model-training data.
#
# Author: Dmytro Denisiuc
# ============================================

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

VALID_EXTENSIONS = {".png"}
LAMA_PAD_MODULO = 8  # big-lama.pt requires height and width divisible by 8.
MODEL_FILENAME = "big-lama.pt"


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"


def unique_paths(paths: list[Path]) -> list[Path]:
    seen, result = set(), []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_project_root(cli_value: str | None, script_dir: Path) -> Path:
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


def resolve_model_path(
    cli_value: str | None, project_root: Path, script_dir: Path
) -> Path:
    """Find big-lama.pt, searching sensible locations for host and Docker."""
    searched: list[Path] = []

    explicit = cli_value or os.environ.get("LAMA_MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        options = [path] if path.is_absolute() else [
            project_root / path, Path.cwd() / path, script_dir / path
        ]
        # Allow the user to point at a folder that contains big-lama.pt.
        options += [option / MODEL_FILENAME for option in list(options)]
        for option in unique_paths(options):
            searched.append(option)
            if option.is_file():
                return option
    else:
        roots = [
            project_root, script_dir, script_dir.parent, Path.cwd(),
            Path("/workspace"), Path("/app"), Path("/"),
        ]
        for root in unique_paths(roots):
            for folder in ("Models", "models"):
                searched.append(root / folder / MODEL_FILENAME)
        searched.append(Path("/models") / MODEL_FILENAME)
        for option in unique_paths(searched):
            if option.is_file():
                return option

    listing = "\n  ".join(str(p) for p in unique_paths(searched))
    raise FileNotFoundError(
        f"{MODEL_FILENAME} not found. Searched:\n  {listing}\n"
        "Place the model in one of these locations, mount it into the "
        "container (e.g. -v /host/Models:/workspace/Models), or pass "
        "--model-path / set LAMA_MODEL_PATH."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LaMa-based artefact removal, background transparency, "
            "and edge rounding as one pipeline."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "1", "2", "3"),
        default="all",
        help="Run all stages or one individual stage (default: all).",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable file logging.")
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Folder containing Images/ (default: $PROJECT_ROOT, else "
            "auto-detected: /workspace in Docker, otherwise the current directory)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Log folder (default: $LOG_DIR, else <project root>/Logs).",
    )

    # Folder names under <ProjectRoot>/Images
    parser.add_argument("--input-folder", default="02_Frames")
    parser.add_argument("--cleaned-folder", default="04_Cleaned")
    parser.add_argument("--mask-folder", default="04_Masks")
    parser.add_argument("--transparent-folder", default="05_Transparent")
    parser.add_argument("--output-folder", default="06_LaMa_Inference_Output")

    # LaMa model settings
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to big-lama.pt (or the folder containing it). Relative "
            "paths are resolved from the project root. Default: $LAMA_MODEL_PATH, "
            "else search Models/big-lama.pt in the usual locations."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto, i.e. CUDA if available).",
    )

    # Stage 1 settings
    parser.add_argument("--red-threshold", type=int, default=180)
    parser.add_argument("--blue-threshold", type=int, default=180)
    parser.add_argument("--green-threshold", type=int, default=200)
    parser.add_argument(
        "--dilate-px",
        type=int,
        default=2,
        help="Mask dilation in pixels, so the inpainted area covers anti-aliased edges.",
    )

    # Stage 2 settings
    parser.add_argument("--threshold-full", type=int, default=235)
    parser.add_argument("--threshold-start", type=int, default=180)
    parser.add_argument("--erode-size", type=int, default=3)

    # Stage 3 settings
    parser.add_argument("--alpha-cutoff", type=int, default=128)
    parser.add_argument("--supersample", type=int, default=4)

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    byte_values = {
        "red-threshold": args.red_threshold,
        "blue-threshold": args.blue_threshold,
        "green-threshold": args.green_threshold,
        "threshold-full": args.threshold_full,
        "threshold-start": args.threshold_start,
        "alpha-cutoff": args.alpha_cutoff,
    }
    for name, value in byte_values.items():
        if not 0 <= value <= 255:
            raise ValueError(f"--{name} must be between 0 and 255.")

    if args.threshold_start >= args.threshold_full:
        raise ValueError("--threshold-start must be lower than --threshold-full.")
    if args.dilate_px < 0:
        raise ValueError("--dilate-px cannot be negative.")
    if args.erode_size < 3 or args.erode_size % 2 == 0:
        raise ValueError("--erode-size must be an odd integer of 3 or greater.")
    if args.supersample < 1:
        raise ValueError("--supersample must be at least 1.")


def configure_logger(log_file: Path, enable_file_logging: bool) -> logging.Logger:
    logger = logging.getLogger("image_preprocessing_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if enable_file_logging:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            # e.g. read-only or unmounted volume in a container: keep going.
            logger.warning("File logging disabled (cannot write %s): %s", log_file, exc)

    return logger


def log(logger: logging.Logger, message: str, level: str = "INFO") -> None:
    level = level.upper()
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "SUCCESS":
        logger.info("SUCCESS: %s", message)
    else:
        logger.info(message)


def png_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


# ----------------------------------------------------------------------
# LaMa model wrapper (big-lama.pt, TorchScript)
# ----------------------------------------------------------------------
class LamaInpainter:
    """Thin wrapper around the TorchScript big-lama.pt checkpoint.

    The model accepts an RGB image tensor of shape (1, 3, H, W) with values
    in [0, 1] and a binary mask tensor of shape (1, 1, H, W) in which 1 marks
    the region to be filled. It returns an RGB tensor of shape (1, 3, H, W)
    with values in [0, 1]. H and W must be divisible by 8.
    """

    def __init__(self, model_path: Path, device_name: str, logger: logging.Logger):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for LaMa inference. "
                "Install it with: pip install torch"
            ) from exc

        if not model_path.is_file():
            raise FileNotFoundError(
                f"LaMa model not found: {model_path}. "
                "Download big-lama.pt and place it there, or use --model-path."
            )

        self._torch = torch
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        self.device = torch.device(device_name)

        log(logger, f"Loading LaMa model: {model_path}")
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.model.to(self.device)
        log(logger, f"LaMa model ready on device: {self.device}")

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill the masked region of a BGR uint8 image and return a BGR uint8 image."""
        torch = self._torch
        height, width = mask.shape

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        binary_mask = (mask > 0).astype(np.uint8)

        pad_h = (-height) % LAMA_PAD_MODULO
        pad_w = (-width) % LAMA_PAD_MODULO
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
            binary_mask = np.pad(binary_mask, ((0, pad_h), (0, pad_w)), mode="symmetric")

        image_tensor = (
            torch.from_numpy(rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        mask_tensor = (
            torch.from_numpy(binary_mask.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            output = self.model(image_tensor, mask_tensor)

        result = output[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = result[:height, :width]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# Stage 1: remove magenta numbering artefacts with LaMa
# ----------------------------------------------------------------------
def build_magenta_mask(
    img_bgr: np.ndarray,
    red_threshold: int,
    blue_threshold: int,
    green_threshold: int,
) -> np.ndarray:
    blue, green, red = cv2.split(img_bgr)
    selected = (
        (red >= red_threshold)
        & (blue >= blue_threshold)
        & (green <= green_threshold)
    )
    return selected.astype(np.uint8) * 255


def remove_numbering(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    inpainter: LamaInpainter,
    args: argparse.Namespace,
) -> int:
    """Inpaint magenta artefacts and return the number of masked pixels."""
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    mask = build_magenta_mask(
        image,
        args.red_threshold,
        args.blue_threshold,
        args.green_threshold,
    )

    if args.dilate_px > 0:
        size = 2 * args.dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    masked_pixels = int(np.count_nonzero(mask))
    if masked_pixels == 0:
        cleaned = image
    else:
        inpainted = inpainter.inpaint(image, mask)
        # Replace only the masked region so that every other pixel is
        # preserved exactly as in the source frame.
        cleaned = image.copy()
        region = mask > 0
        cleaned[region] = inpainted[region]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Failed to save mask: {mask_path}")
    if not cv2.imwrite(str(output_path), cleaned):
        raise OSError(f"Failed to save cleaned image: {output_path}")

    return masked_pixels


def run_stage_1(
    source_dir: Path,
    cleaned_dir: Path,
    mask_dir: Path,
    inpainter: LamaInpainter,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[int, int]:
    log(logger, "Stage 1 of 3: Removing magenta numbering artefacts with LaMa (big-lama.pt)")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Stage 1 of 3 source folder not found: {source_dir}")

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    files = png_files(source_dir)
    log(logger, f"Stage 1 of 3 found {len(files)} PNG file(s).")

    processed = failed = 0
    for index, input_path in enumerate(files, 1):
        try:
            masked_pixels = remove_numbering(
                input_path,
                cleaned_dir / input_path.name,
                mask_dir / f"{input_path.stem}_mask.png",
                inpainter,
                args,
            )
            processed += 1
            log(
                logger,
                f"Stage 1 of 3 [{index}/{len(files)}] processed: "
                f"{input_path.name} ({masked_pixels} pixel(s) inpainted)",
            )
        except Exception as exc:
            failed += 1
            log(logger, f"Stage 1 of 3 failed for {input_path.name}: {exc}", "ERROR")

    log(logger, f"Stage 1 of 3 complete: {processed} processed, {failed} failed.", "SUCCESS")
    return processed, failed


# ----------------------------------------------------------------------
# Stage 2: make white / near-white background transparent
# ----------------------------------------------------------------------
def make_transparent(img: Image.Image, args: argparse.Namespace) -> Image.Image:
    array = np.asarray(img.convert("RGBA"), dtype=np.float64).copy()
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    whiteness = np.minimum(np.minimum(red, green), blue)

    alpha = np.full_like(whiteness, 255.0)
    alpha[whiteness >= args.threshold_full] = 0.0

    fade_mask = (
        (whiteness >= args.threshold_start)
        & (whiteness < args.threshold_full)
    )
    alpha[fade_mask] = 255.0 * (
        1.0
        - (whiteness[fade_mask] - args.threshold_start)
        / (args.threshold_full - args.threshold_start)
    )

    array[..., 3] = alpha
    output = Image.fromarray(array.astype(np.uint8), "RGBA")
    eroded_alpha = output.getchannel("A").filter(ImageFilter.MinFilter(args.erode_size))
    output.putalpha(eroded_alpha)
    return output


def run_stage_2(
    cleaned_dir: Path,
    transparent_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[int, int]:
    log(logger, "Stage 2 of 3: Creating transparent-background PNG images")
    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Stage 2 of 3 source folder not found: {cleaned_dir}")

    transparent_dir.mkdir(parents=True, exist_ok=True)
    files = png_files(cleaned_dir)
    log(logger, f"Stage 2 of 3 found {len(files)} PNG file(s).")

    processed = failed = 0
    for index, input_path in enumerate(files, 1):
        try:
            with Image.open(input_path) as image:
                result = make_transparent(image, args)
            output_path = transparent_dir / input_path.name
            result.save(output_path, format="PNG")
            processed += 1
            log(logger, f"Stage 2 of 3 [{index}/{len(files)}] processed: {input_path.name}")
        except Exception as exc:
            failed += 1
            log(logger, f"Stage 2 of 3 failed for {input_path.name}: {exc}", "ERROR")

    log(logger, f"Stage 2 of 3 complete: {processed} processed, {failed} failed.", "SUCCESS")
    return processed, failed


# ----------------------------------------------------------------------
# Stage 3: replace jagged opaque blobs with anti-aliased circles
# ----------------------------------------------------------------------
def make_perfect_circles(
    img: Image.Image,
    alpha_cutoff: int,
    supersample: int,
) -> tuple[Image.Image, int]:
    array = np.asarray(img.convert("RGBA"), dtype=np.float64).copy()
    red, green, blue, alpha = [array[..., channel] for channel in range(4)]
    height, width = alpha.shape
    original_mask = alpha > alpha_cutoff

    labelled, blob_count = ndimage.label(original_mask)
    new_mask = np.zeros((height, width), dtype=bool)
    alpha_super = np.zeros(
        (height * supersample, width * supersample), dtype=np.float64
    )
    yy, xx = np.mgrid[0 : height * supersample, 0 : width * supersample]
    y_base, x_base = np.ogrid[0:height, 0:width]

    for label_number in range(1, blob_count + 1):
        ys, xs = np.where(labelled == label_number)
        if xs.size == 0:
            continue

        centre_x = (xs.min() + xs.max()) / 2.0
        centre_y = (ys.min() + ys.max()) / 2.0
        radius = ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4.0

        distance = np.sqrt(
            (x_base - centre_x) ** 2 + (y_base - centre_y) ** 2
        )
        new_mask |= distance <= radius

        centre_x_super = centre_x * supersample
        centre_y_super = centre_y * supersample
        radius_super = radius * supersample
        distance_super = np.sqrt(
            (xx - centre_x_super - supersample / 2.0) ** 2
            + (yy - centre_y_super - supersample / 2.0) ** 2
        )
        alpha_super[distance_super <= radius_super] = 255.0

    alpha_final = alpha_super.reshape(
        height, supersample, width, supersample
    ).mean(axis=(1, 3))

    need_fill = new_mask & ~original_mask
    if need_fill.any() and original_mask.any():
        _, indices = ndimage.distance_transform_edt(
            ~original_mask, return_indices=True
        )
        nearest_y, nearest_x = indices
        red[need_fill] = red[nearest_y[need_fill], nearest_x[need_fill]]
        green[need_fill] = green[nearest_y[need_fill], nearest_x[need_fill]]
        blue[need_fill] = blue[nearest_y[need_fill], nearest_x[need_fill]]

    output_rgb = np.dstack((red, green, blue)).astype(np.uint8)
    output = Image.fromarray(output_rgb, "RGB").convert("RGBA")
    output.putalpha(Image.fromarray(alpha_final.astype(np.uint8), "L"))
    return output, int(blob_count)


def run_stage_3(
    transparent_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    log(logger, "Stage 3 of 3: Rounding opaque blob edges")
    if not transparent_dir.is_dir():
        raise FileNotFoundError(f"Stage 3 of 3 source folder not found: {transparent_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = png_files(transparent_dir)
    log(logger, f"Stage 3 of 3 found {len(files)} PNG file(s).")

    processed = failed = total_blobs = 0
    for index, input_path in enumerate(files, 1):
        try:
            with Image.open(input_path) as image:
                result, blob_count = make_perfect_circles(
                    image, args.alpha_cutoff, args.supersample
                )
            output_path = output_dir / input_path.name
            result.save(output_path, format="PNG")
            processed += 1
            total_blobs += blob_count
            log(
                logger,
                f"Stage 3 of 3 [{index}/{len(files)}] processed: "
                f"{input_path.name} ({blob_count} blob(s))",
            )
        except Exception as exc:
            failed += 1
            log(logger, f"Stage 3 of 3 failed for {input_path.name}: {exc}", "ERROR")

    log(
        logger,
        f"Stage 3 of 3 complete: {processed} processed, {failed} failed, "
        f"{total_blobs} total blob(s) rounded.",
        "SUCCESS",
    )
    return processed, failed, total_blobs


def main() -> int:
    args = parse_arguments()
    start_time = time.perf_counter()

    script_dir = Path(__file__).resolve().parent
    project_root = resolve_project_root(args.project_root, script_dir)
    images_dir = project_root / "Images"
    log_dir = Path(args.log_dir or os.environ.get("LOG_DIR") or project_root / "Logs")
    log_file = log_dir / "06_Run_LaMa_Inference.log"
    logger = configure_logger(log_file, not args.no_log)

    source_dir = images_dir / args.input_folder
    cleaned_dir = images_dir / args.cleaned_folder
    mask_dir = images_dir / args.mask_folder
    transparent_dir = images_dir / args.transparent_folder
    output_dir = images_dir / args.output_folder

    try:
        validate_arguments(args)

        model_path = None
        if args.stage in ("all", "1"):
            model_path = resolve_model_path(args.model_path, project_root, script_dir)

        log(logger, "=" * 60)
        log(logger, "Image preprocessing pipeline started")
        log(logger, f"Selected stage: {args.stage}")
        log(logger, f"Project root: {project_root}")
        log(logger, f"Input folder: {source_dir}")
        log(logger, f"Cleaned folder: {cleaned_dir}")
        log(logger, f"Mask folder: {mask_dir}")
        log(logger, f"Transparent folder: {transparent_dir}")
        log(logger, f"Final output folder: {output_dir}")
        log(logger, f"LaMa model: {model_path if model_path else 'not needed for this stage'}")
        log(logger, f"Running in Docker: {running_in_docker()}")
        log(logger, f"Log file: {log_file if not args.no_log else 'Disabled'}")

        total_failed = 0

        if args.stage in ("all", "1"):
            inpainter = LamaInpainter(model_path, args.device, logger)
            _, failed = run_stage_1(
                source_dir, cleaned_dir, mask_dir, inpainter, args, logger
            )
            total_failed += failed

        if args.stage in ("all", "2"):
            _, failed = run_stage_2(
                cleaned_dir, transparent_dir, args, logger
            )
            total_failed += failed

        if args.stage in ("all", "3"):
            _, failed, _ = run_stage_3(
                transparent_dir, output_dir, args, logger
            )
            total_failed += failed

        elapsed = time.perf_counter() - start_time
        log(logger, "=" * 60)
        log(logger, f"Pipeline completed in {elapsed:.2f} seconds.", "SUCCESS")
        log(logger, f"Total failed file operations: {total_failed}")
        log(logger, "=" * 60)
        return 1 if total_failed else 0

    except Exception as exc:
        log(logger, f"Pipeline failed: {exc}", "ERROR")
        log(logger, "=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
