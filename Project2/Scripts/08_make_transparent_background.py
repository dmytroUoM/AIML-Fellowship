"""
Batch-remove white backgrounds from images and save with transparency.

Source folder : Frames/Improved_Real-ESRGAN
Output folder : Frames/Transparent_Background (created automatically if missing)

Usage:
    python make_transparent.py
    (run from the folder that contains the "Frames" directory,
     or edit SOURCE_DIR / OUTPUT_DIR below to use absolute paths)
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SOURCE_DIR = Path("../Frames/Improved_Real-ESRGAN")
OUTPUT_DIR = Path("../Frames/Transparent_Background")

THRESHOLD_FULL = 235   # pixels whiter than this -> fully transparent
THRESHOLD_START = 180  # pixels start fading to transparent below this
ERODE_SIZE = 3          # size of the min-filter used to clean edge fringe

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_transparent(img: Image.Image) -> Image.Image:
    """Return a copy of img with white/near-white background made transparent."""
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    whiteness = np.minimum(np.minimum(r, g), b)

    alpha = np.ones_like(whiteness) * 255
    alpha[whiteness >= THRESHOLD_FULL] = 0

    fade_mask = (whiteness >= THRESHOLD_START) & (whiteness < THRESHOLD_FULL)
    alpha[fade_mask] = 255 * (
        1 - (whiteness[fade_mask] - THRESHOLD_START) / (THRESHOLD_FULL - THRESHOLD_START)
    )

    arr[..., 3] = alpha
    out = Image.fromarray(arr.astype(np.uint8), "RGBA")

    # Erode the alpha mask a little to remove stray white-ish fringe pixels
    alpha_img = out.split()[3].filter(ImageFilter.MinFilter(ERODE_SIZE))
    out.putalpha(alpha_img)

    return out


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in {SOURCE_DIR.resolve()}")
        return

    for f in image_files:
        img = Image.open(f)
        result = make_transparent(img)
        out_path = OUTPUT_DIR / (f.stem + ".png")  # always save as PNG to keep alpha
        result.save(out_path)
        print(f"Processed: {f.name} -> {out_path}")

    print(f"\nDone. {len(image_files)} image(s) saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
