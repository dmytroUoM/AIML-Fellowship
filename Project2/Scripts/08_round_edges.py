"""
Turn jagged, staircase-edged blobs into perfect, anti-aliased circles.

For each opaque blob found in an image, this script fits a true circle
(based on the blob's bounding box), fills in any gap between the original
jagged edge and the new perfect circle using the nearest neighbouring
pixel's color, and renders the final circle edge with smooth anti-aliasing.

Source folder : Frames/Transparent_Background
Output folder : Frames/Rounded (created automatically if missing)

Usage:
    python round_edges.py
    (run from the folder that contains the "Frames" directory,
     or edit SOURCE_DIR / OUTPUT_DIR below to use absolute paths)
"""

from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SOURCE_DIR = Path("../Images/04_Transparent")
OUTPUT_DIR = Path("../Images/05_Rounded")

ALPHA_CUTOFF = 128   # alpha value above which a pixel is considered "opaque"
SUPERSAMPLE = 4       # supersampling factor used for smooth (anti-aliased) circle edges

VALID_EXTENSIONS = {".png"}  # transparency requires PNG


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
    return out


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_files:
        print(f"No PNG images found in {SOURCE_DIR.resolve()}")
        return

    for f in image_files:
        img = Image.open(f)
        result = make_perfect_circles(img)
        out_path = OUTPUT_DIR / f.name
        result.save(out_path)
        print(f"Processed: {f.name} -> {out_path}")

    print(f"\nDone. {len(image_files)} image(s) saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
