"""
Pipeline: remove magenta numbering/labels from an image.

Steps:
1. Load image (BGR, OpenCV).
2. Build a color mask that isolates the magenta text (axis numbers like
   12,13,14... and the "P1"/value labels). Magenta pixels have high Blue,
   high Red, low-to-mid Green (R≈255, B≈255, G varies due to anti-aliasing).
3. Dilate the mask slightly so anti-aliased halo pixels around each glyph
   are included (otherwise you get faint pink fringing left behind).
4. Inpaint the masked region using cv2.inpaint (Telea algorithm) so the
   removed text is filled in with plausible surrounding color/texture.
5. Save the cleaned image.

Usage:
    python3 remove_numbering.py input.png output.png
"""

import sys
import cv2
import numpy as np


def build_magenta_mask(img_bgr: np.ndarray,
                        r_thresh: int = 180,
                        b_thresh: int = 180,
                        g_thresh: int = 200) -> np.ndarray:
    """Mask pixels that look magenta/pink (i.e. R and B both high, G lower).

    Using a max(G) threshold rather than a fixed band lets us catch the
    anti-aliased edge pixels that fade toward white (255,255,255) as well
    as the pure magenta core (255,0,255).
    """
    b, g, r = (img_bgr[..., 0].astype(int),
               img_bgr[..., 1].astype(int),
               img_bgr[..., 2].astype(int))

    mask = (r >= r_thresh) & (b >= b_thresh) & (g <= g_thresh) & (g < r) & (g < b)
    return (mask * 255).astype(np.uint8)


def remove_numbering(input_path: str,
                      output_path: str,
                      dilate_px: int = 2,
                      inpaint_radius: int = 3,
                      method: str = "telea") -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    mask = build_magenta_mask(img)

    # Grow the mask a little so faint anti-aliased pixels aren't left behind
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result = cv2.inpaint(img, mask, inpaint_radius, flag)

    cv2.imwrite(output_path, result)

    # Also save the mask for inspection/debugging
    mask_path = output_path.rsplit(".", 1)[0] + "_mask.png"
    cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 remove_numbering.py <input> <output>")
        sys.exit(1)
    remove_numbering(sys.argv[1], sys.argv[2])
    print("Done.")
