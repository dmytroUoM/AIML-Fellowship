import os
from pathlib import Path
import cv2
import numpy as np


def build_magenta_mask(img_bgr: np.ndarray, r_thresh: int = 180, b_thresh: int = 180, g_thresh: int = 200) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    return (((r >= r_thresh) & (b >= b_thresh) & (g <= g_thresh)).astype(np.uint8) * 255)


def remove_numbering(input_path: str, output_path: str, dilate_px: int = 2, inpaint_radius: int = 3, method: str = 'telea') -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {input_path}')

    mask = build_magenta_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Save mask for debugging
    mask_file = MASK_FOLDER / f"{Path(input_path).stem}_mask.png"
    cv2.imwrite(str(mask_file), mask)


    algo = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
    cleaned = cv2.inpaint(img, mask, inpaint_radius, algo)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cleaned)




INPUT_FOLDER = Path("..") / "Images" / "02_Frames"
OUTPUT_FOLDER = Path("..") / "Images" / "03_Cleaned"
MASK_FOLDER = Path("..") / "Images" / "03_Masks"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MASK_FOLDER.mkdir(parents=True, exist_ok=True)

print("Input Folder :", INPUT_FOLDER.resolve())
print("Output Folder:", OUTPUT_FOLDER.resolve())
print("Mask Folder  :", MASK_FOLDER.resolve())



def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

    files = [f for f in INPUT_FOLDER.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    for infile in files:
        outfile = OUTPUT_FOLDER / infile.name
        remove_numbering(str(infile), str(outfile))

        print(f'Processed: {infile.name}')

    print(f'Completed. {len(files)} file(s) saved to {OUTPUT_FOLDER}')
    print(f'Completed. {len(files)} mask file(s) saved to {MASK_FOLDER}')

if __name__ == '__main__':
    main()
