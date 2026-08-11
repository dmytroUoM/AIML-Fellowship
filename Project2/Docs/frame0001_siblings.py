"""
frame0001_siblings.py

Walks through every subfolder of Images/ (including nested ones like
04_Dataset/train/images, 04_Dataset/train/masks, 05_Synthetic_Backgrounds/masks,
06_Inference_Output/masks, etc.), keeps ONLY the LEFT slice of each matching
image, and saves the result into Docs/image-pipeline.

Naming convention: <path-under-Images-with-dashes>-<file-name>.png
e.g. Images/04_Dataset/train/masks/frame_0001.png
     -> 04_Dataset-train-masks-frame_0001.png
(This avoids images/ and masks/ -- or train/ vs val/ -- overwriting each
other once everything is flattened into one output folder.)

By default only frame_0001's siblings are processed (across every folder,
including masks) -- pass --stem "" to process all frames instead.

Cropping mode:
    --crop-px N        Keep a FIXED N pixels from the left (default: 230).
    --crop-percent P   Keep P% of the width from the left instead. Scales
                        with resolution -- use this if some folders have
                        different image widths than others (e.g.
                        06_Final_Upscaled is 2x the width after upscaling),
                        so the crop represents the same relative portion of
                        the frame everywhere. Overrides --crop-px.

Expected layout (relative to this script, which lives in Project2\\Docs):

    Project2\\
        Docs\\
            crop_pipeline.py   <-- run from here
            image-pipeline\\    <-- output goes here (auto-created)
        Images\\
            01_Origin\\
                frame_0001.png
                ...
            03_Masks\\
                frame_0001.png
                ...
            04_Dataset\\
                train\\images\\...
                train\\masks\\...
            ...

Usage (from Project2\\Docs):
    python crop_pipeline.py
    python crop_pipeline.py --crop-percent 25
    python crop_pipeline.py --stem ""          # process every frame, not just frame_0001

Optional arguments:
    --images-dir     Path to the Images folder (default: ..\\Images)
    --output-dir     Path to save output    (default: .\\image-pipeline)
    --crop-px        Fixed pixels to keep from the left (default: 230)
    --crop-percent   Percent of width to keep from the left (overrides --crop-px)
    --stem           Only process files with this exact filename stem (default: frame_0001)
    --ext            Comma-separated list of extensions to process (default: png,jpg,jpeg)
"""

import argparse
from pathlib import Path
from PIL import Image


def crop_left(img: Image.Image, crop_px: int = None, crop_percent: float = None) -> Image.Image:
    """
    Return a copy of img keeping ONLY the leftmost slice.

    Exactly one of crop_px / crop_percent should be provided.
    - crop_px: keep a fixed number of pixels (same absolute width for every image).
    - crop_percent: keep a proportion of the width (0-100), which scales
      automatically with resolution -- useful when some folders (e.g. an
      upscaled output) have different pixel dimensions than others but you
      want the crop to represent the same relative portion of the frame.
    """
    width, height = img.size

    if crop_percent is not None:
        px = round(width * (crop_percent / 100))
    else:
        px = crop_px

    if px is None:
        raise ValueError("Must provide either crop_px or crop_percent.")
    if px > width:
        raise ValueError(f"crop width ({px}) is greater than image width ({width}).")
    if px <= 0:
        raise ValueError(f"computed crop width ({px}) must be > 0.")

    return img.crop((0, 0, px, height))


def iter_image_files(images_dir: Path, extensions: set, stem_filter: str):
    """
    Yield (subfolder_name, file_path) pairs.

    subfolder_name is the name of the *direct* child folder of images_dir
    that the file lives under (e.g. '01_Origin'), used for the output
    naming convention <folder-name>-<file-name>.png even when --recursive
    is used and the file is nested deeper.

    stem_filter, if set, restricts results to files whose filename stem
    matches exactly (e.g. 'frame_0001'), so only that frame's siblings
    across all folders (including any Masks folders) are processed.
    """
    for top_level in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        # Always recurse into nested subfolders (e.g. 04_Dataset/train/images,
        # 05_Synthetic_Backgrounds/masks, 06_Inference_Output/masks) so that
        # masks and nested image sets are included, not just top-level files.
        files = [f for f in top_level.rglob("*") if f.is_file()]

        for f in sorted(files):
            if f.suffix.lower().lstrip(".") not in extensions:
                continue
            if stem_filter and stem_filter.lower() not in f.stem.lower():
                continue
            # Build a name-prefix from every folder between images_dir and the
            # file, e.g. Images/04_Dataset/train/masks/frame_0001.png ->
            # "04_Dataset-train-masks". This keeps images/ and masks/ (and
            # train/ vs val/) from colliding when flattened into one folder.
            rel_parts = f.relative_to(images_dir).parent.parts
            prefix = "-".join(rel_parts)
            yield prefix, f


def main():
    parser = argparse.ArgumentParser(description="Crop 230px from the left of images and rename.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("..") / "Images",
        help="Path to the Images folder (default: ..\\Images, i.e. Project2\\Images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frame0001_siblings"),
        help="Where to save cropped images (default: .\\frame0001_siblings)",
    )
    parser.add_argument(
        "--crop-px",
        type=int,
        default=230,
        help="Fixed number of pixels to keep from the left edge (default: 230). "
             "Ignored if --crop-percent is given.",
    )
    parser.add_argument(
        "--crop-percent",
        type=float,
        default=None,
        help="Keep this percentage (0-100) of the width from the left edge instead "
             "of a fixed pixel count. Scales with resolution -- recommended when "
             "folders have different image widths (e.g. 06_Final_Upscaled is 2x "
             "the width of other folders after upscaling). Overrides --crop-px.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="png,jpg,jpeg",
        help="Comma-separated list of file extensions to process (default: png,jpg,jpeg)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="frame_0001",
        help="Only process files whose filename (without extension) CONTAINS this "
             "text, e.g. 'frame_0001' matches both 'frame_0001.png' and "
             "'frame_0001_mask.png' -- applies across ALL folders, including "
             "Masks/Masks_lama folders, so you get every sibling of that one frame. "
             "Pass an empty string ('') to process all files instead.",
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    extensions = {e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}

    if not images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for folder_prefix, file_path in iter_image_files(images_dir, extensions, args.stem):
        out_name = f"{folder_prefix}-{file_path.stem}.png"
        out_path = output_dir / out_name

        try:
            with Image.open(file_path) as img:
                img.load()
                if args.crop_percent is not None:
                    cropped = crop_left(img, crop_percent=args.crop_percent)
                else:
                    cropped = crop_left(img, crop_px=args.crop_px)
                cropped.save(out_path, "PNG")
            processed += 1
            print(f"OK   {file_path.relative_to(images_dir)}  ->  {out_path.name}")
        except Exception as e:
            skipped += 1
            print(f"SKIP {file_path.relative_to(images_dir)}  ({e})")

    print(f"\nDone. Processed: {processed}  Skipped: {skipped}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
