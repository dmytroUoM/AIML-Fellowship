# ============================================
# Script: aiml_09_run_trained_model_transparent.py
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Load a trained segmentation model (from aiml_09_train_segmentation_demo.py)
#   - Run it on new frames to predict the foreground/background mask
#   - Composite each frame onto a transparent background using the predicted
#     mask (background removal via the trained model, not classical rules)
#   - Save the transparent-background image + predicted mask per frame
#   - Optionally score against known ground-truth masks (e.g. the val set),
#     since that is the only way to get a real accuracy number rather than
#     just a visual impression
#   - Save log to Logs folder beside script
#   - Create output folders if they do not exist
#
# Output layout:
#   Images\06_Inference_Output\images\*.png   (RGBA, transparent background)
#   Images\06_Inference_Output\masks\*.png    (predicted mask, grayscale)
# ============================================

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------
# Argument parsing (mirrors the -NoLog switch in the .ps1 scripts)
# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run a trained segmentation model on new frames to predict "
                "foreground/background masks and produce transparent-background images."
)
parser.add_argument(
    "--no-log",
    action="store_true",
    help="Disable saving the log file (equivalent to -NoLog in the .ps1 scripts)."
)
parser.add_argument(
    "--model-path", type=str, default="results/run_model.pt",
    help="Path to the trained model weights (.pt file) saved by "
         "08_train_segmentation_demo.py (default: results/run_model.pt)."
)
parser.add_argument(
    "--input-folder", type=str, default="Images/02_Frames",
    help="Folder (relative to the project root, i.e. one level above Scripts) containing "
         "the frames to run inference on (default: Images/02_Frames)."
)
parser.add_argument(
    "--output-folder", type=str, default="Images/06_Inference_Output",
    help="Folder (relative to the project root) to write predicted output into "
         "(default: Images/06_Inference_Output)."
)
parser.add_argument(
    "--ground-truth-masks-folder", type=str, default=None,
    help="Optional folder (relative to the project root) of known ground-truth masks, "
         "matched by filename, to score prediction accuracy (IoU) against. Use this with "
         "a validation set where the correct answer is already known. Omit for frames with "
         "no known ground truth (pure inference only)."
)
parser.add_argument(
    "--threshold", type=float, default=0.5,
    help="Probability threshold above which a pixel is classified as foreground (default: 0.5)."
)
parser.add_argument(
    "--device", type=str, default=None, choices=["cpu", "cuda"],
    help="Force inference device. Default: cuda if available, else cpu."
)
args = parser.parse_args()

ENABLE_LOGGING = not args.no_log

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "Logs"
LOG_FILE = LOGS_DIR / "aiml_09_run_trained_model_transparent.log"

PROJECT_ROOT = SCRIPT_DIR / ".."
MODEL_PATH = SCRIPT_DIR / args.model_path
INPUT_FOLDER = PROJECT_ROOT / args.input_folder
OUTPUT_ROOT = PROJECT_ROOT / args.output_folder
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_MASKS_DIR = OUTPUT_ROOT / "masks"
GT_MASKS_FOLDER = (PROJECT_ROOT / args.ground_truth_masks_folder) if args.ground_truth_masks_folder else None

# ----------------------------------------------------
# Logging setup (mirrors Write-Log in the .ps1 script:
# timestamped, leveled, written to console + log file)
# ----------------------------------------------------
logger = logging.getLogger("run_trained_model")
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
# Model definition - must match 08_train_segmentation_demo.py exactly,
# since we are loading weights trained with that architecture.
# ----------------------------------------------------
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
            )
        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)  # logits


def predict_mask(model, img_bgr: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    """Run the model on one image, return a full-resolution binary mask
    (0/255 uint8) matching the original image's dimensions."""
    orig_h, orig_w = img_bgr.shape[:2]

    resized = cv2.resize(img_bgr, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (probs > threshold).astype(np.uint8) * 255
    mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_full


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask > 127
    gt = gt_mask > 127
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def main() -> int:
    try:
        log("============================================")
        log("Script started.")
        log("Script name: 09_run_trained_model.py")
        log("Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks")
        log(f"Script folder: {SCRIPT_DIR}")
        log(f"Model path: {MODEL_PATH.resolve()}")
        log(f"Input folder: {INPUT_FOLDER.resolve()}")
        log(f"Output folder: {OUTPUT_ROOT.resolve()}")
        log(f"Threshold: {args.threshold}")
        if GT_MASKS_FOLDER:
            log(f"Ground-truth masks folder (for scoring): {GT_MASKS_FOLDER.resolve()}")
        else:
            log("No ground-truth masks folder provided - pure inference only, no accuracy scoring.")

        if ENABLE_LOGGING:
            log(f"Logs folder: {LOGS_DIR}")
            log(f"Log file: {LOG_FILE}")
        else:
            log("File logging disabled by --no-log option.", "WARNING")

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH.resolve()}. "
                f"Run 08_train_segmentation_demo.py first, or check --model-path."
            )

        if not INPUT_FOLDER.is_dir():
            raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER.resolve()}")

        if GT_MASKS_FOLDER and not GT_MASKS_FOLDER.is_dir():
            raise FileNotFoundError(f"Ground-truth masks folder not found: {GT_MASKS_FOLDER.resolve()}")

        OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)

        device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        log(f"Inference device: {device_str}")

        model = TinyUNet().to(device)
        model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
        model.eval()
        log(f"Model loaded successfully from {MODEL_PATH.resolve()}", "SUCCESS")

        extensions = {'.png'}
        files = sorted(
            f for f in INPUT_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total_files = len(files)
        log(f"Found {total_files} PNG file(s) to run inference on.")

        if total_files == 0:
            log("No frames found, nothing to do.", "WARNING")
            log("Script completed successfully.", "SUCCESS")
            log("============================================")
            return 0

        processed = 0
        failed = 0
        iou_scores = []
        t_batch_start = time.time()

        for f in files:
            try:
                img = cv2.imread(str(f))
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {f}")

                pred_mask = predict_mask(model, img, device, args.threshold)

                # Save predicted mask
                cv2.imwrite(str(OUTPUT_MASKS_DIR / f.name), pred_mask)

                # Composite to transparent-background RGBA
                b, g, r = cv2.split(img)
                rgba = cv2.merge([b, g, r, pred_mask])
                cv2.imwrite(str(OUTPUT_IMAGES_DIR / f.name), rgba)

                # Optional scoring against known ground truth
                score_note = ""
                if GT_MASKS_FOLDER:
                    gt_path = GT_MASKS_FOLDER / f.name
                    if gt_path.is_file():
                        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                        score = iou(pred_mask, gt_mask)
                        iou_scores.append(score)
                        score_note = f", IoU={score:.4f}"
                    else:
                        log(f"No ground-truth mask found for {f.name}, skipping scoring for this file.", "WARNING")

                processed += 1
                log(f"Processed: {f.name}{score_note}")

            except Exception as file_err:
                failed += 1
                log(f"Failed to process {f.name}: {file_err}", "ERROR")

        total_elapsed = time.time() - t_batch_start

        log(f"Inference output saved: {processed} image(s) to {OUTPUT_IMAGES_DIR.resolve()}", "SUCCESS")
        log(f"Predicted masks saved to {OUTPUT_MASKS_DIR.resolve()}", "SUCCESS")
        log(f"Total inference time: {total_elapsed:.1f}s "
            f"({(total_elapsed/processed):.3f}s/frame average)" if processed else "No frames processed.")

        if iou_scores:
            mean_iou = float(np.mean(iou_scores))
            log(f"Mean IoU vs ground truth over {len(iou_scores)} scored frame(s): {mean_iou:.4f}", "SUCCESS")

        if failed > 0:
            log(f"{failed} file(s) failed to process.", "WARNING")

        log("Script completed successfully.", "SUCCESS")
        log("============================================")
        return 0

    except Exception as e:
        log(f"Script failed: {e}", "ERROR")
        log("============================================")
        return 1


if __name__ == '__main__':
    sys.exit(main())
