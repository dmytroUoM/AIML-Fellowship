"""
aiml_08b_train_with_full_metrics.py

Extension of aiml_08_train_segmentation_demo.py that implements a
comprehensive set of quantitative performance metrics for the
TinyUNet segmentation model, addressing Module 8 Milestone Task 1
("Performance Metric Selection and Implementation").

WHY THESE METRICS FOR THIS BUSINESS PROBLEM:
    The task is pixel-level binary segmentation (tank content vs.
    background), so classification metrics are computed per-pixel
    rather than per-image:

    - IoU (Intersection-over-Union): the standard segmentation metric.
      Penalises both over- and under-segmentation in one score. Maps
      directly to "how clean does the final transparent composite
      look" -- the actual business deliverable.
    - Dice coefficient: closely related to IoU but more sensitive to
      small foreground regions, which matters since tank content can
      occupy a small fraction of the frame. Reporting both guards
      against over-interpreting a single number.
    - Precision: (TP / (TP+FP)) -- how much of the predicted
      foreground is actually correct. Low precision = visible stray
      artefacts around tank edges in the output.
    - Recall: (TP / (TP+FN)) -- how much of the true foreground was
      captured. Low recall = real tank content silently erased, which
      is a worse failure mode for a measurement/research pipeline than
      a slightly noisy edge, since it represents genuine data loss.

    All four are computed every epoch, on the held-out validation set,
    and logged into history.json alongside the existing loss curves so
    they can be tracked over time and visualised in the dashboard.

This script is intentionally a copy-and-extend of the original
training script (rather than a diff/patch) so the original remains
untouched and this can be run side-by-side to regenerate a fuller
history.json from the same data.

Usage (identical CLI to the original):
    py aiml_08b_train_with_full_metrics.py --epochs 15 --lr 1e-3 \
        --batch-size 4 --device cpu \
        --images-dir Images\\04_Dataset\\train\\images \
        --masks-dir Images\\04_Dataset\\train\\masks \
        --run-name lr1e-3_bs4_full_metrics
  py aiml_08b_train_with_full_metrics.py --epochs 15 --lr 1e-3 --batch-size 4 --device cpu --images-dir ..\Images\04_Dataset\train\images --masks-dir ..\Images\04_Dataset\train\masks --run-name lr1e-3_bs4_full_metrics --seed 43      
"""

import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------
# Hyperparameters (unchanged from the original script)
# ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--images-dir", type=str, default="images")
parser.add_argument("--masks-dir", type=str, default="masks")
parser.add_argument("--run-name", type=str, default="run")
parser.add_argument("--seed", type=int, default=43)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ----------------------------------------------------
# Dataset (unchanged)
# ----------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_files[idx]))
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t


# ----------------------------------------------------
# TinyUNet (unchanged -- identical architecture, so checkpoints
# trained by either script are interchangeable)
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


# ----------------------------------------------------
# Metric suite -- all computed from the same confusion-matrix counts
# (TP/FP/FN) so they're consistent with each other.
# ----------------------------------------------------
def segmentation_metrics(pred_logits, target, thresh=0.5, eps=1e-6):
    """
    Returns a dict of {iou, dice, precision, recall}, each averaged
    over the batch, computed from a single shared confusion matrix
    per sample -- avoids redundant thresholding/summing per metric.
    """
    pred = (torch.sigmoid(pred_logits) > thresh).float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


def main():
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: --device cuda requested but no GPU available, falling back to cpu.")

    image_files = sorted(Path(args.images_dir).glob("*.png"))
    mask_files = sorted(Path(args.masks_dir).glob("*.png"))
    assert len(image_files) == len(mask_files) and len(image_files) > 0

    n_val = max(1, int(len(image_files) * args.val_fraction))
    train_imgs, val_imgs = image_files[n_val:], image_files[:n_val]
    train_masks, val_masks = mask_files[n_val:], mask_files[:n_val]

    train_ds = SegmentationDataset(train_imgs, train_masks)
    val_ds = SegmentationDataset(val_imgs, val_masks)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Device: {device} | Train images: {len(train_ds)} | Val images: {len(val_ds)}")
    print(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")

    model = TinyUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_iou": [], "val_dice": [], "val_precision": [], "val_recall": [],
        "epoch_seconds": [],
    }

    total_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        agg = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                m = segmentation_metrics(logits, masks)
                for k in agg:
                    agg[k] += m[k] * imgs.size(0)
        val_loss /= len(val_ds)
        for k in agg:
            agg[k] /= len(val_ds)

        epoch_seconds = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(agg["iou"])
        history["val_dice"].append(agg["dice"])
        history["val_precision"].append(agg["precision"])
        history["val_recall"].append(agg["recall"])
        history["epoch_seconds"].append(epoch_seconds)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_iou={agg['iou']:.4f} "
              f"val_dice={agg['dice']:.4f} val_precision={agg['precision']:.4f} "
              f"val_recall={agg['recall']:.4f} ({epoch_seconds:.2f}s)")

    total_seconds = time.time() - total_t0
    print(f"Total training time: {total_seconds:.2f}s "
          f"({total_seconds/args.epochs:.2f}s/epoch average)")

    Path("results").mkdir(exist_ok=True)
    history["device"] = str(device)
    history["lr"] = args.lr
    history["batch_size"] = args.batch_size
    history["total_seconds"] = total_seconds
    history["n_params"] = n_params
    with open(f"results/{args.run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), f"results/{args.run_name}_model.pt")
    print(f"Saved: results/{args.run_name}_history.json, results/{args.run_name}_model.pt")


if __name__ == "__main__":
    main()
