"""
aiml_08_train_segmentation_demo.py

A minimal, REAL, working training loop for a small U-Net-style segmentation
model, built to demonstrate the training process end to end: data loading,
model definition, the training loop, loss curves, hyperparameters, and
CPU vs GPU timing.

This is intentionally small (small dataset, small model, few epochs) so it
runs in seconds-to-minutes on a laptop CPU for demonstration purposes.
A production model for this task would use a much larger, more diverse
dataset (thousands of real images, not augmented synthetic ones) and a
larger model, trained for longer.

Usage:
    py aiml_08_train_segmentation_demo --epochs 15 --lr 1e-3 --batch-size 4 --device cpu --images-dir ..\Images\04_Dataset\train\images --masks-dir ..\Images\04_Dataset\train\masks
    python aiml_08_train_segmentation_demo --epochs 15 --lr 1e-3 --batch-size 4 --device cuda
"""

import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------
# Hyperparameters (all exposed as CLI args on purpose,
# so their effect can be shown directly)
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
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ----------------------------------------------------
# Dataset
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

        img_t = torch.from_numpy(img).permute(2, 0, 1)      # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)
        return img_t, mask_t


# ----------------------------------------------------
# A small U-Net (deliberately tiny: this is a teaching demo,
# not a production architecture)
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


def iou_score(pred_logits, target, thresh=0.5):
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


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

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "epoch_seconds": []}

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
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                val_iou += iou_score(logits, masks) * imgs.size(0)
        val_loss /= len(val_ds)
        val_iou /= len(val_ds)

        epoch_seconds = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["epoch_seconds"].append(epoch_seconds)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} ({epoch_seconds:.2f}s)")

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
