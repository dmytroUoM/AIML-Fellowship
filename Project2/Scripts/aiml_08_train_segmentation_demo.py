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

Regularisation:
    Two standard regularisation techniques are included and exposed as
    CLI arguments, so their effect on the train/val curves can be
    demonstrated directly:
      - Dropout: nn.Dropout2d is applied inside each encoder/decoder
        block. Controlled by --dropout (probability, default 0.0 = off).
      - Early stopping: training monitors validation loss and stops once
        it fails to improve by at least --min-delta for --patience
        consecutive epochs. The best-performing model (by val_loss) is
        checkpointed separately and used for the final saved model.
        Set --patience 0 to disable early stopping (train the full
        number of --epochs, as before).

Usage:
    py aiml_08_train_segmentation_demo --epochs 15 --lr 1e-3 --batch-size 4 --dropout 0.2 --patience 5 --device cpu --images-dir ..\Images\04_Dataset\train\images --masks-dir ..\Images\04_Dataset\train\masks
    python aiml_08_train_segmentation_demo --epochs 15 --lr 1e-3 --batch-size 4 --dropout 0.2 --patience 5 --device cuda
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
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--dropout", type=float, default=0.0,
                     help="Dropout probability applied inside each encoder/decoder "
                          "block (0.0 disables dropout).")
parser.add_argument("--patience", type=int, default=5,
                     help="Early stopping patience: number of consecutive epochs "
                          "without a val_loss improvement of at least --min-delta "
                          "before training stops. Set to 0 to disable early stopping.")
parser.add_argument("--min-delta", type=float, default=1e-4,
                     help="Minimum decrease in val_loss counted as an improvement "
                          "for early stopping purposes.")
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
    def __init__(self, dropout=0.0):
        super().__init__()
        def block(cin, cout):
            layers = [
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
            ]
            # Dropout2d zeroes whole feature-map channels (rather than individual
            # pixels), which is the standard way to regularise convolutional
            # layers. Placed at the end of each block, after the activations.
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            return nn.Sequential(*layers)
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
    print(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, "
          f"dropout={args.dropout}, patience={args.patience}, min_delta={args.min_delta}")

    model = TinyUNet(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "epoch_seconds": []}

    # ----------------------------------------------------
    # Early stopping state
    # ----------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state_dict = None
    stopped_early = False

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

        # ------------------------------------------------
        # Early stopping: track the best val_loss seen so far and count
        # how many epochs have passed without a meaningful improvement.
        # ------------------------------------------------
        improved = val_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        flag = " *" if improved else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} ({epoch_seconds:.2f}s){flag}")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping: no val_loss improvement > {args.min_delta} for "
                  f"{args.patience} consecutive epochs. Stopping after epoch {epoch} "
                  f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}).")
            stopped_early = True
            break

    total_seconds = time.time() - total_t0
    print(f"Total training time: {total_seconds:.2f}s "
          f"({total_seconds/len(history['epoch']):.2f}s/epoch average)")

    # If early stopping triggered (or even if it didn't), restore the best
    # checkpoint by val_loss so the saved model isn't an overfit final epoch.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model weights from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.4f}).")

    Path("results").mkdir(exist_ok=True)

    # ----------------------------------------------------
    # Persist the full parameter set used for this run, not just a subset,
    # so every history.json is a complete, self-contained record of exactly
    # how the run was invoked (useful for comparing runs later without
    # needing to keep the original command line around separately).
    # ----------------------------------------------------
    history["params"] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "device": args.device,
        "images_dir": args.images_dir,
        "masks_dir": args.masks_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "dropout": args.dropout,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    # Keep the previously-used top-level keys too, for backward compatibility
    # with any tooling/notebooks that already read history["lr"] etc. directly.
    history["device"] = str(device)
    history["lr"] = args.lr
    history["batch_size"] = args.batch_size
    history["dropout"] = args.dropout
    history["patience"] = args.patience
    history["min_delta"] = args.min_delta
    history["seed"] = args.seed
    history["val_fraction"] = args.val_fraction
    history["images_dir"] = args.images_dir
    history["masks_dir"] = args.masks_dir
    history["run_name"] = args.run_name
    history["total_seconds"] = total_seconds
    history["n_params"] = n_params
    history["n_train_images"] = len(train_ds)
    history["n_val_images"] = len(val_ds)
    history["stopped_early"] = stopped_early
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    with open(f"results/{args.run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), f"results/{args.run_name}_model.pt")
    print(f"Saved: results/{args.run_name}_history.json, results/{args.run_name}_model.pt")


if __name__ == "__main__":
    main()
