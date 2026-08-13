"""
aiml_10_explain_gradcam.py

Explainable AI (XAI) script for the trained TinyUNet segmentation model,
addressing requirement [K20]: "Apply Explainable AI (XAI) techniques
(e.g., SHAP, LIME) to interpret model decisions and identify features
contributing most to predictions, especially those that might be linked
to bias."

WHY GRAD-CAM RATHER THAN SHAP/LIME FOR THIS MODEL:
    SHAP's DeepExplainer / GradientExplainer and LIME's image explainer
    are both designed and validated primarily for image *classification*
    models (a single scalar output per class). TinyUNet is a *dense
    prediction* (segmentation) model: it outputs a full-resolution logit
    map, not a class score. Applying SHAP/LIME directly would require
    reducing the output to a single scalar per explanation (e.g. mean
    logit over the mask), which is common practice but produces a
    coarse, indirect explanation and is also the standard justification
    in the interpretability literature for preferring Grad-CAM-style,
    gradient-based saliency methods on segmentation/detection heads.
    Grad-CAM:
      - Works natively on any convolutional layer's activations, so it
        integrates directly with TinyUNet's actual architecture and its
        skip connections.
      - Directly answers the required question ("which features / which
        parts of the input contribute most to the prediction, and does
        that suggest bias?") for pixel-dense outputs.
      - Is far cheaper to compute than SHAP's Shapley-value sampling
        (a single backward pass vs. thousands of perturbed forward
        passes), which matters given this pipeline explicitly targets
        laptop CPU runtimes.
    This choice, and the trade-off against SHAP/LIME, should be stated
    explicitly in the report as the justification for the XAI method
    selected.

WHAT THIS SCRIPT DOES:
    1. Loads a trained TinyUNet checkpoint (produced by
       aiml_08_train_segmentation_demo.py, e.g. results/<run_name>_model.pt).
    2. Re-creates the exact TinyUNet architecture (copied verbatim from
       the training script so state_dict keys match).
    3. Runs one or more validation images through the model using the
       same preprocessing as aiml_09_run_trained_model_transparent.py
       (resize to 224x224, BGR->RGB, /255, CHW tensor).
    4. Registers a forward hook + backward hook on a chosen convolutional
       layer (default: enc3, the deepest encoder block -- the layer most
       Grad-CAM implementations target because it holds the most
       semantically-aggregated features before the network starts
       upsampling back out).
    5. Backpropagates from the mean predicted logit inside the predicted
       foreground mask (a standard adaptation of Grad-CAM for dense
       prediction / segmentation targets, since there is no single
       class score to differentiate against as in classification).
    6. Computes the Grad-CAM heatmap: ReLU(sum_k( alpha_k * A_k )),
       where alpha_k is the global-average-pooled gradient for feature
       map channel k, and A_k is that channel's activation map.
    7. Upsamples the heatmap to input resolution and overlays it on the
       original image, saving:
         - <name>_original.png
         - <name>_predicted_mask.png
         - <name>_gradcam_heatmap.png
         - <name>_gradcam_overlay.png
       plus a JSON summary of pixel-level statistics for the report.

USAGE:
    python aiml_10_explain_gradcam.py \
        --model-path results/lr1e-3_bs4_GPU_model.pt \
        --images-dir Images\\04_Dataset\\val\\images \
        --num-samples 4 \
        --output-dir Images\\06_XAI_GradCAM

    # Explain a single named image instead of sampling:
    python aiml_10_explain_gradcam.py \
        --model-path results/run_model.pt \
        --single-image Images\\04_Dataset\\val\\images\\frame_0032.png \
        --output-dir Images\\06_XAI_GradCAM

INTERPRETING THE OUTPUT (for the report):
    - A Grad-CAM overlay that concentrates *inside the mixing-tank
      cylinder region* (the true foreground) is evidence the model is
      making its decision based on the intended object, not spurious
      background cues.
    - A Grad-CAM overlay that lights up on the tank rig, frame borders,
      lighting glare, or the fixed on-screen overlay/label region
      instead of the tank contents itself would be evidence of a
      potential bias: the model may be relying on incidental,
      position-fixed visual cues (e.g. always-present rig hardware or a
      timestamp overlay) rather than genuine tank-content features. This
      is the kind of finding [K20] and [B5] ask you to document, since
      such a shortcut would not generalise to camera setups where those
      fixed cues are absent or differently placed.
    - Compare Grad-CAM maps across several validation frames: if the
      hot region consistently sits in the *same fixed image coordinates*
      regardless of where the tank contents actually are in each frame,
      that is a strong, concrete piece of evidence for position-based
      shortcut learning / bias, worth flagging explicitly.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------
# TinyUNet -- copied verbatim from aiml_08_train_segmentation_demo.py
# so that torch.load(...).load_state_dict(...) matches exactly.
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
# Grad-CAM
# ----------------------------------------------------
class GradCAM:
    """
    Generic Grad-CAM wrapper: hooks a target conv layer, captures its
    forward activations and backward gradients, and computes the
    class-activation-map style heatmap from them.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        input_tensor: (1, 3, H, W), requires no external grad setup --
        this method handles zero_grad/backward internally.
        Returns a (H, W) float32 heatmap normalised to [0, 1], at the
        model's input resolution.
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # (1, 1, H, W)

        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()

        # Segmentation adaptation of Grad-CAM: differentiate the mean
        # logit over the *predicted foreground region* (falls back to
        # the mean logit over the whole map if nothing is predicted
        # foreground, e.g. very early in training / a blank prediction).
        if pred_mask.sum() > 0:
            target_scalar = (logits * pred_mask).sum() / pred_mask.sum()
        else:
            target_scalar = logits.mean()

        target_scalar.backward()

        # alpha_k = global-average-pooled gradient for channel k
        gradients = self.gradients[0]          # (C, h, w)
        activations = self.activations[0]      # (C, h, w)
        alpha = gradients.mean(dim=(1, 2))     # (C,)

        cam = torch.relu((alpha[:, None, None] * activations).sum(dim=0))
        cam = cam.cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Upsample from feature-map resolution to input resolution
        h, w = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam, probs.detach().cpu().numpy()[0, 0], pred_mask.detach().cpu().numpy()[0, 0]


# ----------------------------------------------------
# Preprocessing -- matches aiml_09_run_trained_model_transparent.py
# ----------------------------------------------------
def load_and_preprocess(image_path: Path, size: int, device: str):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    resized = cv2.resize(img_bgr, (size, size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor.requires_grad_(False)  # gradients flow from conv weights, not input

    display_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # for overlay, uint8
    return tensor, display_rgb, (orig_h, orig_w)


def overlay_heatmap_on_image(rgb_uint8: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb_uint8.astype(np.float32) + 0.45 * heatmap_color.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def explain_one_image(model, target_layer_name, image_path, size, device, output_dir):
    layer_map = {
        "enc1": model.enc1, "enc2": model.enc2, "enc3": model.enc3,
        "dec2": model.dec2, "dec1": model.dec1,
    }
    target_layer = layer_map[target_layer_name]

    tensor, display_rgb, (orig_h, orig_w) = load_and_preprocess(image_path, size, device)

    cam_engine = GradCAM(model, target_layer)
    cam, probs, pred_mask = cam_engine.generate(tensor)

    overlay = overlay_heatmap_on_image(display_rgb, cam)
    heatmap_only = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_only = cv2.cvtColor(heatmap_only, cv2.COLOR_BGR2RGB)
    mask_vis = (pred_mask * 255).astype(np.uint8)

    stem = image_path.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"{stem}_original.png"), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_predicted_mask.png"), mask_vis)
    cv2.imwrite(str(out_dir / f"{stem}_gradcam_heatmap.png"), cv2.cvtColor(heatmap_only, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_gradcam_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Simple, report-friendly quantitative summary: where is the CAM's
    # "hot" energy concentrated, spatially, as a coarse bias check
    # (e.g. is it centred, or pinned to one corner/edge across images?).
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total_weight = cam.sum() + 1e-8
    centroid_y = float((ys * cam).sum() / total_weight)
    centroid_x = float((xs * cam).sum() / total_weight)

    summary = {
        "image": str(image_path),
        "target_layer": target_layer_name,
        "predicted_foreground_fraction": float(pred_mask.mean()),
        "mean_confidence_in_predicted_region": float(
            (probs * pred_mask).sum() / max(pred_mask.sum(), 1e-8)
        ),
        "gradcam_centroid_xy_normalised": [centroid_x / w, centroid_y / h],
    }
    with open(out_dir / f"{stem}_gradcam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True,
                         help="Path to a trained TinyUNet state_dict, "
                              "e.g. results/<run_name>_model.pt")
    parser.add_argument("--images-dir", type=str, default=None,
                         help="Directory of images to sample from (e.g. "
                              "Images/04_Dataset/val/images)")
    parser.add_argument("--single-image", type=str, default=None,
                         help="Explain exactly one image instead of sampling")
    parser.add_argument("--num-samples", type=int, default=4,
                         help="How many images to sample from --images-dir")
    parser.add_argument("--target-layer", type=str, default="enc3",
                         choices=["enc1", "enc2", "enc3", "dec2", "dec1"],
                         help="Which TinyUNet block to explain (default: "
                              "enc3, the deepest / most semantic layer)")
    parser.add_argument("--size", type=int, default=224,
                         help="Model input resolution (must match training)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="Images/06_XAI_GradCAM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Grad-CAM needs gradients w.r.t. activations, so keep autograd on
    # even though we're not training; only disable grad on the input.
    for p in model.parameters():
        p.requires_grad_(True)

    if args.single_image:
        targets = [Path(args.single_image)]
    else:
        if not args.images_dir:
            raise ValueError("Provide either --single-image or --images-dir")
        all_images = sorted(Path(args.images_dir).glob("*.png")) + \
                     sorted(Path(args.images_dir).glob("*.jpg"))
        if not all_images:
            raise FileNotFoundError(f"No .png/.jpg images found in {args.images_dir}")
        targets = random.sample(all_images, min(args.num_samples, len(all_images)))

    all_summaries = []
    for img_path in targets:
        print(f"Explaining: {img_path}")
        summary = explain_one_image(
            model, args.target_layer, img_path, args.size, device, args.output_dir
        )
        all_summaries.append(summary)
        print(f"  predicted_foreground_fraction = {summary['predicted_foreground_fraction']:.3f}")
        print(f"  gradcam_centroid_xy_normalised = {summary['gradcam_centroid_xy_normalised']}")

    # Aggregate centroid check: if every image's Grad-CAM centroid
    # clusters in nearly the same spot, that's a red flag for
    # position-based shortcut learning (documented for [K20]/[B5]).
    if len(all_summaries) > 1:
        centroids = np.array([s["gradcam_centroid_xy_normalised"] for s in all_summaries])
        spread = centroids.std(axis=0)
        print("\n--- Cross-image bias check ---")
        print(f"Grad-CAM centroid std-dev across {len(all_summaries)} images "
              f"(x, y, normalised 0-1): {spread.tolist()}")
        print("Low spread (<< 0.1) suggests the model's attention is pinned "
              "to a fixed image location regardless of content -- worth "
              "investigating as a potential bias / shortcut. Higher spread "
              "suggests attention tracks actual tank-content location.")

    out_dir = Path(args.output_dir)
    with open(out_dir / "gradcam_batch_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved {len(all_summaries)} Grad-CAM explanation(s) to {out_dir}")


if __name__ == "__main__":
    main()
