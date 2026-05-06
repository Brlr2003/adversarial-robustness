"""
Make a single nice One Pixel attack example for the thesis figure.

Tries a few CIFAR-10 images until it finds one that:
  - Is correctly classified by the standard model on clean
  - Gets flipped by a single-pixel attack
Then renders a 4-panel figure (clean / pixel / adv / preds).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.one_pixel import _apply_perturbation, one_pixel_attack
from src.models.resnet import resnet18_cifar10
from src.utils.data import CIFAR10_CLASSES


def load(path, device):
    m = resnet18_cifar10()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard", default="checkpoints/standard_best.pt")
    ap.add_argument("--robust", default="checkpoints/robust_best.pt")
    ap.add_argument("--output", default="overleaf/figures/onepixel_example.png")
    ap.add_argument("--max-images", type=int, default=30)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device", device)
    std = load(args.standard, device)
    rob = load(args.robust, device)

    ds = datasets.CIFAR10("./data", train=False, download=False, transform=transforms.ToTensor())

    # Search for a successful k=1 example where robust resists; fall back to any.
    candidates = []
    for idx in range(args.max_images):
        img, label = ds[idx]
        img_b = img.unsqueeze(0).to(device)
        with torch.no_grad():
            std_clean_pred = int(std(img_b).argmax(1).item())
            rob_clean_pred = int(rob(img_b).argmax(1).item())
        if std_clean_pred != label or rob_clean_pred != label:
            continue

        adv, info = one_pixel_attack(
            std, img.unsqueeze(0), torch.tensor([label]),
            k=1, pop_size=400, max_iter=75, device=device, seed=idx,
        )
        if not info["success"][0].item():
            continue
        with torch.no_grad():
            adv_pred_std = int(std(adv.to(device)).argmax(1).item())
            adv_pred_rob = int(rob(adv.to(device)).argmax(1).item())
            std_clean_probs = F.softmax(std(img_b), dim=1)[0].cpu().numpy()
            std_adv_probs = F.softmax(std(adv.to(device)), dim=1)[0].cpu().numpy()
            rob_clean_probs = F.softmax(rob(img_b), dim=1)[0].cpu().numpy()
            rob_adv_probs = F.softmax(rob(adv.to(device)), dim=1)[0].cpu().numpy()
        if adv_pred_std == label:
            continue
        bundle = (idx, img, adv[0].cpu(), label, adv_pred_std,
                  std_clean_probs, std_adv_probs, rob_clean_probs, rob_adv_probs)
        if adv_pred_rob == label:
            # Ideal case: robust resists.
            print(f"ideal example: image {idx} (label={CIFAR10_CLASSES[label]}, std-adv={CIFAR10_CLASSES[adv_pred_std]}, rob resists)")
            found = bundle
            break
        candidates.append(bundle)
    else:
        found = candidates[0] if candidates else None
        if found is not None:
            idx0, _, _, label0, adv_pred_std0, _, _, _, _ = found
            print(f"fallback: image {idx0} (label={CIFAR10_CLASSES[label0]}, std-adv={CIFAR10_CLASSES[adv_pred_std0]}, robust also fooled)")

    if found is None:
        print("No successful k=1 example found; aborting.")
        sys.exit(1)

    idx, clean, adv, label, adv_pred, scp, sap, rcp, rap = found

    # Compute the perturbation map (where pixel changed).
    diff = (adv - clean).abs().sum(dim=0)  # (H, W)
    yy, xx = (diff > 1e-6).nonzero(as_tuple=True)
    print(f"  pixel changed at: x={xx.tolist()}, y={yy.tolist()}")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    axes[0].imshow(clean.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Clean\nlabel: {CIFAR10_CLASSES[label]}\nstd: {scp[label]*100:.1f}% / rob: {rcp[label]*100:.1f}%", fontsize=10)
    axes[0].axis("off")

    pert = (adv - clean).clone()
    pert_vis = pert.abs().sum(dim=0).numpy()
    axes[1].imshow(pert_vis, cmap="hot")
    axes[1].set_title("Perturbation\n(1 pixel changed)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(adv.permute(1, 2, 0).numpy().clip(0, 1))
    axes[2].set_title(f"Adversarial\nstd predicts: {CIFAR10_CLASSES[adv_pred]} ({sap[adv_pred]*100:.1f}%)", fontsize=10)
    axes[2].axis("off")

    classes = CIFAR10_CLASSES
    x = np.arange(len(classes))
    width = 0.35
    axes[3].bar(x - width / 2, sap * 100, width, label="Standard")
    axes[3].bar(x + width / 2, rap * 100, width, label="Adv-Trained")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    axes[3].set_ylabel("Confidence (%)")
    axes[3].set_title("Predictions on adversarial", fontsize=10)
    axes[3].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    pdf_path = args.output.replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
    print(f"saved {args.output} and {pdf_path}")


if __name__ == "__main__":
    main()
