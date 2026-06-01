"""
Render a sample-images figure for the thesis (Chapter 3, Datasets).

Shows one example image per CIFAR-10 class in a 2x5 grid, with the class
name above each image. Saved as PNG and PDF under overleaf/figures/.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import CIFAR10_CLASSES


def main():
    ds = datasets.CIFAR10(
        "./data", train=False, download=False, transform=transforms.ToTensor()
    )

    # First image found for each of the 10 classes.
    per_class = {}
    for img, label in ds:
        if label not in per_class:
            per_class[label] = img
        if len(per_class) == 10:
            break

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for cls in range(10):
        ax = axes[cls // 5][cls % 5]
        ax.imshow(per_class[cls].permute(1, 2, 0).numpy())
        ax.set_title(CIFAR10_CLASSES[cls], fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    out = "overleaf/figures/cifar10_samples.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved {out} and {out.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
