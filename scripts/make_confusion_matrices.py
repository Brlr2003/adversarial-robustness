"""
Confusion matrices for the standard and adversarially trained models on the full
CIFAR-10 test set (10,000 images). Requested by the thesis examiners.

The matrices use a grayscale colormap with the count annotated in every non-empty
cell, so they are fully readable on a black-and-white print (the number carries the
information, not the colour). Saved at 300 DPI.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.resnet import resnet18_cifar10

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(path):
    m = resnet18_cifar10()
    ck = torch.load(path, map_location=dev, weights_only=True)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(dev).eval()


@torch.no_grad()
def predict(model, loader):
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(dev)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def confusion(y, p, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, pr in zip(y, p):
        cm[t, pr] += 1
    return cm


def plot_cm(cm, title, path):
    acc = np.trace(cm) / cm.sum() * 100
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"{title}  (accuracy {acc:.1f}%)", fontsize=12)
    thr = cm.max() * 0.5
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=8, fontweight="bold" if i == j else "normal",
                        color="white" if v > thr else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved {path}  (accuracy {acc:.1f}%)")


def main():
    tf = transforms.ToTensor()
    test = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(test, batch_size=500, num_workers=2)
    os.makedirs("overleaf/figures", exist_ok=True)
    for tag, ck, title in [
        ("standard", "checkpoints/standard_best.pt", "Standard model"),
        ("robust", "checkpoints/robust_best.pt", "Adversarially trained model"),
    ]:
        y, p = predict(load(ck), loader)
        plot_cm(confusion(y, p), title, f"overleaf/figures/confusion_{tag}.png")


if __name__ == "__main__":
    main()
