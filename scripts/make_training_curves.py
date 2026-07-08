"""
Render the training-dynamics figure for the thesis (Chapter 5, Section 5.5.1)
from the real MLflow logs under mlruns/.

Left panel: training loss per epoch (standard vs adversarial).
Right panel: test accuracy per epoch (standard clean, adv clean, adv PGD robust).

Reads MLflow metric files directly (no mlflow dependency). Each metric file
has one "timestamp value step" line per epoch.
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_metric(run_dir: str, name: str):
    """Return (steps, values) sorted by step for one MLflow metric."""
    path = os.path.join(run_dir, "metrics", name)
    if not os.path.exists(path):
        return [], []
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rows.append((int(parts[2]), float(parts[1])))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def run_name(run_dir: str) -> str:
    tag = os.path.join(run_dir, "tags", "mlflow.runName")
    return open(tag).read().strip() if os.path.exists(tag) else ""


def find_runs():
    """Pick the 50-epoch standard run and the adversarial run."""
    std, adv = None, None
    for run_dir in glob.glob(os.path.join(ROOT, "mlruns", "*", "*")):
        if not os.path.isdir(os.path.join(run_dir, "metrics")):
            continue
        name = run_name(run_dir)
        n_epochs = len(read_metric(run_dir, "train_loss")[0])
        if name == "standard_training" and n_epochs >= 50:
            std = run_dir
        elif name == "adversarial_training" and n_epochs >= 50:
            adv = run_dir
    return std, adv


def main():
    std, adv = find_runs()
    if std is None or adv is None:
        sys.exit(f"could not locate both runs (standard={std}, adversarial={adv})")

    s_ep, s_loss = read_metric(std, "train_loss")
    _, s_testacc = read_metric(std, "test_accuracy")
    a_ep, a_loss = read_metric(adv, "train_loss")
    _, a_clean = read_metric(adv, "test_clean_accuracy")
    _, a_pgd = read_metric(adv, "test_pgd_accuracy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Black lines distinguished by line style and marker shape, so the figure
    # reads on a black-and-white printout (no reliance on colour). Markers are
    # thinned with markevery so the 50-epoch curves stay legible.
    me = 6

    # Left: training loss
    ax1.plot(s_ep, s_loss, color="black", lw=1.8, ls="-", marker="o",
             markevery=me, ms=5, label="Standard")
    ax1.plot(a_ep, a_loss, color="black", lw=1.8, ls="--", marker="s",
             markevery=me, ms=5, markerfacecolor="white", label="Adversarial")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: test accuracy
    ax2.plot(s_ep, s_testacc, color="black", lw=1.8, ls="-", marker="o",
             markevery=me, ms=5, label="Standard (clean)")
    ax2.plot(a_ep, a_clean, color="black", lw=1.8, ls="--", marker="s",
             markevery=me, ms=5, markerfacecolor="white", label="Adv-Trained (clean)")
    ax2.plot(a_ep, a_pgd, color="black", lw=1.8, ls=":", marker="^",
             markevery=me, ms=6, label="Adv-Trained (PGD robust)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_title("Test accuracy")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out = os.path.join(ROOT, "overleaf/figures/training_curves.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", out, "and the .pdf")
    # report key values for caption/text consistency
    print(f"std loss {s_loss[0]:.2f} -> {s_loss[-1]:.4f}; std test acc end {s_testacc[-1]:.2f} (best {max(s_testacc):.2f})")
    print(f"adv loss {a_loss[0]:.2f} -> {a_loss[-1]:.2f}; adv clean end {a_clean[-1]:.2f}; adv pgd peak {max(a_pgd):.2f}")


if __name__ == "__main__":
    main()
