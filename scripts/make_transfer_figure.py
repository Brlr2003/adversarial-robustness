"""
Render the transferability figure for the thesis: one 2x2 heatmap per attack
(FGSM, PGD, C&W), where cell (source, target) is the target model's accuracy
on adversarial examples crafted against the source model. The diagonal is the
white-box robust accuracy; the off-diagonal is the transfer setting.

Reads results/transfer.json (produced by scripts/evaluate_transfer.py).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {"fgsm": "FGSM", "pgd": "PGD", "cw": "C&W"}
LABELS = {"standard": "Standard", "robust": "Adv-Trained"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/transfer.json")
    ap.add_argument("--output", default="overleaf/figures/transfer_matrix.png")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    matrices = data["matrices"]
    attacks = [a for a in ["fgsm", "pgd", "cw"] if a in matrices]
    order = ["standard", "robust"]

    fig, axes = plt.subplots(1, len(attacks), figsize=(4.2 * len(attacks), 3.8))
    if len(attacks) == 1:
        axes = [axes]

    for ax, attack in zip(axes, attacks):
        m = matrices[attack]
        grid = np.array([[m[s][t] for t in order] for s in order])
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="equal")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels([LABELS[t] for t in order])
        ax.set_yticklabels([LABELS[s] for s in order])
        ax.set_xlabel("Target (evaluated)")
        if ax is axes[0]:
            ax.set_ylabel("Source (crafted on)")
        ax.set_title(PRETTY[attack])
        for i in range(len(order)):
            for j in range(len(order)):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")

    fig.suptitle("Transfer accuracy (%): lower = stronger transfer", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    pdf = args.output.replace(".png", ".pdf")
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    print(f"saved {args.output} and {pdf}")


if __name__ == "__main__":
    main()
