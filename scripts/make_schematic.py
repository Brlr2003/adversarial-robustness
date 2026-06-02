"""
Schematic of the two pipelines in the thesis, for the defense slide and the
methodology chapter:

  (top)    adversarial example generation: clean image -> model -> loss ->
           gradient/decision -> perturbation -> adversarial image -> wrong class
  (bottom) PGD adversarial training (min-max): for each batch, an inner PGD
           attack builds the worst-case input, the model is trained on it, and
           the weights are updated; the loop yields the robust model.

Saves overleaf/figures/schematic_flow.pdf (+ .png slide).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#2c7fb8"
RED = "#c0392b"
GREY = "#34495e"
LIGHT = "#eaf2f8"
LIGHTR = "#fdEDEC"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=GREY, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#1b2631")


def arrow(ax, x1, y1, x2, y2, color=GREY, text=None, fs=8.5, rad=0.0, off=(0, 0)):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text,
                ha="center", va="center", fontsize=fs, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # ---------- Top: adversarial example generation ----------
    ax.text(0.1, 6.5, "Adversarial example generation", fontsize=13,
            fontweight="bold", color=RED)
    y = 5.2
    h, w = 0.85, 1.9
    box(ax, 0.1, y, w, h, "Clean image\n$x$  (label $y$)")
    box(ax, 2.5, y, w, h, "ResNet-18\n$f_\\theta$", fc=LIGHT)
    box(ax, 4.9, y, w, h, "Loss\n$L(f_\\theta(x), y)$")
    box(ax, 7.7, y, 2.2, h, "Perturbation $\\delta$\nFGSM / PGD / DeepFool /\nC&W / HSJ / One Pixel", fc=LIGHTR, ec=RED, fs=8.3)
    box(ax, 10.2, y, 1.7, h, "Adversarial\n$x^{*}=x+\\delta$", fc=LIGHTR, ec=RED)

    arrow(ax, 2.0, y + h / 2, 2.5, y + h / 2)
    arrow(ax, 4.4, y + h / 2, 4.9, y + h / 2)
    arrow(ax, 6.8, y + h / 2, 7.7, y + h / 2, color=RED,
          text="$\\nabla_x L$ / decision", off=(0, 0.33))
    arrow(ax, 9.9, y + h / 2, 10.2, y + h / 2, color=RED)
    # constraint note
    ax.text(8.8, y - 0.28, "$\\|\\delta\\| \\leq \\epsilon$ ($L_\\infty,\\,L_2,\\,L_0$)",
            ha="center", fontsize=8.5, color=RED)

    # adversarial back through model -> wrong class
    box(ax, 10.2, 3.55, 1.7, h, "ResNet-18\n$f_\\theta$")
    box(ax, 7.7, 3.55, 1.9, h, "Wrong class\n$\\hat{y}\\neq y$", fc=LIGHTR, ec=RED, bold=True)
    arrow(ax, 11.05, y, 11.05, 3.55 + h, color=RED, rad=0.0)
    arrow(ax, 10.2, 3.55 + h / 2, 9.6, 3.55 + h / 2, color=RED)

    # divider
    ax.plot([0.1, 11.9], [3.05, 3.05], color="#bdc3c7", lw=1, ls="--")

    # ---------- Bottom: PGD adversarial training (min-max) ----------
    ax.text(0.1, 2.75, "PGD adversarial training  (min-max objective)", fontsize=13,
            fontweight="bold", color=BLUE)
    yb = 1.35
    box(ax, 0.1, yb, w, h, "Clean batch\n$(x, y)$")
    box(ax, 2.5, yb, 2.2, h, "Inner max: 7-step PGD\n$x^{*}=\\arg\\max_\\delta L$", fc="#e8f4fb", ec=BLUE, fs=8.6)
    box(ax, 5.3, yb, 1.7, h, "ResNet-18\n$f_\\theta$")
    box(ax, 7.4, yb, 1.7, h, "Loss\n$L(f_\\theta(x^{*}), y)$")
    box(ax, 9.5, yb, 2.4, h, "Outer min: update $\\theta$\n$\\theta \\leftarrow \\theta-\\eta\\nabla_\\theta L$", fc="#e8f4fb", ec=BLUE, fs=8.6)

    arrow(ax, 2.0, yb + h / 2, 2.5, yb + h / 2)
    arrow(ax, 4.7, yb + h / 2, 5.3, yb + h / 2, color=BLUE)
    arrow(ax, 7.0, yb + h / 2, 7.4, yb + h / 2, color=BLUE)
    arrow(ax, 9.1, yb + h / 2, 9.5, yb + h / 2, color=BLUE, text="$\\nabla_\\theta L$", off=(0, 0.3))
    # loop back
    arrow(ax, 10.7, yb, 10.7, yb - 0.55, color=BLUE)
    ax.add_patch(FancyArrowPatch((10.7, yb - 0.55), (1.05, yb - 0.55),
                 arrowstyle="-", linewidth=1.5, color=BLUE))
    arrow(ax, 1.05, yb - 0.55, 1.05, yb, color=BLUE)
    ax.text(6.0, yb - 0.78, "repeat for every batch, 50 epochs  $\\rightarrow$  robust model (83.9% clean, 45.1% PGD)",
            ha="center", fontsize=9, color=BLUE, style="italic")

    plt.tight_layout()
    os.makedirs("overleaf/figures", exist_ok=True)
    plt.savefig("overleaf/figures/schematic_flow.pdf", bbox_inches="tight")
    plt.savefig("overleaf/figures/schematic_flow.png", dpi=200, bbox_inches="tight")
    print("saved overleaf/figures/schematic_flow.pdf (+ .png)")


if __name__ == "__main__":
    main()
