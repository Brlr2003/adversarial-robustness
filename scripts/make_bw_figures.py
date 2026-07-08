"""
Regenerate the two figures whose source experiments are expensive to re-run,
straight from their saved JSON results, in a black-and-white-print-safe style:

  overleaf/figures/hopfield_defense.pdf   from results/hopfield_defense.json
  overleaf/figures/hopskipjump_curve.pdf  from results/hopskipjump_l2.json

Bars are separated by gray level and hatch; the CDF curves by line style and
marker, so nothing relies on colour. Numbers are unchanged: this only restyles.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "overleaf/figures")


def hopfield_bars():
    with open(os.path.join(ROOT, "results/hopfield_defense.json")) as f:
        d = json.load(f)
    acc = d["accuracy"]
    cols = ["clean", "k1", "k3", "k5"]
    xticklabels = ["Clean", "One Pixel k=1", "One Pixel k=3", "One Pixel k=5"]
    methods = ["standard", "hopfield-nn", "hopfield-fc"]
    labels_disp = {
        "standard": "Standard head",
        "hopfield-nn": "Hopfield (nearest attractor)",
        "hopfield-fc": "Hopfield denoise + head",
    }
    # Three gray levels plus hatching so the bars separate in grayscale.
    bar_style = {
        "standard":    dict(facecolor="white", hatch="////", edgecolor="black"),
        "hopfield-nn": dict(facecolor="0.55",  hatch="",     edgecolor="black"),
        "hopfield-fc": dict(facecolor="0.2",   hatch="....", edgecolor="white"),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(cols))
    width = 0.26
    for j, m in enumerate(methods):
        vals = [acc[m][c] for c in cols]
        ax.bar([xi + (j - 1) * width for xi in x], vals, width,
               label=labels_disp[m], linewidth=0.8, **bar_style[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Accuracy on the 100-image subset (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "hopfield_defense.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG, "hopfield_defense.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved hopfield_defense.pdf/.png")


def cdf(vals):
    xs = sorted(vals)
    n = len(xs)
    x, y = [0.0], [0.0]
    for i, v in enumerate(xs):
        x.append(v)
        y.append(i + 1)
    return x, y, n


def hopskipjump_curve():
    with open(os.path.join(ROOT, "results/hopskipjump_l2.json")) as f:
        d = json.load(f)
    styles = {
        "standard": dict(ls="-", marker="o", mfc="black", label="Standard"),
        "robust": dict(ls="--", marker="s", mfc="white", label="Adv-Trained"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, st in styles.items():
        rec = d[tag]
        n_total = rec["n_evaluated"]
        solved = [rec["l2"][i] for i in range(len(rec["l2"]))
                  if rec["success"][i] and math.isfinite(rec["l2"][i])]
        if not solved:
            continue
        # cumulative fraction of ALL attacked images (denominator = n_evaluated)
        xs = sorted(solved)
        x = [0.0] + xs
        y = [0.0] + [(i + 1) / n_total * 100 for i in range(len(xs))]
        ax.plot(x, y, color="black", lw=1.8, ls=st["ls"], marker=st["marker"],
                markevery=max(1, len(x) // 8), ms=5, markerfacecolor=st["mfc"],
                label=st["label"])
        med = xs[len(xs) // 2]
        ax.axvline(med, color="black", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel(r"$L_2$ perturbation")
    ax.set_ylabel("Cumulative % of attacked images")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(r"HopSkipJump $L_2$: final perturbation within the query budget")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "hopskipjump_curve.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG, "hopskipjump_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved hopskipjump_curve.pdf/.png")


if __name__ == "__main__":
    hopfield_bars()
    hopskipjump_curve()
