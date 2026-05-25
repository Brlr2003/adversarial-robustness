"""
Render the Carlini-Wagner figure for the thesis: the cumulative distribution
of the final L2 perturbation over successfully attacked images, for the
standard and adversarially trained models. This parallels the HopSkipJump
curve and shows directly how much more perturbation C&W needs against the
robust model.

Reads results/cw_l2.json (produced by scripts/evaluate_cw.py).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cdf_points(values):
    v = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/cw_l2.json")
    ap.add_argument("--output", default="overleaf/figures/cw_curve.png")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {
        "standard": dict(color="#c0392b", label="Standard"),
        "robust": dict(color="#2c7fb8", label="Adv-Trained"),
    }
    for tag, st in styles.items():
        if tag not in data:
            continue
        succ = data[tag]["success"]
        l2 = data[tag]["l2"]
        solved = [l2[i] for i in range(len(l2)) if succ[i] and np.isfinite(l2[i])]
        if not solved:
            continue
        x, y = cdf_points(solved)
        ax.plot(x, y * 100, lw=2, **st)
        med = float(np.median(solved))
        ax.axvline(med, color=st["color"], ls=":", lw=1, alpha=0.7)

    ax.set_xlabel(r"$L_2$ perturbation")
    ax.set_ylabel("Cumulative % of attacked images")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Carlini-Wagner $L_2$: final perturbation (successful attacks)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    pdf = args.output.replace(".png", ".pdf")
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    print(f"saved {args.output} and {pdf}")


if __name__ == "__main__":
    main()
