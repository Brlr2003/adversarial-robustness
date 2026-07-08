"""
Render the DeepFool result panel (Figure 5.8) as a light, black-and-white-safe
figure that mirrors the demo layout: three images on top (original, adversarial,
amplified perturbation) with the L2/Linf norms, and below them the standard and
robust models' class-probability bars on the adversarial image.

Data is precomputed by the offline DeepFool run and saved to .shots/deepfool_data.pt.
This exists because the browser screenshot tool disconnected mid-capture; FGSM and
PGD (Figure 5.7) are live light-theme app screenshots.
"""
from __future__ import annotations
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
           "frog", "horse", "ship", "truck"]
RED = "#b91c1c"     # predicted class (dark: reads in grayscale)
GREY = "#cbd5e1"    # other classes (light)

d = torch.load("results/deepfool_panel_data.pt", weights_only=False)
orig = d["orig"].permute(1, 2, 0).numpy()
adv = d["adv"].permute(1, 2, 0).numpy().clip(0, 1)
pert = np.abs(d["adv"] - d["orig"]).sum(0).numpy()
sp_a, rp_a = d["sp_a"].numpy(), d["rp_a"].numpy()
sp_c, rp_c = d["sp_c"].numpy(), d["rp_c"].numpy()

fig = plt.figure(figsize=(11, 7.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.15], hspace=0.42, wspace=0.28)

# --- top row: three images ---
titles = ["Original Image", "Adversarial Image", "Perturbation (amplified)"]
subs = ["DeepFool", f"$L_2$ norm: {d['l2']:.4f}", f"$L_\\infty$ norm: {d['linf']:.4f}"]
for j, (im, t, s) in enumerate(zip([orig, adv, None], titles, subs)):
    ax = fig.add_subplot(gs[0, j])
    if im is not None:
        ax.imshow(im)
    else:
        ax.imshow(pert, cmap="Greys", vmin=0, vmax=max(pert.max(), 1e-6))
    ax.set_title(t, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel(s, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

# --- bottom row: standard and robust probability bars ---
def bars(ax, probs, clean, title):
    y = np.arange(len(CLASSES))[::-1]
    pred = int(probs.argmax())
    colors = [RED if i == pred else GREY for i in range(len(CLASSES))]
    ax.barh(y, probs * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("Probability (%)", fontsize=10)
    verdict = "FOOLED" if pred != 5 else "Correct"   # dog = index 5
    ax.set_title(f"{title}", fontsize=12, fontweight="bold", pad=30)
    ax.text(0.5, 1.02,
            f"Clean: {CLASSES[int(clean.argmax())]} ({clean.max()*100:.1f}%)\n"
            f"Attacked: {CLASSES[pred]} ({probs[pred]*100:.1f}%) {verdict}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5)
    ax.grid(True, axis="x", alpha=0.3)

bars(fig.add_subplot(gs[1, 0]), sp_a, sp_c, "Standard Model")
bars(fig.add_subplot(gs[1, 1]), rp_a, rp_c, "Robust Model")
fig.add_subplot(gs[1, 2]).axis("off")

plt.savefig("overleaf/figures/DeepFool.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig("overleaf/figures/DeepFool.pdf", bbox_inches="tight", facecolor="white")
print("saved overleaf/figures/DeepFool.png and .pdf")
