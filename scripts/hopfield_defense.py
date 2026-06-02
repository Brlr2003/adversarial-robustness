"""
Hopfield associative-memory defense against the One Pixel attack.

Motivation. The standard ResNet-18 maps a 32x32 image to a 512-d feature
vector (the global-average-pool output) before a single linear layer assigns
class logits. A One-Pixel perturbation nudges this 512-d embedding slightly
off the clean data manifold, and because the standard model has small margins
(DeepFool needs only L2 ~ 0.25 to cross a boundary), that nudge can flip the
linear head's decision.

A modern Hopfield network (Ramsauer et al., 2020) stores patterns X and, given
a query xi, retrieves a softmax-weighted combination

    retrieved = X^T . softmax(beta * X . xi),

which for a suitable beta settles the query onto the nearest stored attractor.
Storing the clean training embeddings as attractors therefore pulls a perturbed
query back toward the clean embedding of its class, undoing the one-pixel nudge
before classification. This is the associative-memory / pattern-completion view
of robustness: the memory "cleans" the input.

Three readouts are compared on the same 100-image test subset (the One-Pixel
images saved by scripts/save_one_pixel_images.py):

  standard       the standard model's own linear head on the raw embedding
                 (this is the One-Pixel result to beat)
  hopfield-nn    class = argmax of the per-class softmax mass over stored labels
                 (a pure associative-memory classifier, no trained head)
  hopfield-fc    retrieve/denoise the embedding, then apply the trained head

beta is selected on a validation split disjoint from the 100 evaluation images,
so it is not tuned on the test set. The One-Pixel images were crafted against
the standard head and are NOT adaptive to the Hopfield readout; this is exactly
the question the supervisor posed (classify the existing perturbed images with a
Hopfield classifier), and the non-adaptive caveat is reported.

Outputs: results/hopfield_defense.json and overleaf/figures/hopfield_defense.pdf
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet import resnet18_cifar10

IMG_DIR = "results/one_pixel_images"
K_VALUES = [1, 3, 5]
BETAS = [2, 4, 8, 16, 24, 32, 48, 64]
MEM_SIZE = 50000          # number of clean training embeddings stored
VAL_SIZE = 1000           # disjoint test images used only to pick beta
SUBSET_SEED = 0


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def load_standard(path: str, device: torch.device) -> torch.nn.Module:
    model = resnet18_cifar10(num_classes=10)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """512-d global-average-pool embedding (the input to model.fc)."""
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = model.avg_pool(out)
    return torch.flatten(out, 1)


@torch.no_grad()
def embed(model, imgs: torch.Tensor, device, bs: int = 500) -> torch.Tensor:
    feats = []
    for i in range(0, len(imgs), bs):
        feats.append(features(model, imgs[i:i + bs].to(device)).cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def hopfield_predict(
    model, query_feat, S_raw, S_norm, mem_onehot, beta, device, chunk=200,
):
    """Return (nn_pred, fc_pred) for a batch of query embeddings."""
    nn_preds, fc_preds = [], []
    fc = model.fc.to(device)
    S_raw_d = S_raw.to(device)
    S_norm_d = S_norm.to(device)
    onehot_d = mem_onehot.to(device)
    for i in range(0, len(query_feat), chunk):
        q = query_feat[i:i + chunk].to(device)
        q_norm = F.normalize(q, dim=1)
        sim = q_norm @ S_norm_d.t()                  # (b, M) cosine
        w = torch.softmax(beta * sim, dim=1)         # (b, M)
        class_mass = w @ onehot_d                    # (b, 10)
        nn_preds.append(class_mass.argmax(1).cpu())
        retrieved = w @ S_raw_d                       # (b, 512) denoised
        fc_preds.append(fc(retrieved).argmax(1).cpu())
    return torch.cat(nn_preds), torch.cat(fc_preds)


@torch.no_grad()
def standard_predict(model, query_feat, device, chunk=500):
    preds = []
    fc = model.fc.to(device)
    for i in range(0, len(query_feat), chunk):
        preds.append(fc(query_feat[i:i + chunk].to(device)).argmax(1).cpu())
    return torch.cat(preds)


def acc(pred, labels) -> float:
    return 100.0 * (pred == labels).float().mean().item()


def main():
    device = get_device()
    print(f"Device: {device}")
    model = load_standard("checkpoints/standard_best.pt", device)

    # --- stored memory: clean training embeddings ---
    print("Building memory from training embeddings ...")
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    mem_idx = torch.randperm(len(train), generator=g)[:MEM_SIZE].tolist()
    mem_imgs = torch.stack([train[i][0] for i in mem_idx])
    mem_labels = torch.tensor([train[i][1] for i in mem_idx])
    S_raw = embed(model, mem_imgs, device)
    S_norm = F.normalize(S_raw, dim=1)
    mem_onehot = F.one_hot(mem_labels, 10).float()
    print(f"  stored {len(S_raw)} patterns of dim {S_raw.shape[1]}")

    # --- validation split (disjoint from the 100 eval images) for beta ---
    tf_test = transforms.Compose([transforms.ToTensor()])
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)
    gt = torch.Generator().manual_seed(SUBSET_SEED)
    order = torch.randperm(len(test), generator=gt)
    val_idx = order[100:100 + VAL_SIZE].tolist()           # 100.. avoids the eval 0..100
    val_imgs = torch.stack([test[i][0] for i in val_idx])
    val_labels = torch.tensor([test[i][1] for i in val_idx])
    val_feat = embed(model, val_imgs, device)

    print("Selecting beta on the validation split (clean accuracy) ...")
    best = {"nn": (None, -1.0), "fc": (None, -1.0)}
    for beta in BETAS:
        nn_p, fc_p = hopfield_predict(model, val_feat, S_raw, S_norm, mem_onehot, beta, device)
        a_nn, a_fc = acc(nn_p, val_labels), acc(fc_p, val_labels)
        print(f"  beta={beta:>3}: hopfield-nn {a_nn:.1f}%  hopfield-fc {a_fc:.1f}%")
        if a_nn > best["nn"][1]:
            best["nn"] = (beta, a_nn)
        if a_fc > best["fc"][1]:
            best["fc"] = (beta, a_fc)
    beta_nn, beta_fc = best["nn"][0], best["fc"][0]
    val_std = acc(standard_predict(model, val_feat, device), val_labels)
    print(f"  selected beta_nn={beta_nn} (val {best['nn'][1]:.1f}%), "
          f"beta_fc={beta_fc} (val {best['fc'][1]:.1f}%); standard val {val_std:.1f}%")

    # --- evaluation on the 100-image subset: clean + One Pixel k=1,3,5 ---
    labels = torch.load(os.path.join(IMG_DIR, "labels.pt"))
    clean = torch.load(os.path.join(IMG_DIR, "clean.pt"))
    img_sets = {"clean": clean}
    for k in K_VALUES:
        img_sets[f"k{k}"] = torch.load(os.path.join(IMG_DIR, f"adv_k{k}.pt"))

    rows = {"standard": {}, "hopfield-nn": {}, "hopfield-fc": {}}
    for name, imgs in img_sets.items():
        feat = embed(model, imgs, device)
        std_p = standard_predict(model, feat, device)
        nn_p, _ = hopfield_predict(model, feat, S_raw, S_norm, mem_onehot, beta_nn, device)
        _, fc_p = hopfield_predict(model, feat, S_raw, S_norm, mem_onehot, beta_fc, device)
        rows["standard"][name] = acc(std_p, labels)
        rows["hopfield-nn"][name] = acc(nn_p, labels)
        rows["hopfield-fc"][name] = acc(fc_p, labels)

    cols = ["clean"] + [f"k{k}" for k in K_VALUES]
    print("\n=== Accuracy on the 100-image subset (%) ===")
    header = f"{'method':<14}" + "".join(f"{c:>9}" for c in cols)
    print(header)
    for m in ["standard", "hopfield-nn", "hopfield-fc"]:
        print(f"{m:<14}" + "".join(f"{rows[m][c]:>9.1f}" for c in cols))

    result = {
        "beta_nn": beta_nn, "beta_fc": beta_fc,
        "mem_size": MEM_SIZE, "val_size": VAL_SIZE,
        "val_standard_acc": val_std,
        "accuracy": rows, "columns": cols,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/hopfield_defense.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved results/hopfield_defense.json")

    # --- figure: grouped bars, accuracy vs perturbation ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    methods = ["standard", "hopfield-nn", "hopfield-fc"]
    labels_disp = {"standard": "Standard head", "hopfield-nn": "Hopfield (nearest attractor)",
                   "hopfield-fc": "Hopfield denoise + head"}
    colors = {"standard": "#c0392b", "hopfield-nn": "#7f8c8d", "hopfield-fc": "#2c7fb8"}
    x = range(len(cols))
    width = 0.26
    xticklabels = ["Clean", "One Pixel k=1", "One Pixel k=3", "One Pixel k=5"]
    for j, m in enumerate(methods):
        vals = [rows[m][c] for c in cols]
        ax.bar([xi + (j - 1) * width for xi in x], vals, width,
               label=labels_disp[m], color=colors[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Accuracy on the 100-image subset (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    os.makedirs("overleaf/figures", exist_ok=True)
    plt.savefig("overleaf/figures/hopfield_defense.pdf", bbox_inches="tight")
    plt.savefig("overleaf/figures/hopfield_defense.png", dpi=200, bbox_inches="tight")
    print("Saved overleaf/figures/hopfield_defense.pdf")


if __name__ == "__main__":
    main()
