"""
Regenerate and SAVE the One Pixel adversarial images for the fixed 100-image
test subset (seed 0) against the standard model, for k in {1, 3, 5}.

The black-box evaluation script (scripts/evaluate_blackbox.py) only recorded
success statistics and discarded the perturbed images. Two things now need the
actual images: the Hopfield-defense experiment, and the supervisor's request for
the 100 perturbed images. They are regenerated here with the exact same
configuration as the thesis run (per-image seed = index, pop_size = 400,
max_iter = 50), so the per-image outcomes match results/one_pixel.json.

These are evaluation adversarial examples crafted against the frozen, fully
trained model, so they are static once created. Saving them is sound (unlike
the PGD adversarial-training images, which must be regenerated on the fly
against the still-changing weights).

Outputs (results/one_pixel_images/):
    clean.pt            (100, 3, 32, 32) clean test images, [0, 1]
    labels.pt           (100,) ground-truth labels
    adv_k1.pt / adv_k3.pt / adv_k5.pt   (100, 3, 32, 32) perturbed images
    clean_pred.pt       (100,) standard-model prediction on the clean image
    meta.json           per-image label, clean_pred, and per-k adv_pred/success
    png/clean/*.png, png/k{1,3,5}/*.png   individual images
    png/montage_k{1,3,5}.png              first-24 grids for quick inspection
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.one_pixel import one_pixel_attack
from src.models.resnet import resnet18_cifar10

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

NUM_IMAGES = 100
SUBSET_SEED = 0
K_VALUES = [1, 3, 5]
POP_SIZE = 400
MAX_ITER = 50
OUT_DIR = "results/one_pixel_images"


def get_subset(data_dir: str):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    g = torch.Generator().manual_seed(SUBSET_SEED)
    idx = torch.randperm(len(dataset), generator=g)[:NUM_IMAGES].tolist()
    imgs = torch.stack([dataset[i][0] for i in idx])           # (100,3,32,32)
    labels = torch.tensor([dataset[i][1] for i in idx])        # (100,)
    return imgs, labels


def load_standard(path: str, device: torch.device) -> torch.nn.Module:
    model = resnet18_cifar10(num_classes=10)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for sub in ["png/clean"] + [f"png/k{k}" for k in K_VALUES]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    imgs, labels = get_subset("./data")
    model = load_standard("checkpoints/standard_best.pt", device)

    with torch.no_grad():
        clean_pred = model(imgs.to(device)).argmax(dim=1).cpu()

    torch.save(imgs, os.path.join(OUT_DIR, "clean.pt"))
    torch.save(labels, os.path.join(OUT_DIR, "labels.pt"))
    torch.save(clean_pred, os.path.join(OUT_DIR, "clean_pred.pt"))

    meta = []
    for i in range(NUM_IMAGES):
        meta.append({
            "index": i,
            "label": int(labels[i]),
            "label_name": CLASSES[int(labels[i])],
            "clean_pred": int(clean_pred[i]),
            "clean_correct": bool(clean_pred[i] == labels[i]),
        })

    for i in range(NUM_IMAGES):
        save_image(imgs[i], os.path.join(OUT_DIR, f"png/clean/{i:03d}_{CLASSES[int(labels[i])]}.png"))

    for k in K_VALUES:
        print(f"\n=== One Pixel, k={k} (pop={POP_SIZE}, iter={MAX_ITER}) ===")
        adv = imgs.clone()
        adv_pred = clean_pred.clone()
        n_correct = 0
        n_success = 0
        t0 = time.time()
        for i in range(NUM_IMAGES):
            # Match the thesis run exactly: per-image attack with seed = index.
            a, info = one_pixel_attack(
                model, imgs[i:i + 1], labels[i:i + 1],
                k=k, pop_size=POP_SIZE, max_iter=MAX_ITER, device=device, seed=i,
            )
            adv[i] = a[0].cpu()
            with torch.no_grad():
                p = int(model(a.to(device)).argmax(dim=1).item())
            adv_pred[i] = p
            correct_clean = bool(clean_pred[i] == labels[i])
            if correct_clean:
                n_correct += 1
                if p != int(labels[i]):
                    n_success += 1
            meta[i][f"adv_pred_k{k}"] = p
            meta[i][f"adv_pred_name_k{k}"] = CLASSES[p]
            meta[i][f"success_k{k}"] = bool(correct_clean and p != int(labels[i]))
            if (i + 1) % 20 == 0:
                sr = 100 * n_success / max(n_correct, 1)
                print(f"  [{i + 1}/{NUM_IMAGES}] correct-clean={n_correct} success={sr:.1f}%  ({time.time() - t0:.0f}s)")

        torch.save(adv, os.path.join(OUT_DIR, f"adv_k{k}.pt"))
        for i in range(NUM_IMAGES):
            save_image(adv[i], os.path.join(OUT_DIR, f"png/k{k}/{i:03d}_{CLASSES[int(labels[i])]}_to_{CLASSES[int(adv_pred[i])]}.png"))
        grid = make_grid(adv[:24], nrow=6, padding=2)
        save_image(grid, os.path.join(OUT_DIR, f"png/montage_k{k}.png"))
        sr = 100 * n_success / max(n_correct, 1)
        print(f"  done k={k}: success {sr:.1f}% over {n_correct} correctly-classified ({time.time() - t0:.0f}s)")

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved tensors, PNGs, and meta.json under {OUT_DIR}/")


if __name__ == "__main__":
    main()
