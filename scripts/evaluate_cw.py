"""
Carlini & Wagner L_2 evaluation.

Runs the C&W L_2 attack on a fixed CIFAR-10 subset for the standard and
adversarially trained models. C&W is white-box and batched, so it is far
cheaper per image than HopSkipJump/One Pixel and can run on a larger subset.

For each model it reports the attack success rate and the distribution of the
final L_2 perturbation over the images that were successfully attacked, and
writes everything to JSON.

Usage:
    python scripts/evaluate_cw.py \
        --standard checkpoints/standard_best.pt \
        --robust   checkpoints/robust_best.pt \
        --num-images 500 \
        --output    results/cw_l2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.cw import cw_l2_attack
from src.models.resnet import resnet18_cifar10


def get_test_subset(data_dir: str, num_images: int, seed: int = 0):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:num_images].tolist()
    return Subset(dataset, indices)


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    model = resnet18_cifar10(num_classes=10)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def collect_correct(model, subset, device, batch_size):
    """Return a (images, labels) tensor pair of correctly classified images."""
    imgs = torch.stack([subset[i][0] for i in range(len(subset))])
    lbls = torch.tensor([subset[i][1] for i in range(len(subset))])
    keep_img, keep_lbl = [], []
    for i in range(0, len(imgs), batch_size):
        xb = imgs[i : i + batch_size].to(device)
        yb = lbls[i : i + batch_size].to(device)
        with torch.no_grad():
            pred = model(xb).argmax(dim=1)
        mask = (pred == yb).cpu()
        keep_img.append(imgs[i : i + batch_size][mask])
        keep_lbl.append(lbls[i : i + batch_size][mask])
    return torch.cat(keep_img), torch.cat(keep_lbl)


def evaluate_cw(model, subset, device, args) -> dict:
    images, labels = collect_correct(model, subset, device, args.batch_size)
    n = images.shape[0]

    success_all = []
    l2_all = []
    linf_all = []

    t0 = time.time()
    for i in range(0, n, args.batch_size):
        xb = images[i : i + args.batch_size]
        yb = labels[i : i + args.batch_size]
        _, info = cw_l2_attack(
            model, xb, yb,
            confidence=args.confidence,
            binary_search_steps=args.steps,
            max_iterations=args.iters,
            learning_rate=args.lr,
            initial_const=args.const,
            device=device,
        )
        success_all.extend(info["success"].tolist())
        l2_all.extend(info["l2"].tolist())
        linf_all.extend(info["linf"].tolist())
        done = min(i + args.batch_size, n)
        print(
            f"  [{done}/{n}] "
            f"success-rate={sum(success_all) / len(success_all) * 100:.1f}%  "
            f"({time.time() - t0:.1f}s)"
        )

    # L_2 statistics over successfully attacked images only.
    solved = [l2_all[j] for j in range(len(l2_all)) if success_all[j]]
    solved.sort()
    if solved:
        median_l2 = solved[len(solved) // 2]
        mean_l2 = sum(solved) / len(solved)
    else:
        median_l2 = float("nan")
        mean_l2 = float("nan")

    return {
        "n_evaluated": n,
        "n_total": len(subset),
        "success": success_all,
        "l2": l2_all,
        "linf": linf_all,
        "success_rate": sum(success_all) / max(len(success_all), 1) * 100,
        "median_l2": median_l2,
        "mean_l2": mean_l2,
        "wall_time_sec": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard", type=str, default="checkpoints/standard_best.pt")
    p.add_argument("--robust", type=str, default="checkpoints/robust_best.pt")
    p.add_argument("--num-images", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--confidence", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=8, help="binary search steps")
    p.add_argument("--iters", type=int, default=200, help="Adam iters per step")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--const", type=float, default=1e-2)
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Attack: carlini-wagner L2")
    print(f"Num images: {args.num_images} (batch {args.batch_size})")

    subset = get_test_subset(args.data_dir, args.num_images, seed=args.seed)

    out: dict = {
        "attack": "cw_l2",
        "num_images": args.num_images,
        "seed": args.seed,
        "device": str(device),
        "binary_search_steps": args.steps,
        "max_iterations": args.iters,
        "confidence": args.confidence,
    }

    for tag, ckpt in [("standard", args.standard), ("robust", args.robust)]:
        if not os.path.exists(ckpt):
            print(f"  [skip] {tag}: checkpoint not found at {ckpt}")
            continue
        print(f"\n=== Model: {tag} ({ckpt}) ===")
        model = load_model(ckpt, device)
        out[tag] = evaluate_cw(model, subset, device, args)
        print(
            f"  -> success {out[tag]['success_rate']:.1f}%  "
            f"median L2 {out[tag]['median_l2']:.3f}"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
