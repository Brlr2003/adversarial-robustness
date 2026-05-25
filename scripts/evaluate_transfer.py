"""
Transferability study.

Adversarial examples crafted against one model are evaluated on another. This
is the transfer-based black-box threat model: the attacker has white-box access
to a *substitute* model (here, one of my two trained models) but only queries
the victim with the resulting examples. It complements HopSkipJump, which is
decision-based black-box on the victim directly.

For each (source, target, attack) combination I report the target model's
accuracy on the adversarial examples. The diagonal (source == target) is the
ordinary white-box robust accuracy; the off-diagonal entries measure transfer.
To keep the baseline clean, all examples are crafted from the set of images
that *both* models classify correctly, so any drop is attributable to the
attack rather than to an already-wrong prediction.

Usage:
    python scripts/evaluate_transfer.py \
        --standard checkpoints/standard_best.pt \
        --robust   checkpoints/robust_best.pt \
        --num-images 1000 \
        --output    results/transfer.json
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
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
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


def both_correct(models, images, labels, device, batch_size):
    """Mask of images that every model in `models` classifies correctly."""
    mask = torch.ones(len(images), dtype=torch.bool)
    for m in models:
        preds = []
        for i in range(0, len(images), batch_size):
            xb = images[i : i + batch_size].to(device)
            with torch.no_grad():
                preds.append(m(xb).argmax(dim=1).cpu())
        preds = torch.cat(preds)
        mask &= preds == labels
    return mask


def craft(attack, model, images, labels, device, batch_size, eps, alpha, steps):
    """Craft adversarial examples for a whole tensor in batches."""
    out = []
    for i in range(0, len(images), batch_size):
        xb = images[i : i + batch_size].to(device)
        yb = labels[i : i + batch_size].to(device)
        if attack == "fgsm":
            adv = fgsm_attack(model, xb, yb, epsilon=eps, device=device)
        elif attack == "pgd":
            adv = pgd_attack(
                model, xb, yb, epsilon=eps, alpha=alpha, steps=steps, device=device
            )
        elif attack == "cw":
            adv, _ = cw_l2_attack(
                model, xb, yb,
                binary_search_steps=6, max_iterations=100, device=device,
            )
        else:
            raise ValueError(attack)
        out.append(adv.detach().cpu())
    return torch.cat(out)


def accuracy(model, images, labels, device, batch_size):
    correct = 0
    for i in range(0, len(images), batch_size):
        xb = images[i : i + batch_size].to(device)
        yb = labels[i : i + batch_size].to(device)
        with torch.no_grad():
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / len(images) * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard", type=str, default="checkpoints/standard_best.pt")
    p.add_argument("--robust", type=str, default="checkpoints/robust_best.pt")
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "cw"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    subset = get_test_subset(args.data_dir, args.num_images, seed=args.seed)
    images = torch.stack([subset[i][0] for i in range(len(subset))])
    labels = torch.tensor([subset[i][1] for i in range(len(subset))])

    models = {
        "standard": load_model(args.standard, device),
        "robust": load_model(args.robust, device),
    }

    mask = both_correct(list(models.values()), images, labels, device, args.batch_size)
    images, labels = images[mask], labels[mask]
    print(f"Images both models classify correctly: {len(images)}/{args.num_images}")

    out: dict = {
        "num_images": args.num_images,
        "n_common_correct": int(len(images)),
        "epsilon": args.epsilon,
        "pgd_steps": args.pgd_steps,
        "seed": args.seed,
        "device": str(device),
        "matrices": {},
    }

    for attack in args.attacks:
        print(f"\n=== Attack: {attack} ===")
        matrix = {}
        for src_name, src_model in models.items():
            t0 = time.time()
            adv = craft(
                attack, src_model, images, labels, device,
                args.batch_size, args.epsilon, args.alpha, args.pgd_steps,
            )
            row = {}
            for tgt_name, tgt_model in models.items():
                row[tgt_name] = accuracy(tgt_model, adv, labels, device, args.batch_size)
            matrix[src_name] = row
            print(
                f"  source={src_name:8s} -> "
                + ", ".join(f"{t}={a:.1f}%" for t, a in row.items())
                + f"  ({time.time() - t0:.1f}s)"
            )
        out["matrices"][attack] = matrix

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
