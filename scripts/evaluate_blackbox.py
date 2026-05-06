"""
Black-box attack evaluation: HopSkipJump and One Pixel.

Both attacks take longer per image than FGSM/PGD and need a per-image
budget cap. This script runs each attack on a fixed test subset for
the standard and adversarially trained models and writes the results
to JSON.

Usage:
    python scripts/evaluate_blackbox.py \
        --standard checkpoints/standard_best.pt \
        --robust   checkpoints/robust_best.pt \
        --attack   hopskipjump \
        --num-images 200 \
        --output    results/hopskipjump.json
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

from src.attacks.hopskipjump import hopskipjump_attack
from src.attacks.one_pixel import one_pixel_attack
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


def evaluate_hopskipjump(
    model: torch.nn.Module,
    subset,
    device: torch.device,
    norm: str,
    max_queries: int,
    num_iterations: int,
) -> dict:
    queries_all = []
    success_all = []
    l2_all = []
    linf_all = []
    qtt_all = []

    correct_only = 0
    n_total = len(subset)

    t0 = time.time()
    for idx in range(n_total):
        img, label = subset[idx]
        img = img.unsqueeze(0)
        label_t = torch.tensor([label])

        # Skip images that are already misclassified clean (their attack
        # success metric is degenerate).
        with torch.no_grad():
            clean_pred = model(img.to(device)).argmax(dim=1).item()
        if clean_pred != label:
            continue
        correct_only += 1

        adv, info = hopskipjump_attack(
            model, img, label_t,
            norm=norm,
            max_queries=max_queries,
            num_iterations=num_iterations,
            device=device,
        )
        queries_all.append(int(info["queries"][0].item()))
        success_all.append(bool(info["success"][0].item()))
        l2_all.append(float(info["l2"][0].item()))
        linf_all.append(float(info["linf"][0].item()))
        qtt_all.append(info["queries_to_threshold"][0].tolist())

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{idx + 1}/{n_total}] "
                f"correct-only={correct_only} "
                f"avg-queries={sum(queries_all) / max(len(queries_all), 1):.0f} "
                f"success-rate={sum(success_all) / max(len(success_all), 1) * 100:.1f}%  "
                f"({elapsed:.1f}s)"
            )

    return {
        "n_evaluated": correct_only,
        "n_total": n_total,
        "queries": queries_all,
        "success": success_all,
        "l2": l2_all,
        "linf": linf_all,
        "queries_to_threshold": qtt_all,
        "thresholds": info["thresholds"] if queries_all else [],
        "norm": norm,
        "max_queries": max_queries,
        "num_iterations": num_iterations,
    }


def evaluate_one_pixel(
    model: torch.nn.Module,
    subset,
    device: torch.device,
    k_values: list[int],
    pop_size: int,
    max_iter: int,
) -> dict:
    results: dict = {"per_k": {}}
    n_total = len(subset)

    for k in k_values:
        success = []
        queries = []
        fitness = []
        correct_only = 0
        t0 = time.time()
        for idx in range(n_total):
            img, label = subset[idx]
            img_b = img.unsqueeze(0)
            label_t = torch.tensor([label])
            with torch.no_grad():
                clean_pred = model(img_b.to(device)).argmax(dim=1).item()
            if clean_pred != label:
                continue
            correct_only += 1

            adv, info = one_pixel_attack(
                model, img_b, label_t,
                k=k, pop_size=pop_size, max_iter=max_iter, device=device,
                seed=idx,
            )
            success.append(bool(info["success"][0].item()))
            queries.append(int(info["queries"][0].item()))
            fitness.append(float(info["fitness"][0].item()))

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(
                    f"  k={k} [{idx + 1}/{n_total}] "
                    f"success={sum(success) / max(len(success), 1) * 100:.1f}%  "
                    f"({elapsed:.1f}s)"
                )

        results["per_k"][k] = {
            "n_evaluated": correct_only,
            "n_total": n_total,
            "success": success,
            "queries": queries,
            "fitness": fitness,
        }
    results["pop_size"] = pop_size
    results["max_iter"] = max_iter
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard", type=str, default="checkpoints/standard_best.pt")
    p.add_argument("--robust", type=str, default="checkpoints/robust_best.pt")
    p.add_argument(
        "--attack", type=str, choices=["hopskipjump", "one_pixel"], required=True
    )
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=0)
    # HSJ
    p.add_argument("--hsj-norm", type=str, default="l2", choices=["l2", "linf"])
    p.add_argument("--hsj-max-queries", type=int, default=2500)
    p.add_argument("--hsj-iterations", type=int, default=25)
    # OP
    p.add_argument("--op-k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--op-pop", type=int, default=400)
    p.add_argument("--op-iter", type=int, default=75)
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Attack: {args.attack}")
    print(f"Num images: {args.num_images}")

    subset = get_test_subset(args.data_dir, args.num_images, seed=args.seed)

    out: dict = {
        "attack": args.attack,
        "num_images": args.num_images,
        "seed": args.seed,
        "device": str(device),
    }

    for tag, ckpt in [("standard", args.standard), ("robust", args.robust)]:
        if not os.path.exists(ckpt):
            print(f"  [skip] {tag}: checkpoint not found at {ckpt}")
            continue
        print(f"\n=== Model: {tag} ({ckpt}) ===")
        model = load_model(ckpt, device)

        if args.attack == "hopskipjump":
            t0 = time.time()
            r = evaluate_hopskipjump(
                model, subset, device,
                norm=args.hsj_norm,
                max_queries=args.hsj_max_queries,
                num_iterations=args.hsj_iterations,
            )
            r["wall_time_sec"] = time.time() - t0
            out[tag] = r
        elif args.attack == "one_pixel":
            t0 = time.time()
            r = evaluate_one_pixel(
                model, subset, device,
                k_values=args.op_k,
                pop_size=args.op_pop,
                max_iter=args.op_iter,
            )
            r["wall_time_sec"] = time.time() - t0
            out[tag] = r

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
