"""
Evaluation entry point.

Usage:
    python scripts/evaluate.py --model-path checkpoints/standard_best.pt
    python scripts/evaluate.py --model-path checkpoints/robust_best.pt --attacks fgsm pgd
    python scripts/evaluate.py --model-path checkpoints/standard_best.pt --max-batches 10
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet import resnet18_cifar10
from src.utils.data import get_dataloaders
from src.utils.metrics import full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["fgsm", "pgd", "deepfool"],
        help="Attacks to evaluate",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for speed")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = resnet18_cifar10(num_classes=config["model"]["num_classes"])
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from: {args.model_path}")

    if "test_accuracy" in checkpoint:
        print(f"  Checkpoint accuracy: {checkpoint['test_accuracy']:.2f}%")
    if "test_pgd_accuracy" in checkpoint:
        print(f"  Checkpoint PGD accuracy: {checkpoint['test_pgd_accuracy']:.2f}%")

    # Data
    _, test_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Build attack configs (only for requested attacks)
    attack_configs = {}
    for attack in args.attacks:
        if attack in config["attacks"]:
            attack_configs[attack] = config["attacks"][attack]

    # Run evaluation
    print(f"\nEvaluating against: {list(attack_configs.keys())}")
    if args.max_batches:
        print(f"  (limited to {args.max_batches} batches)")

    results = full_evaluation(
        model, test_loader, attack_configs, device, max_batches=args.max_batches
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for key, value in results.items():
        if "accuracy" in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")

    # Save to JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
