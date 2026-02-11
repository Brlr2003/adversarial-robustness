"""
Training entry point.

Usage:
    python scripts/train.py --mode standard --epochs 50
    python scripts/train.py --mode adversarial --epochs 50
    python scripts/train.py --mode both --epochs 50
"""

import argparse
import os
import sys

import mlflow
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet import resnet18_cifar10
from src.training.adversarial import train_adversarial
from src.training.standard import train_standard
from src.utils.data import get_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train models for adversarial robustness")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "adversarial", "both"],
        default="both",
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    training_config = config["training"]
    if args.epochs:
        training_config["epochs"] = args.epochs
    if args.batch_size:
        training_config["batch_size"] = args.batch_size

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    train_loader, test_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=training_config["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    print(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    checkpoint_dir = config["paths"]["checkpoints"]

    # --- Standard Training ---
    if args.mode in ("standard", "both"):
        print("\n" + "=" * 60)
        print("STANDARD TRAINING")
        print("=" * 60)

        model = resnet18_cifar10(num_classes=config["model"]["num_classes"])
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        with mlflow.start_run(run_name="standard_training"):
            mlflow.log_params({
                "mode": "standard",
                "epochs": training_config["epochs"],
                "learning_rate": training_config["learning_rate"],
                "batch_size": training_config["batch_size"],
                "architecture": config["model"]["architecture"],
            })

            train_standard(
                model, train_loader, test_loader,
                config=training_config,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )

    # --- Adversarial Training ---
    if args.mode in ("adversarial", "both"):
        print("\n" + "=" * 60)
        print("ADVERSARIAL TRAINING (PGD-AT)")
        print("=" * 60)

        model = resnet18_cifar10(num_classes=config["model"]["num_classes"])
        adv_config = config["adversarial_training"]

        with mlflow.start_run(run_name="adversarial_training"):
            mlflow.log_params({
                "mode": "adversarial",
                "epochs": training_config["epochs"],
                "learning_rate": training_config["learning_rate"],
                "batch_size": training_config["batch_size"],
                "architecture": config["model"]["architecture"],
                "adv_epsilon": adv_config["epsilon"],
                "adv_alpha": adv_config["alpha"],
                "adv_steps": adv_config["steps"],
            })

            train_adversarial(
                model, train_loader, test_loader,
                config=training_config,
                adv_config=adv_config,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print("Run `mlflow ui` to view experiment results.")


if __name__ == "__main__":
    main()
