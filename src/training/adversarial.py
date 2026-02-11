"""
Adversarial Training (PGD-AT) - Madry et al., 2018

Instead of training on clean examples, we generate PGD adversarial examples
on-the-fly during training and train the model to correctly classify them.

This is the gold-standard defense method and produces models that are
provably robust (within the threat model) to first-order attacks.
"""

import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from src.attacks.pgd import pgd_attack


def train_adversarial(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: dict,
    adv_config: dict,
    device: torch.device,
    checkpoint_dir: str = "./checkpoints",
) -> nn.Module:
    """
    Adversarial training using PGD-AT (Madry et al., 2018).

    The key idea: at each training step, generate PGD adversarial examples
    from the current batch, then train the model on these adversarial examples.

    Args:
        model: Model to train
        train_loader: Training data
        test_loader: Test data
        config: Training hyperparameters
        adv_config: Adversarial training parameters (epsilon, alpha, steps)
        device: Device
        checkpoint_dir: Checkpoint save directory

    Returns:
        Adversarially trained model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    if config.get("lr_schedule") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.get("step_milestones", [25, 40]),
            gamma=config.get("step_gamma", 0.1),
        )

    best_accuracy = 0.0

    for epoch in range(1, config["epochs"] + 1):
        # --- Adversarial Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [Adv Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples using PGD
            model.eval()  # BN in eval mode for attack generation
            adv_images = pgd_attack(
                model,
                images,
                labels,
                epsilon=adv_config["epsilon"],
                alpha=adv_config["alpha"],
                steps=adv_config["steps"],
                random_start=adv_config.get("random_start", True),
                device=device,
            )
            model.train()  # Back to train mode

            # Train on adversarial examples
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * train_correct / train_total:.2f}%",
            )

        scheduler.step()

        train_accuracy = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / train_total

        # --- Validation (clean accuracy) ---
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * test_correct / test_total

        # --- Validation (PGD accuracy - quick check) ---
        pgd_correct = 0
        pgd_total = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 5:  # Quick check on 5 batches
                break
            images, labels = images.to(device), labels.to(device)
            adv_images = pgd_attack(
                model, images, labels,
                epsilon=adv_config["epsilon"],
                alpha=adv_config["alpha"],
                steps=20,
                random_start=True,
                device=device,
            )
            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = outputs.max(1)
                pgd_total += labels.size(0)
                pgd_correct += predicted.eq(labels).sum().item()

        pgd_accuracy = 100.0 * pgd_correct / pgd_total

        # Log metrics
        mlflow.log_metrics(
            {
                "train_loss": avg_train_loss,
                "train_adv_accuracy": train_accuracy,
                "test_clean_accuracy": test_accuracy,
                "test_pgd_accuracy": pgd_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
            f"Adv Train Acc={train_accuracy:.2f}%, "
            f"Clean Test={test_accuracy:.2f}%, PGD Test={pgd_accuracy:.2f}%"
        )

        # Save best model (based on PGD accuracy for robust model)
        if pgd_accuracy > best_accuracy:
            best_accuracy = pgd_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_clean_accuracy": test_accuracy,
                    "test_pgd_accuracy": pgd_accuracy,
                },
                os.path.join(checkpoint_dir, "robust_best.pt"),
            )
            print(f"  ✓ New best robust model (PGD acc: {pgd_accuracy:.2f}%)")

    mlflow.log_metric("best_pgd_accuracy", best_accuracy)
    return model
