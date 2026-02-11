"""Standard training loop for CIFAR-10."""

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm


def train_standard(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: str = "./checkpoints",
) -> nn.Module:
    """
    Standard training loop for CIFAR-10 classification.

    Args:
        model: Model to train
        train_loader: Training data
        test_loader: Test data for validation
        config: Training configuration dict
        device: Device to train on
        checkpoint_dir: Where to save checkpoints

    Returns:
        Trained model
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler
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
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
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

        # --- Validation ---
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

        # Log metrics
        mlflow.log_metrics(
            {
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
            f"Train Acc={train_accuracy:.2f}%, Test Acc={test_accuracy:.2f}%"
        )

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_accuracy": test_accuracy,
                },
                os.path.join(checkpoint_dir, "standard_best.pt"),
            )
            print(f"  ✓ New best model saved (accuracy: {test_accuracy:.2f}%)")

    mlflow.log_metric("best_test_accuracy", best_accuracy)
    return model
