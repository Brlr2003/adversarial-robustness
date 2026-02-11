"""Evaluation metrics for model robustness."""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.attacks.deepfool import deepfool_attack
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


def evaluate_clean(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model accuracy on clean (unperturbed) test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Clean evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return {"clean_accuracy": accuracy, "clean_correct": correct, "clean_total": total}


def evaluate_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_name: str,
    attack_params: dict,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate model robustness against a specific attack.

    Args:
        model: Target model
        test_loader: Test data
        attack_name: One of 'fgsm', 'pgd', 'deepfool'
        attack_params: Attack hyperparameters
        device: Device
        max_batches: Limit evaluation to N batches (for speed)

    Returns:
        Dict with accuracy and other metrics under attack
    """
    model.eval()
    correct = 0
    total = 0
    total_perturbation_norm = 0.0

    desc = f"{attack_name.upper()} attack"
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=desc)):
        if max_batches and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples
        if attack_name == "fgsm":
            adv_images = fgsm_attack(
                model, images, labels, epsilon=attack_params["epsilon"], device=device
            )
        elif attack_name == "pgd":
            adv_images = pgd_attack(
                model, images, labels,
                epsilon=attack_params["epsilon"],
                alpha=attack_params["alpha"],
                steps=attack_params["steps"],
                random_start=attack_params.get("random_start", True),
                device=device,
            )
        elif attack_name == "deepfool":
            adv_images, norms = deepfool_attack(
                model, images,
                max_iterations=attack_params.get("max_iterations", 50),
                overshoot=attack_params.get("overshoot", 0.02),
                device=device,
            )
            total_perturbation_norm += norms.sum().item()
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    result = {
        f"{attack_name}_accuracy": accuracy,
        f"{attack_name}_correct": correct,
        f"{attack_name}_total": total,
    }

    if attack_name == "deepfool" and total > 0:
        result["deepfool_avg_perturbation_norm"] = total_perturbation_norm / total

    return result


def full_evaluation(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_configs: dict,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run full evaluation: clean + all attacks."""
    results = evaluate_clean(model, test_loader, device)

    for attack_name, attack_params in attack_configs.items():
        attack_results = evaluate_attack(
            model, test_loader, attack_name, attack_params, device, max_batches
        )
        results.update(attack_results)

    return results
