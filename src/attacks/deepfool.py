"""
DeepFool - Moosavi-Dezfooli et al., 2016

An iterative attack that finds the minimum perturbation needed to change
the model's prediction. Unlike FGSM/PGD which use a fixed ε budget,
DeepFool computes the closest decision boundary and pushes the input
just past it.

This gives a measure of the model's "robustness radius" per sample.
"""

import torch
import torch.nn as nn


def deepfool_attack(
    model: nn.Module,
    images: torch.Tensor,
    max_iterations: int = 50,
    overshoot: float = 0.02,
    num_classes: int = 10,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples using DeepFool.

    Args:
        model: Target neural network
        images: Clean input images, shape (B, C, H, W), values in [0, 1]
        max_iterations: Maximum number of iterations
        overshoot: Overshoot parameter to ensure misclassification
        num_classes: Number of classes
        device: Device to run on

    Returns:
        Tuple of (adversarial_images, perturbation_norms)
    """
    model.eval()
    images = images.clone().detach().to(device)
    batch_size = images.shape[0]

    adv_images = images.clone()
    perturbation_norms = torch.zeros(batch_size, device=device)

    # Process each image individually (DeepFool is inherently per-sample)
    for idx in range(batch_size):
        x = images[idx : idx + 1].clone().detach().requires_grad_(True)
        original_output = model(x)
        original_class = original_output.argmax(dim=1).item()

        total_perturbation = torch.zeros_like(x)

        for _ in range(max_iterations):
            x_pert = (images[idx : idx + 1] + total_perturbation).requires_grad_(True)
            output = model(x_pert)

            current_class = output.argmax(dim=1).item()
            if current_class != original_class:
                break

            # Compute gradients for all classes
            gradients = []
            for k in range(num_classes):
                if x_pert.grad is not None:
                    x_pert.grad.zero_()
                output[0, k].backward(retain_graph=True)
                gradients.append(x_pert.grad.clone())

            # Find closest decision boundary
            f_original = output[0, original_class]
            min_dist = float("inf")
            best_perturbation = None

            for k in range(num_classes):
                if k == original_class:
                    continue

                # Difference in logits and gradients
                f_k = output[0, k]
                w_k = gradients[k] - gradients[original_class]

                f_diff = (f_k - f_original).abs().item()
                w_norm = w_k.flatten().norm().item()

                if w_norm == 0:
                    continue

                dist = f_diff / w_norm

                if dist < min_dist:
                    min_dist = dist
                    best_perturbation = (f_diff / (w_norm**2 + 1e-8)) * w_k

            if best_perturbation is None:
                break

            total_perturbation += best_perturbation.detach()

        # Apply overshoot
        total_perturbation = (1 + overshoot) * total_perturbation
        adv_images[idx] = (images[idx] + total_perturbation.squeeze(0)).clamp(0, 1)
        perturbation_norms[idx] = total_perturbation.flatten().norm()

    return adv_images.detach(), perturbation_norms.detach()
