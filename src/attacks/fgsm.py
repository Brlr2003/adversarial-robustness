"""
Fast Gradient Sign Method (FGSM) - Goodfellow et al., 2014

The simplest adversarial attack: perturb input in the direction of the
gradient of the loss w.r.t. the input, scaled by epsilon.

    x_adv = x + ε * sign(∇_x L(θ, x, y))
"""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Generate adversarial examples using FGSM.

    Args:
        model: Target neural network (must be in eval mode)
        images: Clean input images, shape (B, C, H, W), values in [0, 1]
        labels: True labels, shape (B,)
        epsilon: Maximum perturbation magnitude
        device: Device to run on

    Returns:
        Adversarial images, shape (B, C, H, W), values in [0, 1]
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Enable gradient computation on input
    images.requires_grad = True

    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # Backward pass to get gradient w.r.t. input
    model.zero_grad()
    loss.backward()

    # Create perturbation
    perturbation = epsilon * images.grad.sign()

    # Apply perturbation and clamp to valid range
    adversarial_images = (images + perturbation).clamp(0, 1)

    return adversarial_images.detach()
