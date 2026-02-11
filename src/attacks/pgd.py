"""
Projected Gradient Descent (PGD) - Madry et al., 2018

An iterative version of FGSM. Takes multiple small steps in the gradient
direction and projects back into the ε-ball around the original input.

This is considered the strongest first-order attack and the basis for
adversarial training.
"""

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Generate adversarial examples using PGD (L∞ threat model).

    Args:
        model: Target neural network
        images: Clean input images, shape (B, C, H, W), values in [0, 1]
        labels: True labels, shape (B,)
        epsilon: Maximum perturbation magnitude (L∞ bound)
        alpha: Step size per iteration
        steps: Number of PGD iterations
        random_start: Whether to start from a random point in the ε-ball
        device: Device to run on

    Returns:
        Adversarial images, shape (B, C, H, W), values in [0, 1]
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Initialize adversarial images
    adv_images = images.clone().detach()

    if random_start:
        # Start from a random point in the ε-ball
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = adv_images.clamp(0, 1).detach()

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv_images.requires_grad = True

        # Forward pass
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Take step in gradient direction
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project back into ε-ball around original image
        perturbation = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = (images + perturbation).clamp(0, 1).detach()

    return adv_images
