"""
Carlini & Wagner L_2 attack - Carlini and Wagner, 2017 (IEEE S&P)

A strong optimisation-based white-box attack. Instead of working within a
fixed epsilon budget like FGSM/PGD, C&W searches for the smallest L_2
perturbation that changes the prediction. It rests on three ideas:

  1. Change of variables. Rather than constraining x_adv to [0, 1] with
     clipping, optimise over w with x_adv = 1/2 (tanh(w) + 1). The box
     constraint then holds automatically and the optimiser is unconstrained.
  2. A margin loss f(x') = max(Z(x')_y - max_{i != y} Z(x')_i, -kappa) that
     reaches its floor once the example is misclassified by margin kappa, so
     the optimiser stops pushing the logits and spends the rest of its
     budget shrinking the perturbation.
  3. Binary search over the constant c that trades the perturbation size
     against the margin loss.

This is the canonical strong L_2 attack and the standard against which
defences are measured. It complements DeepFool (also L_2-minimising but
greedier and weaker) and PGD (L_inf, fixed budget).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _to_tanh_space(x: torch.Tensor) -> torch.Tensor:
    """Map x in [0, 1] to w in R via the inverse of 1/2 (tanh(w) + 1)."""
    x = (x * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6)
    return 0.5 * torch.log((1 + x) / (1 - x))  # arctanh


def _from_tanh_space(w: torch.Tensor) -> torch.Tensor:
    """Map w in R back to [0, 1]."""
    return 0.5 * (torch.tanh(w) + 1)


def cw_l2_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    confidence: float = 0.0,
    binary_search_steps: int = 6,
    max_iterations: int = 200,
    learning_rate: float = 0.01,
    initial_const: float = 1e-2,
    targeted: bool = False,
    target_labels: torch.Tensor | None = None,
    abort_early: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """
    Carlini & Wagner L_2 attack.

    Args:
        model: Target network in eval mode.
        images: (B, C, H, W) in [0, 1].
        labels: (B,) ground-truth labels.
        confidence: kappa; larger values produce higher-confidence adversarial
            examples at the cost of a larger perturbation.
        binary_search_steps: outer binary-search steps over the constant c.
        max_iterations: Adam steps per binary-search step.
        learning_rate: Adam learning rate.
        initial_const: starting value of c.
        targeted: if True, drive predictions to target_labels.
        target_labels: (B,) target labels for a targeted attack.
        abort_early: stop a search step once the loss plateaus.
        device: device.

    Returns:
        (adv_images, info) where info contains:
            'success': (B,) bool tensor.
            'l2':      (B,) float tensor, final L_2 perturbation (inf if failed).
            'linf':    (B,) float tensor.
            'const':   (B,) float tensor, the final constant c per sample.
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    B = images.shape[0]
    if targeted and target_labels is not None:
        target_labels = target_labels.clone().detach().to(device)

    with torch.no_grad():
        num_classes = model(images[:1]).shape[1]

    # One-hot of the class to suppress (untargeted) or promote (targeted).
    focus = target_labels if targeted else labels
    one_hot = torch.zeros(B, num_classes, device=device)
    one_hot[torch.arange(B, device=device), focus] = 1.0

    lower_bound = torch.zeros(B, device=device)
    const = torch.full((B,), initial_const, device=device)
    upper_bound = torch.full((B,), 1e10, device=device)

    # Best adversarial example found across all binary-search steps.
    best_l2 = torch.full((B,), float("inf"), device=device)
    best_adv = images.clone()
    best_success = torch.zeros(B, dtype=torch.bool, device=device)

    w_init = _to_tanh_space(images)

    for _ in range(binary_search_steps):
        w = w_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        prev_loss = float("inf")
        step_success = torch.zeros(B, dtype=torch.bool, device=device)

        for it in range(max_iterations):
            adv = _from_tanh_space(w)
            l2_sq = (adv - images).flatten(1).pow(2).sum(dim=1)

            logits = model(adv)
            real = (one_hot * logits).sum(dim=1)
            other = ((1 - one_hot) * logits - one_hot * 1e4).max(dim=1)[0]

            if targeted:
                f = torch.clamp(other - real, min=-confidence)
            else:
                f = torch.clamp(real - other, min=-confidence)

            loss = (l2_sq + const * f).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                is_adv = (pred == target_labels) if targeted else (pred != labels)
                l2 = l2_sq.sqrt()
                step_success |= is_adv
                improve = is_adv & (l2 < best_l2)
                best_l2 = torch.where(improve, l2, best_l2)
                best_adv = torch.where(improve.view(-1, 1, 1, 1), adv.detach(), best_adv)
                best_success |= is_adv

            # Abort the step early if the loss has stopped improving.
            if abort_early and it % max(1, max_iterations // 10) == 0:
                if loss.item() > prev_loss * 0.9999:
                    break
                prev_loss = loss.item()

        # Binary search on c: shrink for solved samples, grow for the rest.
        with torch.no_grad():
            found = step_success
            upper_bound = torch.where(found, torch.minimum(upper_bound, const), upper_bound)
            lower_bound = torch.where(~found, torch.maximum(lower_bound, const), lower_bound)
            has_upper = upper_bound < 1e9
            midpoint = (lower_bound + upper_bound) / 2
            const = torch.where(has_upper, midpoint, const * 10)

    delta = (best_adv - images).flatten(1)
    linf = delta.abs().max(dim=1)[0]

    info = {
        "success": best_success.cpu(),
        "l2": best_l2.cpu(),
        "linf": linf.cpu(),
        "const": const.cpu(),
    }
    return best_adv.detach(), info
