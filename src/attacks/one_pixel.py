"""
One Pixel Attack - Su, Vargas, Sakurai, 2019 (IEEE TEVC)

An L_0-bounded attack: change at most k pixels of the input. Differential
evolution searches over (x, y, r, g, b) tuples. Each candidate solution
specifies k pixels and their new values; fitness is the model's confidence
on the true class (lower is better, untargeted).

This attack is interesting because it does not need gradients, but unlike
HopSkipJump it does need score-level access (probabilities) to drive the
DE fitness function. It is the cheapest possible black-box probe and gives
a very different threat model from PGD: only a handful of pixels change,
but those pixels can change arbitrarily within [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_perturbation(image: torch.Tensor, candidate: np.ndarray) -> torch.Tensor:
    """
    Apply a One-Pixel candidate to a single image.

    image: (C, H, W), values in [0, 1].
    candidate: flat array of length 5k. Each block of 5 is (x, y, r, g, b).
    """
    perturbed = image.clone()
    H, W = image.shape[-2:]
    k = candidate.size // 5
    coords = candidate.reshape(k, 5)
    for x, y, r, g, b in coords:
        xi = int(np.clip(x, 0, W - 1))
        yi = int(np.clip(y, 0, H - 1))
        perturbed[0, yi, xi] = float(np.clip(r, 0.0, 1.0))
        perturbed[1, yi, xi] = float(np.clip(g, 0.0, 1.0))
        perturbed[2, yi, xi] = float(np.clip(b, 0.0, 1.0))
    return perturbed


def _evaluate_population(
    model: nn.Module,
    image: torch.Tensor,
    label: int,
    population: np.ndarray,
    targeted: bool,
    target: int | None,
    device: torch.device | str,
) -> np.ndarray:
    """
    Evaluate a whole DE population in one batched forward pass.

    Returns the fitness for each candidate. Untargeted: true-class probability
    (lower = better). Targeted: 1 - target-class probability (lower = better).
    """
    pop_size = population.shape[0]
    perturbed_batch = torch.stack(
        [_apply_perturbation(image, population[i]) for i in range(pop_size)], dim=0
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(model(perturbed_batch), dim=1)
    if targeted:
        return (1.0 - probs[:, target]).cpu().numpy()
    return probs[:, label].cpu().numpy()


def _differential_evolution_one_pixel(
    model: nn.Module,
    image: torch.Tensor,
    label: int,
    k: int,
    pop_size: int,
    max_iter: int,
    F_param: float,
    CR: float,
    targeted: bool,
    target: int | None,
    device: torch.device | str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int, bool]:
    """
    Run DE/rand/1/bin to minimise the fitness function.

    Returns (best_candidate, best_fitness, queries, succeeded).
    Early stops when the candidate fools the model.
    """
    H, W = image.shape[-2:]
    queries = 0

    # Init population uniformly.
    bounds_low = np.tile([0, 0, 0.0, 0.0, 0.0], k)
    bounds_high = np.tile([W - 1, H - 1, 1.0, 1.0, 1.0], k)
    population = rng.uniform(bounds_low, bounds_high, size=(pop_size, 5 * k))

    fitness = _evaluate_population(model, image, label, population, targeted, target, device)
    queries += pop_size

    best_idx = int(np.argmin(fitness))
    best = population[best_idx].copy()
    best_fit = float(fitness[best_idx])

    # Check the initial population for an immediate success.
    succeeded = _is_misclassified(
        model, image, label, best, targeted, target, device
    )
    queries += 1
    if succeeded:
        return best, best_fit, queries, True

    for _ in range(max_iter):
        # Generate trial population: DE/rand/1/bin.
        idxs = rng.integers(0, pop_size, size=(pop_size, 3))
        a = population[idxs[:, 0]]
        b = population[idxs[:, 1]]
        c = population[idxs[:, 2]]
        mutants = a + F_param * (b - c)
        # Crossover.
        cross = rng.random(population.shape) < CR
        # Ensure at least one dim from mutant.
        force = rng.integers(0, population.shape[1], size=pop_size)
        cross[np.arange(pop_size), force] = True
        trials = np.where(cross, mutants, population)
        # Clip to bounds.
        trials = np.clip(trials, bounds_low, bounds_high)

        trial_fitness = _evaluate_population(
            model, image, label, trials, targeted, target, device
        )
        queries += pop_size

        # Selection.
        improved = trial_fitness < fitness
        population = np.where(improved[:, None], trials, population)
        fitness = np.where(improved, trial_fitness, fitness)

        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_fit:
            best_fit = float(fitness[cur_best_idx])
            best = population[cur_best_idx].copy()

        # Early stop on success.
        if _is_misclassified(model, image, label, best, targeted, target, device):
            queries += 1
            return best, best_fit, queries, True
        queries += 1

    return best, best_fit, queries, False


def _is_misclassified(
    model: nn.Module,
    image: torch.Tensor,
    label: int,
    candidate: np.ndarray,
    targeted: bool,
    target: int | None,
    device: torch.device | str,
) -> bool:
    """One forward pass to check whether the candidate succeeds."""
    perturbed = _apply_perturbation(image, candidate).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(model(perturbed).argmax(dim=1).item())
    if targeted:
        return pred == target
    return pred != label


def one_pixel_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
    pop_size: int = 400,
    max_iter: int = 75,
    F_param: float = 0.5,
    CR: float = 0.7,
    targeted: bool = False,
    target_labels: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = 0,
) -> tuple[torch.Tensor, dict]:
    """
    One Pixel Attack via differential evolution.

    Args:
        model: Target network in eval mode.
        images: (B, C, H, W) in [0, 1].
        labels: (B,) ground-truth labels.
        k: Number of pixels to perturb (1, 3, 5 in the original paper).
        pop_size: DE population size.
        max_iter: DE generations.
        F_param: DE differential weight.
        CR: DE crossover rate.
        targeted: If True, drive predictions to target_labels.
        target_labels: (B,) target labels for targeted attack.
        device: Device.
        seed: Optional RNG seed for reproducibility.

    Returns:
        (adv_images, info) where info contains:
            'success': (B,) bool tensor.
            'queries': (B,) int tensor.
            'fitness': (B,) float tensor (best-class probability seen).
            'k': int (pixels modified).
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    if targeted and target_labels is not None:
        target_labels = target_labels.clone().detach().to(device)

    B = images.shape[0]
    adv_out = images.clone()
    success_mask = torch.zeros(B, dtype=torch.bool)
    query_counts = torch.zeros(B, dtype=torch.long)
    fitness_out = torch.zeros(B)

    rng = np.random.default_rng(seed)

    for i in range(B):
        x = images[i].cpu()  # DE works on CPU; only forward passes go to device.
        y = int(labels[i].item())
        y_tgt = (
            int(target_labels[i].item()) if (targeted and target_labels is not None) else None
        )

        # Skip if already misclassified (untargeted) / already on-target (targeted).
        with torch.no_grad():
            init_pred = int(model(images[i : i + 1]).argmax(dim=1).item())
        if (not targeted and init_pred != y) or (targeted and init_pred == y_tgt):
            adv_out[i] = images[i]
            success_mask[i] = True
            continue

        best, fit, queries, ok = _differential_evolution_one_pixel(
            model, x, y, k, pop_size, max_iter, F_param, CR,
            targeted, y_tgt, device, rng,
        )
        adv_out[i] = _apply_perturbation(x, best).to(device)
        success_mask[i] = ok
        query_counts[i] = queries
        fitness_out[i] = fit

    info = {
        "success": success_mask,
        "queries": query_counts,
        "fitness": fitness_out,
        "k": k,
    }
    return adv_out.detach(), info
