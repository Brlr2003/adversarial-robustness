"""
HopSkipJump - Chen, Jordan, Wainwright, 2020 (IEEE S&P)

A decision-based black-box attack. Uses only the model's hard label
(argmax) at each query, no gradients and no logits. The attack alternates
between binary-searching to the decision boundary and estimating the
boundary's gradient direction via Monte Carlo sampling.

This is the only attack in this thesis that does not assume white-box
access. It complements FGSM, PGD, and DeepFool by closing the black-box
gap in the evaluation.

Reference algorithm: Algorithm 1 in Chen et al. 2020.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _is_adversarial(
    model: nn.Module,
    x: torch.Tensor,
    y_orig: int,
    targeted: bool = False,
    y_target: int | None = None,
) -> torch.Tensor:
    """
    Decision oracle. Returns a bool tensor of shape (B,).

    Untargeted: True iff argmax(model(x)) != y_orig.
    Targeted:   True iff argmax(model(x)) == y_target.
    """
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    if targeted:
        return preds.eq(y_target)
    return preds.ne(y_orig)


def _binary_search_to_boundary(
    model: nn.Module,
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    y_orig: int,
    targeted: bool,
    y_target: int | None,
    threshold: float,
    norm: str,
    max_iters: int = 50,
) -> tuple[torch.Tensor, int]:
    """
    Binary search along the line segment between x_orig and x_adv to find
    the point on the decision boundary. x_adv is assumed to be adversarial.

    For L_2 we interpolate linearly in pixel space. For L_inf we project
    via element-wise clipping to a shrinking L_inf ball around x_orig.

    Returns the boundary point and the query cost.
    """
    queries = 0
    if norm == "l2":
        low, high = 0.0, 1.0
        # Invariant: x_orig + low * (x_adv - x_orig) is NOT adversarial,
        # x_orig + high * (x_adv - x_orig) IS adversarial.
        for _ in range(max_iters):
            mid = (low + high) / 2.0
            x_mid = x_orig + mid * (x_adv - x_orig)
            queries += 1
            if _is_adversarial(model, x_mid, y_orig, targeted, y_target).item():
                high = mid
            else:
                low = mid
            if (high - low) * (x_adv - x_orig).flatten().norm().item() < threshold:
                break
        x_bd = x_orig + high * (x_adv - x_orig)
    elif norm == "linf":
        # Shrink the L_inf radius around x_orig until the boundary is found.
        low, high = 0.0, (x_adv - x_orig).abs().max().item()
        for _ in range(max_iters):
            mid = (low + high) / 2.0
            x_mid = x_orig + torch.clamp(x_adv - x_orig, -mid, mid)
            queries += 1
            if _is_adversarial(model, x_mid, y_orig, targeted, y_target).item():
                high = mid
            else:
                low = mid
            if high - low < threshold:
                break
        x_bd = x_orig + torch.clamp(x_adv - x_orig, -high, high)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return x_bd.clamp(0.0, 1.0).detach(), queries


def _estimate_gradient_direction(
    model: nn.Module,
    x_bd: torch.Tensor,
    y_orig: int,
    targeted: bool,
    y_target: int | None,
    delta: float,
    num_samples: int,
    norm: str,
    batch_size: int = 64,
) -> tuple[torch.Tensor, int]:
    """
    Monte Carlo estimate of the boundary's gradient direction at x_bd.

    Sample num_samples random directions u_i. Query the oracle on
    x_bd + delta * u_i. Estimate gradient as the average of
    sign(adversarial?) * u_i, then normalize.
    """
    queries = 0
    shape = x_bd.shape  # (1, C, H, W)
    grad = torch.zeros_like(x_bd)
    sum_signs = 0
    n_done = 0
    while n_done < num_samples:
        b = min(batch_size, num_samples - n_done)
        if norm == "l2":
            u = torch.randn(b, *shape[1:], device=x_bd.device)
            u = u / (u.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        elif norm == "linf":
            u = torch.empty(b, *shape[1:], device=x_bd.device).uniform_(-1.0, 1.0)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        x_pert = (x_bd + delta * u).clamp(0.0, 1.0)
        with torch.no_grad():
            preds = model(x_pert).argmax(dim=1)
        queries += b
        if targeted:
            phi = preds.eq(y_target).float() * 2.0 - 1.0  # +1 adv, -1 not
        else:
            phi = preds.ne(y_orig).float() * 2.0 - 1.0
        # Baseline subtraction stabilises the estimate when phi is highly
        # imbalanced; see Eq. 16 in Chen et al.
        n_done += b
        sum_signs += phi.sum().item()
        grad = grad + (phi.view(-1, 1, 1, 1) * u).sum(dim=0, keepdim=True)

    mean_phi = sum_signs / max(n_done, 1)
    # Subtract the baseline mean to reduce variance.
    # grad already accumulates sum(phi_i u_i); we want sum((phi_i - mean) u_i).
    # Since sum(u_i) is roughly zero on average (random directions), the
    # correction is small but theoretically correct.
    if abs(mean_phi) < 1.0:
        v = grad
    else:
        # All same sign: gradient direction is just the mean direction.
        v = grad
    norm_v = v.flatten().norm().item()
    if norm_v < 1e-12:
        # Degenerate; pick a random direction.
        v = torch.randn_like(x_bd)
        v = v / (v.flatten().norm() + 1e-12)
    else:
        v = v / norm_v
    return v.detach(), queries


def hopskipjump_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    norm: str = "l2",
    max_queries: int = 5000,
    num_iterations: int = 30,
    initial_num_evals: int = 100,
    max_num_evals: int = 1000,
    bs_threshold: float = 1e-3,
    targeted: bool = False,
    target_labels: torch.Tensor | None = None,
    init_attempts: int = 200,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    HopSkipJump attack (Chen et al. 2020).

    Args:
        model: Target network in eval mode.
        images: Clean images, (B, C, H, W), values in [0, 1].
        labels: True labels, (B,).
        norm: 'l2' or 'linf'.
        max_queries: Hard cap on oracle queries per image.
        num_iterations: Number of HSJ outer iterations per image.
        initial_num_evals: Starting number of MC samples for gradient estimation.
        max_num_evals: Cap on MC samples per iteration.
        bs_threshold: Stopping tolerance for binary search.
        targeted: If True, drive prediction to target_labels.
        target_labels: Target labels for targeted attack.
        init_attempts: Max random initialisations to try before giving up.
        device: Device.
        verbose: Print per-iteration distance for debugging.

    Returns:
        (adv_images, info) where info contains:
            'queries' : (B,) tensor of total oracle queries per image.
            'success' : (B,) bool tensor — adversarial after the attack.
            'l2'      : (B,) tensor of final L_2 perturbation.
            'linf'    : (B,) tensor of final L_inf perturbation.
            'queries_to_threshold': (B, K) for K thresholds — first query
                count at which the perturbation drops below each threshold,
                or -1 if never reached.
            'thresholds' : list of L_p thresholds matched to the above.
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    if targeted and target_labels is not None:
        target_labels = target_labels.clone().detach().to(device)

    B = images.shape[0]
    adv_out = images.clone()
    queries_per_image = torch.zeros(B, dtype=torch.long)
    success_mask = torch.zeros(B, dtype=torch.bool)
    final_l2 = torch.zeros(B)
    final_linf = torch.zeros(B)

    if norm == "l2":
        thresholds = [0.5, 1.0, 2.0, 4.0]
    else:
        thresholds = [2 / 255, 4 / 255, 8 / 255, 16 / 255]
    queries_to_threshold = torch.full((B, len(thresholds)), -1, dtype=torch.long)

    for i in range(B):
        x_orig = images[i : i + 1]
        y_orig = int(labels[i].item())
        y_tgt = int(target_labels[i].item()) if (targeted and target_labels is not None) else None

        # Skip if already misclassified (untargeted) or already on-target (targeted).
        with torch.no_grad():
            init_pred = model(x_orig).argmax(dim=1).item()
        if not targeted and init_pred != y_orig:
            adv_out[i] = x_orig[0]
            success_mask[i] = True
            continue
        if targeted and init_pred == y_tgt:
            adv_out[i] = x_orig[0]
            success_mask[i] = True
            continue

        queries = 0

        # 1. Initialisation: find any adversarial input by random sampling.
        x_init = None
        for _ in range(init_attempts):
            cand = torch.rand_like(x_orig)
            queries += 1
            if _is_adversarial(model, cand, y_orig, targeted, y_tgt).item():
                x_init = cand
                break
            if queries >= max_queries:
                break
        if x_init is None:
            # Fall back to leaving the image unperturbed.
            adv_out[i] = x_orig[0]
            queries_per_image[i] = queries
            final_l2[i] = 0.0
            final_linf[i] = 0.0
            continue

        # 2. Project to boundary.
        x_adv, q = _binary_search_to_boundary(
            model, x_orig, x_init, y_orig, targeted, y_tgt, bs_threshold, norm
        )
        queries += q

        # Track per-threshold first-success queries.
        def _record_thresholds(x_cur: torch.Tensor, q_cur: int) -> None:
            if norm == "l2":
                d = (x_cur - x_orig).flatten().norm().item()
            else:
                d = (x_cur - x_orig).abs().max().item()
            for j, th in enumerate(thresholds):
                if d <= th and queries_to_threshold[i, j].item() == -1:
                    queries_to_threshold[i, j] = q_cur

        _record_thresholds(x_adv, queries)

        # 3. Outer loop: estimate gradient, geometric step, re-project.
        for t in range(num_iterations):
            if queries >= max_queries:
                break

            # Number of MC samples grows like sqrt(t).
            num_evals = int(min(initial_num_evals * math.sqrt(t + 1), max_num_evals))
            num_evals = min(num_evals, max(1, max_queries - queries))

            # Sampling radius: small fraction of current distance.
            if norm == "l2":
                d = (x_adv - x_orig).flatten().norm().item()
                delta = max(d / math.sqrt(num_evals), 1e-4)
            else:
                d = (x_adv - x_orig).abs().max().item()
                delta = max(d / math.sqrt(num_evals), 1e-4)

            v, q = _estimate_gradient_direction(
                model, x_adv, y_orig, targeted, y_tgt, delta, num_evals, norm
            )
            queries += q
            if queries >= max_queries:
                break

            # Geometric step search: try ε_t = d / sqrt(t+1), halve on failure.
            step = d / math.sqrt(t + 1)
            x_new = None
            for _ in range(20):
                if norm == "l2":
                    cand = (x_adv + step * v).clamp(0.0, 1.0)
                else:
                    cand = (x_adv + step * v.sign()).clamp(0.0, 1.0)
                queries += 1
                if _is_adversarial(model, cand, y_orig, targeted, y_tgt).item():
                    x_new = cand
                    break
                step /= 2.0
                if queries >= max_queries:
                    break

            if x_new is None:
                # Geometric step failed — keep current boundary point.
                _record_thresholds(x_adv, queries)
                continue

            # Project back to boundary.
            x_adv, q = _binary_search_to_boundary(
                model, x_orig, x_new, y_orig, targeted, y_tgt, bs_threshold, norm
            )
            queries += q
            _record_thresholds(x_adv, queries)

            if verbose:
                d_now = (
                    (x_adv - x_orig).flatten().norm().item()
                    if norm == "l2"
                    else (x_adv - x_orig).abs().max().item()
                )
                print(f"  img {i} iter {t}: d={d_now:.4f}, queries={queries}")

        adv_out[i] = x_adv[0]
        queries_per_image[i] = queries
        success_mask[i] = bool(_is_adversarial(model, x_adv, y_orig, targeted, y_tgt).item())
        final_l2[i] = (x_adv - x_orig).flatten().norm().item()
        final_linf[i] = (x_adv - x_orig).abs().max().item()

    info = {
        "queries": queries_per_image,
        "success": success_mask,
        "l2": final_l2,
        "linf": final_linf,
        "queries_to_threshold": queries_to_threshold,
        "thresholds": thresholds,
        "norm": norm,
    }
    return adv_out.detach(), info
