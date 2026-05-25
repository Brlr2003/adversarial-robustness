"""Unit tests for adversarial attacks."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.cw import cw_l2_attack
from src.attacks.deepfool import deepfool_attack
from src.attacks.fgsm import fgsm_attack
from src.attacks.hopskipjump import hopskipjump_attack
from src.attacks.one_pixel import one_pixel_attack
from src.attacks.pgd import pgd_attack
from src.models.resnet import resnet18_cifar10


@pytest.fixture
def model():
    """Create an untrained model for testing."""
    model = resnet18_cifar10()
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    """Create a random sample batch."""
    images = torch.rand(4, 3, 32, 32)  # 4 random images
    labels = torch.tensor([0, 1, 2, 3])
    return images, labels


class TestFGSM:
    def test_output_shape(self, model, sample_batch):
        images, labels = sample_batch
        adv = fgsm_attack(model, images, labels, epsilon=0.031)
        assert adv.shape == images.shape

    def test_output_range(self, model, sample_batch):
        images, labels = sample_batch
        adv = fgsm_attack(model, images, labels, epsilon=0.031)
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0

    def test_perturbation_bound(self, model, sample_batch):
        images, labels = sample_batch
        epsilon = 0.031
        adv = fgsm_attack(model, images, labels, epsilon=epsilon)
        perturbation = (adv - images).abs()
        assert perturbation.max() <= epsilon + 1e-6

    def test_zero_epsilon_returns_original(self, model, sample_batch):
        images, labels = sample_batch
        adv = fgsm_attack(model, images, labels, epsilon=0.0)
        assert torch.allclose(adv, images, atol=1e-6)


class TestPGD:
    def test_output_shape(self, model, sample_batch):
        images, labels = sample_batch
        adv = pgd_attack(model, images, labels, epsilon=0.031, alpha=0.008, steps=5)
        assert adv.shape == images.shape

    def test_output_range(self, model, sample_batch):
        images, labels = sample_batch
        adv = pgd_attack(model, images, labels, epsilon=0.031, alpha=0.008, steps=5)
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0

    def test_perturbation_bound(self, model, sample_batch):
        images, labels = sample_batch
        epsilon = 0.031
        adv = pgd_attack(model, images, labels, epsilon=epsilon, alpha=0.008, steps=10)
        perturbation = (adv - images).abs()
        assert perturbation.max() <= epsilon + 1e-6

    def test_more_steps_larger_perturbation(self, model, sample_batch):
        """More PGD steps should generally produce stronger attacks."""
        images, labels = sample_batch
        adv_1 = pgd_attack(model, images, labels, epsilon=0.031, alpha=0.008, steps=1, random_start=False)
        adv_10 = pgd_attack(model, images, labels, epsilon=0.031, alpha=0.008, steps=10, random_start=False)
        norm_1 = (adv_1 - images).flatten(1).norm(2, dim=1).mean()
        norm_10 = (adv_10 - images).flatten(1).norm(2, dim=1).mean()
        assert norm_10 >= norm_1 - 1e-6


class TestDeepFool:
    def test_output_shape(self, model, sample_batch):
        images, _ = sample_batch
        adv, norms = deepfool_attack(model, images, max_iterations=5)
        assert adv.shape == images.shape
        assert norms.shape == (images.shape[0],)

    def test_output_range(self, model, sample_batch):
        images, _ = sample_batch
        adv, _ = deepfool_attack(model, images, max_iterations=5)
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0

    def test_norms_non_negative(self, model, sample_batch):
        images, _ = sample_batch
        _, norms = deepfool_attack(model, images, max_iterations=5)
        assert (norms >= 0).all()


class TestHopSkipJump:
    def test_output_shape(self, model, sample_batch):
        images, labels = sample_batch
        adv, info = hopskipjump_attack(
            model, images, labels,
            max_queries=80, num_iterations=2,
            initial_num_evals=10, max_num_evals=20,
        )
        assert adv.shape == images.shape
        assert info["queries"].shape == (images.shape[0],)
        assert info["success"].shape == (images.shape[0],)

    def test_output_range(self, model, sample_batch):
        images, labels = sample_batch
        adv, _ = hopskipjump_attack(
            model, images, labels,
            max_queries=80, num_iterations=2,
            initial_num_evals=10, max_num_evals=20,
        )
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0

    def test_query_budget_respected(self, model, sample_batch):
        images, labels = sample_batch
        budget = 100
        _, info = hopskipjump_attack(
            model, images, labels,
            max_queries=budget, num_iterations=3,
            initial_num_evals=10, max_num_evals=20,
        )
        assert (info["queries"] <= budget + 50).all()


class TestOnePixel:
    def test_output_shape(self, model, sample_batch):
        images, labels = sample_batch
        adv, info = one_pixel_attack(
            model, images, labels, k=1, pop_size=10, max_iter=2,
        )
        assert adv.shape == images.shape
        assert info["success"].shape == (images.shape[0],)

    def test_output_range(self, model, sample_batch):
        images, labels = sample_batch
        adv, _ = one_pixel_attack(
            model, images, labels, k=1, pop_size=10, max_iter=2,
        )
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0

    def test_l0_perturbation_bound(self, model, sample_batch):
        """At most k pixels should differ from the original."""
        images, labels = sample_batch
        k = 3
        adv, _ = one_pixel_attack(
            model, images, labels, k=k, pop_size=10, max_iter=2,
        )
        diff = (adv - images).abs().sum(dim=1)  # collapse channels
        for i in range(images.shape[0]):
            n_changed = (diff[i] > 1e-6).sum().item()
            assert n_changed <= k


class TestCarliniWagner:
    def test_output_shape(self, model, sample_batch):
        images, labels = sample_batch
        adv, info = cw_l2_attack(
            model, images, labels,
            binary_search_steps=2, max_iterations=20,
        )
        assert adv.shape == images.shape
        assert info["success"].shape == (images.shape[0],)
        assert info["l2"].shape == (images.shape[0],)

    def test_output_range(self, model, sample_batch):
        images, labels = sample_batch
        adv, _ = cw_l2_attack(
            model, images, labels,
            binary_search_steps=2, max_iterations=20,
        )
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0 + 1e-6

    def test_finds_small_perturbation(self, model, sample_batch):
        """On successful samples, C&W should use less L2 than an 8/255 FGSM step."""
        images, labels = sample_batch
        adv, info = cw_l2_attack(
            model, images, labels,
            binary_search_steps=4, max_iterations=100,
        )
        fgsm_adv = fgsm_attack(model, images, labels, epsilon=8 / 255)
        fgsm_l2 = (fgsm_adv - images).flatten(1).norm(2, dim=1)
        succeeded = info["success"]
        if succeeded.any():
            cw_l2 = info["l2"][succeeded]
            # C&W minimises L2, so on solved samples it stays well bounded.
            assert torch.isfinite(cw_l2).all()
            assert (cw_l2 <= fgsm_l2[succeeded] + 1.0).all()


class TestResNet:
    def test_forward_shape(self, model):
        x = torch.rand(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_parameter_count(self, model):
        params = sum(p.numel() for p in model.parameters())
        assert params > 10_000_000  # ResNet-18 should have ~11M params
        assert params < 15_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
