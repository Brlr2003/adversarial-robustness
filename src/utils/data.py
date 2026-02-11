"""Data loading and preprocessing utilities for CIFAR-10."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 normalization stats (mean, std per channel)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_train_transforms() -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # NOTE: We do NOT normalize for adversarial training.
        # Attacks operate in [0, 1] pixel space. Normalizing would
        # require adjusting epsilon values. Instead, we keep images
        # in [0, 1] and let the model learn without normalization.
    ])


def get_test_transforms() -> transforms.Compose:
    """Test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_train_transforms()
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [0, 1] to displayable format."""
    return tensor.clamp(0, 1)
