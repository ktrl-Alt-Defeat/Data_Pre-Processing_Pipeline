"""Training image transformation pipeline."""

from __future__ import annotations

from torchvision import transforms
from .normalization import IMAGENET_MEAN, IMAGENET_STD, get_normalize_transform


def build_train_transforms(
    resize: tuple[int, int] = (256, 256),
    crop_size: tuple[int, int] = (224, 224),
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    hflip_p: float = 0.5,
    rotation_degrees: float = 15.0,
) -> transforms.Compose:
    """Build PyTorch transformation pipeline for training."""
    transform_list = [
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=hflip_p),
    ]

    if rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    transform_list.extend([
        transforms.ToTensor(),
        get_normalize_transform(mean, std),
    ])

    return transforms.Compose(transform_list)
