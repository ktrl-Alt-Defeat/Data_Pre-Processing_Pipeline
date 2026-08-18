"""Inference image transformation pipeline."""

from __future__ import annotations

from torchvision import transforms
from .normalization import IMAGENET_MEAN, IMAGENET_STD, get_normalize_transform


def build_inference_transforms(
    resize: tuple[int, int] = (256, 256),
    crop_size: tuple[int, int] = (224, 224),
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> transforms.Compose:
    """Build PyTorch transformation pipeline for inference."""
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        get_normalize_transform(mean, std),
    ])
