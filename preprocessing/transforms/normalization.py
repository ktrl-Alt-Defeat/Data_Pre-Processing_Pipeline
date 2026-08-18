"""Normalization parameters and torchvision transform builders."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalize_transform(mean: tuple[float, ...] = IMAGENET_MEAN, std: tuple[float, ...] = IMAGENET_STD) -> transforms.Normalize:
    return transforms.Normalize(mean=list(mean), std=list(std))
