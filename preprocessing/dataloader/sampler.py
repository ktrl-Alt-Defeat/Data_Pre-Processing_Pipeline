"""Custom samplers for class imbalance handling."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from ..core.records import ImageRecord


def build_weighted_sampler(records: Sequence[ImageRecord]) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler to balance class sampling."""
    class_counts: dict[int, int] = {}
    for rec in records:
        if rec.class_index is not None:
            class_counts[rec.class_index] = class_counts.get(rec.class_index, 0) + 1

    weights: list[float] = []
    for rec in records:
        idx = rec.class_index if rec.class_index is not None else 0
        count = class_counts.get(idx, 1)
        weights.append(1.0 / float(count))

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights), replacement=True)
