"""PyTorch Dataset wrapper for processed crop disease image records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

from ..core.records import ImageRecord


class CropDiseaseDataset(Dataset):
    """PyTorch Dataset for ImageRecord items."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        image_path = Path(record.source_path)

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = record.class_index if record.class_index is not None else 0
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
