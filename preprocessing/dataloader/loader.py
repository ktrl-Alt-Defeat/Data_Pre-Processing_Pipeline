"""PyTorch DataLoader builder and dataset stage runner."""

from __future__ import annotations

from typing import Any, Sequence
from torch.utils.data import DataLoader

from ..core.config import Config
from ..core.context import RunContext
from ..core.records import ImageRecord, Split
from ..transforms import build_inference_transforms, build_train_transforms, build_validation_transforms
from .dataset import CropDiseaseDataset
from .sampler import build_weighted_sampler


def build_dataloaders(
    splits: dict[Split, list[ImageRecord]],
    config: Config,
) -> dict[Split, DataLoader]:
    """Create DataLoaders for train, val, and test splits."""
    batch_size = config.dataloader.batch_size
    num_workers = config.dataloader.num_workers
    shuffle_train = config.dataloader.shuffle_train

    dataloaders: dict[Split, DataLoader] = {}

    train_recs = splits.get(Split.TRAIN, [])
    if train_recs:
        train_ds = CropDiseaseDataset(train_recs, transform=build_train_transforms())
        sampler = build_weighted_sampler(train_recs) if config.dataloader.sampler == "weighted" else None
        dataloaders[Split.TRAIN] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(shuffle_train if sampler is None else False),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=config.dataloader.pin_memory,
            drop_last=config.dataloader.drop_last,
        )

    val_recs = splits.get(Split.VAL, [])
    if val_recs:
        val_ds = CropDiseaseDataset(val_recs, transform=build_validation_transforms())
        dataloaders[Split.VAL] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.dataloader.pin_memory,
        )

    test_recs = splits.get(Split.TEST, [])
    if test_recs:
        test_ds = CropDiseaseDataset(test_recs, transform=build_inference_transforms())
        dataloaders[Split.TEST] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.dataloader.pin_memory,
        )

    return dataloaders
