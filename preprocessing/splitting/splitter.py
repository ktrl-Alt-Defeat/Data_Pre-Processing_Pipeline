"""Dataset splitting module with stratified and group-stratified splitting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.context import RunContext
from ..core.logging import StageTracker, stage_scope
from ..core.records import Corpus, ImageRecord, PipelineStage, Split


@dataclass
class SplitResult:
    """Result of dataset splitting stage."""

    splits: dict[Split, list[ImageRecord]] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def count_for(self, split: Split) -> int:
        return len(self.splits.get(split, []))

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": {s.value: len(recs) for s, recs in self.splits.items()},
            "statistics": self.statistics,
        }


def split_corpus(config: Config, context: RunContext, corpus: Corpus) -> SplitResult:
    """Split corpus accepted images into train, val, test splits."""
    with stage_scope(PipelineStage.SPLITTING, context.logger(PipelineStage.SPLITTING)) as tracker:
        accepted_records = corpus.accepted()
        if not accepted_records:
            tracker.warn("splitting.empty_corpus", message="No accepted records to split")
            return SplitResult()

        strategy = config.split.strategy
        ratios = config.split.ratios
        seed = config.split.seed or config.seed

        train_ratio = ratios.train
        val_ratio = ratios.val
        test_ratio = ratios.test

        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        # Group records by class
        class_records: dict[str, list[ImageRecord]] = {}
        for rec in accepted_records:
            class_records.setdefault(rec.label, []).append(rec)

        split_dict: dict[Split, list[ImageRecord]] = {
            Split.TRAIN: [],
            Split.VAL: [],
            Split.TEST: [],
        }

        # Perform stratified / group-stratified splitting per class
        rng = np.random.default_rng(seed)

        for label, records in class_records.items():
            if len(records) < config.split.min_samples_per_class:
                if config.split.drop_classes_below_minimum:
                    tracker.warn("splitting.class_dropped", label=label, count=len(records))
                    continue

            # Check if group splitting is required
            if strategy in ("stratified_group", "group") and config.split.group_by == "duplicate_group":
                groups: dict[str, list[ImageRecord]] = {}
                for rec in records:
                    grp_key = rec.duplicate_group or rec.image_id
                    groups.setdefault(grp_key, []).append(rec)

                group_keys = list(groups.keys())
                rng.shuffle(group_keys)

                n_groups = len(group_keys)
                n_train = max(1, int(round(n_groups * train_ratio)))
                n_val = int(round(n_groups * val_ratio))
                if n_train + n_val >= n_groups and n_groups > 2:
                    n_val = max(1, n_groups - n_train - 1)

                train_groups = group_keys[:n_train]
                val_groups = group_keys[n_train : n_train + n_val]
                test_groups = group_keys[n_train + n_val :]

                for g in train_groups:
                    for rec in groups[g]:
                        rec.split = Split.TRAIN
                        split_dict[Split.TRAIN].append(rec)
                for g in val_groups:
                    for rec in groups[g]:
                        rec.split = Split.VAL
                        split_dict[Split.VAL].append(rec)
                for g in test_groups:
                    for rec in groups[g]:
                        rec.split = Split.TEST
                        split_dict[Split.TEST].append(rec)
            else:
                # Standard stratified split
                recs = list(records)
                rng.shuffle(recs)

                n = len(recs)
                n_train = max(1, int(round(n * train_ratio)))
                n_val = int(round(n * val_ratio))
                if n_train + n_val >= n and n > 2:
                    n_val = max(1, n - n_train - 1)

                train_recs = recs[:n_train]
                val_recs = recs[n_train : n_train + n_val]
                test_recs = recs[n_train + n_val :]

                for rec in train_recs:
                    rec.split = Split.TRAIN
                    split_dict[Split.TRAIN].append(rec)
                for rec in val_recs:
                    rec.split = Split.VAL
                    split_dict[Split.VAL].append(rec)
                for rec in test_recs:
                    rec.split = Split.TEST
                    split_dict[Split.TEST].append(rec)

        stats = {
            "train_count": len(split_dict[Split.TRAIN]),
            "val_count": len(split_dict[Split.VAL]),
            "test_count": len(split_dict[Split.TEST]),
            "total_split": sum(len(v) for v in split_dict.values()),
            "strategy": strategy,
        }

        # Save split manifest
        split_json = context.layout.metadata_dir / "splits.json"
        split_json.parent.mkdir(parents=True, exist_ok=True)
        split_manifest = {
            s.value: [rec.image_id for rec in recs] for s, recs in split_dict.items()
        }
        split_json.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

        tracker.metrics(**stats)
        tracker.info("splitting.completed", **stats)

        return SplitResult(splits=split_dict, statistics=stats)
