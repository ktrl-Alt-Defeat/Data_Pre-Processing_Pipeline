"""Dataset diversity analysis.

Diversity is measured on the perceptual hashes the quality stage already
computed, so no image is decoded again. Two views:

* **global** - how visually distinct the corpus is overall
* **per class** - whether individual classes are made of near-identical shots,
  which inflates accuracy without teaching the model anything

Pairwise distances over a full corpus are quadratic, so a deterministic sample
is drawn and, above a small size, random pairs are evaluated instead of all
pairs. The sample and the pairs are seeded, so results are reproducible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.config import DiversityConfig
from ..core.records import Metric, MetricStatus

_FULL_PAIRWISE_LIMIT = 1200
_SAMPLED_PAIRS = 200_000


@dataclass(frozen=True, slots=True)
class DiversityReport:
    """Visual-diversity findings."""

    sample_size: int
    hash_bits: int
    mean_distance: float
    normalized_diversity: float
    unique_hash_ratio: float
    duplicate_ratio: float
    near_duplicate_ratio: float
    pairs_evaluated: int
    least_diverse_classes: tuple[tuple[str, float], ...] = ()
    per_class_diversity: dict[str, float] = field(default_factory=dict)
    metrics: tuple[Metric, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "hash_bits": self.hash_bits,
            "mean_hamming_distance": self.mean_distance,
            "normalized_diversity": self.normalized_diversity,
            "unique_hash_ratio": self.unique_hash_ratio,
            "duplicate_ratio": self.duplicate_ratio,
            "near_duplicate_ratio": self.near_duplicate_ratio,
            "pairs_evaluated": self.pairs_evaluated,
            "least_diverse_classes": [{"label": label, "diversity": value}
                                      for label, value in self.least_diverse_classes],
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


def analyze_diversity(frame: pd.DataFrame, config: DiversityConfig) -> DiversityReport:
    """Measure visual diversity from perceptual hashes already on the manifest."""
    hashes, labels = _hash_array(frame)
    duplicate_ratio, near_ratio = _duplicate_ratios(frame)

    if not config.enabled or hashes.size == 0:
        return DiversityReport(0, config.hash_bits, 0.0, 0.0, 0.0, duplicate_ratio, near_ratio, 0)

    sample_indices = _sample_indices(hashes.size, config.sample_size, seed=hashes.size)
    sample = hashes[sample_indices]
    bits = config.hash_bits
    mean_distance, pairs = _mean_pairwise_distance(sample, bits)

    per_class = _per_class_diversity(hashes, labels, bits)
    least = tuple(sorted(per_class.items(), key=lambda item: (item[1], item[0]))[:10])

    report = DiversityReport(
        sample_size=int(sample.size),
        hash_bits=bits,
        mean_distance=round(mean_distance, 4),
        normalized_diversity=round(mean_distance / bits, 6) if bits else 0.0,
        unique_hash_ratio=round(float(np.unique(hashes).size / hashes.size), 6),
        duplicate_ratio=duplicate_ratio,
        near_duplicate_ratio=near_ratio,
        pairs_evaluated=pairs,
        least_diverse_classes=least,
        per_class_diversity={label: round(value, 6) for label, value in sorted(per_class.items())},
    )
    return dataclasses.replace(report, metrics=_metrics(report))


def _hash_array(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or "perceptual_hash" not in frame.columns:
        return np.array([], dtype=np.uint64), np.array([], dtype=object)
    subset = frame[["perceptual_hash", "label"]].dropna(subset=["perceptual_hash"])
    if subset.empty:
        return np.array([], dtype=np.uint64), np.array([], dtype=object)
    values = np.array([int(str(value), 16) for value in subset["perceptual_hash"]], dtype=np.uint64)
    return values, subset["label"].astype("string").to_numpy()


def _duplicate_ratios(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty or "quality_duplicate_status" not in frame.columns:
        return 0.0, 0.0
    statuses = frame["quality_duplicate_status"].dropna().astype("string")
    total = len(frame)
    if not total:
        return 0.0, 0.0
    exact = int((statuses == "exact_duplicate").sum())
    near = int((statuses == "near_duplicate").sum())
    return round(exact / total, 6), round(near / total, 6)


def _sample_indices(size: int, limit: int, seed: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, size=limit, replace=False))


def _popcount(values: np.ndarray) -> np.ndarray:
    """Bit counts of a uint64 array, via a uint8 view and a lookup table."""
    table = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)
    return table[values.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def _mean_pairwise_distance(sample: np.ndarray, bits: int) -> tuple[float, int]:
    size = sample.size
    if size < 2:
        return 0.0, 0

    if size <= _FULL_PAIRWISE_LIMIT:
        rows, cols = np.triu_indices(size, k=1)
        distances = _popcount(np.bitwise_xor(sample[rows], sample[cols]))
        return float(distances.mean()), int(distances.size)

    rng = np.random.default_rng(size)
    left = rng.integers(0, size, _SAMPLED_PAIRS)
    right = rng.integers(0, size, _SAMPLED_PAIRS)
    keep = left != right
    distances = _popcount(np.bitwise_xor(sample[left[keep]], sample[right[keep]]))
    return float(distances.mean()), int(distances.size)


def _per_class_diversity(hashes: np.ndarray, labels: np.ndarray, bits: int) -> dict[str, float]:
    """Normalised mean pairwise distance within each class."""
    diversity: dict[str, float] = {}
    for label in np.unique(labels):
        member_hashes = hashes[labels == label]
        if member_hashes.size < 2:
            continue
        sample = member_hashes[_sample_indices(member_hashes.size, 400, seed=member_hashes.size)]
        distance, _ = _mean_pairwise_distance(sample, bits)
        diversity[str(label)] = distance / bits if bits else 0.0
    return diversity


def _metrics(report: DiversityReport) -> tuple[Metric, ...]:
    return (
        Metric(
            key="diversity.normalized",
            name="Visual diversity",
            definition="Mean pairwise perceptual-hash distance, normalised by hash length.",
            method=f"mean Hamming distance over {report.pairs_evaluated:,} sampled pairs / {report.hash_bits} bits",
            value=report.normalized_diversity,
            threshold=0.25,
            status=_status(report.normalized_diversity >= 0.25, report.normalized_diversity >= 0.15),
            interpretation=(
                f"Images differ by {report.mean_distance:.1f} of {report.hash_bits} hash bits on average "
                f"({report.normalized_diversity:.1%} normalised)."
            ),
            recommendation=(
                "Low diversity means many images look alike; expect optimistic validation scores unless "
                "duplicate clusters are kept inside a single split."
                if report.normalized_diversity < 0.25
                else "Diversity is healthy for a benchmark corpus."
            ),
        ),
        Metric(
            key="diversity.unique_hashes",
            name="Unique perceptual hashes",
            definition="Share of images with a perceptual hash no other image shares exactly.",
            method="unique(perceptual_hash) / images",
            value=report.unique_hash_ratio,
            threshold=0.95,
            status=_status(report.unique_hash_ratio >= 0.95, report.unique_hash_ratio >= 0.85),
            interpretation=f"{report.unique_hash_ratio:.2%} of images have a distinct perceptual hash.",
            recommendation=("Investigate the repeated hashes; they are usually re-uploads of the same photograph."
                            if report.unique_hash_ratio < 0.95 else "No action needed."),
        ),
        Metric(
            key="diversity.duplicate_ratio",
            name="Duplicate ratio",
            definition="Share of images flagged as exact duplicates by the quality gate.",
            method="count(duplicate_status = exact_duplicate) / images",
            value=report.duplicate_ratio,
            threshold=0.02,
            status=_status(report.duplicate_ratio <= 0.02, report.duplicate_ratio <= 0.10),
            interpretation=f"{report.duplicate_ratio:.2%} of images are exact duplicates of another image.",
            recommendation=("Keep the duplicate policy at reject, or group duplicates when splitting."
                            if report.duplicate_ratio > 0.02 else "No action needed."),
        ),
        Metric(
            key="diversity.near_duplicate_ratio",
            name="Near-duplicate ratio",
            definition="Share of images flagged as perceptually near-identical to a retained image.",
            method="count(duplicate_status = near_duplicate) / images",
            value=report.near_duplicate_ratio,
            threshold=0.05,
            status=_status(report.near_duplicate_ratio <= 0.05, report.near_duplicate_ratio <= 0.15),
            interpretation=f"{report.near_duplicate_ratio:.2%} of images are near duplicates.",
            recommendation=("Near duplicates split across train and test are the most common cause of inflated "
                            "benchmark scores; keep their clusters together."
                            if report.near_duplicate_ratio > 0.05 else "No action needed."),
        ),
        Metric(
            key="diversity.least_diverse_classes",
            name="Least diverse classes",
            definition="Classes whose members are most visually similar to each other.",
            method="Normalised mean pairwise hash distance within each class",
            value=report.least_diverse_classes[0][0] if report.least_diverse_classes else None,
            threshold=None,
            status=MetricStatus.INFORMATIONAL,
            interpretation=(
                "Lowest-diversity class: "
                + ", ".join(f"{label} ({value:.2f})" for label, value in report.least_diverse_classes[:5])
                if report.least_diverse_classes else "Not enough images per class to measure."
            ),
            recommendation="Classes below 0.15 are often a single photo session; treat their scores with caution.",
            detail={"classes": [{"label": label, "diversity": value}
                                for label, value in report.least_diverse_classes]},
        ),
    )


def _status(healthy: bool, tolerable: bool) -> MetricStatus:
    if healthy:
        return MetricStatus.OK
    return MetricStatus.WARNING if tolerable else MetricStatus.CRITICAL


__all__ = ["DiversityReport", "analyze_diversity"]
