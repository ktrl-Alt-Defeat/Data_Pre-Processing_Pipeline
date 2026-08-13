"""Descriptive statistics primitives.

Thin, deterministic wrappers over pandas that produce serialisable summaries.
Everything here is pure: given the same frame and the same configured
percentiles, the output is byte-identical between runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Distributional summary of one numeric column."""

    name: str
    count: int
    missing: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    percentiles: Mapping[str, float] = field(default_factory=dict)

    @property
    def median(self) -> float | None:
        return self.percentiles.get("p50")

    @property
    def coefficient_of_variation(self) -> float | None:
        """Spread relative to magnitude; comparable across differently scaled metrics."""
        if not self.mean or self.std is None:
            return None
        return abs(self.std / self.mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "missing": self.missing,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "cv": self.coefficient_of_variation,
            **dict(self.percentiles),
        }


@dataclass(frozen=True, slots=True)
class CategoricalSummary:
    """Frequency summary of one categorical column."""

    name: str
    count: int
    missing: int
    unique: int
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def top(self) -> str | None:
        return next(iter(self.counts), None)

    @property
    def shares(self) -> dict[str, float]:
        total = sum(self.counts.values())
        return {key: value / total for key, value in self.counts.items()} if total else {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "missing": self.missing,
            "unique": self.unique,
            "top": self.top,
            "counts": dict(self.counts),
        }


def summarize_numeric(series: pd.Series, name: str, percentiles: Sequence[float]) -> NumericSummary:
    """Summarise a numeric column, ignoring nulls."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    missing = int(len(series) - len(values))
    if values.empty:
        return NumericSummary(name, 0, missing, None, None, None, None, {})

    quantiles = {f"p{int(round(q * 100)):02d}": float(values.quantile(q)) for q in percentiles}
    return NumericSummary(
        name=name,
        count=int(len(values)),
        missing=missing,
        mean=float(values.mean()),
        std=float(values.std(ddof=0)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        percentiles=quantiles,
    )


def summarize_categorical(series: pd.Series, name: str, top_n: int | None = None) -> CategoricalSummary:
    """Summarise a categorical column, ordered by frequency then label."""
    values = series.dropna().astype("string")
    counts = values.value_counts()
    ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    limited = ordered[:top_n] if top_n else ordered
    return CategoricalSummary(
        name=name,
        count=int(len(values)),
        missing=int(len(series) - len(values)),
        unique=int(counts.size),
        counts={str(key): int(value) for key, value in limited},
    )


def summaries_to_frame(summaries: Sequence[NumericSummary]) -> pd.DataFrame:
    """Tabular view of numeric summaries; the basis of image_statistics.csv."""
    if not summaries:
        return pd.DataFrame(columns=["name", "count", "missing", "mean", "std", "min", "max", "cv"])
    return pd.DataFrame([summary.as_dict() for summary in summaries])


def gini(values: Sequence[float] | np.ndarray) -> float:
    """Gini coefficient: 0 is a perfectly even distribution, 1 maximally skewed."""
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    total = array.sum()
    if array.size == 0 or total <= 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    return float((2 * np.sum(index * array)) / (array.size * total) - (array.size + 1) / array.size)


def shannon_entropy(counts: Sequence[float] | np.ndarray, base: float = 2.0) -> float:
    """Entropy of a count vector, in the given base."""
    array = np.asarray(list(counts), dtype=np.float64)
    total = array.sum()
    if total <= 0:
        return 0.0
    probabilities = array[array > 0] / total
    return float(-np.sum(probabilities * (np.log(probabilities) / np.log(base))))


def normalized_entropy(counts: Sequence[float] | np.ndarray) -> float:
    """Entropy scaled to [0, 1]; 1 means every category is equally represented."""
    array = np.asarray(list(counts), dtype=np.float64)
    non_zero = int((array > 0).sum())
    if non_zero <= 1:
        return 0.0 if non_zero == 0 else 1.0
    return float(shannon_entropy(array) / np.log2(non_zero))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


__all__ = [
    "CategoricalSummary",
    "NumericSummary",
    "clamp",
    "gini",
    "normalized_entropy",
    "shannon_entropy",
    "summaries_to_frame",
    "summarize_categorical",
    "summarize_numeric",
]
