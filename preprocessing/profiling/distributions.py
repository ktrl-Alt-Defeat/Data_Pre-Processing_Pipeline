"""Dataset distributions.

Turns the corpus manifest into the histograms and frequency tables the reports
and visualisations consume. Bin edges are derived from the data and the
configured bin count only, so a rerun on the same corpus reproduces identical
bins — a requirement for comparing two dataset versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.config import ProfilingConfig


@dataclass(frozen=True, slots=True)
class Histogram:
    """A binned numeric distribution."""

    name: str
    edges: tuple[float, ...]
    counts: tuple[int, ...]
    total: int

    @classmethod
    def from_series(cls, name: str, series: pd.Series, bins: int) -> "Histogram":
        values = pd.to_numeric(series, errors="coerce").dropna().astype("float64").to_numpy()
        if values.size == 0:
            return cls(name, (), (), 0)
        low, high = float(values.min()), float(values.max())
        if low == high:
            # A degenerate range still deserves a bin, otherwise numpy widens it
            # arbitrarily and the report shows a range the data does not have.
            return cls(name, (low, high), (int(values.size),), int(values.size))
        counts, edges = np.histogram(values, bins=bins, range=(low, high))
        return cls(name, tuple(float(edge) for edge in edges), tuple(int(count) for count in counts), int(values.size))

    @property
    def centers(self) -> tuple[float, ...]:
        if len(self.edges) < 2:
            return tuple(self.edges)
        return tuple((self.edges[i] + self.edges[i + 1]) / 2 for i in range(len(self.edges) - 1))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "edges": list(self.edges), "counts": list(self.counts), "total": self.total}


@dataclass(frozen=True, slots=True)
class CategoryDistribution:
    """A frequency table over a categorical column."""

    name: str
    counts: Mapping[str, int]

    @classmethod
    def from_series(cls, name: str, series: pd.Series, top_n: int | None = None) -> "CategoryDistribution":
        values = series.dropna().astype("string").value_counts()
        ordered = sorted(values.items(), key=lambda item: (-item[1], str(item[0])))
        return cls(name, {str(key): int(value) for key, value in (ordered[:top_n] if top_n else ordered)})

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def shares(self) -> dict[str, float]:
        total = self.total
        return {key: value / total for key, value in self.counts.items()} if total else {}

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "counts": dict(self.counts), "total": self.total}


_NUMERIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("width", "width"),
    ("height", "height"),
    ("megapixels", "megapixels"),
    ("aspect_ratio", "aspect_ratio"),
    ("file_size_bytes", "file_size_bytes"),
    ("quality_brightness", "brightness"),
    ("quality_contrast", "contrast"),
    ("quality_entropy", "entropy"),
    ("quality_blur_score", "sharpness"),
    ("quality_score", "quality_score"),
)

_CATEGORICAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "class"),
    ("dataset_name", "source"),
    ("image_format", "format"),
    ("color_mode", "color_mode"),
    ("quality_grade", "quality_grade"),
    ("quality_duplicate_status", "duplicate_status"),
    ("crop", "crop"),
)


def build_histograms(frame: pd.DataFrame, config: ProfilingConfig) -> dict[str, Histogram]:
    """Histograms for every numeric image property present in the manifest."""
    return {
        alias: Histogram.from_series(alias, frame[column], config.histogram_bins)
        for column, alias in _NUMERIC_FIELDS
        if column in frame.columns
    }


def build_category_distributions(frame: pd.DataFrame, config: ProfilingConfig) -> dict[str, CategoryDistribution]:
    """Frequency tables for every categorical image property."""
    limits = {"class": None, "source": None}
    return {
        alias: CategoryDistribution.from_series(alias, frame[column], limits.get(alias, config.top_classes))
        for column, alias in _CATEGORICAL_FIELDS
        if column in frame.columns
    }


def resolution_distribution(frame: pd.DataFrame, top_n: int) -> CategoryDistribution:
    """Most common exact ``width x height`` pairs."""
    if not {"width", "height"} <= set(frame.columns):
        return CategoryDistribution("resolution", {})
    pairs = (
        frame[["width", "height"]]
        .dropna()
        .astype("int64")
        .agg(lambda row: f"{row.width}x{row.height}", axis=1)
        if len(frame)
        else pd.Series(dtype="string")
    )
    return CategoryDistribution.from_series("resolution", pairs, top_n)


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-class table: counts, share, sources and mean quality."""
    if frame.empty or "label" not in frame.columns:
        return pd.DataFrame(columns=["label", "images", "share"])

    grouped = frame.groupby("label", dropna=False)
    table = pd.DataFrame(
        {
            "images": grouped.size(),
            "sources": grouped["dataset_name"].nunique() if "dataset_name" in frame.columns else 0,
            "class_index": grouped["class_index"].first() if "class_index" in frame.columns else None,
        }
    )
    for column, alias in (("quality_score", "mean_quality_score"), ("megapixels", "mean_megapixels"),
                          ("quality_blur_score", "mean_sharpness"), ("quality_brightness", "mean_brightness")):
        if column in frame.columns:
            table[alias] = grouped[column].mean()

    table["share"] = table["images"] / len(frame)
    table = table.reset_index().sort_values(["images", "label"], ascending=[False, True], ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def source_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-source table: counts, class coverage and mean image properties."""
    if frame.empty or "dataset_name" not in frame.columns:
        return pd.DataFrame(columns=["dataset_name", "images", "share"])

    grouped = frame.groupby("dataset_name", dropna=False)
    table = pd.DataFrame({"images": grouped.size(), "classes": grouped["label"].nunique()})
    for column, alias in (("megapixels", "mean_megapixels"), ("quality_score", "mean_quality_score"),
                          ("file_size_bytes", "mean_file_size_bytes")):
        if column in frame.columns:
            table[alias] = grouped[column].mean()
    if "dataset_version" in frame.columns:
        table["version"] = grouped["dataset_version"].first()

    table["share"] = table["images"] / len(frame)
    return table.reset_index().sort_values(["images", "dataset_name"], ascending=[False, True], ignore_index=True)


def duplicate_cluster_sizes(frame: pd.DataFrame) -> dict[int, int]:
    """Cluster size -> number of clusters, for the duplicate visualisation."""
    if "duplicate_group" not in frame.columns:
        return {}
    groups = frame["duplicate_group"].dropna()
    if groups.empty:
        return {}
    sizes = groups.value_counts().value_counts()
    return {int(size): int(count) for size, count in sorted(sizes.items())}


__all__ = [
    "CategoryDistribution",
    "Histogram",
    "build_category_distributions",
    "build_histograms",
    "class_distribution",
    "duplicate_cluster_sizes",
    "resolution_distribution",
    "source_distribution",
]
