"""Dataset profiler.

Assembles one :class:`DatasetProfile` describing the corpus: totals, per-class
and per-source composition, distributional summaries of every image property,
and channel statistics.

Read-only. Records are never modified; everything is derived from the manifest
frame the corpus stage already produced, which is why profiling a corpus of any
size costs one vectorised pass plus the sampled RGB decode.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from ..core.config import Config
from ..core.context import RunContext
from ..core.records import Corpus, RunManifest
from .distributions import (
    CategoryDistribution,
    Histogram,
    build_category_distributions,
    build_histograms,
    class_distribution,
    duplicate_cluster_sizes,
    resolution_distribution,
    source_distribution,
)
from .rgb import RgbProfiler, RgbStatistics
from .statistics import (
    CategoricalSummary,
    NumericSummary,
    summaries_to_frame,
    summarize_categorical,
    summarize_numeric,
)

_NUMERIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("width", "width"),
    ("height", "height"),
    ("megapixels", "megapixels"),
    ("aspect_ratio", "aspect_ratio"),
    ("file_size_bytes", "file_size_bytes"),
    ("quality_brightness", "brightness"),
    ("quality_contrast", "contrast"),
    ("quality_entropy", "entropy"),
    ("quality_blur_score", "sharpness"),
    ("quality_colorfulness", "colorfulness"),
    ("quality_score", "quality_score"),
)

_CATEGORICAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "class"),
    ("dataset_name", "source"),
    ("image_format", "format"),
    ("color_mode", "color_mode"),
    ("quality_grade", "quality_grade"),
    ("quality_duplicate_status", "duplicate_status"),
)


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Everything profiling measured about the corpus."""

    manifest: RunManifest
    dataset_fingerprint: str | None
    config_fingerprint: str
    generated_at: str
    totals: Mapping[str, Any]
    numeric: Mapping[str, NumericSummary]
    categorical: Mapping[str, CategoricalSummary]
    histograms: Mapping[str, Histogram]
    categories: Mapping[str, CategoryDistribution]
    resolutions: CategoryDistribution
    duplicate_clusters: Mapping[int, int]
    classes: pd.DataFrame = field(default_factory=pd.DataFrame)
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    rgb: RgbStatistics | None = None

    @property
    def image_count(self) -> int:
        return int(self.totals.get("images", 0))

    @property
    def class_count(self) -> int:
        return int(self.totals.get("classes", 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "generated_at": self.generated_at,
            "totals": dict(self.totals),
            "numeric": {name: summary.as_dict() for name, summary in self.numeric.items()},
            "categorical": {name: summary.as_dict() for name, summary in self.categorical.items()},
            "histograms": {name: histogram.as_dict() for name, histogram in self.histograms.items()},
            "categories": {name: distribution.as_dict() for name, distribution in self.categories.items()},
            "resolutions": self.resolutions.as_dict(),
            "duplicate_clusters": {str(size): count for size, count in self.duplicate_clusters.items()},
            "rgb": self.rgb.as_dict() if self.rgb else None,
        }


class DatasetProfiler:
    """Builds a :class:`DatasetProfile` from a corpus."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._profiling = config.profiling

    def profile(self, corpus: Corpus, context: RunContext) -> DatasetProfile:
        """Profile the corpus. Never mutates records."""
        records = corpus.records if self._profiling.include_rejected else corpus.accepted()
        frame = _frame_for(corpus, self._profiling.include_rejected)
        percentiles = self._profiling.percentiles

        rgb_profiler = RgbProfiler(
            self._profiling.rgb,
            cache_dir=self._config.paths.cache_dir,
            workers=self._config.execution.resolved_workers,
        )

        return DatasetProfile(
            manifest=context.manifest,
            dataset_fingerprint=corpus.fingerprint,
            config_fingerprint=context.config_fingerprint,
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            totals=self._totals(corpus, frame),
            numeric={
                alias: summarize_numeric(frame[column], alias, percentiles)
                for column, alias in _NUMERIC_COLUMNS
                if column in frame.columns
            },
            categorical={
                alias: summarize_categorical(frame[column], alias, self._profiling.top_classes)
                for column, alias in _CATEGORICAL_COLUMNS
                if column in frame.columns
            },
            histograms=build_histograms(frame, self._profiling),
            categories=build_category_distributions(frame, self._profiling),
            resolutions=resolution_distribution(frame, self._profiling.top_resolutions),
            duplicate_clusters=duplicate_cluster_sizes(frame),
            classes=class_distribution(frame),
            sources=source_distribution(frame),
            rgb=rgb_profiler.profile(records, corpus.fingerprint),
        )

    def _totals(self, corpus: Corpus, frame: pd.DataFrame) -> dict[str, Any]:
        totals: dict[str, Any] = {
            "images": int(len(frame)),
            "images_in_corpus": len(corpus.records),
            "images_accepted": len(corpus.accepted()),
            "images_rejected": len(corpus.rejected()),
            "classes": int(frame["label"].nunique()) if "label" in frame.columns else 0,
            "sources": int(frame["dataset_name"].nunique()) if "dataset_name" in frame.columns else 0,
            "dataset_version": corpus.version,
        }
        if "file_size_bytes" in frame.columns:
            totals["bytes"] = int(pd.to_numeric(frame["file_size_bytes"], errors="coerce").fillna(0).sum())
        if "megapixels" in frame.columns:
            totals["megapixels"] = float(pd.to_numeric(frame["megapixels"], errors="coerce").fillna(0).sum())
        if "duplicate_group" in frame.columns:
            totals["duplicate_clusters"] = int(frame["duplicate_group"].dropna().nunique())
        return totals


def _frame_for(corpus: Corpus, include_rejected: bool) -> pd.DataFrame:
    frame = corpus.to_frame(accepted_only=not include_rejected)
    return frame if not frame.empty else pd.DataFrame(columns=["label", "dataset_name"])


def image_statistics_frame(profile: DatasetProfile) -> pd.DataFrame:
    """One row per measured image property; written to image_statistics.csv."""
    return summaries_to_frame(list(profile.numeric.values()))


__all__ = ["DatasetProfile", "DatasetProfiler", "image_statistics_frame"]
