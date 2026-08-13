"""Profiling stage: describe the corpus exhaustively, change nothing.

Reads the manifest the corpus stage produced, summarises every image property,
and writes ``class_distribution.csv``, ``image_statistics.csv`` and
``profiling_report.html``. The only pixel access is the sampled, cached RGB
pass in :mod:`preprocessing.profiling.rgb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import Config
from ..core.context import RunContext
from ..core.io import ensure_dir
from ..core.logging import StageTracker, stage_scope
from ..core.records import Corpus, PipelineStage
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
from .profiler import DatasetProfile, DatasetProfiler, image_statistics_frame
from .rgb import RgbProfiler, RgbStatistics
from .statistics import (
    CategoricalSummary,
    NumericSummary,
    clamp,
    gini,
    normalized_entropy,
    shannon_entropy,
    summaries_to_frame,
    summarize_categorical,
    summarize_numeric,
)

_STAGE = PipelineStage.PROFILING


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    """The profile plus the artefacts written for it."""

    profile: DatasetProfile
    artifacts: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile.as_dict(), "artifacts": dict(self.artifacts)}


def profile_dataset(config: Config, context: RunContext, corpus: Corpus) -> ProfilingResult:
    """Profile the corpus and write the profiling artefacts."""
    with stage_scope(_STAGE, context.logger(_STAGE)) as tracker:
        profile = DatasetProfiler(config).profile(corpus, context)
        tracker.processed(profile.image_count)
        artifacts = _write_artifacts(context, profile)
        _record_metrics(tracker, profile)

    return ProfilingResult(profile=profile, artifacts=artifacts)


def render_profiling_report(context: RunContext, profile: DatasetProfile, figures: dict[str, Path] | None = None) -> Path:
    """Render ``reports/profiling_report.html`` for an existing profile."""
    from reports.report import HtmlDocument, manifest_rows

    document = HtmlDocument(
        title="Dataset profiling report",
        subtitle=f"{profile.image_count:,} images across {profile.class_count} classes",
    )
    document.key_values("Provenance", manifest_rows(profile.manifest, profile.dataset_fingerprint,
                                                    profile.config_fingerprint))
    document.key_values("Totals", {key: value for key, value in profile.totals.items()})

    if profile.rgb:
        document.key_values("RGB channel statistics", {
            "Mean (0-1)": ", ".join(f"{value:.5f}" for value in profile.rgb.mean),
            "Std (0-1)": ", ".join(f"{value:.5f}" for value in profile.rgb.std),
            "Mean (0-255)": ", ".join(f"{value:.2f}" for value in profile.rgb.mean_255),
            "Std (0-255)": ", ".join(f"{value:.2f}" for value in profile.rgb.std_255),
            "Images sampled": f"{profile.rgb.sample_size:,}",
            "Pixels accumulated": f"{profile.rgb.pixels:,}",
            "Decode failures": profile.rgb.failures,
        })

    document.table("Image statistics", image_statistics_frame(profile), limit=None,
                   note="One row per measured image property, over the profiled images.")
    document.table("Class distribution", profile.classes, limit=200)
    document.table("Source contribution", profile.sources, limit=None)
    document.key_values("Most common resolutions", profile.resolutions.counts)
    document.key_values("Formats", profile.categories["format"].counts if "format" in profile.categories else {})
    if figures:
        document.figures("Figures", figures, relative_to=context.layout.reports_dir.parent)
    return document.write(context.layout.profiling_report, profile.manifest)


def _write_artifacts(context: RunContext, profile: DatasetProfile) -> dict[str, str]:
    layout = context.layout
    ensure_dir(layout.analytics_dir)

    profile.classes.to_csv(layout.class_distribution_csv, index=False, encoding="utf-8")
    image_statistics_frame(profile).to_csv(layout.image_statistics_csv, index=False, encoding="utf-8")
    return {
        "class_distribution": str(layout.class_distribution_csv),
        "image_statistics": str(layout.image_statistics_csv),
    }


def _record_metrics(tracker: StageTracker, profile: DatasetProfile) -> None:
    quality = profile.numeric.get("quality_score")
    megapixels = profile.numeric.get("megapixels")
    tracker.metrics(
        images=profile.image_count,
        classes=profile.class_count,
        sources=profile.totals.get("sources", 0),
        mean_quality_score=round(quality.mean, 6) if quality and quality.mean is not None else None,
        mean_megapixels=round(megapixels.mean, 6) if megapixels and megapixels.mean is not None else None,
        rgb_mean=list(profile.rgb.mean) if profile.rgb else None,
        rgb_std=list(profile.rgb.std) if profile.rgb else None,
        rgb_sample_size=profile.rgb.sample_size if profile.rgb else 0,
    )


__all__ = [
    "CategoricalSummary",
    "CategoryDistribution",
    "DatasetProfile",
    "DatasetProfiler",
    "Histogram",
    "NumericSummary",
    "ProfilingResult",
    "RgbProfiler",
    "RgbStatistics",
    "build_category_distributions",
    "build_histograms",
    "clamp",
    "class_distribution",
    "duplicate_cluster_sizes",
    "gini",
    "image_statistics_frame",
    "normalized_entropy",
    "profile_dataset",
    "render_profiling_report",
    "resolution_distribution",
    "shannon_entropy",
    "source_distribution",
    "summaries_to_frame",
    "summarize_categorical",
    "summarize_numeric",
]
