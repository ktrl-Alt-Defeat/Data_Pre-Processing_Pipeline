"""Source and label bias analysis.

A merged benchmark inherits the habits of each contributing dataset: one source
may supply every image of a class, or shoot at a different resolution, so a
model can learn the source rather than the disease. This module quantifies that
risk, plus the label consistency of the harmonised space.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import BiasConfig
from ..core.records import Metric, MetricStatus
from ..profiling.statistics import normalized_entropy


@dataclass(frozen=True, slots=True)
class BiasReport:
    """Source-contribution and label-consistency findings."""

    source_count: int
    source_shares: dict[str, float] = field(default_factory=dict)
    source_diversity: float = 0.0
    single_source_classes: tuple[str, ...] = ()
    dominated_classes: tuple[tuple[str, str, float], ...] = ()
    shared_classes: tuple[str, ...] = ()
    inconsistent_labels: tuple[tuple[str, tuple[str, ...]], ...] = ()
    label_consistency: float = 1.0
    resolution_by_source: dict[str, float] = field(default_factory=dict)
    quality_by_source: dict[str, float] = field(default_factory=dict)
    metrics: tuple[Metric, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "source_shares": dict(self.source_shares),
            "source_diversity": self.source_diversity,
            "single_source_classes": list(self.single_source_classes),
            "dominated_classes": [{"label": label, "source": source, "share": share}
                                  for label, source, share in self.dominated_classes],
            "shared_classes": list(self.shared_classes),
            "inconsistent_labels": [{"raw_label": raw, "canonical": list(values)}
                                    for raw, values in self.inconsistent_labels],
            "label_consistency": self.label_consistency,
            "resolution_by_source": dict(self.resolution_by_source),
            "quality_by_source": dict(self.quality_by_source),
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


def analyze_bias(frame: pd.DataFrame, config: BiasConfig) -> BiasReport:
    """Measure source contribution, per-class dominance and label consistency."""
    if not config.enabled or frame.empty or "dataset_name" not in frame.columns:
        return BiasReport(source_count=0)

    shares = _source_shares(frame)
    dominated, single_source, shared = _class_source_mix(frame, config.max_source_share)
    inconsistent, consistency = _label_consistency(frame)

    report = BiasReport(
        source_count=len(shares),
        source_shares=shares,
        source_diversity=round(normalized_entropy(list(shares.values())), 6),
        single_source_classes=single_source,
        dominated_classes=dominated,
        shared_classes=shared,
        inconsistent_labels=inconsistent,
        label_consistency=round(consistency, 6),
        resolution_by_source=_mean_by_source(frame, "megapixels"),
        quality_by_source=_mean_by_source(frame, "quality_score"),
    )
    return dataclasses.replace(report, metrics=_metrics(report, config))


def _source_shares(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["dataset_name"].dropna().astype("string").value_counts()
    total = int(counts.sum())
    return {str(name): round(int(count) / total, 6) for name, count in sorted(counts.items())} if total else {}


def _class_source_mix(
    frame: pd.DataFrame, max_share: float
) -> tuple[tuple[tuple[str, str, float], ...], tuple[str, ...], tuple[str, ...]]:
    if "label" not in frame.columns:
        return (), (), ()

    grouped = frame.groupby(["label", "dataset_name"], dropna=False).size().rename("images").reset_index()
    totals = grouped.groupby("label")["images"].transform("sum")
    grouped["share"] = grouped["images"] / totals

    dominant = grouped.sort_values(["label", "share"], ascending=[True, False]).groupby("label").head(1)
    dominated = tuple(
        (str(row.label), str(row.dataset_name), round(float(row.share), 6))
        for row in dominant.itertuples(index=False)
        if row.share > max_share
    )
    per_class_sources = grouped.groupby("label")["dataset_name"].nunique()
    single = tuple(sorted(str(label) for label, count in per_class_sources.items() if count == 1))
    shared = tuple(sorted(str(label) for label, count in per_class_sources.items() if count > 1))
    return dominated, single, shared


def _label_consistency(frame: pd.DataFrame) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], float]:
    """A raw folder label resolving to two canonical labels is an ontology conflict."""
    if not {"source_class", "label"} <= set(frame.columns):
        return (), 1.0
    subset = frame[["source_class", "label"]].dropna()
    if subset.empty:
        return (), 1.0

    grouped = subset.groupby("source_class")["label"].unique()
    conflicts = tuple(
        (str(raw), tuple(sorted(str(value) for value in values)))
        for raw, values in sorted(grouped.items())
        if len(values) > 1
    )
    consistency = 1.0 - (len(conflicts) / grouped.size) if grouped.size else 1.0
    return conflicts, consistency


def _mean_by_source(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in frame.columns:
        return {}
    values = frame.groupby("dataset_name")[column].mean().dropna()
    return {str(name): round(float(value), 6) for name, value in sorted(values.items())}


def _metrics(report: BiasReport, config: BiasConfig) -> tuple[Metric, ...]:
    dominated_share = len(report.dominated_classes) / max(1, len(report.single_source_classes) + len(report.shared_classes))
    resolution_spread = _spread(report.resolution_by_source)
    quality_spread = _spread(report.quality_by_source)

    return (
        Metric(
            key="bias.source_diversity",
            name="Source diversity",
            definition="Entropy of the per-source image counts, normalised to [0, 1].",
            method="H(source shares) / log2(number of sources)",
            value=report.source_diversity,
            threshold=0.6,
            status=(MetricStatus.INFORMATIONAL if report.source_count < 2
                    else _status(report.source_diversity >= 0.6, report.source_diversity >= 0.35)),
            interpretation=(
                f"{report.source_count} source(s) contribute: "
                + ", ".join(f"{name} {share:.1%}" for name, share in report.source_shares.items())
            ),
            recommendation=("A single-source corpus cannot support cross-source generalisation claims."
                            if report.source_count < 2
                            else "Balance contributions if one source dominates the benchmark."),
        ),
        Metric(
            key="bias.dominated_classes",
            name="Source-dominated classes",
            definition=f"Classes where one source supplies more than {config.max_source_share:.0%} of the images.",
            method=f"max(per-class source share) > {config.max_source_share}",
            value=len(report.dominated_classes),
            threshold=0,
            status=(MetricStatus.INFORMATIONAL if report.source_count < 2
                    else _status(not report.dominated_classes, dominated_share <= 0.5)),
            interpretation=(
                f"{len(report.dominated_classes)} classes are dominated by a single source."
                if report.dominated_classes else "No class is dominated by one source."
            ),
            recommendation="A model can learn the source's capture conditions instead of the disease; "
                           "evaluate cross-source held-out performance.",
            detail={"classes": [{"label": label, "source": source, "share": share}
                                for label, source, share in report.dominated_classes[:50]]},
        ),
        Metric(
            key="bias.single_source_classes",
            name="Single-source classes",
            definition="Classes whose images all come from one contributing dataset.",
            method="count(distinct dataset_name per label) == 1",
            value=len(report.single_source_classes),
            threshold=None,
            status=MetricStatus.INFORMATIONAL,
            interpretation=(
                f"{len(report.single_source_classes)} of "
                f"{len(report.single_source_classes) + len(report.shared_classes)} classes come from one source."
            ),
            recommendation="Expected when datasets cover different crops; a concern when they overlap.",
            detail={"classes": list(report.single_source_classes[:50])},
        ),
        Metric(
            key="bias.label_consistency",
            name="Label consistency",
            definition="Share of raw folder labels that map to exactly one canonical label.",
            method="1 - (conflicting raw labels / distinct raw labels)",
            value=report.label_consistency,
            threshold=1.0,
            status=_status(report.label_consistency >= 1.0, report.label_consistency >= 0.95),
            interpretation=(
                f"{len(report.inconsistent_labels)} raw labels resolve to more than one canonical label."
                if report.inconsistent_labels else "Every raw label resolves to exactly one canonical label."
            ),
            recommendation=("Pin the mapping with corpus.label_aliases so harmonisation is unambiguous."
                            if report.inconsistent_labels else "No action needed."),
            detail={"conflicts": [{"raw_label": raw, "canonical": list(values)}
                                  for raw, values in report.inconsistent_labels[:50]]},
        ),
        Metric(
            key="bias.resolution_consistency",
            name="Resolution consistency across sources",
            definition="Relative spread of mean megapixels between sources.",
            method="(max - min) / max of per-source mean megapixels",
            value=round(resolution_spread, 6),
            threshold=0.5,
            status=(MetricStatus.INFORMATIONAL if report.source_count < 2
                    else _status(resolution_spread <= 0.5, resolution_spread <= 0.8)),
            interpretation=(
                "Mean megapixels per source: "
                + ", ".join(f"{name} {value:.2f}" for name, value in report.resolution_by_source.items())
                if report.resolution_by_source else "No resolution data available."
            ),
            recommendation="Large gaps let a model identify the source from image size alone; "
                           "resizing at packaging time removes the shortcut.",
        ),
        Metric(
            key="bias.quality_consistency",
            name="Quality consistency across sources",
            definition="Relative spread of mean quality score between sources.",
            method="(max - min) / max of per-source mean quality score",
            value=round(quality_spread, 6),
            threshold=0.3,
            status=(MetricStatus.INFORMATIONAL if report.source_count < 2
                    else _status(quality_spread <= 0.3, quality_spread <= 0.5)),
            interpretation=(
                "Mean quality per source: "
                + ", ".join(f"{name} {value:.3f}" for name, value in report.quality_by_source.items())
                if report.quality_by_source else "No quality data available."
            ),
            recommendation="Systematic quality gaps between sources bias any cross-source comparison.",
        ),
    )


def _spread(values: dict[str, float]) -> float:
    if len(values) < 2:
        return 0.0
    array = np.array(list(values.values()), dtype=np.float64)
    top = float(array.max())
    return float((top - array.min()) / top) if top > 0 else 0.0


def _status(healthy: bool, tolerable: bool) -> MetricStatus:
    if healthy:
        return MetricStatus.OK
    return MetricStatus.WARNING if tolerable else MetricStatus.CRITICAL


__all__ = ["BiasReport", "analyze_bias"]
