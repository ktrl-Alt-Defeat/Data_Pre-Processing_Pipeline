"""Train/test leakage risk analysis.

Splitting has not happened yet, so this module measures *risk*: the duplicate
clusters that would leak across partitions if the split ignored them, and the
grouping constraint that prevents it. It also surfaces clusters spanning classes
or sources, which are label conflicts and cross-dataset overlap rather than
leakage as such.

The output feeds two consumers: ``analytics/leakage_report.csv`` for review, and
the grouped splitter, which must keep every cluster inside one split.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import LeakageConfig, SplitConfig
from ..core.records import Metric, MetricStatus

_COLUMNS = (
    "group_id",
    "group_size",
    "risk",
    "labels",
    "sources",
    "image_ids",
    "distinct_labels",
    "distinct_sources",
    "retained_images",
)


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Duplicate-cluster leakage risk."""

    clusters: int
    clustered_images: int
    cross_class_clusters: int
    cross_source_clusters: int
    largest_cluster: int
    at_risk_images: int
    random_split_leak_probability: float
    grouping_required: bool
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: tuple[Metric, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "clusters": self.clusters,
            "clustered_images": self.clustered_images,
            "cross_class_clusters": self.cross_class_clusters,
            "cross_source_clusters": self.cross_source_clusters,
            "largest_cluster": self.largest_cluster,
            "at_risk_images": self.at_risk_images,
            "random_split_leak_probability": self.random_split_leak_probability,
            "grouping_required": self.grouping_required,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


def analyze_leakage(frame: pd.DataFrame, config: LeakageConfig, split: SplitConfig) -> LeakageReport:
    """Quantify how much duplicate content would leak across a naive split."""
    empty = LeakageReport(0, 0, 0, 0, 0, 0, 0.0, False, pd.DataFrame(columns=list(_COLUMNS)))
    if not config.enabled or frame.empty or "duplicate_group" not in frame.columns:
        return empty

    clustered = frame[frame["duplicate_group"].notna()]
    if clustered.empty:
        return dataclasses.replace(empty, metrics=_metrics(empty, split))

    table = _cluster_table(clustered)
    sizes = table["group_size"].to_numpy(dtype=np.float64)
    report = LeakageReport(
        clusters=int(len(table)),
        clustered_images=int(sizes.sum()),
        cross_class_clusters=int((table["distinct_labels"] > 1).sum()),
        cross_source_clusters=int((table["distinct_sources"] > 1).sum()),
        largest_cluster=int(sizes.max()),
        at_risk_images=int(table.loc[table["retained_images"] > 1, "retained_images"].sum()),
        random_split_leak_probability=round(_leak_probability(table, split), 6),
        grouping_required=bool((table["retained_images"] > 1).any()),
        table=table,
    )
    return dataclasses.replace(report, metrics=_metrics(report, split))


def _cluster_table(clustered: pd.DataFrame) -> pd.DataFrame:
    """One row per duplicate cluster, with the members that survived the gate."""
    accepted = clustered["status"].eq("accepted") if "status" in clustered.columns else pd.Series(True, index=clustered.index)
    grouped = clustered.assign(_accepted=accepted).groupby("duplicate_group", dropna=True)

    rows = []
    for group_id, members in grouped:
        labels = sorted({str(value) for value in members["label"].dropna()}) if "label" in members else []
        sources = sorted({str(value) for value in members["dataset_name"].dropna()}) if "dataset_name" in members else []
        retained = int(members["_accepted"].sum())
        rows.append(
            {
                "group_id": str(group_id),
                "group_size": int(len(members)),
                "risk": _risk(len(labels), len(sources), retained),
                "labels": " | ".join(labels),
                "sources": " | ".join(sources),
                "image_ids": " ".join(str(value) for value in members["image_id"].head(20)),
                "distinct_labels": len(labels),
                "distinct_sources": len(sources),
                "retained_images": retained,
            }
        )
    table = pd.DataFrame(rows, columns=list(_COLUMNS))
    return table.sort_values(["group_size", "group_id"], ascending=[False, True], ignore_index=True)


def _risk(labels: int, sources: int, retained: int) -> str:
    if labels > 1:
        return "label_conflict"
    if retained > 1:
        return "split_leakage"
    if sources > 1:
        return "cross_source_overlap"
    return "contained"


def _leak_probability(table: pd.DataFrame, split: SplitConfig) -> float:
    """Chance that a cluster with >1 retained image straddles two splits.

    For a cluster of *n* retained images assigned independently at the configured
    ratios, the probability of them not all landing together is 1 - sum(r_i^n).
    """
    ratios = np.array([split.ratios.train, split.ratios.val, split.ratios.test], dtype=np.float64)
    ratios = ratios[ratios > 0]
    at_risk = table.loc[table["retained_images"] > 1, "retained_images"].to_numpy(dtype=np.int64)
    if at_risk.size == 0:
        return 0.0
    together = np.array([np.sum(ratios**int(n)) for n in at_risk], dtype=np.float64)
    return float(np.mean(1.0 - together))


def _metrics(report: LeakageReport, split: SplitConfig) -> tuple[Metric, ...]:
    grouped_split = split.strategy == "stratified_group" and split.group_by == "duplicate_group"
    return (
        Metric(
            key="leakage.clusters",
            name="Duplicate clusters",
            definition="Groups of images identified as identical or perceptually near-identical.",
            method="Clusters produced by the quality stage's duplicate detector",
            value=report.clusters,
            threshold=None,
            status=MetricStatus.INFORMATIONAL,
            interpretation=(
                f"{report.clusters:,} clusters cover {report.clustered_images:,} images; "
                f"the largest holds {report.largest_cluster:,}."
            ),
            recommendation="Clusters are the unit that must stay inside one split.",
        ),
        Metric(
            key="leakage.at_risk_images",
            name="Images at risk of leaking",
            definition="Images in clusters that still hold more than one retained image after the quality gate.",
            method="sum(retained members) over clusters with more than one retained member",
            value=report.at_risk_images,
            threshold=0,
            status=(MetricStatus.OK if not report.grouping_required
                    else MetricStatus.OK if grouped_split else MetricStatus.CRITICAL),
            interpretation=(
                f"{report.at_risk_images:,} images could be separated from their duplicates by a naive split."
                if report.grouping_required
                else "The quality gate already retained at most one image per cluster; nothing can leak."
            ),
            recommendation=(
                "Already covered: split.strategy is stratified_group over duplicate_group."
                if grouped_split and report.grouping_required
                else "Set split.strategy to stratified_group and split.group_by to duplicate_group."
                if report.grouping_required
                else "No action needed."
            ),
        ),
        Metric(
            key="leakage.random_split_probability",
            name="Naive-split leak probability",
            definition="Chance an at-risk cluster would straddle two partitions under an ungrouped split.",
            method="mean over clusters of 1 - sum(ratio ^ retained members)",
            value=report.random_split_leak_probability,
            threshold=0.0,
            status=(MetricStatus.OK if report.random_split_leak_probability == 0 or grouped_split
                    else MetricStatus.CRITICAL),
            interpretation=(
                f"An ungrouped split would separate {report.random_split_leak_probability:.1%} of at-risk clusters, "
                "putting near-identical images in both train and test."
                if report.random_split_leak_probability
                else "No cluster can straddle a split."
            ),
            recommendation="Grouped splitting reduces this to zero by construction.",
        ),
        Metric(
            key="leakage.cross_class_clusters",
            name="Cross-class duplicate clusters",
            definition="Clusters whose members carry more than one canonical label.",
            method="count(distinct labels within cluster) > 1",
            value=report.cross_class_clusters,
            threshold=0,
            status=_status(report.cross_class_clusters == 0,
                           report.cross_class_clusters <= max(1, report.clusters // 20)),
            interpretation=(
                f"{report.cross_class_clusters:,} clusters contain images labelled differently."
                if report.cross_class_clusters else "No duplicate cluster spans two classes."
            ),
            recommendation=(
                "These are either annotation conflicts or perceptual-hash false positives. Review "
                "analytics/leakage_report.csv, then either fix the labels or tighten "
                "quality.duplicates.max_hamming_distance / set across_classes to false."
                if report.cross_class_clusters else "No action needed."
            ),
        ),
        Metric(
            key="leakage.cross_source_clusters",
            name="Cross-dataset overlap",
            definition="Clusters whose members come from more than one contributing dataset.",
            method="count(distinct dataset_name within cluster) > 1",
            value=report.cross_source_clusters,
            threshold=None,
            status=MetricStatus.INFORMATIONAL,
            interpretation=(
                f"{report.cross_source_clusters:,} clusters span two or more sources, i.e. the datasets overlap."
                if report.cross_source_clusters else "No image appears in more than one source."
            ),
            recommendation="Overlap inflates the apparent size of a merged corpus; report the deduplicated count.",
        ),
    )


def _status(healthy: bool, tolerable: bool) -> MetricStatus:
    if healthy:
        return MetricStatus.OK
    return MetricStatus.WARNING if tolerable else MetricStatus.CRITICAL


__all__ = ["LeakageReport", "analyze_leakage"]
