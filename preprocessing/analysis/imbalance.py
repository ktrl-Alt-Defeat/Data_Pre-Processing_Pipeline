"""Class imbalance and long-tail analysis.

Answers whether the class distribution is fit to train and evaluate on: how
skewed it is, which classes are too rare to learn, and whether every class has
enough images to survive the configured split ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.config import ImbalanceConfig, SplitConfig
from ..core.records import Metric, MetricStatus
from ..profiling.statistics import gini, normalized_entropy

_LONG_TAIL_HEAD_SHARE = 0.8


@dataclass(frozen=True, slots=True)
class ImbalanceReport:
    """Class-distribution findings."""

    class_count: int
    total_images: int
    largest: int
    smallest: int
    imbalance_ratio: float
    gini: float
    evenness: float
    rare_classes: tuple[str, ...] = ()
    undersized_classes: tuple[str, ...] = ()
    infeasible_classes: tuple[str, ...] = ()
    head_classes: int = 0
    tail_share: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    metrics: tuple[Metric, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_count": self.class_count,
            "total_images": self.total_images,
            "largest_class": self.largest,
            "smallest_class": self.smallest,
            "imbalance_ratio": self.imbalance_ratio,
            "gini": self.gini,
            "evenness": self.evenness,
            "rare_classes": list(self.rare_classes),
            "undersized_classes": list(self.undersized_classes),
            "infeasible_classes": list(self.infeasible_classes),
            "head_classes": self.head_classes,
            "tail_share": self.tail_share,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


def analyze_imbalance(frame: pd.DataFrame, config: ImbalanceConfig, split: SplitConfig) -> ImbalanceReport:
    """Measure class balance, rare classes, long-tail shape and split feasibility."""
    counts = _counts(frame)
    if not counts:
        return ImbalanceReport(0, 0, 0, 0, 0.0, 0.0, 0.0)

    values = np.array(list(counts.values()), dtype=np.float64)
    total = int(values.sum())
    largest, smallest = int(values.max()), int(values.min())
    ratio = float(largest / smallest) if smallest else float("inf")
    mean = float(values.mean())

    rare = tuple(sorted(label for label, count in counts.items() if count < config.rare_class_ratio * mean))
    undersized = tuple(sorted(label for label, count in counts.items() if count < config.min_class_size))
    required = _required_per_class(split)
    infeasible = tuple(sorted(label for label, count in counts.items() if count < required))
    head, tail_share = _long_tail(values)

    report = ImbalanceReport(
        class_count=len(counts),
        total_images=total,
        largest=largest,
        smallest=smallest,
        imbalance_ratio=round(ratio, 4),
        gini=round(gini(values), 6),
        evenness=round(normalized_entropy(values), 6),
        rare_classes=rare,
        undersized_classes=undersized,
        infeasible_classes=infeasible,
        head_classes=head,
        tail_share=round(tail_share, 6),
        counts={label: int(count) for label, count in counts.items()},
    )
    return _with_metrics(report, config, split, required, mean)


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "label" not in frame.columns:
        return {}
    counts = frame["label"].dropna().astype("string").value_counts()
    return {str(label): int(count) for label, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))}


def _required_per_class(split: SplitConfig) -> int:
    """Smallest class size that can still yield one image in every split."""
    ratios = [split.ratios.train, split.ratios.val, split.ratios.test]
    active = [ratio for ratio in ratios if ratio > 0]
    if not active:
        return split.min_samples_per_class
    return max(split.min_samples_per_class, int(np.ceil(1 / min(active))))


def _long_tail(values: np.ndarray) -> tuple[int, float]:
    """Classes covering 80% of images (head), and the share held by the rest."""
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    head = int(np.searchsorted(cumulative, _LONG_TAIL_HEAD_SHARE) + 1)
    head = min(head, ordered.size)
    return head, float(ordered[head:].sum() / ordered.sum()) if head < ordered.size else 0.0


def _with_metrics(
    report: ImbalanceReport,
    config: ImbalanceConfig,
    split: SplitConfig,
    required: int,
    mean: float,
) -> ImbalanceReport:
    metrics = (
        Metric(
            key="imbalance.ratio",
            name="Class imbalance ratio",
            definition="Images in the largest class divided by images in the smallest class.",
            method="max(class counts) / min(class counts)",
            value=report.imbalance_ratio,
            threshold=config.max_imbalance_ratio,
            status=_status(report.imbalance_ratio <= config.max_imbalance_ratio,
                           report.imbalance_ratio <= config.max_imbalance_ratio * 2),
            interpretation=(
                f"The largest class has {report.largest:,} images and the smallest {report.smallest:,}, "
                f"a {report.imbalance_ratio:.1f}x spread."
            ),
            recommendation=(
                "Use class-weighted loss or a balanced sampler, and report per-class metrics rather than accuracy."
                if report.imbalance_ratio > config.max_imbalance_ratio
                else "No action needed; the distribution is within the configured tolerance."
            ),
        ),
        Metric(
            key="imbalance.gini",
            name="Gini coefficient",
            definition="Inequality of the class-size distribution, 0 perfectly even and 1 maximally skewed.",
            method="Gini coefficient over class counts",
            value=report.gini,
            threshold=0.5,
            status=_status(report.gini <= 0.5, report.gini <= 0.7),
            interpretation=f"Class sizes have a Gini coefficient of {report.gini:.3f}.",
            recommendation=("Consider capping over-represented classes when building the benchmark."
                            if report.gini > 0.5 else "Distribution inequality is moderate."),
        ),
        Metric(
            key="imbalance.evenness",
            name="Distribution evenness",
            definition="Shannon entropy of class sizes normalised to [0, 1]; 1 means all classes are equal.",
            method="H(class counts) / log2(number of classes)",
            value=report.evenness,
            threshold=0.8,
            status=_status(report.evenness >= 0.8, report.evenness >= 0.6),
            interpretation=f"Evenness is {report.evenness:.3f} across {report.class_count} classes.",
            recommendation="Evenness below 0.6 usually needs resampling before benchmark use.",
        ),
        Metric(
            key="imbalance.rare_classes",
            name="Rare classes",
            definition=f"Classes holding fewer than {config.rare_class_ratio:.0%} of the mean class size.",
            method=f"count < {config.rare_class_ratio} x mean class size ({mean:.1f})",
            value=len(report.rare_classes),
            threshold=0,
            status=_status(not report.rare_classes, len(report.rare_classes) <= report.class_count * 0.1),
            interpretation=(
                f"{len(report.rare_classes)} of {report.class_count} classes are rare."
                if report.rare_classes else "No class falls below the rare-class threshold."
            ),
            recommendation=("Collect more images for these classes or merge them into a coarser label."
                            if report.rare_classes else "No action needed."),
            detail={"classes": list(report.rare_classes[:50])},
        ),
        Metric(
            key="imbalance.undersized_classes",
            name="Undersized classes",
            definition=f"Classes with fewer than the configured minimum of {config.min_class_size} images.",
            method=f"count < {config.min_class_size}",
            value=len(report.undersized_classes),
            threshold=0,
            status=_status(not report.undersized_classes, len(report.undersized_classes) <= 3),
            interpretation=(
                f"{len(report.undersized_classes)} classes hold fewer than {config.min_class_size} images."
                if report.undersized_classes else "Every class meets the configured minimum size."
            ),
            recommendation=("Statistical estimates for these classes will be unreliable; consider excluding them "
                            "via corpus.exclude_labels." if report.undersized_classes else "No action needed."),
            detail={"classes": list(report.undersized_classes[:50])},
        ),
        Metric(
            key="imbalance.long_tail",
            name="Long-tail concentration",
            definition="Number of classes needed to cover 80% of all images, and the share left to the tail.",
            method="Cumulative share over classes sorted by size",
            value=report.head_classes,
            threshold=None,
            status=MetricStatus.INFORMATIONAL,
            interpretation=(
                f"{report.head_classes} of {report.class_count} classes cover 80% of images; "
                f"the remaining tail holds {report.tail_share:.1%}."
            ),
            recommendation="Report tail-class performance separately; aggregate accuracy hides it.",
        ),
        Metric(
            key="imbalance.split_feasibility",
            name="Split feasibility",
            definition=(
                f"Classes too small to place at least one image in each split at ratios "
                f"{split.ratios.train:.2f}/{split.ratios.val:.2f}/{split.ratios.test:.2f}."
            ),
            method=f"count < max(min_samples_per_class, ceil(1 / smallest ratio)) = {required}",
            value=len(report.infeasible_classes),
            threshold=0,
            status=_status(not report.infeasible_classes, False),
            interpretation=(
                f"{len(report.infeasible_classes)} classes cannot be split into all three partitions."
                if report.infeasible_classes else "Every class can be split at the configured ratios."
            ),
            recommendation=(
                "Enable split.drop_classes_below_minimum, widen the ratios, or exclude these classes."
                if report.infeasible_classes else "No action needed."
            ),
            detail={"classes": list(report.infeasible_classes[:50]), "required_per_class": required},
        ),
    )
    return _replace_metrics(report, metrics)


def _status(healthy: bool, tolerable: bool) -> MetricStatus:
    if healthy:
        return MetricStatus.OK
    return MetricStatus.WARNING if tolerable else MetricStatus.CRITICAL


def _replace_metrics(report: ImbalanceReport, metrics: Sequence[Metric]) -> ImbalanceReport:
    import dataclasses

    return dataclasses.replace(report, metrics=tuple(metrics))


__all__ = ["ImbalanceReport", "analyze_imbalance"]
