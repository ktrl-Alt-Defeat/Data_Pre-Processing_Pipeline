"""Analysis stage: judge the corpus, change nothing.

Runs the four analyses over the manifest and the profile, combines their
findings into a configurable dataset quality score, and writes
``leakage_report.csv``, ``dataset_score.json`` and ``analysis_report.html``.

Read-only, like profiling: every input is already on the manifest, so no image
is opened here at all.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from ..core.config import Config, DatasetScoreConfig
from ..core.context import RunContext
from ..core.io import ensure_dir, write_json
from ..core.logging import StageTracker, stage_scope
from ..core.records import Corpus, Metric, MetricStatus, PipelineStage, RunManifest
from ..profiling.profiler import DatasetProfile
from ..profiling.statistics import clamp
from .bias import BiasReport, analyze_bias
from .diversity import DiversityReport, analyze_diversity
from .imbalance import ImbalanceReport, analyze_imbalance
from .leakage import LeakageReport, analyze_leakage

_STAGE = PipelineStage.ANALYSIS


@dataclass(frozen=True, slots=True)
class DatasetScore:
    """Weighted dataset quality score and the components behind it."""

    value: float
    grade: str
    components: Mapping[str, float]
    weights: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.value,
            "grade": self.grade,
            "components": dict(self.components),
            "weights": dict(self.weights),
            "contributions": {
                name: round(self.components[name] * weight / sum(self.weights.values()), 6)
                for name, weight in self.weights.items()
                if name in self.components
            },
        }


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Everything the analysis stage concluded."""

    manifest: RunManifest
    dataset_fingerprint: str | None
    config_fingerprint: str
    generated_at: str
    imbalance: ImbalanceReport
    diversity: DiversityReport
    bias: BiasReport
    leakage: LeakageReport
    score: DatasetScore
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return (*self.imbalance.metrics, *self.diversity.metrics, *self.bias.metrics, *self.leakage.metrics)

    @property
    def critical(self) -> tuple[Metric, ...]:
        return tuple(metric for metric in self.metrics if metric.status is MetricStatus.CRITICAL)

    @property
    def warnings(self) -> tuple[Metric, ...]:
        return tuple(metric for metric in self.metrics if metric.status is MetricStatus.WARNING)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "generated_at": self.generated_at,
            "score": self.score.as_dict(),
            "imbalance": self.imbalance.as_dict(),
            "diversity": self.diversity.as_dict(),
            "bias": self.bias.as_dict(),
            "leakage": self.leakage.as_dict(),
            "summary": {
                "metrics": len(self.metrics),
                "critical": [metric.key for metric in self.critical],
                "warnings": [metric.key for metric in self.warnings],
            },
            "artifacts": dict(self.artifacts),
        }


def analyze_dataset(config: Config, context: RunContext, corpus: Corpus, profile: DatasetProfile) -> AnalysisReport:
    """Run every analysis and write the analytical artefacts."""
    with stage_scope(_STAGE, context.logger(_STAGE)) as tracker:
        frame = corpus.to_frame(accepted_only=not config.profiling.include_rejected)
        analysis = config.analysis

        imbalance = analyze_imbalance(frame, analysis.imbalance, config.split)
        diversity = analyze_diversity(frame, analysis.diversity)
        bias = analyze_bias(frame, analysis.bias)
        # Leakage sees every record: cluster membership is a property of the corpus,
        # and a duplicate the gate rejected still evidences a label or source conflict.
        leakage = analyze_leakage(corpus.to_frame(), analysis.leakage, config.split)
        score = compute_dataset_score(analysis.score, profile, imbalance, diversity, bias, leakage)

        report = AnalysisReport(
            manifest=context.manifest,
            dataset_fingerprint=corpus.fingerprint,
            config_fingerprint=context.config_fingerprint,
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            imbalance=imbalance,
            diversity=diversity,
            bias=bias,
            leakage=leakage,
            score=score,
        )
        report = dataclasses.replace(report, artifacts=_write_artifacts(context, report))

        tracker.processed(len(frame))
        _record_metrics(tracker, report)
        _log_findings(tracker, report)

    return report


def compute_dataset_score(
    config: DatasetScoreConfig,
    profile: DatasetProfile,
    imbalance: ImbalanceReport,
    diversity: DiversityReport,
    bias: BiasReport,
    leakage: LeakageReport,
) -> DatasetScore:
    """Combine analytical components into one configurable [0, 1] score."""
    quality = profile.numeric.get("quality_score")
    megapixels = profile.numeric.get("megapixels")
    resolution_cv = megapixels.coefficient_of_variation if megapixels else None

    components = {
        "class_balance": clamp(1.0 - imbalance.gini),
        "duplicate_ratio": clamp(1.0 - (diversity.duplicate_ratio + diversity.near_duplicate_ratio)),
        "diversity": clamp(diversity.normalized_diversity / 0.5),
        "quality_score": clamp(quality.mean if quality and quality.mean is not None else 0.0),
        "label_consistency": clamp(bias.label_consistency),
        "resolution_consistency": clamp(1.0 - (resolution_cv or 0.0)),
        "source_diversity": clamp(bias.source_diversity),
        "leakage_containment": clamp(1.0 - leakage.random_split_leak_probability),
    }

    weights = {name: weight for name, weight in config.weights.items() if name in components}
    total = sum(weights.values())
    if not config.enabled or total <= 0:
        return DatasetScore(0.0, config.fallback_grade, components, weights)

    value = round(sum(components[name] * weight for name, weight in weights.items()) / total, 6)
    return DatasetScore(value, _grade(value, config), components, weights)


def _grade(score: float, config: DatasetScoreConfig) -> str:
    for grade, threshold in sorted(config.grade_thresholds.items(), key=lambda item: -item[1]):
        if score >= threshold:
            return grade
    return config.fallback_grade


def _write_artifacts(context: RunContext, report: AnalysisReport) -> dict[str, str]:
    layout = context.layout
    ensure_dir(layout.analytics_dir)

    table = report.leakage.table if not report.leakage.table.empty else pd.DataFrame(
        columns=["group_id", "group_size", "risk", "labels", "sources", "image_ids",
                 "distinct_labels", "distinct_sources", "retained_images"]
    )
    table.to_csv(layout.leakage_report_csv, index=False, encoding="utf-8")

    score_payload = report.score.as_dict()
    score_payload.update(
        {
            "manifest": report.manifest.as_dict(),
            "dataset_fingerprint": report.dataset_fingerprint,
            "config_fingerprint": report.config_fingerprint,
            "generated_at": report.generated_at,
            "metrics": [metric.as_dict() for metric in report.metrics],
        }
    )
    write_json(layout.dataset_score_json, score_payload)
    return {
        "leakage_report": str(layout.leakage_report_csv),
        "dataset_score": str(layout.dataset_score_json),
    }


def render_analysis_report(context: RunContext, report: AnalysisReport, profile: DatasetProfile) -> str:
    """Render ``reports/analysis_report.html``."""
    from reports.report import HtmlDocument, manifest_rows

    document = HtmlDocument(
        title="Dataset analysis report",
        subtitle=f"Dataset score {report.score.value:.3f} (grade {report.score.grade})",
    )
    document.key_values("Provenance", manifest_rows(report.manifest, report.dataset_fingerprint,
                                                    report.config_fingerprint))
    document.key_values("Dataset score", {
        "Score": f"{report.score.value:.4f}",
        "Grade": report.score.grade,
        **{f"Component: {name}": f"{value:.4f}" for name, value in sorted(report.score.components.items())},
    })
    document.key_values("Findings", {
        "Metrics evaluated": len(report.metrics),
        "Critical": ", ".join(metric.key for metric in report.critical) or "none",
        "Warnings": ", ".join(metric.key for metric in report.warnings) or "none",
    })
    document.metrics("Class imbalance", report.imbalance.metrics)
    document.metrics("Diversity", report.diversity.metrics)
    document.metrics("Source and label bias", report.bias.metrics)
    document.metrics("Leakage risk", report.leakage.metrics)

    if not report.leakage.table.empty:
        document.table("Largest duplicate clusters", report.leakage.table.drop(columns=["image_ids"]), limit=40,
                       note="Full listing in analytics/leakage_report.csv.")
    document.table("Class distribution", profile.classes, limit=100)
    return str(document.write(context.layout.analysis_report, report.manifest))


def _record_metrics(tracker: StageTracker, report: AnalysisReport) -> None:
    tracker.metrics(
        dataset_score=report.score.value,
        dataset_grade=report.score.grade,
        classes=report.imbalance.class_count,
        imbalance_ratio=report.imbalance.imbalance_ratio,
        diversity=report.diversity.normalized_diversity,
        duplicate_clusters=report.leakage.clusters,
        cross_class_clusters=report.leakage.cross_class_clusters,
        critical_findings=[metric.key for metric in report.critical],
        warning_findings=[metric.key for metric in report.warnings],
    )


def _log_findings(tracker: StageTracker, report: AnalysisReport) -> None:
    for metric in report.critical:
        tracker.warn("analysis.critical", metric=metric.key, value=metric.value, threshold=metric.threshold,
                     interpretation=metric.interpretation)
    tracker.info("analysis.completed", score=report.score.value, grade=report.score.grade,
                 critical=len(report.critical), warnings=len(report.warnings))


__all__ = [
    "AnalysisReport",
    "BiasReport",
    "DatasetScore",
    "DiversityReport",
    "ImbalanceReport",
    "LeakageReport",
    "analyze_bias",
    "analyze_dataset",
    "analyze_diversity",
    "analyze_imbalance",
    "analyze_leakage",
    "compute_dataset_score",
    "render_analysis_report",
]
