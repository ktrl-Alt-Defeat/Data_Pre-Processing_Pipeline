"""Quality gate: measure every image objectively, then apply a configured policy.

Three phases, in this order because each depends on the previous:

1. **Analysis** - one decode per image yields blur, exposure, geometry and all
   three hashes. Parallel across threads; results are collected in input order.
2. **Duplicate detection** - a corpus-wide pass over the hashes gathered in (1).
   No image is re-read.
3. **Decision** - the gate combines metrics with the duplicate verdict and
   produces a score, a grade and an explained accept/warn/reject.

Every image leaves this stage with a complete :class:`QualityMetrics` object,
whether it was accepted or not.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import html
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..core.config import Config
from ..core.context import RunContext
from ..core.io import atomic_write_text, ensure_dir
from ..core.logging import StageTracker, stage_scope
from ..core.records import Corpus, DuplicateStatus, ImageRecord, PipelineStage, QualityMetrics
from ..corpus.metadata import MetadataWriter, build_frame
from .blur import BlurAnalyzer, BlurResult, laplacian_variance, tenengrad
from .brightness import ExposureAnalyzer, ExposureResult, colorfulness, shannon_entropy
from .duplicates import (
    DuplicateDetector,
    DuplicateEntry,
    DuplicateGroup,
    DuplicateLink,
    DuplicateResult,
    hamming_distance,
    perceptual_hash,
    to_hex,
)
from .quality_gate import (
    ACCEPT,
    REJECT,
    WARN,
    ImageAnalysis,
    ImageAnalyzer,
    QualityDecision,
    QualityGate,
    QualityReason,
    apply_decision,
)
from .resolution import ResolutionAnalyzer, ResolutionResult

_STAGE = PipelineStage.QUALITY


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Outcome of the quality stage, ready for serialisation and reporting."""

    run_id: str
    dataset_version: str | None
    started_at: str
    duration_seconds: float
    processed: int
    accepted: int
    warned: int
    rejected: int
    failed_analysis: int
    average_score: float
    median_score: float
    grades: dict[str, int] = field(default_factory=dict)
    rejections_by_code: dict[str, int] = field(default_factory=dict)
    duplicates: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "counts": {
                "processed": self.processed,
                "accepted": self.accepted,
                "warned": self.warned,
                "rejected": self.rejected,
                "failed_analysis": self.failed_analysis,
            },
            "scores": {"average": self.average_score, "median": self.median_score, "grades": self.grades},
            "rejections_by_code": self.rejections_by_code,
            "duplicates": self.duplicates,
            "thresholds": self.thresholds,
            "artifacts": self.artifacts,
        }


def assess_quality(config: Config, context: RunContext, corpus: Corpus) -> QualityReport:
    """Score every accepted image and apply the configured quality policy."""
    started = dt.datetime.now(dt.timezone.utc)

    with stage_scope(_STAGE, context.logger(_STAGE)) as tracker:
        candidates = corpus.accepted()
        analyses = _analyze(candidates, config, tracker)
        _store_hashes(candidates, analyses, config)

        gate = QualityGate(config.quality)
        duplicates = _detect_duplicates(candidates, analyses, config, gate, tracker)
        decisions = _decide(candidates, analyses, duplicates, gate, tracker)

        report = _build_report(context, corpus, decisions, duplicates, config, started)
        report = _write_artifacts(config, context, corpus, decisions, duplicates, report)
        _record_metrics(tracker, report)

    return report


# --------------------------------------------------------------------------- #
# Phase 1: measurement
# --------------------------------------------------------------------------- #


def _analyze(records: Sequence[ImageRecord], config: Config, tracker: StageTracker) -> list[ImageAnalysis]:
    if not records:
        return []

    analyzer = ImageAnalyzer(config.quality)
    workers = config.quality.workers or config.execution.resolved_workers
    if workers > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quality") as pool:
            analyses = list(pool.map(analyzer.analyze, records))
    else:
        analyses = [analyzer.analyze(record) for record in records]

    failed = sum(1 for analysis in analyses if analysis.failed)
    tracker.metrics(analysed=len(analyses), analysis_failures=failed, analysis_workers=workers)
    if failed:
        tracker.warn("quality.analysis_failures", count=failed)
    return analyses


def _store_hashes(records: Sequence[ImageRecord], analyses: Sequence[ImageAnalysis], config: Config) -> None:
    """Persist hashes on the records; leakage detection reuses them later."""
    hash_size = config.quality.duplicates.hash_size
    for record, analysis in zip(records, analyses, strict=True):
        if analysis.failed:
            continue
        record.content_hash = analysis.content_hash or record.content_hash
        record.pixel_hash = analysis.pixel_hash or record.pixel_hash
        record.perceptual_hash = to_hex(analysis.perceptual_hash, hash_size) or record.perceptual_hash


# --------------------------------------------------------------------------- #
# Phase 2: duplicates
# --------------------------------------------------------------------------- #


def _detect_duplicates(
    records: Sequence[ImageRecord],
    analyses: Sequence[ImageAnalysis],
    config: Config,
    gate: QualityGate,
    tracker: StageTracker,
) -> DuplicateResult:
    if not config.quality.duplicates.enabled:
        return DuplicateResult()

    # Representative selection can depend on quality, so score first — the gate
    # is pure, and this costs no additional decoding.
    entries = [
        DuplicateEntry(
            image_id=record.image_id,
            label=record.label,
            content_hash=analysis.content_hash,
            pixel_hash=analysis.pixel_hash,
            perceptual_hash=analysis.perceptual_hash,
            quality_score=gate.metrics(analysis, None, False).score,
            megapixels=analysis.resolution.megapixels if analysis.resolution else None,
        )
        for record, analysis in zip(records, analyses, strict=True)
        if not analysis.failed
    ]

    result = DuplicateDetector(config.quality.duplicates).detect(entries)
    statistics = result.statistics()
    tracker.metrics(**{f"duplicate_{key}": value for key, value in statistics.items()})
    if result.groups:
        tracker.info("quality.duplicates_found", **statistics)
    return result


# --------------------------------------------------------------------------- #
# Phase 3: decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Outcome:
    record: ImageRecord
    metrics: QualityMetrics
    decision: QualityDecision
    link: DuplicateLink | None


def _decide(
    records: Sequence[ImageRecord],
    analyses: Sequence[ImageAnalysis],
    duplicates: DuplicateResult,
    gate: QualityGate,
    tracker: StageTracker,
) -> list[_Outcome]:
    groups = {member: group.group_id for group in duplicates.groups for member in group.members}
    outcomes: list[_Outcome] = []
    for record, analysis in zip(records, analyses, strict=True):
        link = duplicates.links.get(record.image_id)
        metrics = gate.metrics(analysis, link, record.image_id in duplicates.representatives)
        decision = gate.decide(analysis, metrics, link)
        apply_decision(record, metrics, decision, _STAGE)

        # Cluster identity lives on the record so leakage analysis and grouped
        # splitting can keep a duplicate cluster inside one split.
        record.duplicate_group = groups.get(record.image_id)
        record.duplicate_of = link.duplicate_of if link else None

        if decision.action == REJECT:
            tracker.rejected()
        else:
            tracker.processed()
            if decision.action == WARN:
                # Counted, not logged: one line per borderline image would bury the
                # stage summary on a large corpus. The reasons live on the record
                # and in analytics/quality_metrics.csv.
                tracker.report.warnings += 1
        outcomes.append(_Outcome(record, metrics, decision, link))
    return outcomes


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _build_report(
    context: RunContext,
    corpus: Corpus,
    outcomes: Sequence[_Outcome],
    duplicates: DuplicateResult,
    config: Config,
    started: dt.datetime,
) -> QualityReport:
    scores = sorted(outcome.decision.score for outcome in outcomes)
    actions = Counter(outcome.decision.action for outcome in outcomes)
    quality = config.quality
    return QualityReport(
        run_id=context.run_id,
        dataset_version=corpus.version,
        started_at=started.isoformat(timespec="seconds"),
        duration_seconds=(dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
        processed=len(outcomes),
        accepted=actions[ACCEPT],
        warned=actions[WARN],
        rejected=actions[REJECT],
        failed_analysis=sum(
            1 for outcome in outcomes if any(reason.check == "decode" for reason in outcome.decision.reasons)
        ),
        average_score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        median_score=round(scores[len(scores) // 2], 6) if scores else 0.0,
        grades=dict(sorted(Counter(outcome.decision.grade for outcome in outcomes).items())),
        rejections_by_code=dict(
            sorted(Counter(o.decision.code.value for o in outcomes if o.decision.rejected and o.decision.code).items())
        ),
        duplicates=duplicates.statistics(),
        thresholds={
            "blur_min_score": quality.blur.min_score,
            "blur_method": quality.blur.method,
            "brightness_range": [quality.brightness.min_mean, quality.brightness.max_mean],
            "contrast_min": quality.contrast.min_value,
            "contrast_method": quality.contrast.method,
            "min_resolution": [quality.resolution.min_width, quality.resolution.min_height],
            "max_resolution": [quality.resolution.max_width, quality.resolution.max_height],
            "aspect_ratio_range": [quality.resolution.min_aspect_ratio, quality.resolution.max_aspect_ratio],
            "min_quality_score": quality.scoring.min_score,
            "warn_quality_score": quality.scoring.warn_score,
            "near_duplicate_distance": quality.duplicates.max_hamming_distance,
            "duplicate_policy": {"exact": quality.duplicates.exact_action, "near": quality.duplicates.near_action},
        },
    )


def _write_artifacts(
    config: Config,
    context: RunContext,
    corpus: Corpus,
    outcomes: Sequence[_Outcome],
    duplicates: DuplicateResult,
    report: QualityReport,
) -> QualityReport:
    layout = context.layout
    ensure_dir(layout.analytics_dir)

    metrics_path = _write_quality_metrics(layout.quality_metrics_csv, outcomes)
    duplicates_path = _write_duplicate_report(layout.duplicate_report_csv, duplicates, corpus)

    frame = build_frame(corpus)
    writer = MetadataWriter(layout, config)
    artifacts = {
        "quality_metrics": str(metrics_path),
        "duplicate_report": str(duplicates_path),
        "metadata_csv": str(writer.write_csv(frame)),
    }
    if config.packaging.manifest_parquet:
        artifacts["image_manifest"] = str(writer.write_parquet(frame))

    completed = dataclasses.replace(report, artifacts=artifacts)
    if config.reports.html:
        artifacts["quality_report"] = str(_write_html(layout.quality_report, completed, duplicates, context))
        completed = dataclasses.replace(completed, artifacts=artifacts)
    return completed


def _write_quality_metrics(path: Path, outcomes: Sequence[_Outcome]) -> Path:
    rows = []
    for outcome in outcomes:
        record, metrics, decision = outcome.record, outcome.metrics, outcome.decision
        row: dict[str, Any] = {
            "image_id": record.image_id,
            "dataset_name": record.dataset_name,
            "source_relpath": record.source_relpath,
            "label": record.label,
            "decision": decision.action,
            "grade": decision.grade,
            "quality_score": decision.score,
            "rejection_code": decision.code.value if decision.rejected and decision.code else None,
            "reason": decision.summary,
            "decided_at": decision.timestamp,
            "duplicate_of": outcome.link.duplicate_of if outcome.link else None,
        }
        row.update({key.removeprefix("quality_"): value for key, value in metrics.as_dict("quality_").items()})
        rows.append(row)

    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["image_id", "decision", "quality_score"])
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _write_duplicate_report(path: Path, duplicates: DuplicateResult, corpus: Corpus) -> Path:
    index = corpus.index()
    rows = []
    for group in duplicates.groups:
        for image_id in group.members:
            link = duplicates.links.get(image_id)
            record = index.get(image_id)
            rows.append(
                {
                    "group_id": group.group_id,
                    "group_size": group.size,
                    "image_id": image_id,
                    "role": "representative" if link is None else link.status.value,
                    "duplicate_of": link.duplicate_of if link else None,
                    "hamming_distance": link.distance if link else None,
                    "similarity": link.similarity if link else 1.0,
                    "dataset_name": record.dataset_name if record else None,
                    "label": record.label if record else None,
                    "source_relpath": record.source_relpath if record else None,
                    "perceptual_hash": record.perceptual_hash if record else None,
                    "status": record.status.value if record else None,
                }
            )
    columns = ["group_id", "group_size", "image_id", "role", "duplicate_of", "hamming_distance", "similarity",
               "dataset_name", "label", "source_relpath", "perceptual_hash", "status"]
    frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _write_html(path: Path, report: QualityReport, duplicates: DuplicateResult, context: RunContext) -> Path:
    manifest = context.manifest
    sections = [
        _html_table("Run", {
            "Run id": manifest.run_id,
            "Pipeline version": manifest.pipeline_version,
            "Dataset version": report.dataset_version,
            "Config hash": manifest.config_hash[:16],
            "Git commit": manifest.git_commit or "not a git checkout",
            "Started": report.started_at,
            "Duration": f"{report.duration_seconds:.2f}s",
        }),
        _html_table("Outcome", {
            "Images assessed": f"{report.processed:,}",
            "Accepted": f"{report.accepted:,}",
            "Accepted with warnings": f"{report.warned:,}",
            "Rejected": f"{report.rejected:,}",
            "Average score": f"{report.average_score:.4f}",
            "Median score": f"{report.median_score:.4f}",
        }),
        _html_table("Grades", {grade: f"{count:,}" for grade, count in report.grades.items()}),
        _html_table("Rejections", {code: f"{count:,}" for code, count in report.rejections_by_code.items()}
                    or {"none": "0"}),
        _html_table("Duplicates", {key: f"{value:,}" for key, value in report.duplicates.items()}),
        _html_table("Thresholds", {key: str(value) for key, value in report.thresholds.items()}),
        _html_group_table(duplicates),
    ]
    document = _HTML_TEMPLATE.format(
        title=f"Quality report - {manifest.run_id}",
        generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        body="\n".join(sections),
    )
    return atomic_write_text(path, document)


def _html_table(title: str, rows: dict[str, str]) -> str:
    body = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows.items()
    )
    return f"<section><h2>{html.escape(title)}</h2><table>{body}</table></section>"


def _html_group_table(duplicates: DuplicateResult, limit: int = 25) -> str:
    if not duplicates.groups:
        return "<section><h2>Duplicate groups</h2><p>No duplicate groups were found.</p></section>"
    header = "<tr><th>Group</th><th>Size</th><th>Exact</th><th>Near</th><th>Representative</th></tr>"
    rows = "\n".join(
        f"<tr><td>{html.escape(group.group_id)}</td><td>{group.size}</td><td>{group.exact_members}</td>"
        f"<td>{group.near_members}</td><td>{html.escape(group.representative)}</td></tr>"
        for group in sorted(duplicates.groups, key=lambda item: -item.size)[:limit]
    )
    note = (
        f"<p>Showing the {limit} largest of {len(duplicates.groups):,} groups; "
        "the complete listing is in analytics/duplicate_report.csv.</p>"
        if len(duplicates.groups) > limit
        else ""
    )
    return f"<section><h2>Duplicate groups</h2><table>{header}{rows}</table>{note}</section>"


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.05rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
 th, td {{ text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #e5e5e5; }}
 th {{ width: 16rem; font-weight: 600; color: #444; }}
 footer {{ margin-top: 3rem; font-size: 0.8rem; color: #777; }}
</style></head><body>
<h1>{title}</h1>
{body}
<footer>Generated {generated} by the crop-disease dataset preprocessing framework.</footer>
</body></html>
"""


def _record_metrics(tracker: StageTracker, report: QualityReport) -> None:
    tracker.metrics(
        images_processed=report.processed,
        accepted=report.accepted,
        warned=report.warned,
        rejected=report.rejected,
        average_quality_score=report.average_score,
        median_quality_score=report.median_score,
        duplicate_groups=report.duplicates.get("groups", 0),
        grades=report.grades,
        rejections_by_code=report.rejections_by_code,
    )


__all__ = [
    "ACCEPT",
    "REJECT",
    "WARN",
    "BlurAnalyzer",
    "BlurResult",
    "DuplicateDetector",
    "DuplicateEntry",
    "DuplicateGroup",
    "DuplicateLink",
    "DuplicateResult",
    "ExposureAnalyzer",
    "ExposureResult",
    "ImageAnalysis",
    "ImageAnalyzer",
    "QualityDecision",
    "QualityGate",
    "QualityReason",
    "QualityReport",
    "ResolutionAnalyzer",
    "ResolutionResult",
    "assess_quality",
    "colorfulness",
    "hamming_distance",
    "laplacian_variance",
    "perceptual_hash",
    "shannon_entropy",
    "tenengrad",
    "to_hex",
]
