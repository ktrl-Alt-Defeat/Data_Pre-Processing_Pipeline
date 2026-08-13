"""Validation stage: is each image and its metadata structurally sound?

This is verification, not cleaning. Nothing is repaired, resized or scored. The
stage runs the four validators cheapest-first — metadata and labels are pure
in-memory checks, so only the records that survive them ever get decoded — then
applies their findings to the corpus:

* an **error** rejects the record, preserving its provenance and recording the
  code, description, stage, validator and timestamp
* a **warning** annotates the record's operation trail and increments counters

Validators are pure functions of a record, so results are identical across runs
and independent of thread scheduling.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.config import Config, ValidationConfig
from ..core.context import RunContext
from ..core.io import ensure_dir, write_json
from ..core.logging import StageTracker, stage_scope
from ..core.records import (
    Corpus,
    ImageRecord,
    PipelineStage,
    RecordStatus,
    Rejection,
    ValidationIssue,
)
from ..corpus.metadata import MetadataWriter, build_frame
from .images import ImageObservation, ImageValidationResult, ImageValidator
from .integrity import IntegrityIssue, IntegrityValidator
from .labels import LabelValidator
from .metadata import MetadataValidator

_STAGE = PipelineStage.VALIDATION


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything the validation stage learned, in one serialisable object."""

    run_id: str
    dataset_version: str | None
    started_at: str
    duration_seconds: float
    total_records: int
    validated: int
    accepted: int
    rejected: int
    skipped: int
    warnings: int
    rejections_by_code: dict[str, int] = field(default_factory=dict)
    rejections_by_validator: dict[str, int] = field(default_factory=dict)
    validator_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    integrity_statistics: dict[str, Any] = field(default_factory=dict)
    integrity_issues: list[dict[str, Any]] = field(default_factory=list)
    mapping_issues: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "counts": {
                "total_records": self.total_records,
                "validated": self.validated,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "skipped": self.skipped,
                "warnings": self.warnings,
            },
            "rejections_by_code": self.rejections_by_code,
            "rejections_by_validator": self.rejections_by_validator,
            "validator_metrics": self.validator_metrics,
            "integrity": {"statistics": self.integrity_statistics, "issues": self.integrity_issues},
            "mapping_issues": self.mapping_issues,
            "artifacts": self.artifacts,
        }


def validate_corpus(config: Config, context: RunContext, corpus: Corpus) -> ValidationReport:
    """Validate every record in the corpus and return a structured report.

    Mutates records only to record validation outcomes: status, rejections and
    the operation trail. Canonical metadata written by the corpus stage is never
    touched.
    """
    settings = config.validation
    started = dt.datetime.now(dt.timezone.utc)

    with stage_scope(_STAGE, context.logger(_STAGE)) as tracker:
        metadata_validator = MetadataValidator(settings.metadata)
        label_validator = LabelValidator(settings.labels, corpus.labels)
        image_validator = ImageValidator(settings)

        mapping_issues = label_validator.validate_mapping() if settings.labels.enabled else []
        _report_mapping_issues(mapping_issues, tracker)

        candidates = _screen_records(corpus, settings, metadata_validator, label_validator, tracker)
        if settings.validate_images:
            _validate_images(candidates, config, image_validator, tracker)

        integrity_validator = IntegrityValidator(settings.integrity, files_already_checked=settings.validate_images)
        integrity_issues = integrity_validator.validate(corpus, context) if settings.integrity.enabled else []
        _apply_integrity_issues(corpus, integrity_issues, tracker)

        _finalize_statuses(corpus)
        tracker.processed(len(corpus.accepted()))
        report = _build_report(
            corpus=corpus,
            context=context,
            tracker=tracker,
            started=started,
            mapping_issues=mapping_issues,
            integrity_issues=integrity_issues,
            integrity_statistics=integrity_validator.statistics,
            validators={
                "metadata": metadata_validator.metrics,
                "labels": label_validator.metrics,
                "images": image_validator.metrics,
                "integrity": integrity_validator.metrics,
            },
        )
        report = _write_artifacts(config, context, corpus, report)
        _record_stage_metrics(tracker, report)

    return report


# --------------------------------------------------------------------------- #
# Record screening
# --------------------------------------------------------------------------- #


def _screen_records(
    corpus: Corpus,
    settings: ValidationConfig,
    metadata_validator: MetadataValidator,
    label_validator: LabelValidator,
    tracker: StageTracker,
) -> list[ImageRecord]:
    """Run the in-memory validators; returns the records worth decoding."""
    candidates: list[ImageRecord] = []

    for record in corpus.records:
        if record.is_rejected:
            tracker.skipped()
            continue

        issues: list[ValidationIssue] = []
        if settings.metadata.enabled:
            issues.extend(metadata_validator.validate(record))
        if settings.labels.enabled and not issues:
            issues.extend(label_validator.validate(record))

        if _apply_issues(record, issues, tracker):
            candidates.append(record)
    return candidates


def _validate_images(
    records: Sequence[ImageRecord],
    config: Config,
    validator: ImageValidator,
    tracker: StageTracker,
) -> None:
    """Decode and validate images, in parallel when it pays off."""
    if not records:
        return

    workers = config.validation.workers or config.execution.resolved_workers
    if workers > 1 and len(records) > 1:
        # Decoding is IO- and C-bound, so threads scale without pickling records;
        # executor.map preserves input order, keeping results deterministic.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="validate") as pool:
            results = list(pool.map(validator.validate, records))
    else:
        results = [validator.validate(record) for record in records]

    for record, result in zip(records, results, strict=True):
        _apply_image_result(record, result, config.validation.record_observations, tracker)


def _apply_image_result(
    record: ImageRecord,
    result: ImageValidationResult,
    record_observations: bool,
    tracker: StageTracker,
) -> None:
    if _apply_issues(record, list(result.issues), tracker) and result.observation and record_observations:
        _store_observation(record, result.observation)


def _store_observation(record: ImageRecord, observation: ImageObservation) -> None:
    """Persist observed technical facts; canonical fields are never overwritten."""
    record.width = observation.width
    record.height = observation.height
    record.channels = observation.channels
    record.color_mode = observation.color_mode
    record.image_format = observation.image_format
    record.exif_orientation = observation.exif_orientation
    record.file_size_bytes = observation.file_size_bytes
    record.record_operation(
        _STAGE,
        "validate_image",
        width=observation.width,
        height=observation.height,
        color_mode=observation.color_mode,
        exif_transposed=observation.exif_transposed,
    )


# --------------------------------------------------------------------------- #
# Issue application
# --------------------------------------------------------------------------- #


def _apply_issues(record: ImageRecord, issues: Iterable[ValidationIssue], tracker: StageTracker) -> bool:
    """Apply findings to a record; returns True when the record survives.

    Only rejections and warnings are counted here. ``processed`` is tallied once
    per surviving record at the end of the stage, so a record that passes both
    the in-memory and the decode validators is never counted twice.
    """
    survived = True
    for issue in issues:
        if issue.is_error:
            record.reject(
                _STAGE,
                issue.code,
                issue.message,
                validator=issue.validator,
                timestamp=issue.timestamp,
                **dict(issue.detail),
            )
            tracker.rejected()
            survived = False
        else:
            record.record_operation(
                _STAGE,
                f"warning:{issue.code.value}",
                validator=issue.validator,
                message=issue.message,
                timestamp=issue.timestamp,
            )
            tracker.warn(
                "validation.warning",
                image_id=issue.image_id,
                validator=issue.validator,
                code=issue.code.value,
                message=issue.message,
            )
    return survived


def _apply_integrity_issues(corpus: Corpus, issues: Sequence[IntegrityIssue], tracker: StageTracker) -> None:
    for issue in issues:
        if not issue.is_error:
            tracker.warn("integrity.warning", check=issue.check, message=issue.message, **dict(issue.detail))
            continue

        if not issue.record_indices:
            tracker.error("integrity.failed", check=issue.check, message=issue.message)
            continue

        for position in issue.record_indices:
            record = corpus.records[position]
            if record.is_rejected:
                continue
            record.reject(
                _STAGE,
                issue.code,
                issue.message,
                validator="integrity",
                check=issue.check,
                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            )
            tracker.rejected()


def _report_mapping_issues(issues: Sequence[ValidationIssue], tracker: StageTracker) -> None:
    for issue in issues:
        tracker.warn("labels.mapping_defect", message=issue.message, **dict(issue.detail))


def _finalize_statuses(corpus: Corpus) -> None:
    """Anything still pending after validation has passed every enabled check."""
    for record in corpus.records:
        if record.status is RecordStatus.PENDING:
            record.accept()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _build_report(
    corpus: Corpus,
    context: RunContext,
    tracker: StageTracker,
    started: dt.datetime,
    mapping_issues: Sequence[ValidationIssue],
    integrity_issues: Sequence[IntegrityIssue],
    integrity_statistics: dict[str, Any],
    validators: dict[str, dict[str, int]],
) -> ValidationReport:
    by_code: dict[str, int] = {}
    by_validator: dict[str, int] = {}
    for record in corpus.records:
        rejection = _validation_rejection(record)
        if rejection is None:
            continue
        by_code[rejection.code.value] = by_code.get(rejection.code.value, 0) + 1
        validator = str(rejection.detail.get("validator", "unknown"))
        by_validator[validator] = by_validator.get(validator, 0) + 1

    return ValidationReport(
        run_id=context.run_id,
        dataset_version=corpus.version,
        started_at=started.isoformat(timespec="seconds"),
        duration_seconds=(dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
        total_records=len(corpus.records),
        validated=tracker.report.processed + tracker.report.rejected,
        accepted=len(corpus.accepted()),
        rejected=len(corpus.rejected()),
        skipped=tracker.report.skipped,
        warnings=tracker.report.warnings,
        rejections_by_code=dict(sorted(by_code.items())),
        rejections_by_validator=dict(sorted(by_validator.items())),
        validator_metrics=validators,
        integrity_statistics=integrity_statistics,
        integrity_issues=[issue.as_dict() for issue in integrity_issues],
        mapping_issues=[issue.as_dict() for issue in mapping_issues],
    )


def _validation_rejection(record: ImageRecord) -> Rejection | None:
    return next((r for r in record.rejections if r.stage is _STAGE), None)


def _write_artifacts(config: Config, context: RunContext, corpus: Corpus, report: ValidationReport) -> ValidationReport:
    """Write validation artefacts and refresh the metadata tables with new statuses.

    The report JSON is written last so that the on-disk document lists the
    artefacts that accompany it.
    """
    directory = ensure_dir(Path(config.validation.output_dir))
    frame = build_frame(corpus)

    accepted_path = directory / "accepted_images.csv"
    rejected_path = directory / "rejected_images.csv"
    report_path = directory / "validation_report.json"
    rejected_mask = frame["rejected"].fillna(False).astype(bool)
    frame[~rejected_mask].to_csv(accepted_path, index=False, encoding="utf-8")
    frame[rejected_mask].to_csv(rejected_path, index=False, encoding="utf-8")

    writer = MetadataWriter(context.layout, config)
    artifacts = {
        "validation_report": str(report_path),
        "accepted_images": str(accepted_path),
        "rejected_images": str(rejected_path),
        "metadata_csv": str(writer.write_csv(frame)),
    }
    if config.packaging.manifest_parquet:
        artifacts["image_manifest"] = str(writer.write_parquet(frame))

    completed = dataclasses.replace(report, artifacts=artifacts)
    write_json(report_path, completed.as_dict())
    return completed


def _record_stage_metrics(tracker: StageTracker, report: ValidationReport) -> None:
    tracker.metrics(
        total_records=report.total_records,
        validated=report.validated,
        accepted=report.accepted,
        rejected=report.rejected,
        skipped=report.skipped,
        rejections_by_code=report.rejections_by_code,
        rejections_by_validator=report.rejections_by_validator,
        integrity_issues=len(report.integrity_issues),
    )


__all__ = [
    "ImageObservation",
    "ImageValidationResult",
    "ImageValidator",
    "IntegrityIssue",
    "IntegrityValidator",
    "LabelValidator",
    "MetadataValidator",
    "ValidationReport",
    "validate_corpus",
]
