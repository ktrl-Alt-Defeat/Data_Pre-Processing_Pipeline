"""Canonical in-memory representation of the corpus.

Every stage reads and enriches the same :class:`ImageRecord` objects; nothing is
ever dropped from the list. Rejected images keep their record with a
:class:`Rejection` attached, which is what makes the mandated traceability
guarantees possible.

:class:`Provenance` is frozen and assigned once at ingestion: origin data is
structurally incapable of being overwritten by a later stage.

These are pure containers with no dependencies inside the framework.
Construction logic lives in ``preprocessing.corpus``, enrichment logic in the
stage packages.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas off the import path
    import pandas as pd


class PipelineStage(str, Enum):
    """Pipeline stages, in execution order.

    The single source of truth for stage identity: logging, reports, summaries
    and the orchestrator all key off these members rather than string literals.
    """

    CORPUS = "corpus"
    VALIDATION = "validation"
    QUALITY = "quality"
    PROFILING = "profiling"
    ANALYSIS = "analysis"
    SPLITTING = "splitting"
    TRANSFORMS = "transforms"
    DATALOADER = "dataloader"
    PACKAGING = "packaging"
    REPORTING = "reporting"

    @property
    def title(self) -> str:
        """Display name for reports, e.g. ``Quality``."""
        return self.value.replace("_", " ").title()

    @property
    def logger_name(self) -> str:
        return f"preprocessing.{self.value}"

    @classmethod
    def execution_order(cls) -> tuple["PipelineStage", ...]:
        return tuple(cls)


class RecordStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNASSIGNED = "unassigned"


class RejectionCode(str, Enum):
    """Machine-readable rejection reasons; the value is what lands in reports."""

    # --- corpus / validation ------------------------------------------------ #
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNREADABLE_FILE = "unreadable_file"
    EMPTY_FILE = "empty_file"
    DECODE_FAILED = "decode_failed"
    TRUNCATED_FILE = "truncated_file"
    INVALID_DIMENSIONS = "invalid_dimensions"
    INVALID_COLOR_MODE = "invalid_color_mode"
    OVERSIZED_IMAGE = "oversized_image"
    LABEL_MISSING = "label_missing"
    LABEL_UNMAPPED = "label_unmapped"
    LABEL_EXCLUDED = "label_excluded"
    METADATA_INVALID = "metadata_invalid"

    # --- quality gate ------------------------------------------------------- #
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"
    LOW_RESOLUTION = "low_resolution"
    EXTREME_ASPECT_RATIO = "extreme_aspect_ratio"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    LOW_QUALITY_SCORE = "low_quality_score"

    # --- splitting / packaging ---------------------------------------------- #
    CLASS_BELOW_MINIMUM = "class_below_minimum"
    SPLIT_LEAKAGE = "split_leakage"
    WRITE_FAILED = "write_failed"

    @property
    def description(self) -> str:
        """Human-readable explanation, used in HTML/PDF reports."""
        return _REJECTION_DESCRIPTIONS.get(self, self.value.replace("_", " "))


_REJECTION_DESCRIPTIONS: dict[RejectionCode, str] = {
    RejectionCode.UNSUPPORTED_FORMAT: "File extension or codec is not in the configured allow-list",
    RejectionCode.UNREADABLE_FILE: "File could not be read from disk",
    RejectionCode.EMPTY_FILE: "File is zero bytes",
    RejectionCode.DECODE_FAILED: "Image data could not be decoded",
    RejectionCode.TRUNCATED_FILE: "Image data ends prematurely",
    RejectionCode.INVALID_DIMENSIONS: "Image dimensions are missing, zero, or degenerate",
    RejectionCode.INVALID_COLOR_MODE: "Colour mode cannot be converted to RGB",
    RejectionCode.OVERSIZED_IMAGE: "Pixel count exceeds the configured safety limit",
    RejectionCode.LABEL_MISSING: "No class label could be derived from the folder layout",
    RejectionCode.LABEL_UNMAPPED: "Raw label has no entry in the harmonised label space",
    RejectionCode.LABEL_EXCLUDED: "Label is outside the configured include/exclude filters",
    RejectionCode.METADATA_INVALID: "Harmonised metadata failed validation",
    RejectionCode.BLURRY: "Sharpness below the configured threshold",
    RejectionCode.TOO_DARK: "Mean brightness below the configured threshold",
    RejectionCode.TOO_BRIGHT: "Mean brightness above the configured threshold",
    RejectionCode.LOW_CONTRAST: "Contrast below the configured threshold",
    RejectionCode.LOW_RESOLUTION: "Resolution below the configured minimum",
    RejectionCode.EXTREME_ASPECT_RATIO: "Aspect ratio outside the configured range",
    RejectionCode.EXACT_DUPLICATE: "Byte- or pixel-identical to a retained image",
    RejectionCode.NEAR_DUPLICATE: "Perceptually near-identical to a retained image",
    RejectionCode.LOW_QUALITY_SCORE: "Aggregate quality score below the configured minimum",
    RejectionCode.CLASS_BELOW_MINIMUM: "Class has too few images to be split safely",
    RejectionCode.SPLIT_LEAKAGE: "Image duplicates an image already assigned to another split",
    RejectionCode.WRITE_FAILED: "Image could not be written to the output package",
}


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable identity of one pipeline run; referenced by every report."""

    run_id: str
    pipeline_version: str
    config_hash: str
    python_version: str
    platform: str
    started_at: str
    dataset_version: str | None = None
    git_commit: str | None = None
    finished_at: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def with_dataset_version(self, version: str) -> "RunManifest":
        """New manifest carrying the corpus version resolved during ingestion."""
        return dataclasses.replace(self, dataset_version=version)

    def finished(self, at: str | None = None) -> "RunManifest":
        """New manifest stamped with a completion time."""
        stamp = at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        return dataclasses.replace(self, finished_at=stamp)

    @property
    def duration_seconds(self) -> float | None:
        if not self.finished_at:
            return None
        started = dt.datetime.fromisoformat(self.started_at)
        return (dt.datetime.fromisoformat(self.finished_at) - started).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "dataset_version": self.dataset_version,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "platform": self.platform,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable origin of one image. Assigned at ingestion, never rewritten."""

    dataset_name: str
    dataset_version: str
    source_root: Path
    source_path: Path
    source_relpath: str
    source_class: str
    original_filename: str
    source_split: str | None = None
    first_seen_run_id: str | None = None
    first_seen_at: str | None = None
    size_bytes: int | None = None
    modified_at: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_root": str(self.source_root),
            "source_path": str(self.source_path),
            "source_relpath": self.source_relpath,
            "source_class": self.source_class,
            "source_split": self.source_split,
            "original_filename": self.original_filename,
            "first_seen_run_id": self.first_seen_run_id,
            "first_seen_at": self.first_seen_at,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class Rejection:
    """Why an image left the pipeline, and at which stage."""

    stage: PipelineStage
    code: RejectionCode
    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "code": self.code.value,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class Operation:
    """A preprocessing operation applied to one image (provenance trail)."""

    stage: PipelineStage
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage.value, "name": self.name, "params": dict(self.params)}


class Severity(str, Enum):
    """Whether a finding removes an image from the corpus or merely annotates it."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A finding produced by a validator.

    Validators are pure: they report issues and never mutate records. The owning
    stage decides what an issue means, which keeps validation deterministic and
    trivially testable. ``validator`` and ``timestamp`` travel with the issue so
    every rejection can name its origin.
    """

    image_id: str
    validator: str
    code: RejectionCode
    message: str
    severity: Severity = Severity.ERROR
    detail: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"))

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "validator": self.validator,
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "detail": dict(self.detail),
        }


class MetricStatus(str, Enum):
    """How a measured analytical metric compares against its configured threshold."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    INFORMATIONAL = "informational"


@dataclass(frozen=True, slots=True)
class Metric:
    """One analytical finding, self-describing enough to print in a report.

    Analysis output is only useful if a reader can tell what was measured, how,
    against what threshold, and what to do about it, so every metric carries its
    own definition, interpretation and recommendation.
    """

    key: str
    name: str
    definition: str
    method: str
    value: float | int | str | None
    status: MetricStatus = MetricStatus.INFORMATIONAL
    threshold: float | None = None
    interpretation: str = ""
    recommendation: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status in (MetricStatus.OK, MetricStatus.INFORMATIONAL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "definition": self.definition,
            "method": self.method,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status.value,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
            "detail": dict(self.detail),
        }


class DuplicateStatus(str, Enum):
    """An image's standing within its duplicate cluster."""

    UNIQUE = "unique"
    REPRESENTATIVE = "representative"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass(slots=True)
class QualityMetrics:
    """Per-image quality measurements produced by the quality gate.

    ``blur_score`` is the raw sharpness metric that configured thresholds are
    compared against; ``sharpness`` is the same quantity normalised to [0, 1]
    for the weighted overall score.
    """

    blur_score: float | None = None
    sharpness: float | None = None
    brightness: float | None = None
    contrast: float | None = None
    colorfulness: float | None = None
    entropy: float | None = None
    width: int | None = None
    height: int | None = None
    megapixels: float | None = None
    aspect_ratio: float | None = None
    duplicate_status: DuplicateStatus | None = None
    perceptual_similarity: float | None = None
    score: float | None = None
    grade: str | None = None

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}blur_score": self.blur_score,
            f"{prefix}sharpness": self.sharpness,
            f"{prefix}brightness": self.brightness,
            f"{prefix}contrast": self.contrast,
            f"{prefix}colorfulness": self.colorfulness,
            f"{prefix}entropy": self.entropy,
            f"{prefix}width": self.width,
            f"{prefix}height": self.height,
            f"{prefix}megapixels": self.megapixels,
            f"{prefix}aspect_ratio": self.aspect_ratio,
            f"{prefix}duplicate_status": self.duplicate_status.value if self.duplicate_status else None,
            f"{prefix}perceptual_similarity": self.perceptual_similarity,
            f"{prefix}score": self.score,
            f"{prefix}grade": self.grade,
        }

    @property
    def measured(self) -> bool:
        return self.score is not None


@dataclass(slots=True)
class ImageRecord:
    """One image, tracked from raw source file to packaged output."""

    image_id: str
    provenance: Provenance

    label: str = ""
    class_index: int | None = None
    crop: str | None = None
    condition: str | None = None
    label_rule: str | None = None

    status: RecordStatus = RecordStatus.PENDING
    split: Split = Split.UNASSIGNED

    width: int | None = None
    height: int | None = None
    channels: int | None = None
    image_format: str | None = None
    color_mode: str | None = None
    file_size_bytes: int | None = None
    exif_orientation: int | None = None

    content_hash: str | None = None
    pixel_hash: str | None = None
    perceptual_hash: str | None = None
    duplicate_of: str | None = None
    duplicate_group: str | None = None

    quality: QualityMetrics = field(default_factory=QualityMetrics)
    rejections: list[Rejection] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None

    # --- provenance accessors ----------------------------------------------- #

    @property
    def dataset_name(self) -> str:
        return self.provenance.dataset_name

    @property
    def dataset_version(self) -> str:
        return self.provenance.dataset_version

    @property
    def source_path(self) -> Path:
        return self.provenance.source_path

    @property
    def source_relpath(self) -> str:
        return self.provenance.source_relpath

    @property
    def source_class(self) -> str:
        return self.provenance.source_class

    @property
    def source_split(self) -> str | None:
        return self.provenance.source_split

    @property
    def original_filename(self) -> str:
        return self.provenance.original_filename

    # --- state transitions --------------------------------------------------- #

    def accept(self) -> None:
        """Mark the record as having passed every gate applied so far."""
        if self.status is not RecordStatus.REJECTED:
            self.status = RecordStatus.ACCEPTED

    def reject(
        self,
        stage: PipelineStage,
        code: RejectionCode,
        message: str | None = None,
        **detail: Any,
    ) -> Rejection:
        """Reject the image, keeping the record for reporting and traceability."""
        rejection = Rejection(stage, code, message or code.description, detail)
        self.rejections.append(rejection)
        self.status = RecordStatus.REJECTED
        return rejection

    def record_operation(self, stage: PipelineStage, name: str, **params: Any) -> None:
        """Append an entry to this image's provenance trail."""
        self.operations.append(Operation(stage, name, params))

    # --- derived views -------------------------------------------------------- #

    @property
    def is_accepted(self) -> bool:
        return self.status is RecordStatus.ACCEPTED

    @property
    def is_rejected(self) -> bool:
        return self.status is RecordStatus.REJECTED

    @property
    def primary_rejection(self) -> Rejection | None:
        return self.rejections[0] if self.rejections else None

    @property
    def aspect_ratio(self) -> float | None:
        if not self.width or not self.height:
            return None
        return self.width / self.height

    @property
    def megapixels(self) -> float | None:
        if not self.width or not self.height:
            return None
        return (self.width * self.height) / 1_000_000

    @property
    def resolution(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return self.width, self.height

    def to_row(self) -> dict[str, Any]:
        """Flat, serialisable row for metadata.csv / image_manifest.parquet.

        One row answers every traceability question: origin dataset and version,
        original folder, original label, canonical label, applied operations,
        rejection reason, quality score, split and the run that first saw it.
        """
        primary = self.primary_rejection
        prov = self.provenance
        row: dict[str, Any] = {
            "image_id": self.image_id,
            "dataset_name": prov.dataset_name,
            "dataset_version": prov.dataset_version,
            "source_root": str(prov.source_root),
            "source_path": str(prov.source_path),
            "source_relpath": prov.source_relpath,
            "source_class": prov.source_class,
            "source_split": prov.source_split,
            "original_filename": prov.original_filename,
            "first_seen_run_id": prov.first_seen_run_id,
            "first_seen_at": prov.first_seen_at,
            "source_size_bytes": prov.size_bytes,
            "source_modified_at": prov.modified_at,
            "label": self.label,
            "class_index": self.class_index,
            "crop": self.crop,
            "condition": self.condition,
            "label_rule": self.label_rule,
            "status": self.status.value,
            "split": self.split.value,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "aspect_ratio": self.aspect_ratio,
            "megapixels": self.megapixels,
            "image_format": self.image_format,
            "color_mode": self.color_mode,
            "file_size_bytes": self.file_size_bytes,
            "exif_orientation": self.exif_orientation,
            "content_hash": self.content_hash,
            "pixel_hash": self.pixel_hash,
            "perceptual_hash": self.perceptual_hash,
            "duplicate_of": self.duplicate_of,
            "duplicate_group": self.duplicate_group,
            "output_path": self.output_path,
        }
        row.update(self.quality.as_dict(prefix="quality_"))
        row.update(
            {
                "rejected": self.is_rejected,
                "rejection_stage": primary.stage.value if primary else None,
                "rejection_code": primary.code.value if primary else None,
                "rejection_reason": primary.message if primary else None,
                "rejection_count": len(self.rejections),
                "operations": ";".join(op.name for op in self.operations),
                "operation_count": len(self.operations),
                "source_attributes_json": json.dumps(dict(prov.attributes), sort_keys=True, default=str),
                "metadata_json": json.dumps(self.metadata, sort_keys=True, default=str),
            }
        )
        return row

    def to_trace(self) -> dict[str, Any]:
        """Full, nested provenance document for a single image."""
        row = self.to_row()
        for key in ("metadata_json", "source_attributes_json"):
            row.pop(key, None)
        row["provenance"] = self.provenance.as_dict()
        row["rejections"] = [r.as_dict() for r in self.rejections]
        row["operations"] = [op.as_dict() for op in self.operations]
        row["metadata"] = dict(self.metadata)
        return row


@dataclass(frozen=True, slots=True)
class LabelMapping:
    """Deterministic mapping between harmonised labels and class indices."""

    label_to_index: dict[str, int]
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_labels(cls, labels: Sequence[str], aliases: Mapping[str, str] | None = None) -> "LabelMapping":
        """Build a mapping with indices assigned in sorted order for reproducibility."""
        ordered = sorted({label for label in labels if label})
        return cls({label: index for index, label in enumerate(ordered)}, dict(aliases or {}))

    @property
    def index_to_label(self) -> dict[int, str]:
        return {index: label for label, index in self.label_to_index.items()}

    @property
    def classes(self) -> list[str]:
        return sorted(self.label_to_index, key=lambda label: self.label_to_index[label])

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    def index_of(self, label: str) -> int | None:
        return self.label_to_index.get(label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "classes": self.classes,
            "label_to_index": dict(self.label_to_index),
            "index_to_label": {str(i): label for i, label in self.index_to_label.items()},
            "aliases": dict(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class SourceSummary:
    """Provenance and statistics of one contributing dataset."""

    name: str
    version: str
    root: Path
    image_count: int
    class_count: int
    fingerprint: str
    total_bytes: int = 0
    raw_labels: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "root": str(self.root),
            "image_count": self.image_count,
            "class_count": self.class_count,
            "fingerprint": self.fingerprint,
            "total_bytes": self.total_bytes,
            "raw_labels": list(self.raw_labels),
            "splits": list(self.splits),
            "attributes": dict(self.attributes),
        }


@dataclass(slots=True)
class Corpus:
    """The merged, canonical dataset shared by every downstream stage.

    Downstream stages consume this object exclusively; no stage after ingestion
    inspects raw folders.
    """

    records: list[ImageRecord]
    labels: LabelMapping
    sources: list[SourceSummary] = field(default_factory=list)
    fingerprint: str | None = None
    version: str | None = None
    manifest: RunManifest | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    def accepted(self) -> list[ImageRecord]:
        return [r for r in self.records if r.is_accepted]

    def rejected(self) -> list[ImageRecord]:
        return [r for r in self.records if r.is_rejected]

    def pending(self) -> list[ImageRecord]:
        return [r for r in self.records if r.status is RecordStatus.PENDING]

    def by_id(self, image_id: str) -> ImageRecord | None:
        return self.index().get(image_id)

    def index(self) -> dict[str, ImageRecord]:
        """Identity index; rebuilt on demand because records are mutated in place."""
        return {record.image_id: record for record in self.records}

    def in_split(self, split: Split) -> list[ImageRecord]:
        return [r for r in self.records if r.split is split and r.is_accepted]

    def source(self, name: str) -> SourceSummary | None:
        return next((s for s in self.sources if s.name == name), None)

    def counts_by_label(self, accepted_only: bool = True) -> dict[str, int]:
        records = self.accepted() if accepted_only else self.records
        return dict(sorted(Counter(r.label for r in records).items()))

    def counts_by_source(self, accepted_only: bool = True) -> dict[str, int]:
        records = self.accepted() if accepted_only else self.records
        return dict(sorted(Counter(r.dataset_name for r in records).items()))

    def rejection_counts(self) -> dict[str, int]:
        codes = (r.primary_rejection.code.value for r in self.records if r.primary_rejection)
        return dict(sorted(Counter(codes).items()))

    def to_frame(self, accepted_only: bool = False) -> "pd.DataFrame":
        """Tabular view of the corpus; the basis of every CSV/parquet artefact."""
        import pandas as pd

        records = self.accepted() if accepted_only else self.records
        return pd.DataFrame([record.to_row() for record in records])

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "run_id": self.manifest.run_id if self.manifest else None,
            "total_images": len(self.records),
            "accepted_images": len(self.accepted()),
            "rejected_images": len(self.rejected()),
            "num_classes": self.labels.num_classes,
            "sources": [s.name for s in self.sources],
            "images_per_source": self.counts_by_source(accepted_only=False),
            "rejections_by_code": self.rejection_counts(),
        }


@dataclass(slots=True)
class StageReport:
    """Structured outcome of one pipeline stage, aggregated into the run summary."""

    stage: PipelineStage
    status: str = "completed"
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    processed: int = 0
    rejected: int = 0
    skipped: int = 0
    warnings: int = 0
    errors: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "processed": self.processed,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": dict(self.metrics),
            "messages": list(self.messages),
        }


__all__ = [
    "Corpus",
    "DuplicateStatus",
    "ImageRecord",
    "LabelMapping",
    "Metric",
    "MetricStatus",
    "Operation",
    "PipelineStage",
    "Provenance",
    "QualityMetrics",
    "RecordStatus",
    "Rejection",
    "RejectionCode",
    "RunManifest",
    "Severity",
    "SourceSummary",
    "Split",
    "StageReport",
    "ValidationIssue",
]
