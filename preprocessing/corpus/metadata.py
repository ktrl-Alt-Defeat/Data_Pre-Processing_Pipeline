"""Metadata assembly and the tabular artefacts derived from it.

Produces the two views of the corpus every consumer needs:

* ``metadata/metadata.csv`` - human-inspectable, one row per image
* ``metadata/image_manifest.parquet`` - typed and compressed, for programmatic use

Both are projections of :meth:`ImageRecord.to_row`, so a column can never drift
between them. The same writer is reused by later stages, which is why the
manifest gains quality scores and split assignments without a second schema.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ..core.config import Config
from ..core.context import OutputLayout
from ..core.errors import ImageError
from ..core.io import ensure_dir, probe_image, write_json
from ..core.logging import StageTracker, get_logger
from ..core.records import Corpus, ImageRecord, PipelineStage, RejectionCode, RunManifest

_logger = get_logger(__name__)

CORPUS_INDEX_FILENAME = "corpus_index.parquet"

_INT_COLUMNS = (
    "class_index",
    "width",
    "height",
    "channels",
    "file_size_bytes",
    "source_size_bytes",
    "exif_orientation",
    "rejection_count",
    "operation_count",
    "quality_width",
    "quality_height",
)
_FLOAT_COLUMNS = (
    "aspect_ratio",
    "megapixels",
    "quality_blur_score",
    "quality_sharpness",
    "quality_brightness",
    "quality_contrast",
    "quality_colorfulness",
    "quality_entropy",
    "quality_megapixels",
    "quality_aspect_ratio",
    "quality_perceptual_similarity",
    "quality_score",
)
_BOOL_COLUMNS = ("rejected",)

# Provenance first: the columns that answer "where did this image come from"
# lead the table in both artefacts.
COLUMN_ORDER: tuple[str, ...] = (
    "image_id",
    "dataset_name",
    "dataset_version",
    "source_relpath",
    "source_class",
    "source_split",
    "original_filename",
    "label",
    "class_index",
    "crop",
    "condition",
    "label_rule",
    "status",
    "split",
    "rejected",
    "rejection_stage",
    "rejection_code",
    "rejection_reason",
    "rejection_count",
    "width",
    "height",
    "channels",
    "aspect_ratio",
    "megapixels",
    "image_format",
    "color_mode",
    "file_size_bytes",
    "exif_orientation",
    "content_hash",
    "pixel_hash",
    "perceptual_hash",
    "duplicate_of",
    "duplicate_group",
    "quality_score",
    "quality_grade",
    "quality_blur_score",
    "quality_sharpness",
    "quality_brightness",
    "quality_contrast",
    "quality_colorfulness",
    "quality_entropy",
    "quality_width",
    "quality_height",
    "quality_megapixels",
    "quality_aspect_ratio",
    "quality_duplicate_status",
    "quality_perceptual_similarity",
    "operations",
    "operation_count",
    "output_path",
    "first_seen_run_id",
    "first_seen_at",
    "source_size_bytes",
    "source_modified_at",
    "source_root",
    "source_path",
    "source_attributes_json",
    "metadata_json",
)


def attach_ingest_metadata(records: Iterable[ImageRecord], manifest: RunManifest) -> None:
    """Stamp each record with harmonised ingestion metadata."""
    ingested_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for record in records:
        record.metadata.setdefault("ingested_at", ingested_at)
        record.metadata.setdefault("ingest_run_id", manifest.run_id)
        record.metadata.setdefault("pipeline_version", manifest.pipeline_version)


def probe_dimensions(records: Sequence[ImageRecord], tracker: StageTracker) -> int:
    """Fill width/height/format from image headers without decoding pixels."""
    probed = 0
    for record in records:
        if record.is_rejected or record.width is not None:
            continue
        try:
            probe = probe_image(record.source_path)
        except ImageError as exc:
            record.reject(PipelineStage.CORPUS, RejectionCode.UNREADABLE_FILE, str(exc.message))
            tracker.rejected()
            continue
        record.width = probe.width
        record.height = probe.height
        record.image_format = probe.image_format
        record.color_mode = probe.color_mode
        record.exif_orientation = probe.exif_orientation
        record.record_operation(PipelineStage.CORPUS, "probe_header", width=probe.width, height=probe.height)
        probed += 1
    return probed


def build_frame(corpus: Corpus, accepted_only: bool = False) -> pd.DataFrame:
    """Typed metadata table for the corpus, in canonical column order."""
    records = corpus.accepted() if accepted_only else corpus.records
    frame = pd.DataFrame([record.to_row() for record in records])
    if frame.empty:
        return pd.DataFrame(columns=list(COLUMN_ORDER))
    return _normalize_dtypes(_order_columns(frame))


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    known = [column for column in COLUMN_ORDER if column in frame.columns]
    extra = [column for column in frame.columns if column not in COLUMN_ORDER]
    return frame[known + sorted(extra)]


def _normalize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Nullable dtypes throughout: parquet round-trips without object columns."""
    for column in _INT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in _FLOAT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in _BOOL_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("string")
    return frame


class MetadataWriter:
    """Writes the metadata artefacts of the output package."""

    def __init__(self, layout: OutputLayout, config: Config) -> None:
        self._layout = layout
        self._config = config

    def write_all(self, corpus: Corpus, harmonization: dict[str, Any] | None = None) -> dict[str, Path]:
        """Write metadata.csv, image_manifest.parquet and label_mapping.json."""
        frame = build_frame(corpus)
        written = {
            "metadata_csv": self.write_csv(frame),
            "label_mapping": self.write_label_mapping(corpus, harmonization),
        }
        if self._config.packaging.manifest_parquet:
            written["image_manifest"] = self.write_parquet(frame)
        return written

    def write_csv(self, frame: pd.DataFrame, path: Path | None = None) -> Path:
        target = path or self._layout.metadata_csv
        ensure_dir(target.parent)
        frame.to_csv(target, index=False, encoding="utf-8")
        return target

    def write_parquet(self, frame: pd.DataFrame, path: Path | None = None) -> Path:
        target = path or self._layout.image_manifest
        ensure_dir(target.parent)
        frame.to_parquet(target, engine="pyarrow", index=False, compression="snappy")
        return target

    def write_label_mapping(self, corpus: Corpus, harmonization: dict[str, Any] | None = None) -> Path:
        payload: dict[str, Any] = corpus.labels.as_dict()
        payload["images_per_class"] = corpus.counts_by_label()
        payload["harmonization"] = harmonization or {}
        payload["dataset_version"] = corpus.version
        payload["dataset_fingerprint"] = corpus.fingerprint
        payload["manifest"] = corpus.manifest.as_dict() if corpus.manifest else None
        return write_json(self._layout.label_mapping, payload)


def write_corpus_index(frame: pd.DataFrame, corpus_dir: Path) -> Path:
    """Persist the first-seen index used to preserve image history across runs."""
    ensure_dir(corpus_dir)
    target = corpus_dir / CORPUS_INDEX_FILENAME
    columns = ["image_id", "first_seen_run_id", "first_seen_at"]
    available = [column for column in columns if column in frame.columns]
    frame[available].to_parquet(target, engine="pyarrow", index=False, compression="snappy")
    return target


def load_first_seen(corpus_dir: Path) -> dict[str, tuple[str, str]]:
    """Load ``image_id -> (first run id, first timestamp)`` from a previous run."""
    path = corpus_dir / CORPUS_INDEX_FILENAME
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError) as exc:
        _logger.warning("corpus_index.unreadable", path=str(path), error=str(exc))
        return {}

    required = {"image_id", "first_seen_run_id", "first_seen_at"}
    if not required.issubset(frame.columns):
        _logger.warning("corpus_index.schema_mismatch", path=str(path), columns=list(frame.columns))
        return {}
    return {
        str(row.image_id): (str(row.first_seen_run_id), str(row.first_seen_at))
        for row in frame.itertuples(index=False)
        if pd.notna(row.first_seen_run_id)
    }


__all__ = [
    "COLUMN_ORDER",
    "MetadataWriter",
    "attach_ingest_metadata",
    "build_frame",
    "load_first_seen",
    "probe_dimensions",
    "write_corpus_index",
]
