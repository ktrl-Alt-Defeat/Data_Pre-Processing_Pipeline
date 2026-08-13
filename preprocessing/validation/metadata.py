"""Record metadata validation.

Verifies that every :class:`ImageRecord` the corpus stage produced carries the
complete, well-formed metadata the rest of the pipeline depends on. Purely
in-memory: no file is opened here, which keeps this validator cheap enough to
run over the whole corpus before any decoding starts.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from typing import Any

from ..core.config import MetadataValidationConfig
from ..core.records import ImageRecord, RejectionCode, Severity, ValidationIssue

VALIDATOR = "metadata"

_HASH_LENGTHS = {"content_hash": 64, "pixel_hash": 40}
_HEX = re.compile(r"^[0-9a-f]+$")
_WHITESPACE = re.compile(r"\s")


class MetadataValidator:
    """Validates record identity, provenance completeness and field consistency."""

    def __init__(self, config: MetadataValidationConfig) -> None:
        self._config = config
        self._counter: Counter[str] = Counter()

    @property
    def metrics(self) -> dict[str, int]:
        return dict(sorted(self._counter.items()))

    def validate(self, record: ImageRecord) -> list[ValidationIssue]:
        """Validate one record; returns an empty list when metadata is complete."""
        self._counter["validated"] += 1
        issues: list[ValidationIssue] = []

        issues.extend(self._check_identity(record))
        issues.extend(self._check_provenance(record))
        if self._config.require_timestamps:
            issues.extend(self._check_timestamps(record))
        if self._config.validate_paths:
            issues.extend(self._check_paths(record))
        if self._config.validate_hashes:
            issues.extend(self._check_hashes(record))
        if self._config.validate_dimensions:
            issues.extend(self._check_dimensions(record))
        return issues

    # --- checks ---------------------------------------------------------------- #

    def _check_identity(self, record: ImageRecord) -> list[ValidationIssue]:
        if not record.image_id:
            return [self._issue(record, "record has no image id")]
        if _WHITESPACE.search(record.image_id):
            return [self._issue(record, f"image id '{record.image_id}' contains whitespace")]
        return []

    def _check_provenance(self, record: ImageRecord) -> list[ValidationIssue]:
        provenance = record.provenance
        missing = [
            field
            for field in self._config.required_provenance_fields
            if _is_blank(getattr(provenance, field, None))
        ]
        issues = []
        if missing:
            issues.append(
                self._issue(
                    record,
                    f"provenance is incomplete, missing: {', '.join(missing)}",
                    missing_fields=missing,
                )
            )
        if self._config.require_first_seen and (not provenance.first_seen_run_id or not provenance.first_seen_at):
            issues.append(self._issue(record, "provenance does not record which run first saw the image"))
        return issues

    def _check_timestamps(self, record: ImageRecord) -> list[ValidationIssue]:
        provenance = record.provenance
        candidates = {"modified_at": provenance.modified_at}
        if self._config.require_first_seen:
            candidates["first_seen_at"] = provenance.first_seen_at

        issues = []
        for field, value in candidates.items():
            if _is_blank(value):
                issues.append(self._issue(record, f"provenance.{field} is missing", field=field))
            elif not _is_iso_timestamp(value):
                issues.append(
                    self._issue(record, f"provenance.{field} is not an ISO-8601 timestamp: {value!r}", field=field)
                )
        return issues

    def _check_paths(self, record: ImageRecord) -> list[ValidationIssue]:
        provenance = record.provenance
        issues = []

        if not provenance.source_path.is_absolute():
            issues.append(self._issue(record, f"source_path is not absolute: {provenance.source_path}"))
        elif not provenance.source_path.is_relative_to(provenance.source_root):
            issues.append(
                self._issue(
                    record,
                    f"source_path escapes its source root: {provenance.source_path}",
                    source_root=str(provenance.source_root),
                )
            )

        if provenance.source_path.name != provenance.original_filename:
            issues.append(
                self._issue(
                    record,
                    f"original_filename '{provenance.original_filename}' does not match "
                    f"source_path '{provenance.source_path.name}'",
                )
            )
        if not provenance.source_relpath.endswith(provenance.original_filename):
            issues.append(
                self._issue(
                    record,
                    f"source_relpath '{provenance.source_relpath}' does not end with the original filename",
                )
            )
        return issues

    def _check_hashes(self, record: ImageRecord) -> list[ValidationIssue]:
        """Hashes are optional at this stage; when present they must be well formed."""
        issues = []
        for field, expected_length in _HASH_LENGTHS.items():
            value = getattr(record, field)
            if value is None:
                continue
            if len(value) != expected_length or not _HEX.match(value):
                issues.append(
                    self._issue(
                        record,
                        f"{field} is not a {expected_length}-character hex digest: {value!r}",
                        field=field,
                    )
                )
        if record.perceptual_hash is not None and not _HEX.match(record.perceptual_hash):
            issues.append(self._issue(record, f"perceptual_hash is not hexadecimal: {record.perceptual_hash!r}"))
        return issues

    def _check_dimensions(self, record: ImageRecord) -> list[ValidationIssue]:
        """Dimensions may be unset before image validation; if set they must be sane."""
        width, height = record.width, record.height
        if width is None and height is None:
            return []
        if width is None or height is None:
            return [self._issue(record, f"dimensions are partially set (width={width}, height={height})")]
        if width <= 0 or height <= 0:
            return [self._issue(record, f"dimensions are not positive ({width}x{height})")]

        size = record.provenance.size_bytes
        if size is not None and size <= 0:
            return [self._issue(record, f"provenance records a non-positive file size ({size})")]
        return []

    def _issue(self, record: ImageRecord, message: str, **detail: Any) -> ValidationIssue:
        self._counter["error:metadata_invalid"] += 1
        return ValidationIssue(
            image_id=record.image_id,
            validator=VALIDATOR,
            code=RejectionCode.METADATA_INVALID,
            message=message,
            severity=Severity.ERROR,
            detail=detail,
        )


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_iso_timestamp(value: str) -> bool:
    try:
        dt.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = ["VALIDATOR", "MetadataValidator"]
