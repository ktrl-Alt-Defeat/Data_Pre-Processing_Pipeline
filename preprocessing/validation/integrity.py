"""Corpus-wide integrity validation.

Where the other validators inspect one record at a time, this one inspects the
corpus as a whole: identity collisions, duplicate entries, records that point at
files or sources that do not exist, and agreement between the in-memory corpus
and the artefacts the ingestion stage wrote to disk.

Findings that implicate specific images carry their ids so the stage can reject
exactly those records; findings about the dataset as a whole carry none.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.config import IntegrityValidationConfig
from ..core.context import RunContext
from ..core.io import read_json
from ..core.records import Corpus, ImageRecord, RejectionCode, Severity

VALIDATOR = "integrity"


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    """One integrity finding, optionally scoped to a set of records.

    Rejection targets are carried as positions in ``corpus.records`` rather than
    image ids: duplicate-id findings are precisely the case where an id no longer
    identifies a single record.
    """

    check: str
    severity: Severity
    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    record_indices: tuple[int, ...] = ()
    image_ids: tuple[str, ...] = ()
    code: RejectionCode = RejectionCode.METADATA_INVALID

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "code": self.code.value,
            "affected_images": len(self.record_indices),
            "image_ids": list(self.image_ids[:50]),
            "detail": dict(self.detail),
        }


class IntegrityValidator:
    """Validates structural consistency across the whole corpus."""

    def __init__(self, config: IntegrityValidationConfig, files_already_checked: bool = False) -> None:
        self._config = config
        self._files_already_checked = files_already_checked
        self._counter: Counter[str] = Counter()
        self._statistics: dict[str, Any] = {}

    @property
    def metrics(self) -> dict[str, int]:
        return dict(sorted(self._counter.items()))

    @property
    def statistics(self) -> dict[str, Any]:
        return dict(self._statistics)

    def validate(self, corpus: Corpus, context: RunContext) -> list[IntegrityIssue]:
        """Run every enabled integrity check and collect the findings."""
        candidates: Indexed = [
            (position, record) for position, record in enumerate(corpus.records) if not record.is_rejected
        ]
        issues: list[IntegrityIssue] = []

        if self._config.check_duplicate_ids:
            issues.extend(self._duplicate_ids(candidates))
        if self._config.check_duplicate_entries:
            issues.extend(self._duplicate_entries(candidates))
        if self._config.check_provenance:
            issues.extend(self._provenance_consistency(candidates, corpus))
        if self._config.check_paths:
            issues.extend(self._invalid_paths(candidates))
        if self._config.check_missing_files and not self._files_already_checked:
            issues.extend(self._missing_files(candidates))
        if self._config.check_fingerprint:
            issues.extend(self._fingerprint_consistency(corpus, context))
        if self._config.check_manifest:
            issues.extend(self._manifest_consistency(corpus, context))

        self._statistics = self._build_statistics(corpus, candidates, issues)
        for issue in issues:
            self._counter[f"{issue.severity.value}:{issue.check}"] += 1
        return issues

    # --- identity ------------------------------------------------------------- #

    def _duplicate_ids(self, candidates: Indexed) -> list[IntegrityIssue]:
        groups = _group(candidates, lambda record: record.image_id)
        return [
            IntegrityIssue(
                check="duplicate_image_id",
                severity=Severity.ERROR,
                message=f"image id '{image_id}' is used by {len(duplicates)} records",
                detail={"image_id": image_id, "paths": [record.source_relpath for _, record in duplicates[:10]]},
                **_targets(duplicates[1:]),
            )
            for image_id, duplicates in sorted(groups.items())
            if len(duplicates) > 1
        ]

    def _duplicate_entries(self, candidates: Indexed) -> list[IntegrityIssue]:
        """The same source file must not appear twice within one dataset."""
        groups = _group(candidates, lambda record: (record.dataset_name, record.source_relpath))
        return [
            IntegrityIssue(
                check="duplicate_metadata_entry",
                severity=Severity.ERROR,
                message=f"'{dataset}/{relpath}' is present {len(duplicates)} times",
                detail={"dataset": dataset, "source_relpath": relpath},
                **_targets(duplicates[1:]),
            )
            for (dataset, relpath), duplicates in sorted(groups.items())
            if len(duplicates) > 1
        ]

    # --- provenance ----------------------------------------------------------- #

    def _provenance_consistency(self, candidates: Indexed, corpus: Corpus) -> list[IntegrityIssue]:
        summaries = {source.name: source for source in corpus.sources}
        orphans: list[tuple[int, ImageRecord]] = []
        version_mismatch: dict[str, list[tuple[int, ImageRecord]]] = defaultdict(list)

        for position, record in candidates:
            summary = summaries.get(record.dataset_name)
            if summary is None:
                orphans.append((position, record))
            elif summary.version != record.dataset_version:
                version_mismatch[record.dataset_name].append((position, record))

        issues: list[IntegrityIssue] = []
        if orphans:
            issues.append(
                IntegrityIssue(
                    check="orphan_metadata",
                    severity=Severity.ERROR,
                    message=f"{len(orphans)} records reference a dataset with no source summary",
                    detail={"known_sources": sorted(summaries)},
                    **_targets(orphans),
                )
            )
        for dataset, affected in sorted(version_mismatch.items()):
            issues.append(
                IntegrityIssue(
                    check="inconsistent_provenance",
                    severity=Severity.WARNING,
                    message=f"{len(affected)} records from '{dataset}' disagree with the source's declared version",
                    detail={"dataset": dataset, "source_version": summaries[dataset].version},
                    **_targets(affected),
                )
            )
        return issues

    def _invalid_paths(self, candidates: Indexed) -> list[IntegrityIssue]:
        invalid = [
            (position, record)
            for position, record in candidates
            if not record.source_path.is_absolute()
            or not record.source_path.is_relative_to(record.provenance.source_root)
            or not record.source_relpath
        ]
        if not invalid:
            return []
        return [
            IntegrityIssue(
                check="invalid_path",
                severity=Severity.ERROR,
                message=f"{len(invalid)} records carry a path outside their declared source root",
                code=RejectionCode.UNREADABLE_FILE,
                **_targets(invalid),
            )
        ]

    def _missing_files(self, candidates: Indexed) -> list[IntegrityIssue]:
        missing = [(position, record) for position, record in candidates if not _exists(record.source_path)]
        if not missing:
            return []
        return [
            IntegrityIssue(
                check="missing_file",
                severity=Severity.ERROR,
                message=f"{len(missing)} records reference a file that no longer exists",
                code=RejectionCode.UNREADABLE_FILE,
                **_targets(missing),
            )
        ]

    # --- artefact agreement ---------------------------------------------------- #

    def _fingerprint_consistency(self, corpus: Corpus, context: RunContext) -> list[IntegrityIssue]:
        """The in-memory corpus must agree with the fingerprint written at ingestion."""
        path = context.layout.dataset_fingerprint
        document = _read_document(path)
        if document is None:
            return [
                IntegrityIssue(
                    check="fingerprint_missing",
                    severity=Severity.WARNING,
                    message=f"no dataset fingerprint was found at {path}",
                    detail={"path": str(path)},
                )
            ]

        expected = {
            "fingerprint": corpus.fingerprint,
            "dataset_version": corpus.version,
            "record_count": len(corpus.records),
            "class_count": corpus.labels.num_classes,
        }
        mismatches = {key: (value, document.get(key)) for key, value in expected.items() if document.get(key) != value}
        if not mismatches:
            return []
        return [
            IntegrityIssue(
                check="fingerprint_mismatch",
                severity=Severity.ERROR,
                message=f"{len(mismatches)} fingerprint fields disagree with {path.name}",
                detail={key: {"corpus": corpus_value, "file": file_value} for key, (corpus_value, file_value) in mismatches.items()},
            )
        ]

    def _manifest_consistency(self, corpus: Corpus, context: RunContext) -> list[IntegrityIssue]:
        manifest = corpus.manifest
        if manifest is None:
            return [
                IntegrityIssue(
                    check="manifest_missing",
                    severity=Severity.ERROR,
                    message="corpus carries no run manifest",
                )
            ]

        mismatches: dict[str, Any] = {}
        if manifest.run_id != context.run_id:
            mismatches["run_id"] = {"manifest": manifest.run_id, "context": context.run_id}
        if manifest.config_hash != context.config_fingerprint:
            mismatches["config_hash"] = {"manifest": manifest.config_hash, "context": context.config_fingerprint}
        if manifest.dataset_version != corpus.version:
            mismatches["dataset_version"] = {"manifest": manifest.dataset_version, "corpus": corpus.version}

        if not mismatches:
            return []
        return [
            IntegrityIssue(
                check="manifest_mismatch",
                severity=Severity.ERROR,
                message=f"run manifest disagrees with the current run on: {', '.join(sorted(mismatches))}",
                detail=mismatches,
            )
        ]

    # --- statistics ------------------------------------------------------------ #

    def _build_statistics(
        self,
        corpus: Corpus,
        candidates: Indexed,
        issues: Sequence[IntegrityIssue],
    ) -> dict[str, Any]:
        return {
            "records_total": len(corpus.records),
            "records_checked": len(candidates),
            "records_rejected_before_integrity": len(corpus.records) - len(candidates),
            "unique_image_ids": len({record.image_id for _, record in candidates}),
            "unique_source_entries": len({(r.dataset_name, r.source_relpath) for _, r in candidates}),
            "sources": len(corpus.sources),
            "classes": corpus.labels.num_classes,
            "issues_total": len(issues),
            "issues_by_severity": dict(sorted(Counter(issue.severity.value for issue in issues).items())),
            "issues_by_check": dict(sorted(Counter(issue.check for issue in issues).items())),
            "images_flagged": len({position for issue in issues for position in issue.record_indices}),
            "file_existence_verified_by": "images" if self._files_already_checked else VALIDATOR,
        }


def _group(candidates: Indexed, key: Callable[[ImageRecord], Any]) -> dict[Any, list[tuple[int, ImageRecord]]]:
    grouped: dict[Any, list[tuple[int, ImageRecord]]] = defaultdict(list)
    for position, record in candidates:
        grouped[key(record)].append((position, record))
    return grouped


def _targets(affected: Sequence[tuple[int, ImageRecord]]) -> dict[str, tuple]:
    return {
        "record_indices": tuple(position for position, _ in affected),
        "image_ids": tuple(record.image_id for _, record in affected),
    }


def _exists(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _read_document(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


__all__ = ["VALIDATOR", "IntegrityIssue", "IntegrityValidator"]
