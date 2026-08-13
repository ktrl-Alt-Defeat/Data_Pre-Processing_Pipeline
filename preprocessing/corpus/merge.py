"""Source discovery and canonical record construction.

This is the only module in the framework that understands raw dataset layouts:
folder-per-class, optionally nested, optionally wrapped in the source's own
train/test directories. Everything downstream consumes :class:`ImageRecord`
objects and never looks at the raw tree again.

No pixels are read here. Traversal collects paths, sizes and mtimes from the
directory scan itself, which keeps ingestion of hundreds of thousands of files
bound by one pass over the filesystem metadata.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator, Mapping

from ..core.config import Config, SourceConfig
from ..core.errors import ConfigurationError, SourceError
from ..core.io import FileEntry, iter_subdirectories, sha256_file, slugify, stable_id, walk_image_entries
from ..core.logging import StageTracker
from ..core.records import (
    ImageRecord,
    PipelineStage,
    Provenance,
    RejectionCode,
    SourceSummary,
)

FirstSeenIndex = Mapping[str, tuple[str, str]]


@dataclass(slots=True)
class SourceScan:
    """Result of traversing one source."""

    summary: SourceSummary
    records: list[ImageRecord] = field(default_factory=list)


def resolve_sources(config: Config, tracker: StageTracker | None = None) -> list[SourceConfig]:
    """Configured sources plus, when explicitly enabled, auto-discovered ones.

    Auto-discovery is opt-in: without ``discovery.auto_discover`` a stray backup
    folder under ``raw_root`` can never silently join the benchmark.
    """
    configured = [source for source in config.sources if source.enabled]
    discovered = _auto_discover(config, {source.path.resolve() for source in configured}) if config.discovery.auto_discover else []

    sources = configured + discovered
    if not sources:
        raise ConfigurationError(
            "no dataset sources are configured. Add entries under 'sources' in config/pipeline.yaml, "
            "or set 'discovery.auto_discover: true' to ingest every sub-directory of paths.raw_root."
        )

    duplicates = [name for name, count in Counter(source.name for source in sources).items() if count > 1]
    if duplicates:
        raise ConfigurationError(f"duplicate source names: {', '.join(sorted(duplicates))}")

    if tracker:
        tracker.info(
            "sources.resolved",
            configured=len(configured),
            discovered=len(discovered),
            names=[source.name for source in sources],
        )
    return sources


def _auto_discover(config: Config, claimed: set[Path]) -> list[SourceConfig]:
    discovery = config.discovery
    sources: list[SourceConfig] = []
    for directory in iter_subdirectories(config.paths.raw_root):
        if directory.resolve() in claimed:
            continue
        if any(fnmatch(directory.name.lower(), pattern.lower()) for pattern in discovery.exclude_patterns):
            continue
        sources.append(
            SourceConfig(
                name=slugify(directory.name),
                path=directory,
                version=discovery.version,
                class_depth=discovery.class_depth,
                extensions=discovery.extensions,
            )
        )
    return sources


class CorpusBuilder:
    """Turns configured sources into canonical, fully provenanced records."""

    def __init__(
        self,
        config: Config,
        run_id: str,
        tracker: StageTracker,
        first_seen: FirstSeenIndex | None = None,
    ) -> None:
        self._config = config
        self._corpus_config = config.corpus
        self._execution = config.execution
        self._run_id = run_id
        self._tracker = tracker
        self._first_seen = dict(first_seen or {})
        self._ingested_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self._seen_ids: set[str] = set()
        self._total = 0

    def build(self, sources: list[SourceConfig]) -> tuple[list[ImageRecord], list[SourceSummary]]:
        """Traverse every source in configuration order and merge the results."""
        records: list[ImageRecord] = []
        summaries: list[SourceSummary] = []

        for source in sources:
            scan = self._scan(source)
            records.extend(scan.records)
            summaries.append(scan.summary)
            self._tracker.info(
                "source.ingested",
                source=source.name,
                version=source.version,
                images=scan.summary.image_count,
                classes=scan.summary.class_count,
                bytes=scan.summary.total_bytes,
                fingerprint=scan.summary.fingerprint[:12],
            )
            if self._limit_reached():
                self._tracker.warn("ingest.limit_reached", limit=self._execution.max_images, ingested=self._total)
                break

        return records, summaries

    # --- traversal ------------------------------------------------------------ #

    def _scan(self, source: SourceConfig) -> SourceScan:
        root = self._validate_root(source)
        digest = hashlib.sha256()
        records: list[ImageRecord] = []
        per_class: Counter[str] = Counter()
        splits: set[str] = set()
        total_bytes = 0

        for entry in walk_image_entries(root, source.extensions):
            if self._limit_reached():
                break

            # The source fingerprint covers everything on disk, including files that
            # are later skipped or rejected, so it tracks the source and not our filters.
            digest.update(f"{entry.relpath}\0{entry.size_bytes}\n".encode("utf-8"))
            total_bytes += entry.size_bytes

            split, class_name = self._locate(entry, source)
            if class_name is None:
                record = self._build_record(entry, source, root, "", split)
                record.reject(
                    PipelineStage.CORPUS,
                    RejectionCode.LABEL_MISSING,
                    relpath=entry.relpath,
                    expected_class_depth=source.class_depth,
                )
                records.append(record)
                self._tracker.rejected()
                continue

            cap = self._execution.max_images_per_class
            if cap is not None and per_class[class_name] >= cap:
                self._tracker.skipped()
                continue

            records.append(self._build_record(entry, source, root, class_name, split))
            per_class[class_name] += 1
            if split:
                splits.add(split)
            self._total += 1
            self._tracker.processed()

        if not records:
            self._tracker.warn("source.empty", source=source.name, root=str(root))

        summary = SourceSummary(
            name=source.name,
            version=source.version,
            root=root,
            image_count=len(records),
            class_count=len(per_class),
            fingerprint=digest.hexdigest(),
            total_bytes=total_bytes,
            raw_labels=tuple(sorted(per_class)),
            splits=tuple(sorted(splits)),
            attributes=dict(source.metadata),
        )
        return SourceScan(summary=summary, records=records)

    def _validate_root(self, source: SourceConfig) -> Path:
        root = source.path
        if not root.exists():
            raise SourceError(f"source '{source.name}' does not exist: {root}")
        if not root.is_dir():
            raise SourceError(f"source '{source.name}' is not a directory: {root}")
        return root.resolve()

    def _locate(self, entry: FileEntry, source: SourceConfig) -> tuple[str | None, str | None]:
        """Split a relative path into the source's own split and its class folder."""
        parts = list(entry.parts[:-1])
        split: str | None = None

        if source.detect_splits and parts and parts[0].lower() in {name.lower() for name in source.split_dirs}:
            split = parts.pop(0).lower()

        if len(parts) < source.class_depth:
            return split, None
        return split, parts[source.class_depth - 1]

    def _build_record(
        self,
        entry: FileEntry,
        source: SourceConfig,
        root: Path,
        class_name: str,
        split: str | None,
    ) -> ImageRecord:
        image_id = self._image_id(source, entry)
        first_run, first_at = self._first_seen.get(image_id, (self._run_id, self._ingested_at))
        provenance = Provenance(
            dataset_name=source.name,
            dataset_version=source.version,
            source_root=root,
            source_path=entry.path,
            source_relpath=entry.relpath,
            source_class=class_name,
            original_filename=entry.filename,
            source_split=split,
            first_seen_run_id=first_run,
            first_seen_at=first_at,
            size_bytes=entry.size_bytes,
            modified_at=dt.datetime.fromtimestamp(entry.modified_at, dt.timezone.utc).isoformat(timespec="seconds"),
            attributes=dict(source.metadata),
        )
        record = ImageRecord(
            image_id=image_id,
            provenance=provenance,
            file_size_bytes=entry.size_bytes,
            image_format=entry.path.suffix.lower().lstrip("."),
        )
        record.record_operation(
            PipelineStage.CORPUS,
            "ingest",
            source=source.name,
            source_version=source.version,
            relpath=entry.relpath,
        )
        return record

    def _image_id(self, source: SourceConfig, entry: FileEntry) -> str:
        if self._corpus_config.id_strategy == "content":
            base = stable_id(sha256_file(entry.path))
        else:
            base = stable_id(source.name, entry.relpath)

        if base not in self._seen_ids:
            self._seen_ids.add(base)
            return base

        # Content-addressed ids legitimately collide for identical files; keep both
        # records distinct so neither loses its provenance, and let the quality
        # gate decide which one survives deduplication.
        suffix = 1
        while f"{base}-{suffix}" in self._seen_ids:
            suffix += 1
        collided = f"{base}-{suffix}"
        self._seen_ids.add(collided)
        self._tracker.warn("image_id.collision", image_id=base, resolved_as=collided, path=entry.relpath)
        return collided

    def _limit_reached(self) -> bool:
        limit = self._execution.max_images
        return limit is not None and self._total >= limit


def raw_label_counts(records: Iterator[ImageRecord] | list[ImageRecord]) -> dict[tuple[str, str], int]:
    """``(dataset, source_class) -> image count``, the input to harmonisation."""
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        if record.source_class:
            counts[(record.dataset_name, record.source_class)] += 1
    return dict(counts)


__all__ = ["CorpusBuilder", "SourceScan", "raw_label_counts", "resolve_sources"]
