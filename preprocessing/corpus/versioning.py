"""Dataset fingerprinting and version resolution.

The fingerprint is a deterministic function of four independent components:

* **composition** - the set of images, their labels and their sizes
* **labels** - the canonical label space and its aliases
* **sources** - each contributing dataset's own fingerprint and version
* **config** - the effective configuration hash

Only machine-independent facts feed the hash: no absolute paths, no mtimes, no
run identifiers. The same data with the same configuration therefore fingerprints
identically on any machine, and a changed component tells you *what* changed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.config import VersioningConfig
from ..core.io import hash_mapping, read_json, write_json
from ..core.logging import get_logger
from ..core.records import Corpus, ImageRecord, LabelMapping, RunManifest, SourceSummary

_logger = get_logger(__name__)
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True, slots=True)
class FingerprintComponents:
    """Per-component hashes; a diff pinpoints what changed between runs."""

    composition: str
    labels: str
    sources: str
    config: str

    def as_dict(self) -> dict[str, str]:
        return {
            "composition": self.composition,
            "labels": self.labels,
            "sources": self.sources,
            "config": self.config,
        }


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    """Deterministic identity of a corpus build."""

    value: str
    components: FingerprintComponents
    record_count: int
    class_count: int
    source_count: int

    def short(self, length: int = 12) -> str:
        return self.value[:length]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.value,
            "components": self.components.as_dict(),
            "record_count": self.record_count,
            "class_count": self.class_count,
            "source_count": self.source_count,
        }


def compute_fingerprint(
    records: Sequence[ImageRecord],
    labels: LabelMapping,
    sources: Sequence[SourceSummary],
    config_hash: str,
) -> DatasetFingerprint:
    """Fingerprint a corpus from its composition, labels, sources and config."""
    components = FingerprintComponents(
        composition=_composition_hash(records),
        labels=hash_mapping(labels.as_dict()),
        sources=hash_mapping([source.as_dict() | {"root": None} for source in sources]),
        config=config_hash,
    )
    value = hash_mapping(components.as_dict())
    return DatasetFingerprint(
        value=value,
        components=components,
        record_count=len(records),
        class_count=labels.num_classes,
        source_count=len(sources),
    )


def _composition_hash(records: Iterable[ImageRecord]) -> str:
    """Stream a canonical, machine-independent description of every image."""
    digest = hashlib.sha256()
    identities = sorted(
        f"{record.image_id}\0{record.dataset_name}\0{record.dataset_version}\0"
        f"{record.source_relpath}\0{record.label}\0{record.provenance.size_bytes or 0}\0{record.status.value}"
        for record in records
    )
    for identity in identities:
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_version(
    config: VersioningConfig,
    fingerprint: DatasetFingerprint,
    previous: dict[str, Any] | None,
    started_at: dt.datetime,
) -> tuple[str, str]:
    """Resolve the corpus version; returns ``(version, reason)``."""
    if config.scheme == "timestamp":
        return started_at.strftime("%Y.%m.%d.%H%M%S"), "timestamp"
    if config.scheme == "fingerprint":
        return fingerprint.short(12), "fingerprint"

    if previous is None or not config.bump_on_change:
        return config.version, "configured"

    previous_version = str(previous.get("dataset_version") or config.version)
    if previous.get("fingerprint") == fingerprint.value:
        return previous_version, "unchanged"

    previous_components = previous.get("components") or {}
    if previous_components.get("labels") != fingerprint.components.labels:
        return _bump(previous_version, "minor"), "label_space_changed"
    return _bump(previous_version, "patch"), "composition_changed"


def _bump(version: str, level: str) -> str:
    match = _SEMVER.match(version)
    if not match:
        _logger.warning("version.not_semantic", version=version)
        return version
    major, minor, patch = (int(part) for part in match.groups())
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def load_previous(path: Path) -> dict[str, Any] | None:
    """Read a previous ``dataset_fingerprint.json``; ``None`` when absent or invalid."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        _logger.warning("fingerprint.unreadable", path=str(path), error=str(exc))
        return None
    return payload if isinstance(payload, dict) else None


def write_fingerprint(path: Path, fingerprint: DatasetFingerprint, corpus: Corpus, manifest: RunManifest) -> Path:
    """Write ``dataset_fingerprint.json``, the reproducibility anchor of a build."""
    payload: dict[str, Any] = fingerprint.as_dict()
    payload.update(
        {
            "dataset_version": corpus.version,
            "created_at": corpus.created_at,
            "manifest": manifest.as_dict(),
            "counts": {
                "total_images": len(corpus.records),
                "accepted_images": len(corpus.accepted()),
                "rejected_images": len(corpus.rejected()),
                "classes": corpus.labels.num_classes,
            },
            "sources": [source.as_dict() for source in corpus.sources],
            "images_per_class": corpus.counts_by_label(),
        }
    )
    return write_json(path, payload)


__all__ = [
    "DatasetFingerprint",
    "FingerprintComponents",
    "compute_fingerprint",
    "load_previous",
    "resolve_version",
    "write_fingerprint",
]
