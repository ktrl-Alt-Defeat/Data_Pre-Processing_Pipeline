"""Corpus construction: heterogeneous datasets in, one canonical corpus out.

This package owns dataset-specific knowledge. Downstream stages consume the
:class:`~preprocessing.core.records.Corpus` it returns and never inspect raw
folders again.
"""

from __future__ import annotations

from typing import Sequence

from ..core.config import Config
from ..core.context import RunContext
from ..core.logging import StageTracker, stage_scope
from ..core.records import (
    Corpus,
    ImageRecord,
    LabelMapping,
    PipelineStage,
    RejectionCode,
    SourceSummary,
)
from .harmonize import HarmonizationRule, LabelHarmonizer, LabelResolution
from .merge import CorpusBuilder, SourceScan, raw_label_counts, resolve_sources
from .metadata import (
    MetadataWriter,
    attach_ingest_metadata,
    build_frame,
    load_first_seen,
    probe_dimensions,
    write_corpus_index,
)
from .versioning import (
    DatasetFingerprint,
    FingerprintComponents,
    compute_fingerprint,
    load_previous,
    resolve_version,
    write_fingerprint,
)


def build_corpus(config: Config, context: RunContext) -> Corpus:
    """Ingest every configured source and return the canonical corpus.

    Writes ``metadata.csv``, ``image_manifest.parquet``, ``label_mapping.json``
    and ``dataset_fingerprint.json`` as a side effect: after this stage the
    corpus is fully described on disk even if a later stage fails.
    """
    with stage_scope(PipelineStage.CORPUS, context.logger(PipelineStage.CORPUS)) as tracker:
        sources = resolve_sources(config, tracker)
        first_seen = load_first_seen(config.paths.corpus_dir) if config.corpus.track_first_seen else {}

        builder = CorpusBuilder(config, context.run_id, tracker, first_seen)
        records, summaries = builder.build(sources)

        harmonizer = LabelHarmonizer(config.corpus, {source.name: source for source in sources})
        harmonizer.fit(raw_label_counts(records))
        _apply_labels(records, harmonizer)
        _apply_label_filters(config, records, tracker)
        _enforce_minimum_class_size(config, records, tracker)

        labels = harmonizer.mapping(sorted({record.label for record in records if record.is_accepted}))
        _assign_class_indices(records, labels)
        attach_ingest_metadata(records, context.manifest)

        if config.corpus.probe_dimensions:
            tracker.metric("probed_headers", probe_dimensions(records, tracker))

        corpus = _finalize(config, context, records, labels, summaries, harmonizer, tracker)

    return corpus


def _apply_labels(records: Sequence[ImageRecord], harmonizer: LabelHarmonizer) -> None:
    for record in records:
        if not record.source_class:
            continue
        resolution = harmonizer.canonical(record.dataset_name, record.source_class)
        record.label = resolution.canonical
        record.crop = resolution.crop
        record.condition = resolution.condition
        record.label_rule = resolution.rule.value
        record.record_operation(
            PipelineStage.CORPUS,
            "harmonize_label",
            raw_label=record.source_class,
            canonical=resolution.canonical,
            rule=resolution.rule.value,
        )
        record.accept()


def _apply_label_filters(config: Config, records: Sequence[ImageRecord], tracker: StageTracker) -> None:
    include = set(config.corpus.include_labels)
    exclude = set(config.corpus.exclude_labels)
    if not include and not exclude:
        return

    excluded = 0
    for record in records:
        if not record.is_accepted:
            continue
        if (include and record.label not in include) or record.label in exclude:
            record.reject(PipelineStage.CORPUS, RejectionCode.LABEL_EXCLUDED, label=record.label)
            tracker.rejected()
            excluded += 1
    tracker.metric("label_filtered", excluded)


def _enforce_minimum_class_size(config: Config, records: Sequence[ImageRecord], tracker: StageTracker) -> None:
    minimum = config.corpus.min_images_per_class
    if minimum <= 1:
        return

    counts: dict[str, int] = {}
    for record in records:
        if record.is_accepted:
            counts[record.label] = counts.get(record.label, 0) + 1

    undersized = {label for label, count in counts.items() if count < minimum}
    if not undersized:
        return

    for record in records:
        if record.is_accepted and record.label in undersized:
            record.reject(
                PipelineStage.CORPUS,
                RejectionCode.CLASS_BELOW_MINIMUM,
                label=record.label,
                count=counts[record.label],
                minimum=minimum,
            )
            tracker.rejected()
    tracker.warn("classes.below_minimum", classes=sorted(undersized), minimum=minimum)


def _assign_class_indices(records: Sequence[ImageRecord], labels: LabelMapping) -> None:
    for record in records:
        record.class_index = labels.index_of(record.label) if record.is_accepted else None


def _finalize(
    config: Config,
    context: RunContext,
    records: list[ImageRecord],
    labels: LabelMapping,
    summaries: list[SourceSummary],
    harmonizer: LabelHarmonizer,
    tracker: StageTracker,
) -> Corpus:
    fingerprint = compute_fingerprint(records, labels, summaries, config.semantic_fingerprint())
    previous = load_previous(context.layout.dataset_fingerprint)
    version, reason = resolve_version(config.corpus.versioning, fingerprint, previous, context.started_at)
    manifest = context.set_dataset_version(version)

    corpus = Corpus(
        records=records,
        labels=labels,
        sources=summaries,
        fingerprint=fingerprint.value,
        version=version,
        manifest=manifest,
        statistics=_statistics(records, labels, summaries, harmonizer),
    )

    writer = MetadataWriter(context.layout, config)
    writer.write_all(corpus, harmonization=harmonizer.as_dict())
    write_fingerprint(context.layout.dataset_fingerprint, fingerprint, corpus, manifest)
    if config.corpus.track_first_seen:
        write_corpus_index(build_frame(corpus), config.paths.corpus_dir)

    tracker.metrics(
        sources=len(summaries),
        images_discovered=len(records),
        images_accepted=len(corpus.accepted()),
        images_rejected=len(corpus.rejected()),
        classes=labels.num_classes,
        raw_labels=len(harmonizer.history()),
        dataset_version=version,
        version_reason=reason,
        fingerprint=fingerprint.short(),
    )
    tracker.info(
        "corpus.built",
        dataset_version=version,
        version_reason=reason,
        fingerprint=fingerprint.short(),
        images=len(corpus.accepted()),
        classes=labels.num_classes,
    )
    return corpus


def _statistics(
    records: Sequence[ImageRecord],
    labels: LabelMapping,
    summaries: Sequence[SourceSummary],
    harmonizer: LabelHarmonizer,
) -> dict[str, object]:
    accepted = [record for record in records if record.is_accepted]
    return {
        "images_discovered": len(records),
        "images_accepted": len(accepted),
        "images_rejected": len(records) - len(accepted),
        "num_classes": labels.num_classes,
        "num_sources": len(summaries),
        "raw_label_count": len(harmonizer.history()),
        "harmonization_rules": harmonizer.rule_counts(),
        "crop_vocabulary": list(harmonizer.crop_vocabulary),
        "bytes_ingested": sum(source.total_bytes for source in summaries),
    }


__all__ = [
    "CorpusBuilder",
    "DatasetFingerprint",
    "FingerprintComponents",
    "HarmonizationRule",
    "LabelHarmonizer",
    "LabelResolution",
    "MetadataWriter",
    "SourceScan",
    "attach_ingest_metadata",
    "build_corpus",
    "build_frame",
    "compute_fingerprint",
    "load_first_seen",
    "load_previous",
    "probe_dimensions",
    "raw_label_counts",
    "resolve_sources",
    "resolve_version",
    "write_corpus_index",
    "write_fingerprint",
]
