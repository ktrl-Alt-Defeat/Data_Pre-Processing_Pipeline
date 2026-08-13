"""Label harmonisation across heterogeneous datasets.

Folder names arrive in every shape a public dataset can invent::

    Tomato___Late_Blight    Late Blight    late_blight
    Late-Blight             tomato_late_blight

All of them must collapse onto one canonical label. The resolution order is
fully configuration driven — no mapping is hardcoded — and every decision is
retained as a :class:`LabelResolution` so ``label_mapping.json`` can explain how
each raw folder name became a class.

Resolution order per raw label:

1. ``label_overrides`` on the owning source (most specific)
2. ``label_aliases`` / ``alias_file`` at corpus level
3. a configured delimiter, e.g. ``Tomato___Late_Blight``
4. the crop vocabulary, which is inferred from the delimited labels and then
   used to split undelimited ones (``tomato_late_blight`` -> tomato + late blight)
5. the source's ``default_crop``
6. condition only
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.config import CorpusConfig, SourceConfig
from ..core.errors import ConfigurationError
from ..core.io import read_yaml, slugify
from ..core.logging import get_logger
from ..core.records import LabelMapping

_logger = get_logger(__name__)


class HarmonizationRule(str, Enum):
    """How a canonical label was derived; reported per raw label."""

    SOURCE_OVERRIDE = "source_override"
    ALIAS = "alias"
    DELIMITER = "delimiter"
    CROP_VOCABULARY = "crop_vocabulary"
    DEFAULT_CROP = "default_crop"
    CONDITION_ONLY = "condition_only"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True, slots=True)
class LabelResolution:
    """One entry of the harmonisation history."""

    dataset: str
    raw_label: str
    normalized: str
    crop: str | None
    condition: str
    canonical: str
    rule: HarmonizationRule
    image_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "raw_label": self.raw_label,
            "normalized": self.normalized,
            "crop": self.crop,
            "condition": self.condition,
            "canonical": self.canonical,
            "rule": self.rule.value,
            "image_count": self.image_count,
        }


class LabelHarmonizer:
    """Maps raw folder labels onto a canonical, dataset-independent label space."""

    def __init__(self, config: CorpusConfig, sources: Mapping[str, SourceConfig]) -> None:
        self._config = config
        self._sources = dict(sources)
        self._delimiters = tuple(sorted(config.label_delimiters, key=len, reverse=True))
        self._aliases = self._load_aliases()
        self._crops: tuple[str, ...] = tuple(sorted({slugify(crop) for crop in config.crops if crop}))
        self._resolutions: dict[tuple[str, str], LabelResolution] = {}

    # --- public API ---------------------------------------------------------- #

    def fit(self, label_counts: Mapping[tuple[str, str], int]) -> "LabelHarmonizer":
        """Learn the crop vocabulary, then resolve every ``(dataset, raw_label)`` pair."""
        self._crops = self._build_crop_vocabulary(label_counts)
        self._resolutions = {}
        for (dataset, raw_label), count in sorted(label_counts.items()):
            self._resolutions[(dataset, raw_label)] = self._resolve(dataset, raw_label, count)
        _logger.info(
            "labels.harmonized",
            raw_labels=len(self._resolutions),
            canonical_labels=len({r.canonical for r in self._resolutions.values()}),
            crops=len(self._crops),
            rules=self.rule_counts(),
        )
        return self

    def canonical(self, dataset: str, raw_label: str) -> LabelResolution:
        """Resolution for one raw label; resolves on demand if unseen during fit."""
        key = (dataset, raw_label)
        if key not in self._resolutions:
            self._resolutions[key] = self._resolve(dataset, raw_label, 0)
        return self._resolutions[key]

    @property
    def crop_vocabulary(self) -> tuple[str, ...]:
        return self._crops

    def history(self) -> list[LabelResolution]:
        """Complete mapping history, sorted by dataset then raw label."""
        return sorted(self._resolutions.values(), key=lambda r: (r.dataset, r.raw_label))

    def rule_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(r.rule.value for r in self._resolutions.values()).items()))

    def mapping(self, labels: Iterable[str] | None = None) -> LabelMapping:
        """Build the canonical :class:`LabelMapping`, including raw-label aliases.

        Aliases are keyed by raw label; a raw label that resolves differently per
        dataset is stored namespaced (``dataset:raw_label``) so no mapping is lost.
        """
        canonical_labels = list(labels) if labels is not None else [r.canonical for r in self._resolutions.values()]
        by_raw: dict[str, set[str]] = {}
        for resolution in self._resolutions.values():
            by_raw.setdefault(resolution.raw_label, set()).add(resolution.canonical)

        aliases: dict[str, str] = {}
        for resolution in self.history():
            if len(by_raw[resolution.raw_label]) == 1:
                aliases[resolution.raw_label] = resolution.canonical
            else:
                aliases[f"{resolution.dataset}:{resolution.raw_label}"] = resolution.canonical
        return LabelMapping.from_labels(canonical_labels, aliases)

    def as_dict(self) -> dict[str, Any]:
        """Harmonisation section of ``label_mapping.json``."""
        return {
            "crop_vocabulary": list(self._crops),
            "delimiters": list(self._delimiters),
            "canonical_format": self._config.canonical_format,
            "rules_applied": self.rule_counts(),
            "history": [resolution.as_dict() for resolution in self.history()],
        }

    # --- resolution ---------------------------------------------------------- #

    def _resolve(self, dataset: str, raw_label: str, count: int) -> LabelResolution:
        normalized = slugify(raw_label, fallback="unknown")
        if not self._config.normalize_labels:
            return LabelResolution(
                dataset, raw_label, normalized, None, raw_label, raw_label, HarmonizationRule.PASSTHROUGH, count
            )

        source = self._sources.get(dataset)
        overrides = source.label_overrides if source else {}
        target, rule = self._lookup(raw_label, normalized, overrides, HarmonizationRule.SOURCE_OVERRIDE)
        if target is None:
            target, rule = self._lookup(raw_label, normalized, self._aliases, HarmonizationRule.ALIAS)

        text = target if target is not None else raw_label
        crop, condition, derived_rule = self._derive(text, source)
        rule = rule if target is not None else derived_rule

        condition = self._normalize_condition(condition)
        canonical = self._compose(crop, condition)
        return LabelResolution(dataset, raw_label, normalized, crop, condition, canonical, rule, count)

    def _lookup(
        self,
        raw_label: str,
        normalized: str,
        table: Mapping[str, str],
        rule: HarmonizationRule,
    ) -> tuple[str | None, HarmonizationRule]:
        for key in (raw_label, normalized, slugify(raw_label)):
            if key in table:
                return table[key], rule
        return None, rule

    def _derive(self, text: str, source: SourceConfig | None) -> tuple[str | None, str, HarmonizationRule]:
        crop, condition = self._split_delimited(text)
        if crop:
            return crop, condition, HarmonizationRule.DELIMITER

        normalized = slugify(text, fallback="unknown")
        crop = self._match_crop(normalized)
        if crop:
            return crop, normalized[len(crop) + 1 :], HarmonizationRule.CROP_VOCABULARY

        if source and source.default_crop:
            return slugify(source.default_crop), normalized, HarmonizationRule.DEFAULT_CROP

        return None, normalized, HarmonizationRule.CONDITION_ONLY

    def _split_delimited(self, text: str) -> tuple[str | None, str]:
        for delimiter in self._delimiters:
            if delimiter not in text:
                continue
            head, _, tail = text.partition(delimiter)
            crop, condition = slugify(head), slugify(tail)
            if crop and condition:
                return crop, condition
        return None, slugify(text, fallback="unknown")

    def _match_crop(self, normalized: str) -> str | None:
        """Longest crop prefix that leaves a non-empty condition behind."""
        candidates = [crop for crop in self._crops if normalized.startswith(f"{crop}_")]
        return max(candidates, key=len) if candidates else None

    def _normalize_condition(self, condition: str) -> str:
        healthy = {slugify(term) for term in self._config.healthy_terms}
        if condition in healthy:
            return slugify(self._config.healthy_label)
        return condition

    def _compose(self, crop: str | None, condition: str) -> str:
        template = self._config.canonical_format if crop else self._config.condition_only_format
        label = template.format(crop=crop or "", condition=condition)
        return label.lower() if self._config.lowercase_labels else label

    # --- setup --------------------------------------------------------------- #

    def _build_crop_vocabulary(self, label_counts: Mapping[tuple[str, str], int]) -> tuple[str, ...]:
        vocabulary = set(self._crops)
        vocabulary.update(slugify(source.default_crop) for source in self._sources.values() if source.default_crop)

        if self._config.infer_crop_vocabulary:
            for _, raw_label in label_counts:
                crop, _ = self._split_delimited(raw_label)
                if crop:
                    vocabulary.add(crop)
            for value in list(self._aliases.values()):
                crop, _ = self._split_delimited(value)
                if crop:
                    vocabulary.add(crop)
        return tuple(sorted(term for term in vocabulary if term))

    def _load_aliases(self) -> dict[str, str]:
        aliases = dict(self._config.label_aliases)
        path = self._config.alias_file
        if path is None:
            return aliases

        alias_path = Path(path)
        if not alias_path.exists():
            raise ConfigurationError(f"corpus.alias_file does not exist: {alias_path}")
        try:
            loaded = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.suffix == ".json" else read_yaml(alias_path)
        except (ValueError, OSError) as exc:
            raise ConfigurationError(f"could not read corpus.alias_file {alias_path}: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigurationError(f"corpus.alias_file must contain a mapping: {alias_path}")

        # File entries lose to inline config, which is the more specific override.
        merged = {str(key): str(value) for key, value in loaded.items()}
        merged.update(aliases)
        return merged


__all__ = ["HarmonizationRule", "LabelHarmonizer", "LabelResolution"]
