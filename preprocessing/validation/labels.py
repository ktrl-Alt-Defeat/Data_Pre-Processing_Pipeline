"""Label validation.

Verifies that the label space produced by corpus harmonisation is well formed
and that every record points into it. Labels are never rewritten here: a label
that needs fixing is a corpus-configuration problem, and silently repairing it
would hide the defect from the very report meant to surface it.

Two scopes:

* :meth:`LabelValidator.validate_mapping` - once per run, over the label space
* :meth:`LabelValidator.validate` - per record, a pure dictionary lookup
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from ..core.config import LabelValidationConfig
from ..core.records import ImageRecord, LabelMapping, RejectionCode, Severity, ValidationIssue

VALIDATOR = "labels"

_WHITESPACE = re.compile(r"\s")


class LabelValidator:
    """Validates canonical labels and the label mapping they index into."""

    def __init__(self, config: LabelValidationConfig, labels: LabelMapping) -> None:
        self._config = config
        self._labels = labels
        self._pattern = re.compile(config.pattern)
        self._counter: Counter[str] = Counter()
        self._invalid_labels = self._scan_label_space()

    @property
    def metrics(self) -> dict[str, int]:
        return dict(sorted(self._counter.items()))

    def validate_mapping(self) -> list[ValidationIssue]:
        """Structural checks over the label space itself, independent of records."""
        issues: list[ValidationIssue] = []
        issues.extend(self._mapping_issue(label, reason) for label, reason in sorted(self._invalid_labels.items()))
        issues.extend(self._duplicate_index_issues())
        issues.extend(self._ambiguous_alias_issues())
        self._counter["mapping_issues"] = len(issues)
        return issues

    def validate(self, record: ImageRecord) -> list[ValidationIssue]:
        """Validate one record's label; returns an empty list when it is sound."""
        self._counter["validated"] += 1
        label = record.label

        if not label:
            return [self._issue(record, RejectionCode.LABEL_MISSING, "record carries no canonical label")]

        reason = self._invalid_labels.get(label) or self._label_defect(label)
        if reason:
            return [
                self._issue(
                    record,
                    RejectionCode.LABEL_UNMAPPED,
                    f"canonical label '{label}' is invalid: {reason}",
                    label=label,
                    reason=reason,
                )
            ]

        if self._config.require_mapping and label not in self._labels.label_to_index:
            return [
                self._issue(
                    record,
                    RejectionCode.LABEL_UNMAPPED,
                    f"canonical label '{label}' is not present in the label mapping",
                    label=label,
                )
            ]

        expected = self._labels.index_of(label)
        if self._config.require_class_index and record.class_index != expected:
            return [
                self._issue(
                    record,
                    RejectionCode.LABEL_UNMAPPED,
                    f"class index {record.class_index} does not match the mapping index {expected} for '{label}'",
                    label=label,
                    class_index=record.class_index,
                    expected_index=expected,
                )
            ]
        return []

    # --- label space ---------------------------------------------------------- #

    def _scan_label_space(self) -> dict[str, str]:
        return {
            label: reason
            for label in self._labels.label_to_index
            if (reason := self._label_defect(label)) is not None
        }

    def _label_defect(self, label: str) -> str | None:
        """First structural defect in a label, or ``None`` when it is well formed."""
        if not label:
            return "label is empty"
        if not self._config.allow_whitespace and _WHITESPACE.search(label):
            return "label contains whitespace"
        if label != label.strip():
            return "label has leading or trailing whitespace"
        if len(label) > self._config.max_length:
            return f"label exceeds the maximum length of {self._config.max_length}"

        forbidden = [char for char in self._config.forbidden_characters if char in label]
        if forbidden:
            return f"label contains forbidden characters: {''.join(forbidden)}"
        if not self._pattern.match(label):
            return f"label does not match the configured pattern {self._config.pattern}"
        return None

    def _duplicate_index_issues(self) -> list[ValidationIssue]:
        by_index: dict[int, list[str]] = defaultdict(list)
        for label, index in self._labels.label_to_index.items():
            by_index[index].append(label)

        return [
            self._mapping_issue(
                ", ".join(sorted(labels)),
                f"class index {index} is shared by {len(labels)} labels",
            )
            for index, labels in sorted(by_index.items())
            if len(labels) > 1
        ]

    def _ambiguous_alias_issues(self) -> list[ValidationIssue]:
        """An alias pointing at a label that does not exist breaks reverse lookups."""
        return [
            self._mapping_issue(raw, f"alias resolves to '{canonical}', which is not a known label")
            for raw, canonical in sorted(self._labels.aliases.items())
            if canonical not in self._labels.label_to_index
        ]

    def _mapping_issue(self, subject: str, reason: str) -> ValidationIssue:
        self._counter["mapping_defects"] += 1
        return ValidationIssue(
            image_id="",
            validator=VALIDATOR,
            code=RejectionCode.LABEL_UNMAPPED,
            message=f"label mapping defect for '{subject}': {reason}",
            severity=Severity.WARNING,
            detail={"subject": subject, "reason": reason},
        )

    def _issue(self, record: ImageRecord, code: RejectionCode, message: str, **detail: Any) -> ValidationIssue:
        self._counter[f"error:{code.value}"] += 1
        return ValidationIssue(
            image_id=record.image_id,
            validator=VALIDATOR,
            code=code,
            message=message,
            severity=Severity.ERROR,
            detail=detail,
        )


__all__ = ["VALIDATOR", "LabelValidator"]
