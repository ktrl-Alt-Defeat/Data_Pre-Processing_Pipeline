"""Resolution and geometry analysis.

Pure arithmetic on dimensions the validation stage already observed, so this
analyser costs nothing and never touches the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.config import ResolutionConfig
from ..core.records import RejectionCode

METRIC = "resolution"


@dataclass(frozen=True, slots=True)
class ResolutionFinding:
    """One geometry constraint that the image failed."""

    code: RejectionCode
    metric: str
    value: float
    threshold: float
    comparison: str
    message: str


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Geometry measurements plus every constraint the image violated."""

    width: int
    height: int
    megapixels: float
    aspect_ratio: float
    findings: tuple[ResolutionFinding, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def too_small(self) -> bool:
        return any(f.code is RejectionCode.LOW_RESOLUTION for f in self.findings)

    @property
    def too_large(self) -> bool:
        return any(f.code is RejectionCode.OVERSIZED_IMAGE for f in self.findings)

    @property
    def unusual_aspect_ratio(self) -> bool:
        return any(f.code is RejectionCode.EXTREME_ASPECT_RATIO for f in self.findings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "megapixels": self.megapixels,
            "aspect_ratio": self.aspect_ratio,
            "too_small": self.too_small,
            "too_large": self.too_large,
            "unusual_aspect_ratio": self.unusual_aspect_ratio,
            "passed": self.passed,
        }


class ResolutionAnalyzer:
    """Checks dimensions, megapixels and aspect ratio against configured bounds."""

    def __init__(self, config: ResolutionConfig) -> None:
        self._config = config

    def analyze(self, width: int, height: int) -> ResolutionResult:
        """Measure geometry and collect every violated constraint."""
        megapixels = (width * height) / 1_000_000
        aspect_ratio = width / height if height else 0.0
        findings = tuple(self._findings(width, height, megapixels, aspect_ratio)) if self._config.enabled else ()
        return ResolutionResult(width, height, megapixels, aspect_ratio, findings)

    def _findings(self, width: int, height: int, megapixels: float, aspect_ratio: float):
        config = self._config
        checks = (
            ("width", width, config.min_width, config.max_width),
            ("height", height, config.min_height, config.max_height),
            ("megapixels", megapixels, config.min_megapixels, config.max_megapixels),
        )
        for metric, value, minimum, maximum in checks:
            if minimum is not None and value < minimum:
                yield _finding(RejectionCode.LOW_RESOLUTION, metric, value, minimum, "<")
            if maximum is not None and value > maximum:
                yield _finding(RejectionCode.OVERSIZED_IMAGE, metric, value, maximum, ">")

        if aspect_ratio < config.min_aspect_ratio:
            yield _finding(RejectionCode.EXTREME_ASPECT_RATIO, "aspect_ratio", aspect_ratio, config.min_aspect_ratio, "<")
        elif aspect_ratio > config.max_aspect_ratio:
            yield _finding(RejectionCode.EXTREME_ASPECT_RATIO, "aspect_ratio", aspect_ratio, config.max_aspect_ratio, ">")


def _finding(code: RejectionCode, metric: str, value: float, threshold: float, comparison: str) -> ResolutionFinding:
    rendered = f"{value:.4g}" if isinstance(value, float) else str(value)
    return ResolutionFinding(
        code=code,
        metric=metric,
        value=float(value),
        threshold=float(threshold),
        comparison=comparison,
        message=f"{metric} {rendered} {comparison} configured limit {threshold:g}",
    )


__all__ = ["METRIC", "ResolutionAnalyzer", "ResolutionFinding", "ResolutionResult"]
