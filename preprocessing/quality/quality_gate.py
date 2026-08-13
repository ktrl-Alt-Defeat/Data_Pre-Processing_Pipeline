"""The quality decision engine.

Two responsibilities, deliberately separated:

* :func:`analyze_image` decodes an image **once** and derives every pixel metric
  and every hash from that single array. Nothing else in this stage opens a file.
* :class:`QualityGate` turns those measurements plus the duplicate verdict into
  one :class:`QualityMetrics` object and one auditable decision.

No decision is ever returned without the reasons that produced it: each carries
the metric, its measured value, the configured threshold and the comparison that
failed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ..core.config import QualityConfig, QualityScoringConfig
from ..core.errors import ImageError
from ..core.io import load_rgb, pixel_digest, sha256_file
from ..core.records import DuplicateStatus, ImageRecord, QualityMetrics, RejectionCode
from .blur import BlurAnalyzer, BlurResult
from .brightness import ExposureAnalyzer, ExposureResult
from .duplicates import DuplicateLink, perceptual_hash, to_hex
from .resolution import ResolutionAnalyzer, ResolutionResult

ACCEPT = "accept"
WARN = "warn"
REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    """Everything measured from one decode of one image."""

    image_id: str
    blur: BlurResult | None = None
    exposure: ExposureResult | None = None
    resolution: ResolutionResult | None = None
    content_hash: str | None = None
    pixel_hash: str | None = None
    perceptual_hash: int | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True, slots=True)
class QualityReason:
    """Why the gate reached its verdict, in terms a report can print verbatim."""

    check: str
    metric: str
    value: float | None
    threshold: float | None
    comparison: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class QualityDecision:
    """The gate's verdict for one image."""

    action: str
    score: float
    grade: str
    code: RejectionCode | None = None
    reasons: tuple[QualityReason, ...] = ()
    timestamp: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
    )

    @property
    def rejected(self) -> bool:
        return self.action == REJECT

    @property
    def summary(self) -> str:
        return "; ".join(reason.message for reason in self.reasons) or "all quality checks passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "grade": self.grade,
            "code": self.code.value if self.code else None,
            "timestamp": self.timestamp,
            "reasons": [reason.as_dict() for reason in self.reasons],
        }


class ImageAnalyzer:
    """Decodes each image once and derives every metric from that one array."""

    def __init__(self, config: QualityConfig) -> None:
        self._config = config
        self._blur = BlurAnalyzer(config.blur)
        self._exposure = ExposureAnalyzer(config.brightness, config.contrast)
        self._resolution = ResolutionAnalyzer(config.resolution)

    def analyze(self, record: ImageRecord) -> ImageAnalysis:
        """Measure one image. Never raises; failures come back as an error field."""
        duplicates = self._config.duplicates
        needs_content_hash = duplicates.enabled and duplicates.exact and duplicates.exact_method != "pixel"

        try:
            content_hash = sha256_file(record.source_path) if needs_content_hash else None
            with load_rgb(record.source_path) as image:
                rgb_full = np.asarray(image, dtype=np.uint8)
                width, height = image.width, image.height
                pixel_hash = self._pixel_hash(image)
        except ImageError as exc:
            return ImageAnalysis(record.image_id, error=exc.message)
        except (OSError, ValueError) as exc:
            return ImageAnalysis(record.image_id, error=f"quality analysis failed: {exc}")

        rgb = _downscale(rgb_full, self._config.metric_resize)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        return ImageAnalysis(
            image_id=record.image_id,
            blur=self._blur.analyze(gray) if self._config.blur.enabled else None,
            exposure=self._exposure.analyze(gray, rgb),
            resolution=self._resolution.analyze(width, height),
            content_hash=content_hash,
            pixel_hash=pixel_hash,
            perceptual_hash=self._perceptual_hash(gray),
        )

    def _pixel_hash(self, image) -> str | None:
        duplicates = self._config.duplicates
        if duplicates.enabled and duplicates.exact and duplicates.exact_method != "content":
            return pixel_digest(image)
        return None

    def _perceptual_hash(self, gray: np.ndarray) -> int | None:
        duplicates = self._config.duplicates
        if not duplicates.enabled or not duplicates.near:
            return None
        return perceptual_hash(gray, duplicates.hash_size, duplicates.hash_type)


class QualityGate:
    """Combines every measurement into one score, one grade and one decision."""

    def __init__(self, config: QualityConfig) -> None:
        self._config = config
        self._scoring = config.scoring

    def metrics(self, analysis: ImageAnalysis, link: DuplicateLink | None, is_representative: bool) -> QualityMetrics:
        """Assemble the complete :class:`QualityMetrics` for one image."""
        metrics = QualityMetrics()
        if analysis.blur:
            metrics.blur_score = analysis.blur.score
            metrics.sharpness = _ratio(analysis.blur.score, self._scoring.reference_blur)
        if analysis.exposure:
            metrics.brightness = analysis.exposure.brightness
            metrics.contrast = analysis.exposure.contrast
            metrics.entropy = analysis.exposure.entropy
            metrics.colorfulness = analysis.exposure.colorfulness
        if analysis.resolution:
            metrics.width = analysis.resolution.width
            metrics.height = analysis.resolution.height
            metrics.megapixels = analysis.resolution.megapixels
            metrics.aspect_ratio = analysis.resolution.aspect_ratio

        metrics.duplicate_status = link.status if link else (
            DuplicateStatus.REPRESENTATIVE if is_representative else DuplicateStatus.UNIQUE
        )
        metrics.perceptual_similarity = link.similarity if link else None
        metrics.score = self._score(metrics)
        metrics.grade = self._grade(metrics.score)
        return metrics

    def decide(self, analysis: ImageAnalysis, metrics: QualityMetrics, link: DuplicateLink | None) -> QualityDecision:
        """Apply the configured policy and explain the outcome."""
        if analysis.failed:
            reason = QualityReason("decode", "source_image", None, None, "n/a", analysis.error or "unreadable")
            return QualityDecision(REJECT, 0.0, self._scoring.fallback_grade, RejectionCode.UNREADABLE_FILE, (reason,))

        failures = list(self._gate_failures(analysis, metrics))
        duplicate_failure = self._duplicate_failure(link)
        score = metrics.score if metrics.score is not None else 0.0
        grade = metrics.grade or self._scoring.fallback_grade

        if duplicate_failure and duplicate_failure[0] == REJECT:
            return QualityDecision(REJECT, score, grade, duplicate_failure[1], (duplicate_failure[2], *_reasons(failures)))
        if failures and self._config.reject_on_failure:
            return QualityDecision(REJECT, score, grade, failures[0][0], _reasons(failures))
        if self._scoring.enabled and score < self._scoring.min_score:
            reason = _threshold_reason("quality_score", "score", score, self._scoring.min_score, "<")
            return QualityDecision(REJECT, score, grade, RejectionCode.LOW_QUALITY_SCORE, (reason, *_reasons(failures)))

        warnings = _reasons(failures)
        if duplicate_failure and duplicate_failure[0] == WARN:
            warnings = (duplicate_failure[2], *warnings)
        if self._scoring.enabled and score < self._scoring.warn_score:
            warnings = (*warnings, _threshold_reason("quality_score", "score", score, self._scoring.warn_score, "<"))
        if warnings:
            return QualityDecision(WARN, score, grade, None, warnings)
        return QualityDecision(ACCEPT, score, grade)

    # --- policy ---------------------------------------------------------------- #

    def _gate_failures(self, analysis: ImageAnalysis, metrics: QualityMetrics):
        blur, exposure, resolution = analysis.blur, analysis.exposure, analysis.resolution
        config = self._config

        if blur and config.blur.enabled and not blur.passed:
            yield RejectionCode.BLURRY, _threshold_reason("blur", "blur_score", blur.score, blur.threshold, "<")

        if exposure and config.brightness.enabled:
            if exposure.underexposed:
                yield RejectionCode.TOO_DARK, _threshold_reason(
                    "brightness", "brightness", exposure.brightness, config.brightness.min_mean, "<"
                )
            elif exposure.overexposed:
                yield RejectionCode.TOO_BRIGHT, _threshold_reason(
                    "brightness", "brightness", exposure.brightness, config.brightness.max_mean, ">"
                )
            limit = config.brightness.max_clipped_fraction
            if limit is not None and exposure.clipped_fraction > limit:
                clipping_side = (
                    RejectionCode.TOO_BRIGHT
                    if exposure.highlight_fraction >= exposure.shadow_fraction
                    else RejectionCode.TOO_DARK
                )
                yield clipping_side, _threshold_reason(
                    "exposure", "clipped_fraction", exposure.clipped_fraction, limit, ">"
                )

        if exposure and config.contrast.enabled and exposure.low_contrast:
            yield RejectionCode.LOW_CONTRAST, _threshold_reason(
                "contrast", "contrast", exposure.contrast, config.contrast.min_value, "<"
            )

        if resolution:
            for finding in resolution.findings:
                yield finding.code, QualityReason(
                    "resolution", finding.metric, finding.value, finding.threshold, finding.comparison, finding.message
                )

    def _duplicate_failure(self, link: DuplicateLink | None) -> tuple[str, RejectionCode, QualityReason] | None:
        if link is None:
            return None
        config = self._config.duplicates
        exact = link.status is DuplicateStatus.EXACT_DUPLICATE
        action = config.exact_action if exact else config.near_action
        if action == "keep":
            return None

        code = RejectionCode.EXACT_DUPLICATE if exact else RejectionCode.NEAR_DUPLICATE
        detail = "identical to" if exact else f"{link.similarity:.3f} similar to"
        reason = QualityReason(
            check="duplicates",
            metric="perceptual_similarity",
            value=link.similarity,
            threshold=None if exact else 1.0 - (config.max_hamming_distance / (config.hash_size**2)),
            comparison=">=",
            message=f"{detail} retained image {link.duplicate_of} (group {link.group_id})",
        )
        return (REJECT if action == "reject" else WARN), code, reason

    # --- scoring ---------------------------------------------------------------- #

    def _score(self, metrics: QualityMetrics) -> float:
        components = self._components(metrics)
        weights = {name: weight for name, weight in self._scoring.weights.items() if name in components}
        total = sum(weights.values())
        if not total:
            return 0.0
        weighted = sum(components[name] * weight for name, weight in weights.items())
        return round(weighted / total, 6)

    def _components(self, metrics: QualityMetrics) -> dict[str, float]:
        scoring = self._scoring
        components: dict[str, float] = {}
        if metrics.blur_score is not None:
            components["blur"] = _ratio(metrics.blur_score, scoring.reference_blur)
        if metrics.brightness is not None:
            components["brightness"] = _brightness_score(
                metrics.brightness, self._config.brightness.min_mean, self._config.brightness.max_mean
            )
        if metrics.contrast is not None:
            components["contrast"] = _ratio(metrics.contrast, scoring.reference_contrast)
        if metrics.megapixels is not None:
            components["resolution"] = _ratio(metrics.megapixels, scoring.reference_megapixels)
        if metrics.entropy is not None:
            components["entropy"] = _ratio(metrics.entropy, 8.0)
        if metrics.colorfulness is not None:
            components["colorfulness"] = _ratio(metrics.colorfulness, 100.0)
        return components

    def _grade(self, score: float | None) -> str:
        if score is None:
            return self._scoring.fallback_grade
        for grade, threshold in sorted(self._scoring.grade_thresholds.items(), key=lambda item: -item[1]):
            if score >= threshold:
                return grade
        return self._scoring.fallback_grade


def apply_decision(record: ImageRecord, metrics: QualityMetrics, decision: QualityDecision, stage) -> None:
    """Attach metrics and the verdict to a record, preserving all provenance."""
    record.quality = metrics
    record.record_operation(
        stage,
        "quality_assessment",
        action=decision.action,
        score=decision.score,
        grade=decision.grade,
        timestamp=decision.timestamp,
    )
    if decision.rejected and decision.code is not None:
        record.reject(
            stage,
            decision.code,
            decision.summary,
            validator="quality_gate",
            timestamp=decision.timestamp,
            score=decision.score,
            grade=decision.grade,
            metrics={key: value for key, value in metrics.as_dict().items() if value is not None},
            reasons=[reason.as_dict() for reason in decision.reasons],
        )


def _downscale(rgb: np.ndarray, longest_side: int) -> np.ndarray:
    """Fit the image into ``longest_side`` so metrics are resolution-invariant."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= longest_side:
        return rgb
    scale = longest_side / longest
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return float(min(1.0, max(0.0, value / reference)))


def _brightness_score(brightness: float, minimum: float, maximum: float) -> float:
    """1.0 inside the configured window, decaying linearly to 0 at pure black/white."""
    if minimum <= brightness <= maximum:
        return 1.0
    if brightness < minimum:
        return float(max(0.0, brightness / minimum)) if minimum > 0 else 0.0
    headroom = 255.0 - maximum
    return float(max(0.0, (255.0 - brightness) / headroom)) if headroom > 0 else 0.0


def _threshold_reason(check: str, metric: str, value: float, threshold: float, comparison: str) -> QualityReason:
    return QualityReason(
        check=check,
        metric=metric,
        value=float(value),
        threshold=float(threshold),
        comparison=comparison,
        message=f"{metric} {value:.4g} {comparison} configured threshold {threshold:g}",
    )


def _reasons(failures: Sequence[tuple[RejectionCode, QualityReason]]) -> tuple[QualityReason, ...]:
    return tuple(reason for _, reason in failures)


__all__ = [
    "ACCEPT",
    "REJECT",
    "WARN",
    "ImageAnalysis",
    "ImageAnalyzer",
    "QualityDecision",
    "QualityGate",
    "QualityReason",
    "apply_decision",
]
