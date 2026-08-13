"""Blur analysis.

Measures how sharp an image is and reports the measurement against the
configured threshold. The analyser never rejects; it hands a number, a verdict
and a confidence to the quality gate.

Both algorithms operate on a grayscale array that the gate has already
downscaled to a fixed working size, so scores are comparable between a 6000px
photograph and a 300px thumbnail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..core.config import BlurConfig

METRIC = "blur_score"


@dataclass(frozen=True, slots=True)
class BlurResult:
    """Sharpness measurement for one image."""

    score: float
    threshold: float
    passed: bool
    confidence: float
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "blur_score": self.score,
            "blur_threshold": self.threshold,
            "blur_passed": self.passed,
            "blur_confidence": self.confidence,
            "blur_method": self.method,
        }


class BlurAnalyzer:
    """Computes an objective sharpness score using a configurable algorithm."""

    def __init__(self, config: BlurConfig) -> None:
        self._config = config
        self._measure = _LAPLACIAN if config.method == "laplacian" else _TENENGRAD

    def analyze(self, gray: np.ndarray) -> BlurResult:
        """Measure sharpness on a pre-scaled grayscale array."""
        score = float(self._measure(gray))
        threshold = float(self._config.min_score)
        return BlurResult(
            score=score,
            threshold=threshold,
            passed=score >= threshold,
            confidence=_confidence(score, threshold),
            method=self._config.method,
        )


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian: the standard focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude; less sensitive to isolated noise."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def _confidence(score: float, threshold: float) -> float:
    """How far the measurement sits from the threshold, normalised to [0, 1].

    A score at the threshold is a coin flip (0.0); one at twice or half the
    threshold is decisive (1.0). Borderline images are therefore visible as such
    in the report instead of hiding behind a bare pass/fail.
    """
    if threshold <= 0:
        return 1.0
    return float(min(1.0, abs(score - threshold) / threshold))


_LAPLACIAN = laplacian_variance
_TENENGRAD = tenengrad

__all__ = ["METRIC", "BlurAnalyzer", "BlurResult", "laplacian_variance", "tenengrad"]
