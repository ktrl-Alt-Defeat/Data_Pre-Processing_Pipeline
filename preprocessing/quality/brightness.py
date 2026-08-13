"""Photometric analysis: brightness, contrast, exposure and tonal spread.

Reports numbers only. Whether an underexposed image is acceptable is a policy
question, and policy lives in the quality gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.config import BrightnessConfig, ContrastConfig

METRIC_BRIGHTNESS = "brightness"
METRIC_CONTRAST = "contrast"

_HISTOGRAM_BINS = 256
_MAX_LEVEL = 255.0


@dataclass(frozen=True, slots=True)
class ExposureResult:
    """Photometric measurements for one image."""

    brightness: float
    contrast: float
    entropy: float
    colorfulness: float
    shadow_fraction: float
    highlight_fraction: float
    underexposed: bool
    overexposed: bool
    low_contrast: bool
    contrast_method: str

    @property
    def clipped_fraction(self) -> float:
        return self.shadow_fraction + self.highlight_fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "entropy": self.entropy,
            "colorfulness": self.colorfulness,
            "shadow_fraction": self.shadow_fraction,
            "highlight_fraction": self.highlight_fraction,
            "clipped_fraction": self.clipped_fraction,
            "underexposed": self.underexposed,
            "overexposed": self.overexposed,
            "low_contrast": self.low_contrast,
            "contrast_method": self.contrast_method,
        }


class ExposureAnalyzer:
    """Measures brightness, contrast and exposure without judging them."""

    def __init__(self, brightness: BrightnessConfig, contrast: ContrastConfig) -> None:
        self._brightness = brightness
        self._contrast = contrast

    def analyze(self, gray: np.ndarray, rgb: np.ndarray) -> ExposureResult:
        """Measure a pre-scaled grayscale array and its RGB counterpart."""
        brightness = float(gray.mean())
        contrast = self._measure_contrast(gray)
        return ExposureResult(
            brightness=brightness,
            contrast=contrast,
            entropy=shannon_entropy(gray),
            colorfulness=colorfulness(rgb),
            shadow_fraction=float(np.count_nonzero(gray <= self._brightness.shadow_level) / gray.size),
            highlight_fraction=float(np.count_nonzero(gray >= self._brightness.highlight_level) / gray.size),
            underexposed=brightness < self._brightness.min_mean,
            overexposed=brightness > self._brightness.max_mean,
            low_contrast=contrast < self._contrast.min_value,
            contrast_method=self._contrast.method,
        )

    def _measure_contrast(self, gray: np.ndarray) -> float:
        if self._contrast.method == "michelson":
            return michelson_contrast(gray)
        if self._contrast.method == "rms":
            return rms_contrast(gray)
        return float(gray.std())


def michelson_contrast(gray: np.ndarray) -> float:
    """(max - min) / (max + min), scaled to the 0-255 range for comparability."""
    low, high = float(gray.min()), float(gray.max())
    total = low + high
    if total <= 0:
        return 0.0
    return float(((high - low) / total) * _MAX_LEVEL)


def rms_contrast(gray: np.ndarray) -> float:
    """Root-mean-square deviation from the mean, on the 0-255 range."""
    normalised = gray.astype(np.float64) / _MAX_LEVEL
    return float(np.sqrt(np.mean((normalised - normalised.mean()) ** 2)) * _MAX_LEVEL)


def shannon_entropy(gray: np.ndarray) -> float:
    """Entropy of the intensity histogram in bits; low values mean flat images."""
    histogram = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=_HISTOGRAM_BINS).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler-Süsstrunk colourfulness; separates vivid foliage from washed-out shots."""
    channels = rgb.astype(np.float64)
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


__all__ = [
    "METRIC_BRIGHTNESS",
    "METRIC_CONTRAST",
    "ExposureAnalyzer",
    "ExposureResult",
    "colorfulness",
    "michelson_contrast",
    "rms_contrast",
    "shannon_entropy",
]
