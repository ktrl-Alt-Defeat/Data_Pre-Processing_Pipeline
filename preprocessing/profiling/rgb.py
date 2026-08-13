"""Per-channel RGB statistics.

This is the only part of profiling that touches pixels, so it is also the only
part that can be slow. Three things keep it affordable on a corpus of hundreds
of thousands of images:

* a deterministic sample rather than the whole corpus
* aggressive downscaling before accumulation
* a cache keyed by dataset fingerprint, so a rerun of the same corpus is free

Accumulation is single-pass Welford-style (sum and sum of squares in float64),
and partial results are combined in sample order, so the result does not depend
on thread scheduling.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..core.config import RgbProfilingConfig
from ..core.errors import ImageError
from ..core.io import ensure_dir, load_rgb, read_json, write_json
from ..core.logging import get_logger
from ..core.records import ImageRecord

_logger = get_logger(__name__)
_MAX_LEVEL = 255.0


@dataclass(frozen=True, slots=True)
class RgbStatistics:
    """Channel means, standard deviations and histograms over the sampled images."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    histogram: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    bins: int
    sample_size: int
    pixels: int
    failures: int = 0

    @property
    def mean_255(self) -> tuple[float, float, float]:
        return tuple(round(value * _MAX_LEVEL, 4) for value in self.mean)  # type: ignore[return-value]

    @property
    def std_255(self) -> tuple[float, float, float]:
        return tuple(round(value * _MAX_LEVEL, 4) for value in self.std)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "mean_255": list(self.mean_255),
            "std_255": list(self.std_255),
            "histogram": [list(channel) for channel in self.histogram],
            "bins": self.bins,
            "sample_size": self.sample_size,
            "pixels": self.pixels,
            "failures": self.failures,
        }


@dataclass(frozen=True, slots=True)
class _Accumulation:
    total: np.ndarray
    total_squared: np.ndarray
    histogram: np.ndarray
    pixels: int
    failed: bool = False


class RgbProfiler:
    """Computes dataset RGB statistics from a deterministic sample."""

    def __init__(self, config: RgbProfilingConfig, cache_dir: Path | None = None, workers: int = 1) -> None:
        self._config = config
        self._cache_dir = cache_dir
        self._workers = config.workers or workers

    def profile(self, records: Sequence[ImageRecord], fingerprint: str | None = None) -> RgbStatistics | None:
        """Channel statistics for ``records``; ``None`` when profiling is disabled."""
        if not self._config.enabled or not records:
            return None

        sample = self._sample(records)
        cache_key = self._cache_key(fingerprint, len(sample))
        cached = self._load_cache(cache_key)
        if cached is not None:
            _logger.info("rgb.cache_hit", sample_size=cached.sample_size, key=cache_key)
            return cached

        statistics = self._accumulate(sample)
        self._store_cache(cache_key, statistics)
        return statistics

    def _sample(self, records: Sequence[ImageRecord]) -> list[ImageRecord]:
        """Seeded sample over id-sorted records, so the selection is reproducible."""
        limit = self._config.sample_size
        ordered = sorted(records, key=lambda record: record.image_id)
        if limit is None or len(ordered) <= limit:
            return ordered
        return sorted(random.Random(len(ordered)).sample(ordered, limit), key=lambda record: record.image_id)

    def _accumulate(self, sample: Sequence[ImageRecord]) -> RgbStatistics:
        if self._workers > 1 and len(sample) > 1:
            with ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="rgb") as pool:
                parts = list(pool.map(self._measure, sample))
        else:
            parts = [self._measure(record) for record in sample]

        bins = self._config.histogram_bins
        total = np.zeros(3, dtype=np.float64)
        total_squared = np.zeros(3, dtype=np.float64)
        histogram = np.zeros((3, bins), dtype=np.int64)
        pixels = 0
        failures = 0

        for part in parts:
            if part.failed:
                failures += 1
                continue
            total += part.total
            total_squared += part.total_squared
            histogram += part.histogram
            pixels += part.pixels

        if pixels == 0:
            return RgbStatistics((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), ((), (), ()), bins, len(sample), 0, failures)

        mean = total / pixels
        variance = np.maximum(total_squared / pixels - mean**2, 0.0)
        return RgbStatistics(
            mean=tuple(round(float(value), 6) for value in mean),  # type: ignore[arg-type]
            std=tuple(round(float(value), 6) for value in np.sqrt(variance)),  # type: ignore[arg-type]
            histogram=tuple(tuple(int(count) for count in channel) for channel in histogram),  # type: ignore[arg-type]
            bins=bins,
            sample_size=len(sample) - failures,
            pixels=int(pixels),
            failures=failures,
        )

    def _measure(self, record: ImageRecord) -> _Accumulation:
        bins = self._config.histogram_bins
        try:
            with load_rgb(record.source_path) as image:
                array = self._downscale(np.asarray(image, dtype=np.uint8))
        except (ImageError, OSError, ValueError):
            return _Accumulation(np.zeros(3), np.zeros(3), np.zeros((3, bins), dtype=np.int64), 0, failed=True)

        flat = array.reshape(-1, 3)
        scaled = flat.astype(np.float64) / _MAX_LEVEL
        histogram = np.stack(
            [np.bincount((flat[:, channel].astype(np.int32) * bins) // 256, minlength=bins) for channel in range(3)]
        ).astype(np.int64)
        return _Accumulation(
            total=scaled.sum(axis=0),
            total_squared=np.square(scaled).sum(axis=0),
            histogram=histogram,
            pixels=int(flat.shape[0]),
        )

    def _downscale(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape[:2]
        longest = max(height, width)
        if longest <= self._config.resize:
            return array
        scale = self._config.resize / longest
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(array, target, interpolation=cv2.INTER_AREA)

    # --- cache ----------------------------------------------------------------- #

    def _cache_key(self, fingerprint: str | None, sample_size: int) -> str | None:
        if not self._config.cache or not fingerprint or self._cache_dir is None:
            return None
        return f"rgb_{fingerprint[:16]}_{sample_size}_{self._config.resize}_{self._config.histogram_bins}"

    def _load_cache(self, key: str | None) -> RgbStatistics | None:
        if key is None or self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = read_json(path)
            return RgbStatistics(
                mean=tuple(payload["mean"]),
                std=tuple(payload["std"]),
                histogram=tuple(tuple(channel) for channel in payload["histogram"]),
                bins=int(payload["bins"]),
                sample_size=int(payload["sample_size"]),
                pixels=int(payload["pixels"]),
                failures=int(payload.get("failures", 0)),
            )
        except (OSError, ValueError, KeyError, TypeError):
            _logger.warning("rgb.cache_unreadable", path=str(path))
            return None

    def _store_cache(self, key: str | None, statistics: RgbStatistics) -> None:
        if key is None or self._cache_dir is None:
            return
        write_json(ensure_dir(self._cache_dir) / f"{key}.json", statistics.as_dict())


__all__ = ["RgbProfiler", "RgbStatistics"]
