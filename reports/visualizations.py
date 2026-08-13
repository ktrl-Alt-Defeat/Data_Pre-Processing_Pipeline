"""Publication-quality static visualisations.

Deterministic by construction: no sampling, no randomised layout, fixed figure
sizes and a fixed colour cycle, so re-running on the same corpus produces
pixel-identical figures. Every filename comes from :class:`OutputLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # no display in batch or CI environments

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from preprocessing.core.config import ReportsConfig  # noqa: E402
from preprocessing.core.context import OutputLayout  # noqa: E402
from preprocessing.core.logging import get_logger  # noqa: E402
from preprocessing.profiling.profiler import DatasetProfile  # noqa: E402

_logger = get_logger(__name__)

_PALETTE = ("#2f6f9f", "#c1553b", "#4c9a6a", "#8a6bbf", "#c9a227", "#7a7a7a")
_CHANNEL_COLORS = ("#c0392b", "#27ae60", "#2980b9")
_GRID = {"color": "#e6e6e6", "linewidth": 0.7}


@dataclass(frozen=True, slots=True)
class VisualizationSet:
    """Paths of the figures that were produced."""

    figures: Mapping[str, Path]

    def as_dict(self) -> dict[str, str]:
        return {name: str(path) for name, path in self.figures.items()}


class VisualizationRenderer:
    """Renders the dataset figures of the output package."""

    def __init__(self, config: ReportsConfig, layout: OutputLayout) -> None:
        self._config = config
        self._layout = layout

    def render(self, profile: DatasetProfile, analysis: Any | None = None) -> VisualizationSet:
        """Render every figure; a failure in one never blocks the others."""
        renderers = {
            "class_distribution.png": lambda path: self._class_distribution(profile, path),
            "resolution_distribution.png": lambda path: self._resolution(profile, path),
            "aspect_ratio.png": lambda path: self._aspect_ratio(profile, path),
            "brightness_distribution.png": lambda path: self._brightness(profile, path),
            "rgb_histogram.png": lambda path: self._rgb(profile, path),
            "duplicate_clusters.png": lambda path: self._duplicates(profile, path),
            "quality_dashboard.png": lambda path: self._dashboard(profile, analysis, path),
        }

        produced: dict[str, Path] = {}
        for filename, renderer in renderers.items():
            path = self._layout.visualization(filename)
            try:
                renderer(path)
            except (ValueError, KeyError, IndexError, RuntimeError) as exc:
                _logger.warning("visualization.failed", figure=filename, error=f"{type(exc).__name__}: {exc}")
                continue
            produced[filename] = path
        _logger.info("visualizations.rendered", count=len(produced), directory=str(self._layout.visualizations_dir))
        return VisualizationSet(produced)

    # --- individual figures ------------------------------------------------------ #

    def _class_distribution(self, profile: DatasetProfile, path: Path) -> None:
        classes = profile.classes
        if classes.empty:
            raise ValueError("no classes to plot")
        limit = self._config.max_classes_in_plots
        shown = classes.head(limit)
        height = max(3.2, 0.24 * len(shown) + 1.4)

        figure, axes = plt.subplots(figsize=(9.5, height))
        positions = np.arange(len(shown))
        axes.barh(positions, shown["images"], color=_PALETTE[0], height=0.72)
        axes.set_yticks(positions, shown["label"], fontsize=8)
        axes.invert_yaxis()
        axes.set_xlabel("images")
        axes.set_title(self._title("Class distribution", profile,
                                   f"{profile.class_count} classes, showing top {len(shown)}"))
        axes.grid(axis="x", **_GRID)
        axes.set_axisbelow(True)
        for position, value in zip(positions, shown["images"], strict=True):
            axes.text(value, position, f" {int(value):,}", va="center", fontsize=7, color="#444")
        self._finish(figure, path)

    def _resolution(self, profile: DatasetProfile, path: Path) -> None:
        figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        self._histogram(left, profile.histograms.get("megapixels"), "megapixels", _PALETTE[0])
        left.set_title("Megapixel distribution")

        resolutions = profile.resolutions.counts
        if resolutions:
            labels = list(resolutions)[:15][::-1]
            values = [resolutions[label] for label in labels]
            positions = np.arange(len(labels))
            right.barh(positions, values, color=_PALETTE[1], height=0.7)
            right.set_yticks(positions, labels, fontsize=8)
            right.set_xlabel("images")
        right.set_title("Most common resolutions")
        right.grid(axis="x", **_GRID)
        right.set_axisbelow(True)
        figure.suptitle(self._title("Resolution", profile), fontsize=11)
        self._finish(figure, path)

    def _aspect_ratio(self, profile: DatasetProfile, path: Path) -> None:
        figure, axes = plt.subplots(figsize=(9, 4.2))
        self._histogram(axes, profile.histograms.get("aspect_ratio"), "width / height", _PALETTE[2])
        axes.axvline(1.0, color="#333", linestyle="--", linewidth=1, label="square")
        axes.legend(fontsize=8, frameon=False)
        axes.set_title(self._title("Aspect ratio distribution", profile))
        self._finish(figure, path)

    def _brightness(self, profile: DatasetProfile, path: Path) -> None:
        figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        self._histogram(left, profile.histograms.get("brightness"), "mean luminance (0-255)", _PALETTE[4])
        left.set_title("Brightness")
        self._histogram(right, profile.histograms.get("contrast"), "contrast", _PALETTE[3])
        right.set_title("Contrast")
        figure.suptitle(self._title("Photometric distribution", profile), fontsize=11)
        self._finish(figure, path)

    def _rgb(self, profile: DatasetProfile, path: Path) -> None:
        rgb = profile.rgb
        if rgb is None or not any(rgb.histogram):
            raise ValueError("no RGB statistics available")

        figure, axes = plt.subplots(figsize=(9, 4.4))
        centers = (np.arange(rgb.bins) + 0.5) * (256 / rgb.bins)
        for channel, (counts, colour, name) in enumerate(
            zip(rgb.histogram, _CHANNEL_COLORS, ("red", "green", "blue"), strict=True)
        ):
            axes.plot(centers, np.asarray(counts, dtype=np.float64) / max(1, rgb.pixels),
                      color=colour, linewidth=1.6, label=f"{name} (mean {rgb.mean_255[channel]:.1f})")
        axes.set_xlabel("intensity (0-255)")
        axes.set_ylabel("share of pixels")
        axes.set_xlim(0, 255)
        axes.legend(fontsize=8, frameon=False)
        axes.grid(**_GRID)
        axes.set_axisbelow(True)
        axes.set_title(self._title("RGB channel histogram", profile,
                                   f"{rgb.sample_size:,} images sampled, {rgb.pixels:,} pixels"))
        self._finish(figure, path)

    def _duplicates(self, profile: DatasetProfile, path: Path) -> None:
        clusters = profile.duplicate_clusters
        figure, axes = plt.subplots(figsize=(9, 4.2))
        if clusters:
            sizes = sorted(clusters)
            counts = [clusters[size] for size in sizes]
            axes.bar([str(size) for size in sizes], counts, color=_PALETTE[1], width=0.65)
            for index, value in enumerate(counts):
                axes.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8, color="#444")
            axes.set_xlabel("images per cluster")
            axes.set_ylabel("clusters")
        else:
            axes.text(0.5, 0.5, "No duplicate clusters detected", ha="center", va="center",
                      transform=axes.transAxes, fontsize=11, color="#666")
            axes.set_axis_off()
        total = sum(size * count for size, count in clusters.items()) if clusters else 0
        axes.set_title(self._title("Duplicate clusters", profile,
                                   f"{len(clusters) and sum(clusters.values()):,} clusters covering {total:,} images"))
        axes.grid(axis="y", **_GRID)
        axes.set_axisbelow(True)
        self._finish(figure, path)

    def _dashboard(self, profile: DatasetProfile, analysis: Any | None, path: Path) -> None:
        figure = plt.figure(figsize=(13, 8.5))
        grid = figure.add_gridspec(3, 3, hspace=0.55, wspace=0.28)

        self._histogram(figure.add_subplot(grid[0, 0]), profile.histograms.get("quality_score"),
                        "quality score", _PALETTE[0], title="Quality score")
        self._bar(figure.add_subplot(grid[0, 1]), profile.categories.get("quality_grade"), "Quality grades",
                  sort_keys=True)
        self._bar(figure.add_subplot(grid[0, 2]), profile.categories.get("source"), "Source contribution")
        self._histogram(figure.add_subplot(grid[1, 0]), profile.histograms.get("sharpness"),
                        "sharpness", _PALETTE[2], title="Sharpness", log=True)
        self._histogram(figure.add_subplot(grid[1, 1]), profile.histograms.get("entropy"),
                        "entropy (bits)", _PALETTE[3], title="Entropy")
        self._bar(figure.add_subplot(grid[1, 2]), profile.categories.get("format"), "Image formats")
        self._class_curve(figure.add_subplot(grid[2, 0:2]), profile)
        self._summary_panel(figure.add_subplot(grid[2, 2]), profile, analysis)

        figure.suptitle(self._title("Dataset overview", profile), fontsize=13, y=0.975)
        self._finish(figure, path, tight=False)

    # --- panel helpers ------------------------------------------------------------ #

    def _histogram(self, axes, histogram, xlabel: str, colour: str, title: str = "", log: bool = False) -> None:
        if histogram is None or not histogram.counts:
            axes.text(0.5, 0.5, "no data", ha="center", va="center", transform=axes.transAxes, color="#888")
            axes.set_axis_off()
            return
        centers, counts = histogram.centers, histogram.counts
        width = (histogram.edges[-1] - histogram.edges[0]) / max(1, len(counts))
        axes.bar(centers, counts, width=width * 0.95, color=colour, align="center")
        axes.set_xlabel(xlabel, fontsize=9)
        axes.set_ylabel("images", fontsize=9)
        axes.tick_params(labelsize=8)
        if log:
            axes.set_xscale("symlog")
        if title:
            axes.set_title(title, fontsize=10)
        axes.grid(axis="y", **_GRID)
        axes.set_axisbelow(True)

    def _bar(self, axes, distribution, title: str, sort_keys: bool = False, limit: int = 8) -> None:
        counts = dict(distribution.counts) if distribution else {}
        if not counts:
            axes.text(0.5, 0.5, "no data", ha="center", va="center", transform=axes.transAxes, color="#888")
            axes.set_axis_off()
            return
        items = sorted(counts.items()) if sort_keys else list(counts.items())[:limit]
        labels = [str(key) for key, _ in items]
        values = [value for _, value in items]
        axes.bar(labels, values, color=[_PALETTE[index % len(_PALETTE)] for index in range(len(labels))], width=0.65)
        axes.set_title(title, fontsize=10)
        axes.tick_params(axis="x", labelrotation=30, labelsize=8)
        axes.tick_params(axis="y", labelsize=8)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
        axes.grid(axis="y", **_GRID)
        axes.set_axisbelow(True)

    def _class_curve(self, axes, profile: DatasetProfile) -> None:
        classes = profile.classes
        if classes.empty:
            axes.set_axis_off()
            return
        counts = classes["images"].to_numpy()
        cumulative = np.cumsum(counts) / counts.sum()
        axes.plot(np.arange(1, len(counts) + 1), cumulative, color=_PALETTE[0], linewidth=1.8)
        axes.axhline(0.8, color="#999", linestyle="--", linewidth=1)
        axes.set_xlabel("classes, largest first", fontsize=9)
        axes.set_ylabel("cumulative share", fontsize=9)
        axes.set_ylim(0, 1.02)
        axes.set_title("Long-tail curve", fontsize=10)
        axes.tick_params(labelsize=8)
        axes.grid(**_GRID)
        axes.set_axisbelow(True)

    def _summary_panel(self, axes, profile: DatasetProfile, analysis: Any | None) -> None:
        axes.set_axis_off()
        lines = [
            f"images        {profile.image_count:,}",
            f"classes       {profile.class_count:,}",
            f"sources       {profile.totals.get('sources', 0):,}",
            f"version       {profile.totals.get('dataset_version')}",
        ]
        quality = profile.numeric.get("quality_score")
        if quality and quality.mean is not None:
            lines.append(f"mean quality  {quality.mean:.3f}")
        if analysis is not None and getattr(analysis, "score", None) is not None:
            lines.append(f"dataset score {analysis.score.value:.3f} ({analysis.score.grade})")
        lines.append(f"fingerprint   {(profile.dataset_fingerprint or '')[:12]}")
        axes.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9,
                  transform=axes.transAxes, color="#222")

    # --- shared ------------------------------------------------------------------- #

    def _title(self, heading: str, profile: DatasetProfile, detail: str = "") -> str:
        suffix = f" - {detail}" if detail else ""
        return f"{heading}{suffix}"

    def _finish(self, figure, path: Path, tight: bool = True) -> None:
        if tight:
            figure.tight_layout()
        figure.savefig(path, dpi=self._config.dpi, format=self._config.figure_format, facecolor="white")
        plt.close(figure)


__all__ = ["VisualizationRenderer", "VisualizationSet"]
