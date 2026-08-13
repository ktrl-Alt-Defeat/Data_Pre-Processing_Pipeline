"""Report and visualisation rendering.

Presentation only: this package consumes the objects produced by the profiling
and analysis stages and never computes statistics of its own.
"""

from .report import HtmlDocument, manifest_rows, sections_from_metrics, worst_status
from .visualizations import VisualizationRenderer, VisualizationSet

__all__ = [
    "HtmlDocument",
    "VisualizationRenderer",
    "VisualizationSet",
    "manifest_rows",
    "sections_from_metrics",
    "worst_status",
]
