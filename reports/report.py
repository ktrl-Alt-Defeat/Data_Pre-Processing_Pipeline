"""HTML report construction.

A minimal document builder shared by the profiling and analysis reports: one
stylesheet, one page skeleton, and section helpers that escape everything they
render. Reports are static, self-contained files so they can be attached to a
paper or a review without a server.
"""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from preprocessing.core.io import atomic_write_text
from preprocessing.core.records import Metric, MetricStatus, RunManifest

_STYLE = """
 :root { color-scheme: light; }
 body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem auto; max-width: 62rem;
        color: #1a1a1a; line-height: 1.45; padding: 0 1rem; }
 h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
 h2 { font-size: 1.1rem; margin-top: 2.2rem; border-bottom: 1px solid #e5e5e5; padding-bottom: 0.3rem; }
 h3 { font-size: 0.95rem; margin-top: 1.4rem; color: #333; }
 p.subtitle { color: #666; margin-top: 0; font-size: 0.9rem; }
 table { border-collapse: collapse; width: 100%; font-size: 0.86rem; margin: 0.6rem 0 1rem; }
 th, td { text-align: left; padding: 0.32rem 0.55rem; border-bottom: 1px solid #ececec; vertical-align: top; }
 thead th { background: #fafafa; font-weight: 600; }
 table.kv th { width: 18rem; font-weight: 600; color: #444; }
 td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
 .status { font-weight: 600; padding: 0.05rem 0.4rem; border-radius: 0.25rem; font-size: 0.78rem; }
 .ok { background: #e7f6ec; color: #16643a; }
 .warning { background: #fdf3e2; color: #8a5a08; }
 .critical { background: #fdeaea; color: #8f1d1d; }
 .informational { background: #eef2f8; color: #33507a; }
 figure { margin: 1rem 0; } img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
 footer { margin-top: 3rem; font-size: 0.8rem; color: #777; border-top: 1px solid #eee; padding-top: 0.8rem; }
 code { background: #f5f5f5; padding: 0.05rem 0.25rem; border-radius: 3px; font-size: 0.85em; }
"""


@dataclass(slots=True)
class HtmlDocument:
    """Accumulates sections and renders a self-contained HTML page."""

    title: str
    subtitle: str = ""
    sections: list[str] = field(default_factory=list)

    def add(self, markup: str) -> "HtmlDocument":
        self.sections.append(markup)
        return self

    def key_values(self, title: str, rows: Mapping[str, Any]) -> "HtmlDocument":
        body = "".join(
            f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>" for key, value in rows.items()
        )
        return self.add(f"<section><h2>{_escape(title)}</h2><table class='kv'>{body}</table></section>")

    def table(self, title: str, frame: pd.DataFrame, limit: int | None = 50, note: str = "") -> "HtmlDocument":
        if frame is None or frame.empty:
            return self.add(f"<section><h2>{_escape(title)}</h2><p>No data.</p></section>")
        shown = frame.head(limit) if limit else frame
        suffix = (
            f"<p>Showing {len(shown):,} of {len(frame):,} rows.</p>" if limit and len(frame) > limit else ""
        )
        extra = f"<p>{_escape(note)}</p>" if note else ""
        return self.add(f"<section><h2>{_escape(title)}</h2>{extra}{_frame_table(shown)}{suffix}</section>")

    def metrics(self, title: str, metrics: Sequence[Metric], note: str = "") -> "HtmlDocument":
        if not metrics:
            return self.add(f"<section><h2>{_escape(title)}</h2><p>No metrics produced.</p></section>")
        header = (
            "<thead><tr><th>Metric</th><th class='num'>Value</th><th class='num'>Threshold</th>"
            "<th>Status</th><th>Interpretation &amp; recommendation</th></tr></thead>"
        )
        rows = "".join(_metric_row(metric) for metric in metrics)
        extra = f"<p>{_escape(note)}</p>" if note else ""
        return self.add(f"<section><h2>{_escape(title)}</h2>{extra}<table>{header}<tbody>{rows}</tbody></table></section>")

    def figures(self, title: str, images: Mapping[str, Path], relative_to: Path | None = None) -> "HtmlDocument":
        available = {name: path for name, path in images.items() if path and path.exists()}
        if not available:
            return self
        blocks = []
        for name, path in available.items():
            source = _relative(path, relative_to)
            blocks.append(
                f"<figure><figcaption><strong>{_escape(name)}</strong></figcaption>"
                f"<img src='{_escape(source)}' alt='{_escape(name)}'></figure>"
            )
        return self.add(f"<section><h2>{_escape(title)}</h2>{''.join(blocks)}</section>")

    def render(self, manifest: RunManifest | None = None) -> str:
        generated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        footprint = (
            f"Run <code>{_escape(manifest.run_id)}</code> &middot; pipeline {_escape(manifest.pipeline_version)} "
            f"&middot; dataset {_escape(manifest.dataset_version)} &middot; "
            f"config <code>{_escape(manifest.config_hash[:16])}</code>"
            if manifest
            else ""
        )
        subtitle = f"<p class='subtitle'>{_escape(self.subtitle)}</p>" if self.subtitle else ""
        return (
            "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            f"<title>{_escape(self.title)}</title><style>{_STYLE}</style></head><body>"
            f"<h1>{_escape(self.title)}</h1>{subtitle}"
            f"{''.join(self.sections)}"
            f"<footer>{footprint}<br>Generated {generated} by the crop-disease dataset preprocessing framework."
            "</footer></body></html>\n"
        )

    def write(self, path: Path, manifest: RunManifest | None = None) -> Path:
        return atomic_write_text(path, self.render(manifest))


def manifest_rows(manifest: RunManifest, dataset_fingerprint: str | None, config_fingerprint: str) -> dict[str, Any]:
    """The provenance block every report opens with."""
    return {
        "Run id": manifest.run_id,
        "Pipeline version": manifest.pipeline_version,
        "Dataset version": manifest.dataset_version,
        "Dataset fingerprint": dataset_fingerprint,
        "Configuration fingerprint": config_fingerprint,
        "Git commit": manifest.git_commit or "not a git checkout",
        "Python": manifest.python_version,
        "Platform": manifest.platform,
        "Started at": manifest.started_at,
    }


def _metric_row(metric: Metric) -> str:
    guidance = _escape(metric.interpretation)
    if metric.recommendation:
        guidance += f"<br><em>{_escape(metric.recommendation)}</em>"
    return (
        f"<tr><td><strong>{_escape(metric.name)}</strong><br>"
        f"<span style='color:#666'>{_escape(metric.definition)}</span><br>"
        f"<span style='color:#888;font-size:0.9em'>Method: {_escape(metric.method)}</span></td>"
        f"<td class='num'>{_escape(_number(metric.value))}</td>"
        f"<td class='num'>{_escape(_number(metric.threshold))}</td>"
        f"<td><span class='status {metric.status.value}'>{metric.status.value}</span></td>"
        f"<td>{guidance}</td></tr>"
    )


def _frame_table(frame: pd.DataFrame) -> str:
    numeric = {column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])}
    header = "".join(
        f"<th class='num'>{_escape(column)}</th>" if column in numeric else f"<th>{_escape(column)}</th>"
        for column in frame.columns
    )
    rows = []
    for row in frame.itertuples(index=False):
        cells = "".join(
            f"<td class='num'>{_escape(_number(value))}</td>" if column in numeric else f"<td>{_escape(value)}</td>"
            for column, value in zip(frame.columns, row, strict=True)
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _number(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _relative(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def sections_from_metrics(metrics: Iterable[Metric]) -> dict[str, list[Metric]]:
    """Group metrics by the prefix of their key, e.g. ``imbalance.gini``."""
    grouped: dict[str, list[Metric]] = {}
    for metric in metrics:
        grouped.setdefault(metric.key.split(".", 1)[0], []).append(metric)
    return grouped


def worst_status(metrics: Sequence[Metric]) -> MetricStatus:
    """The most severe status across a set of metrics."""
    order = [MetricStatus.CRITICAL, MetricStatus.WARNING, MetricStatus.OK, MetricStatus.INFORMATIONAL]
    for status in order:
        if any(metric.status is status for metric in metrics):
            return status
    return MetricStatus.INFORMATIONAL


__all__ = ["HtmlDocument", "manifest_rows", "sections_from_metrics", "worst_status"]
