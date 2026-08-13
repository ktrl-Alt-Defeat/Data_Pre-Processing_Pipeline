"""Structured logging and per-stage instrumentation.

Two sinks with different audiences: a JSON-lines file that is machine readable
and reproducible (one object per event, with the run id on every line), and a
compact console stream for humans. Stages never print; they emit events through
:func:`stage_scope`, which guarantees the mandated started/completed record with
counters, duration, warnings and errors.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import logging.handlers
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import LoggingConfig
from .io import ensure_dir, json_default
from .records import PipelineStage, StageReport

_NOISY_LOGGERS = ("matplotlib", "PIL", "fontTools", "asyncio")
_FIELDS_KEY = "fields"
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime", "taskName"}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per log record, suitable for ingestion and diffing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _utcnow(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, _FIELDS_KEY, {}) or {})
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and k != _FIELDS_KEY}
        payload.update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=json_default)


class ConsoleFormatter(logging.Formatter):
    """``12:03:41 INFO  corpus  stage.completed  processed=1200 rejected=14``."""

    def format(self, record: logging.LogRecord) -> str:
        fields: Mapping[str, Any] = getattr(record, _FIELDS_KEY, {}) or {}
        rendered = " ".join(f"{key}={_render(value)}" for key, value in fields.items())
        stamp = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        name = record.name.replace("preprocessing.", "")
        line = f"{stamp} {record.levelname:<7} {name:<22} {record.getMessage()}"
        if rendered:
            line = f"{line}  {rendered}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=json_default)
    return str(value)


class StructuredLogger:
    """Thin wrapper that turns keyword arguments into structured log fields."""

    __slots__ = ("_logger", "_context")

    def __init__(self, logger: logging.Logger, context: Mapping[str, Any] | None = None) -> None:
        self._logger = logger
        self._context = dict(context or {})

    def bind(self, **fields: Any) -> "StructuredLogger":
        """Return a logger that adds ``fields`` to every subsequent event."""
        return StructuredLogger(self._logger, {**self._context, **fields})

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, fields, exc_info=True)

    def _emit(self, level: int, event: str, fields: Mapping[str, Any], exc_info: bool = False) -> None:
        if not self._logger.isEnabledFor(level):
            return
        self._logger.log(level, event, extra={_FIELDS_KEY: {**self._context, **fields}}, exc_info=exc_info)


def get_logger(name: str) -> StructuredLogger:
    """Structured logger for a module, e.g. ``get_logger(__name__)``."""
    return StructuredLogger(logging.getLogger(name))


def configure_logging(config: LoggingConfig, run_id: str, project_root: Path | None = None) -> Path | None:
    """Install console and JSON-lines handlers; returns the log file path if enabled."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(config.level)

    if config.console:
        console = logging.StreamHandler()
        console.setLevel(config.console_level)
        console.setFormatter(ConsoleFormatter())
        root.addHandler(console)

    log_path: Path | None = None
    if config.json_file:
        directory = config.directory
        if not directory.is_absolute() and project_root is not None:
            directory = project_root / directory
        ensure_dir(directory)
        log_path = directory / config.filename_template.format(run_id=run_id)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(config.level)
        file_handler.setFormatter(JsonLinesFormatter())
        root.addHandler(file_handler)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(config.capture_warnings)

    return log_path


@dataclass(slots=True)
class StageTracker:
    """Counters and events for one stage; materialises into a :class:`StageReport`."""

    stage: PipelineStage
    logger: StructuredLogger
    report: StageReport
    _metrics: dict[str, Any] = field(default_factory=dict)

    def processed(self, count: int = 1) -> None:
        self.report.processed += count

    def rejected(self, count: int = 1) -> None:
        self.report.rejected += count

    def skipped(self, count: int = 1) -> None:
        self.report.skipped += count

    def warn(self, event: str, **fields: Any) -> None:
        self.report.warnings += 1
        self.report.messages.append(f"WARNING {event}")
        self.logger.warning(event, **fields)

    def error(self, event: str, exc: BaseException | None = None, **fields: Any) -> None:
        self.report.errors += 1
        self.report.messages.append(f"ERROR {event}")
        if exc is not None:
            fields = {**fields, "error": f"{type(exc).__name__}: {exc}"}
        self.logger.error(event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self.logger.info(event, **fields)

    def metric(self, key: str, value: Any) -> None:
        self._metrics[key] = value

    def metrics(self, **values: Any) -> None:
        self._metrics.update(values)

    def flush_metrics(self) -> None:
        self.report.metrics.update(self._metrics)


@contextmanager
def stage_scope(
    stage: PipelineStage,
    logger: StructuredLogger | None = None,
    **fields: Any,
) -> Iterator[StageTracker]:
    """Bracket a pipeline stage with started/completed events and timing.

    The tracker's :class:`StageReport` is populated on exit, including on
    failure, so the run summary always contains an entry for every stage that
    was attempted.
    """
    log = (logger or get_logger(stage.logger_name)).bind(stage=stage.value, **fields)
    report = StageReport(stage=stage, started_at=_utcnow())
    tracker = StageTracker(stage=stage, logger=log, report=report)
    started = time.perf_counter()
    log.info("stage.started", **fields)
    try:
        yield tracker
    except BaseException as exc:
        report.status = "failed"
        report.errors += 1
        report.messages.append(f"ERROR {type(exc).__name__}: {exc}")
        _finalize(report, tracker, started)
        log.exception(
            "stage.failed",
            duration_seconds=report.duration_seconds,
            processed=report.processed,
            rejected=report.rejected,
        )
        raise
    else:
        _finalize(report, tracker, started)
        log.info(
            "stage.completed",
            duration_seconds=report.duration_seconds,
            processed=report.processed,
            rejected=report.rejected,
            skipped=report.skipped,
            warnings=report.warnings,
            errors=report.errors,
        )


def _finalize(report: StageReport, tracker: StageTracker, started: float) -> None:
    report.duration_seconds = time.perf_counter() - started
    report.completed_at = _utcnow()
    tracker.flush_metrics()


__all__ = [
    "ConsoleFormatter",
    "JsonLinesFormatter",
    "StageTracker",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
    "stage_scope",
]
