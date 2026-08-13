"""Exception hierarchy for the preprocessing framework.

Failures are split into two categories, and the distinction drives the whole
error-handling policy of the pipeline:

* :class:`PipelineError` and subclasses are *fatal*. They mean the run cannot
  produce a trustworthy artefact (bad configuration, missing source root,
  unwritable output directory). The pipeline aborts and reports.
* :class:`ImageError` and subclasses are *scoped to a single image*. The stage
  converts them into a rejection on the image's record and keeps going, so one
  corrupt file can never take down a corpus-wide run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PreprocessingError(Exception):
    """Base class for every error raised by this framework."""


# --------------------------------------------------------------------------- #
# Fatal errors
# --------------------------------------------------------------------------- #


class PipelineError(PreprocessingError):
    """A failure that makes the current run unrecoverable."""


class ConfigurationError(PipelineError):
    """Configuration is missing, malformed, or internally inconsistent."""


class StageError(PipelineError):
    """A pipeline stage failed as a whole rather than for a single image."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class SourceError(PipelineError):
    """A configured dataset source is unusable (missing, empty, unreadable)."""


class PackagingError(PipelineError):
    """The output package could not be written."""


# --------------------------------------------------------------------------- #
# Per-image, recoverable errors
# --------------------------------------------------------------------------- #


class ImageError(PreprocessingError):
    """Base class for recoverable, single-image failures."""

    def __init__(self, path: Path | str, message: str, **detail: Any) -> None:
        super().__init__(f"{path}: {message}")
        self.path = Path(path)
        self.message = message
        self.detail = detail


class ImageReadError(ImageError):
    """The file could not be read from disk."""


class ImageDecodeError(ImageError):
    """The file could not be decoded as an image (corrupt or truncated)."""


class UnsupportedFormatError(ImageError):
    """The file format is not in the configured allow-list."""


__all__ = [
    "PreprocessingError",
    "PipelineError",
    "ConfigurationError",
    "StageError",
    "SourceError",
    "PackagingError",
    "ImageError",
    "ImageReadError",
    "ImageDecodeError",
    "UnsupportedFormatError",
]
