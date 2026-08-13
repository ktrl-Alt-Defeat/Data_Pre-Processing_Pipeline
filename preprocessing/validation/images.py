"""Structural validation of image files.

Answers one question per image: can this file be decoded into a usable RGB
image? Nothing is resized, normalised, scored or repaired here — the validator
is pure and returns findings, and the stage decides what they mean.

Checks run cheapest-first so a corpus of hundreds of thousands of files spends
decode time only on candidates that passed every free check:

    extension -> stat -> header probe -> full decode

Decoding goes exclusively through :func:`preprocessing.core.io.load_rgb`; this
module never opens a second decoder.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from ..core.config import ValidationConfig
from ..core.errors import ImageDecodeError, ImageError
from ..core.io import ImageProbe, load_rgb, probe_image, to_array
from ..core.records import ImageRecord, RejectionCode, Severity, ValidationIssue

VALIDATOR = "images"

_VALID_EXIF_ORIENTATIONS = frozenset(range(1, 9))
_TRANSPOSING_ORIENTATIONS = frozenset({5, 6, 7, 8})
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ImageObservation:
    """Technical facts observed while validating; never canonical metadata."""

    width: int
    height: int
    channels: int
    color_mode: str
    image_format: str
    file_size_bytes: int
    exif_orientation: int | None = None
    exif_transposed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "color_mode": self.color_mode,
            "image_format": self.image_format,
            "file_size_bytes": self.file_size_bytes,
            "exif_orientation": self.exif_orientation,
            "exif_transposed": self.exif_transposed,
        }


@dataclass(frozen=True, slots=True)
class ImageValidationResult:
    """Outcome for one image: issues plus whatever was observed on the way."""

    image_id: str
    issues: tuple[ValidationIssue, ...] = ()
    observation: ImageObservation | None = None

    @property
    def rejected(self) -> bool:
        return any(issue.is_error for issue in self.issues)


class ImageValidator:
    """Validates that a file decodes to a usable RGB image."""

    def __init__(self, config: ValidationConfig) -> None:
        self._config = config
        self._extensions = {ext.lower() for ext in config.allowed_extensions}
        self._formats = {fmt.lower() for fmt in config.allowed_formats}
        self._color_modes = {mode.upper() for mode in config.allowed_color_modes}
        self._counter: Counter[str] = Counter()

    @property
    def metrics(self) -> dict[str, int]:
        return dict(sorted(self._counter.items()))

    def validate(self, record: ImageRecord) -> ImageValidationResult:
        """Validate one image. Never raises; every failure becomes an issue."""
        issues: list[ValidationIssue] = []
        path = record.source_path

        extension_issue = self._check_extension(record, path)
        if extension_issue:
            return self._result(record, [extension_issue])

        size_issue, size_bytes = self._check_file(record, path)
        if size_issue:
            return self._result(record, [size_issue])

        try:
            probe = probe_image(path)
        except ImageError as exc:
            return self._result(record, [self._issue(record, _decode_code(exc), str(exc.message))])

        geometry_issue = self._check_geometry(record, probe)
        if geometry_issue:
            return self._result(record, [geometry_issue])

        format_issue = self._check_format(record, probe)
        if format_issue:
            return self._result(record, [format_issue])

        mode_issue = self._check_color_mode(record, probe)
        if mode_issue:
            return self._result(record, [mode_issue])

        issues.extend(self._inspect_metadata(record, probe))

        if not self._config.full_decode:
            self._counter["header_only"] += 1
            return self._result(record, issues, self._observe(probe, probe.width, probe.height, False))

        decoded, decode_issue = self._decode(record, path)
        if decode_issue is not None or decoded is None:
            return self._result(record, [*issues, decode_issue] if decode_issue else list(issues))

        pixel_issue = self._check_pixels(record, decoded)
        if pixel_issue:
            return self._result(record, [*issues, pixel_issue])

        transposed = self._config.apply_exif_orientation and (probe.exif_orientation in _TRANSPOSING_ORIENTATIONS)
        observation = self._observe(probe, decoded.width, decoded.height, transposed)
        self._counter["decoded"] += 1
        return self._result(record, issues, observation)

    # --- individual checks --------------------------------------------------- #

    def _check_extension(self, record: ImageRecord, path: Path) -> ValidationIssue | None:
        suffix = path.suffix.lower()
        if suffix in self._extensions:
            return None
        return self._issue(
            record,
            RejectionCode.UNSUPPORTED_FORMAT,
            f"extension '{suffix or '<none>'}' is not in the configured allow-list",
            extension=suffix,
            allowed=sorted(self._extensions),
        )

    def _check_file(self, record: ImageRecord, path: Path) -> tuple[ValidationIssue | None, int]:
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            return (
                self._issue(record, RejectionCode.UNREADABLE_FILE, f"file is not accessible: {exc}", path=str(path)),
                0,
            )

        if size_bytes == 0:
            return self._issue(record, RejectionCode.EMPTY_FILE, "file is zero bytes", path=str(path)), 0

        limit = self._config.max_file_size_mb
        if limit is not None and size_bytes > limit * _BYTES_PER_MB:
            message = f"file size {size_bytes / _BYTES_PER_MB:.1f} MB exceeds the configured limit of {limit} MB"
            return self._issue(record, RejectionCode.OVERSIZED_IMAGE, message, size_bytes=size_bytes, limit_mb=limit), size_bytes

        return None, size_bytes

    def _check_geometry(self, record: ImageRecord, probe: ImageProbe) -> ValidationIssue | None:
        if probe.width <= 0 or probe.height <= 0:
            return self._issue(
                record,
                RejectionCode.INVALID_DIMENSIONS,
                f"image reports empty geometry ({probe.width}x{probe.height})",
                width=probe.width,
                height=probe.height,
            )

        minimum = self._config.min_dimension
        if probe.width < minimum or probe.height < minimum:
            return self._issue(
                record,
                RejectionCode.INVALID_DIMENSIONS,
                f"image is {probe.width}x{probe.height}, below the minimum dimension of {minimum}px",
                width=probe.width,
                height=probe.height,
                min_dimension=minimum,
            )

        max_pixels = self._config.max_pixels
        if max_pixels is not None and probe.width * probe.height > max_pixels:
            return self._issue(
                record,
                RejectionCode.OVERSIZED_IMAGE,
                f"image has {probe.width * probe.height} pixels, above the configured limit of {max_pixels}",
                pixels=probe.width * probe.height,
                max_pixels=max_pixels,
            )
        return None

    def _check_format(self, record: ImageRecord, probe: ImageProbe) -> ValidationIssue | None:
        if probe.image_format in self._formats:
            return None
        return self._issue(
            record,
            RejectionCode.UNSUPPORTED_FORMAT,
            f"decoded format '{probe.image_format}' is not in the configured allow-list",
            image_format=probe.image_format,
            allowed=sorted(self._formats),
        )

    def _check_color_mode(self, record: ImageRecord, probe: ImageProbe) -> ValidationIssue | None:
        if not self._color_modes or probe.color_mode.upper() in self._color_modes:
            return None
        return self._issue(
            record,
            RejectionCode.INVALID_COLOR_MODE,
            f"colour mode '{probe.color_mode}' is not in the configured allow-list",
            color_mode=probe.color_mode,
            allowed=sorted(self._color_modes),
        )

    def _inspect_metadata(self, record: ImageRecord, probe: ImageProbe) -> list[ValidationIssue]:
        """Non-fatal findings: the image is usable but worth flagging."""
        issues: list[ValidationIssue] = []
        orientation = probe.exif_orientation

        if orientation is not None and orientation not in _VALID_EXIF_ORIENTATIONS:
            self._counter["invalid_exif_orientation"] += 1
            issues.append(
                self._issue(
                    record,
                    RejectionCode.METADATA_INVALID,
                    f"EXIF orientation {orientation} is outside the valid range 1-8; orientation ignored",
                    severity=Severity.WARNING,
                    exif_orientation=orientation,
                )
            )
        elif orientation in _TRANSPOSING_ORIENTATIONS:
            self._counter["exif_transposed"] += 1

        declared = (record.image_format or "").lower()
        declared = "jpeg" if declared == "jpg" else declared
        if declared and declared != probe.image_format:
            self._counter["format_mismatch"] += 1
            issues.append(
                self._issue(
                    record,
                    RejectionCode.UNSUPPORTED_FORMAT,
                    f"file extension declares '{declared}' but the codec is '{probe.image_format}'",
                    severity=Severity.WARNING,
                    declared=declared,
                    actual=probe.image_format,
                )
            )

        if probe.color_mode != "RGB":
            self._counter["non_rgb_source"] += 1
        return issues

    def _decode(self, record: ImageRecord, path: Path) -> tuple[Image.Image | None, ValidationIssue | None]:
        try:
            return load_rgb(path, apply_exif=self._config.apply_exif_orientation), None
        except ImageError as exc:
            return None, self._issue(record, _decode_code(exc), str(exc.message))
        except (OSError, ValueError) as exc:
            return None, self._issue(record, RejectionCode.DECODE_FAILED, f"decode failed: {exc}")

    def _check_pixels(self, record: ImageRecord, image: Image.Image) -> ValidationIssue | None:
        if not self._config.require_rgb_convertible:
            return None
        try:
            array = to_array(image)
        except (OSError, ValueError) as exc:
            return self._issue(record, RejectionCode.DECODE_FAILED, f"pixel data is unreadable: {exc}")

        if array.ndim != 3 or array.shape[2] != 3:
            return self._issue(
                record,
                RejectionCode.INVALID_COLOR_MODE,
                f"decoded pixel array has shape {array.shape}, expected (H, W, 3)",
                shape=list(array.shape),
            )
        if array.size == 0:
            return self._issue(record, RejectionCode.INVALID_DIMENSIONS, "decoded image contains no pixels")
        return None

    # --- helpers -------------------------------------------------------------- #

    def _observe(self, probe: ImageProbe, width: int, height: int, transposed: bool) -> ImageObservation:
        return ImageObservation(
            width=width,
            height=height,
            channels=_mode_channels(probe.color_mode),
            color_mode=probe.color_mode,
            image_format=probe.image_format,
            file_size_bytes=probe.file_size_bytes,
            exif_orientation=probe.exif_orientation,
            exif_transposed=transposed,
        )

    def _issue(
        self,
        record: ImageRecord,
        code: RejectionCode,
        message: str,
        severity: Severity = Severity.ERROR,
        **detail: Any,
    ) -> ValidationIssue:
        self._counter[f"{severity.value}:{code.value}"] += 1
        return ValidationIssue(
            image_id=record.image_id,
            validator=VALIDATOR,
            code=code,
            message=message,
            severity=severity,
            detail=detail,
        )

    def _result(
        self,
        record: ImageRecord,
        issues: list[ValidationIssue],
        observation: ImageObservation | None = None,
    ) -> ImageValidationResult:
        self._counter["validated"] += 1
        return ImageValidationResult(record.image_id, tuple(issues), observation)


def _mode_channels(mode: str) -> int:
    """Band count of the *source* mode; unknown modes fall back to RGB's three."""
    try:
        return int(Image.getmodebands(mode))
    except (KeyError, ValueError):
        return 3


def _decode_code(exc: ImageError) -> RejectionCode:
    """Map a decode failure onto the most specific structured reason available."""
    if not isinstance(exc, ImageDecodeError):
        return RejectionCode.UNREADABLE_FILE
    message = exc.message.lower()
    if "truncat" in message or "ends prematurely" in message or "not enough image data" in message:
        return RejectionCode.TRUNCATED_FILE
    if "not a recognised image" in message:
        return RejectionCode.UNSUPPORTED_FORMAT
    return RejectionCode.DECODE_FAILED


__all__ = ["VALIDATOR", "ImageObservation", "ImageValidationResult", "ImageValidator"]
