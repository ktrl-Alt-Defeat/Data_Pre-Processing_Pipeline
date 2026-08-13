"""Filesystem, image-decoding and serialisation primitives.

Every stage funnels its disk access through this module so that decoding
policy (EXIF handling, RGB conversion, truncation detection, decompression-bomb
limits) is defined exactly once and behaves identically everywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import yaml
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from .errors import ImageDecodeError, ImageReadError, UnsupportedFormatError

DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
_EXIF_ORIENTATION_TAG = 0x0112
_HASH_CHUNK_BYTES = 1 << 20

# Truncated files must raise rather than silently decode to grey padding: the
# validation stage's job is to detect them, not to paper over them.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def configure_pillow_limits(max_pixels: int | None) -> None:
    """Set Pillow's decompression-bomb ceiling (``None`` disables the check)."""
    Image.MAX_IMAGE_PIXELS = max_pixels


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def iter_files(
    root: Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """Yield files under ``root`` with a matching extension, in sorted order.

    Sorted traversal keeps corpus construction deterministic across machines and
    filesystems, which the fingerprinting and splitting stages rely on.
    """
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames.sort()
        directory = Path(dirpath)
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in allowed:
                yield directory / filename


@dataclass(frozen=True, slots=True)
class FileEntry:
    """A discovered file plus the stat data the OS already handed us."""

    path: Path
    relpath: str
    size_bytes: int
    modified_at: float

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.relpath.split("/"))

    @property
    def filename(self) -> str:
        return self.path.name


def walk_image_entries(
    root: Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    follow_symlinks: bool = False,
) -> Iterator[FileEntry]:
    """Traverse ``root`` yielding image files with size and mtime, sorted.

    Uses ``os.scandir`` and the stat data it caches, so ingesting hundreds of
    thousands of files costs one directory scan and no extra ``stat`` syscalls.
    Traversal is depth-first with names sorted at every level, which is what
    makes corpus construction and fingerprinting reproducible.
    """
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    root = root.resolve()
    stack: list[tuple[Path, str]] = [(root, "")]

    while stack:
        directory, prefix = stack.pop()
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError:
            continue

        subdirectories: list[tuple[Path, str]] = []
        for entry in entries:
            relpath = f"{prefix}{entry.name}"
            try:
                if entry.is_dir(follow_symlinks=follow_symlinks):
                    subdirectories.append((Path(entry.path), f"{relpath}/"))
                    continue
                if not entry.is_file(follow_symlinks=follow_symlinks):
                    continue
                if Path(entry.name).suffix.lower() not in allowed:
                    continue
                stat = entry.stat(follow_symlinks=follow_symlinks)
            except OSError:
                continue
            yield FileEntry(Path(entry.path), relpath, stat.st_size, stat.st_mtime)

        stack.extend(reversed(subdirectories))


def iter_subdirectories(root: Path) -> list[Path]:
    """Immediate sub-directories of ``root``, sorted by name."""
    try:
        with os.scandir(root) as scanner:
            return sorted((Path(e.path) for e in scanner if e.is_dir()), key=lambda p: p.name)
    except OSError:
        return []


def relative_path(path: Path, root: Path) -> str:
    """POSIX-style path of ``path`` relative to ``root``, machine independent."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- #
# Hashing and identity
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    """Streaming SHA-256 of the file bytes (exact-duplicate identity)."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ImageReadError(path, f"could not read file: {exc}") from exc
    return digest.hexdigest()


def pixel_digest(image: Image.Image) -> str:
    """SHA-1 of decoded RGB pixels: identical content re-encoded still matches."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha1(array.tobytes()).hexdigest()


def stable_id(*parts: str, length: int = 16) -> str:
    """Deterministic short identifier derived from stable string components."""
    joined = "\x1f".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def hash_mapping(payload: Any) -> str:
    """SHA-256 over a canonical JSON encoding; used for config/dataset hashes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Image access
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ImageProbe:
    """Header-only image facts; cheap enough to run over an entire corpus."""

    width: int
    height: int
    image_format: str
    color_mode: str
    file_size_bytes: int
    exif_orientation: int | None

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


def probe_image(path: Path) -> ImageProbe:
    """Read image metadata without decoding pixel data.

    Raises :class:`ImageDecodeError` for unreadable headers so callers can turn
    the failure into a rejection instead of crashing.
    """
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise ImageReadError(path, f"could not stat file: {exc}") from exc

    try:
        with Image.open(path) as image:
            orientation = _exif_orientation(image)
            return ImageProbe(
                width=image.width,
                height=image.height,
                image_format=(image.format or "unknown").lower(),
                color_mode=image.mode,
                file_size_bytes=size_bytes,
                exif_orientation=orientation,
            )
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(path, "file is not a recognised image") from exc
    except (OSError, ValueError) as exc:
        raise ImageDecodeError(path, f"header could not be parsed: {exc}") from exc


def load_rgb(path: Path, apply_exif: bool = True) -> Image.Image:
    """Fully decode an image to RGB, honouring EXIF orientation.

    This is the single decode path for the framework: profiling, quality metrics
    and packaging all see pixels in the same orientation and colour space.
    """
    try:
        with Image.open(path) as image:
            image.load()  # forces a full decode so truncation surfaces here
            if apply_exif:
                image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(path, "file is not a recognised image") from exc
    except OSError as exc:
        raise ImageDecodeError(path, f"decode failed: {exc}") from exc
    except ValueError as exc:  # Pillow raises this for bomb-limit violations
        raise ImageDecodeError(path, f"decode rejected: {exc}") from exc


def to_array(image: Image.Image) -> np.ndarray:
    """RGB uint8 array of shape (H, W, 3)."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def check_extension(path: Path, extensions: Sequence[str]) -> None:
    """Raise :class:`UnsupportedFormatError` if the suffix is not allowed."""
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    suffix = path.suffix.lower()
    if suffix not in allowed:
        raise UnsupportedFormatError(path, f"extension '{suffix}' is not supported", extension=suffix)


def _exif_orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
    except (AttributeError, OSError, ValueError):
        return None
    value = exif.get(_EXIF_ORIENTATION_TAG) if exif else None
    return int(value) if isinstance(value, (int, float)) else None


def save_image(image: Image.Image, path: Path, image_format: str, quality: int = 95) -> None:
    """Write an image atomically in the configured output format."""
    ensure_dir(path.parent)
    fmt = image_format.upper()
    fmt = "JPEG" if fmt in {"JPG", "JPEG"} else fmt
    params: dict[str, Any] = {}
    if fmt == "JPEG":
        params = {"quality": quality, "optimize": True, "subsampling": 1}
        image = image.convert("RGB")
    elif fmt == "WEBP":
        params = {"quality": quality, "method": 4}
    elif fmt == "PNG":
        params = {"optimize": True}

    tmp = path.with_name(f".{path.name}.tmp")
    try:
        image.save(tmp, format=fmt, **params)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """JSON encoder fallback for paths, enums, numpy scalars and datetimes."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write via a temporary file + rename so readers never see partial output."""
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    text = json.dumps(payload, indent=indent, sort_keys=False, default=json_default)
    return atomic_write_text(path, text + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: Any) -> Path:
    text = yaml.safe_dump(
        json.loads(json.dumps(payload, default=json_default)),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return atomic_write_text(path, text)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a mapping at the root of {path}, got {type(loaded).__name__}")
    return loaded


def copy_file(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def link_or_copy(src: Path, dst: Path) -> Path:
    """Hard-link when the filesystem allows it, otherwise fall back to a copy."""
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
    return dst


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

_SLUG_SEPARATORS = re.compile(r"[\s\-/\\.]+")
_SLUG_INVALID = re.compile(r"[^a-z0-9_]+")
_SLUG_COLLAPSE = re.compile(r"_{2,}")


def slugify(text: str, fallback: str = "unknown") -> str:
    """Filesystem-safe lowercase identifier: ``Tomato Late-Blight`` -> ``tomato_late_blight``."""
    normalised = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalised = _SLUG_SEPARATORS.sub("_", normalised.strip().lower())
    normalised = _SLUG_INVALID.sub("_", normalised)
    normalised = _SLUG_COLLAPSE.sub("_", normalised).strip("_")
    return normalised or fallback


__all__ = [
    "DEFAULT_EXTENSIONS",
    "FileEntry",
    "ImageProbe",
    "atomic_write_bytes",
    "atomic_write_text",
    "check_extension",
    "configure_pillow_limits",
    "copy_file",
    "ensure_dir",
    "hash_mapping",
    "iter_files",
    "iter_subdirectories",
    "json_default",
    "link_or_copy",
    "load_rgb",
    "pixel_digest",
    "probe_image",
    "read_json",
    "read_yaml",
    "relative_path",
    "save_image",
    "sha256_file",
    "slugify",
    "stable_id",
    "to_array",
    "walk_image_entries",
    "write_json",
    "write_yaml",
]
