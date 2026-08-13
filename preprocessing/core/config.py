"""Typed, validated configuration for the whole framework.

The pipeline has no hardcoded behaviour: thresholds, ratios, augmentations,
paths, seeds and worker counts all come from YAML. ``config/pipeline.yaml`` is
the root document and pulls in ``quality.yaml``, ``split.yaml`` and
``transforms.yaml`` through its ``include`` section.

Models are frozen and reject unknown keys, so a typo in a threshold name fails
the run at load time instead of silently disabling a gate.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigurationError
from .io import DEFAULT_EXTENSIONS, hash_mapping, read_yaml, write_yaml

ENV_PREFIX = "CDP__"
_PATH_KEYS = {"raw_root", "corpus_dir", "cache_dir", "processed_dir", "output_dir", "root", "directory", "path", "alias_file"}
_OPAQUE_KEYS = {"metadata", "label_overrides", "label_aliases", "weights"}


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# --------------------------------------------------------------------------- #
# Paths, execution, logging
# --------------------------------------------------------------------------- #


class PathsConfig(_Section):
    raw_root: Path = Path("data/raw")
    corpus_dir: Path = Path("data/corpus")
    cache_dir: Path = Path("data/cache")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs/preprocessed_dataset")


class SourceConfig(_Section):
    """One contributing dataset in folder-per-class layout."""

    name: str
    path: Path
    version: str = "unversioned"
    enabled: bool = True
    class_depth: int = Field(1, ge=1, description="Directory levels below the root that hold class folders")
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    detect_splits: bool = Field(True, description="Treat a leading train/val/test directory as the source's own split")
    split_dirs: tuple[str, ...] = ("train", "val", "valid", "validation", "test", "eval")
    default_crop: str | None = Field(None, description="Crop assumed when this source's labels carry no crop prefix")
    label_overrides: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(0, description="Higher priority wins when duplicate images collide across sources")


class DiscoveryConfig(_Section):
    """Opt-in discovery of sources that are not listed explicitly."""

    auto_discover: bool = False
    exclude_patterns: tuple[str, ...] = (".*", "_*", "*backup*", "*bak", "*tmp*", "*temp*", "*archive*")
    class_depth: int = Field(1, ge=1)
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    version: str = "unversioned"


class ExecutionConfig(_Section):
    workers: int = Field(0, ge=0, description="0 selects cpu_count() - 1")
    chunk_size: int = Field(64, ge=1)
    progress: bool = True
    fail_fast: bool = False
    max_images: int | None = Field(None, ge=1, description="Global cap for smoke runs")
    max_images_per_class: int | None = Field(None, ge=1)

    @property
    def resolved_workers(self) -> int:
        if self.workers > 0:
            return self.workers
        return max(1, (os.cpu_count() or 2) - 1)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    console: bool = True
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_file: bool = True
    directory: Path = Path("outputs/logs")
    filename_template: str = "run_{run_id}.jsonl"
    capture_warnings: bool = True


class StagesConfig(_Section):
    """Stage toggles; corpus construction always runs."""

    validation: bool = True
    quality: bool = True
    profiling: bool = True
    analysis: bool = True
    splitting: bool = True
    packaging: bool = True
    dataloaders: bool = True
    reports: bool = True
    visualizations: bool = True


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #


class VersioningConfig(_Section):
    scheme: Literal["semantic", "timestamp", "fingerprint"] = "semantic"
    version: str = "1.0.0"
    bump_on_change: bool = True


class CorpusConfig(_Section):
    label_delimiters: tuple[str, ...] = ("___", "__")
    canonical_format: str = "{crop}___{condition}"
    condition_only_format: str = "{condition}"
    normalize_labels: bool = True
    lowercase_labels: bool = True
    label_aliases: dict[str, str] = Field(default_factory=dict)
    alias_file: Path | None = None
    crops: tuple[str, ...] = Field((), description="Explicit crop vocabulary; extends whatever is inferred")
    infer_crop_vocabulary: bool = Field(
        True, description="Learn crop names from delimited labels and use them to split undelimited ones"
    )
    healthy_terms: tuple[str, ...] = ("healthy", "normal", "fresh")
    healthy_label: str = "healthy"
    include_labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()
    min_images_per_class: int = Field(1, ge=1)
    id_strategy: Literal["path", "content"] = "path"
    probe_dimensions: bool = Field(False, description="Read image headers during ingestion (no pixel decode)")
    track_first_seen: bool = Field(True, description="Carry first_seen_run_id over from a previous corpus index")
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)

    @field_validator("canonical_format", "condition_only_format")
    @classmethod
    def _format_has_condition(cls, value: str) -> str:
        if "{condition}" not in value:
            raise ValueError("label format must contain the '{condition}' placeholder")
        return value


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class LabelValidationConfig(_Section):
    enabled: bool = True
    pattern: str = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$"
    max_length: int = Field(128, ge=1)
    forbidden_characters: tuple[str, ...] = ("/", "\\", ":", "*", "?", '"', "<", ">", "|")
    allow_whitespace: bool = False
    require_mapping: bool = True
    require_class_index: bool = True


class MetadataValidationConfig(_Section):
    enabled: bool = True
    required_provenance_fields: tuple[str, ...] = (
        "dataset_name",
        "dataset_version",
        "source_root",
        "source_path",
        "source_relpath",
        "source_class",
        "original_filename",
    )
    require_timestamps: bool = True
    require_first_seen: bool = True
    validate_hashes: bool = True
    validate_paths: bool = True
    validate_dimensions: bool = True


class IntegrityValidationConfig(_Section):
    enabled: bool = True
    check_duplicate_ids: bool = True
    check_duplicate_entries: bool = True
    check_missing_files: bool = True
    check_paths: bool = True
    check_provenance: bool = True
    check_fingerprint: bool = True
    check_manifest: bool = True


class ValidationConfig(_Section):
    """Structural verification settings.

    Image-level options stay flat because Module 1 reads ``max_pixels`` when it
    configures Pillow's decompression-bomb ceiling.
    """

    validate_images: bool = True
    allowed_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    allowed_formats: tuple[str, ...] = ("jpeg", "png", "mpo")
    allowed_color_modes: tuple[str, ...] = Field(
        (), description="Empty means any mode that converts to RGB is accepted"
    )
    full_decode: bool = True
    apply_exif_orientation: bool = True
    require_rgb_convertible: bool = True
    min_dimension: int = Field(2, ge=1)
    max_pixels: int | None = Field(178_956_970, ge=1)
    max_file_size_mb: float | None = Field(None, gt=0)

    record_observations: bool = Field(
        True, description="Store technical facts observed while decoding so later stages need not decode again"
    )
    workers: int | None = Field(None, ge=1, description="Decode threads; defaults to execution.workers")
    output_dir: Path = Path("data/processed/validation")

    labels: LabelValidationConfig = Field(default_factory=LabelValidationConfig)
    metadata: MetadataValidationConfig = Field(default_factory=MetadataValidationConfig)
    integrity: IntegrityValidationConfig = Field(default_factory=IntegrityValidationConfig)


# --------------------------------------------------------------------------- #
# Quality gate (config/quality.yaml)
# --------------------------------------------------------------------------- #


class BlurConfig(_Section):
    enabled: bool = True
    method: Literal["laplacian", "tenengrad"] = "laplacian"
    min_score: float = Field(80.0, ge=0)


class BrightnessConfig(_Section):
    enabled: bool = True
    min_mean: float = Field(25.0, ge=0, le=255)
    max_mean: float = Field(235.0, ge=0, le=255)
    shadow_level: float = Field(16.0, ge=0, le=255, description="Pixels at or below this count as crushed shadows")
    highlight_level: float = Field(240.0, ge=0, le=255)
    max_clipped_fraction: float | None = Field(
        None, ge=0, le=1, description="Reject when clipped shadow+highlight pixels exceed this share"
    )


class ContrastConfig(_Section):
    enabled: bool = True
    method: Literal["std", "michelson", "rms"] = "std"
    min_value: float = Field(15.0, ge=0)


class ResolutionConfig(_Section):
    enabled: bool = True
    min_width: int = Field(64, ge=1)
    min_height: int = Field(64, ge=1)
    max_width: int | None = Field(None, ge=1)
    max_height: int | None = Field(None, ge=1)
    min_megapixels: float | None = None
    max_megapixels: float | None = None
    min_aspect_ratio: float = Field(0.25, gt=0)
    max_aspect_ratio: float = Field(4.0, gt=0)


class DuplicateConfig(_Section):
    enabled: bool = True
    exact: bool = True
    exact_method: Literal["content", "pixel", "both"] = "both"
    near: bool = True
    hash_type: Literal["phash", "dhash", "ahash"] = "phash"
    hash_size: int = Field(8, ge=4, le=16)
    max_hamming_distance: int = Field(5, ge=0)
    keep: Literal["first", "highest_quality", "largest"] = "highest_quality"
    across_classes: bool = Field(True, description="Also compare images from different labels")
    exact_action: Literal["reject", "warn", "keep"] = "reject"
    near_action: Literal["reject", "warn", "keep"] = "reject"


class QualityScoringConfig(_Section):
    enabled: bool = True
    weights: dict[str, float] = Field(
        default_factory=lambda: {"blur": 0.4, "brightness": 0.2, "contrast": 0.25, "resolution": 0.15}
    )
    min_score: float = Field(0.35, ge=0, le=1)
    warn_score: float = Field(0.55, ge=0, le=1, description="Scores below this are accepted with a warning")
    reference_blur: float = Field(400.0, gt=0, description="Blur score mapped to 1.0 when normalising")
    reference_contrast: float = Field(70.0, gt=0)
    reference_megapixels: float = Field(0.25, gt=0)
    grade_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"A": 0.85, "B": 0.70, "C": 0.55, "D": 0.40}
    )
    fallback_grade: str = "F"

    @field_validator("weights")
    @classmethod
    def _weights_positive(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(w < 0 for w in value.values()) or sum(value.values()) <= 0:
            raise ValueError("quality scoring weights must be non-negative and sum to a positive value")
        return value

    @field_validator("grade_thresholds")
    @classmethod
    def _grades_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not 0 <= score <= 1 for score in value.values()):
            raise ValueError("grade thresholds must be non-empty and within [0, 1]")
        return value

    @model_validator(mode="after")
    def _warn_above_reject(self) -> "QualityScoringConfig":
        if self.warn_score < self.min_score:
            raise ValueError("scoring.warn_score must not be below scoring.min_score")
        return self


class QualityConfig(_Section):
    enabled: bool = True
    reject_on_failure: bool = True
    metric_resize: int = Field(
        512, ge=32, description="Longest side of the single decoded array every pixel metric is computed from"
    )
    workers: int | None = Field(None, ge=1, description="Analysis threads; defaults to execution.workers")
    blur: BlurConfig = Field(default_factory=BlurConfig)
    brightness: BrightnessConfig = Field(default_factory=BrightnessConfig)
    contrast: ContrastConfig = Field(default_factory=ContrastConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    duplicates: DuplicateConfig = Field(default_factory=DuplicateConfig)
    scoring: QualityScoringConfig = Field(default_factory=QualityScoringConfig)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


class RgbProfilingConfig(_Section):
    enabled: bool = True
    sample_size: int | None = Field(
        4000, ge=1, description="Images sampled for channel statistics; null profiles the whole corpus"
    )
    resize: int = Field(256, ge=16, description="Longest side each sampled image is reduced to before accumulation")
    histogram_bins: int = Field(64, ge=8, le=256)
    workers: int | None = Field(None, ge=1)
    cache: bool = Field(True, description="Reuse results keyed by dataset fingerprint")


class ProfilingConfig(_Section):
    enabled: bool = True
    include_rejected: bool = Field(False, description="Profile the benchmark-ready corpus only, by default")
    histogram_bins: int = Field(40, ge=5, le=200)
    percentiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    top_classes: int = Field(50, ge=1, description="Classes listed individually in reports")
    top_resolutions: int = Field(25, ge=1)
    rgb: RgbProfilingConfig = Field(default_factory=RgbProfilingConfig)

    @field_validator("percentiles")
    @classmethod
    def _percentiles_in_range(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not 0 < item < 1 for item in value):
            raise ValueError("percentiles must be non-empty and strictly between 0 and 1")
        return tuple(sorted(value))


class DatasetScoreConfig(_Section):
    enabled: bool = True
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "class_balance": 0.25,
            "duplicate_ratio": 0.15,
            "diversity": 0.15,
            "quality_score": 0.2,
            "label_consistency": 0.1,
            "resolution_consistency": 0.05,
            "source_diversity": 0.1,
        }
    )
    grade_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"A": 0.85, "B": 0.70, "C": 0.55, "D": 0.40}
    )
    fallback_grade: str = "F"

    @field_validator("weights")
    @classmethod
    def _weights_positive(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(weight < 0 for weight in value.values()) or sum(value.values()) <= 0:
            raise ValueError("dataset score weights must be non-negative and sum to a positive value")
        return value


class ImbalanceConfig(_Section):
    rare_class_ratio: float = Field(0.25, gt=0, description="Fraction of the mean class size below which a class is rare")
    min_class_size: int = Field(20, ge=1)
    max_imbalance_ratio: float = Field(10.0, gt=1, description="Warn above largest/smallest class ratio")


class DiversityConfig(_Section):
    enabled: bool = True
    sample_size: int = Field(2000, ge=50, description="Per-corpus sample used for pairwise hash distances")
    hash_bits: int = Field(64, ge=16)


class BiasConfig(_Section):
    enabled: bool = True
    max_source_share: float = Field(0.8, gt=0, le=1, description="Warn when one source dominates a class")


class LeakageConfig(_Section):
    enabled: bool = True
    check_exact: bool = True
    check_near: bool = True
    max_hamming_distance: int = Field(3, ge=0)


class AnalysisConfig(_Section):
    enabled: bool = True
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    bias: BiasConfig = Field(default_factory=BiasConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    score: DatasetScoreConfig = Field(default_factory=DatasetScoreConfig)


# --------------------------------------------------------------------------- #
# Splitting (config/split.yaml)
# --------------------------------------------------------------------------- #


class SplitRatios(_Section):
    train: float = Field(0.7, gt=0, lt=1)
    val: float = Field(0.15, gt=0, lt=1)
    test: float = Field(0.15, ge=0, lt=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitRatios":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


class SplitConfig(_Section):
    strategy: Literal["stratified", "stratified_group", "random"] = "stratified_group"
    ratios: SplitRatios = Field(default_factory=SplitRatios)
    seed: int = 42
    group_by: Literal["none", "duplicate_group", "source_dataset"] = "duplicate_group"
    min_samples_per_class: int = Field(3, ge=1)
    drop_classes_below_minimum: bool = True
    verify_leakage: bool = True


# --------------------------------------------------------------------------- #
# Transforms (config/transforms.yaml)
# --------------------------------------------------------------------------- #


class NormalizationConfig(_Section):
    source: Literal["imagenet", "dataset", "custom"] = "imagenet"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @field_validator("std")
    @classmethod
    def _std_positive(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(v <= 0 for v in value):
            raise ValueError("normalization std must be strictly positive")
        return value


class RandomResizedCropConfig(_Section):
    enabled: bool = False
    scale: tuple[float, float] = (0.7, 1.0)
    ratio: tuple[float, float] = (0.75, 1.333)


class RandomCropConfig(_Section):
    enabled: bool = True
    padding: int = Field(0, ge=0)
    pad_if_needed: bool = True


class FlipConfig(_Section):
    enabled: bool = True
    p: float = Field(0.5, ge=0, le=1)


class RotationConfig(_Section):
    enabled: bool = True
    degrees: float = Field(15.0, ge=0)
    expand: bool = False


class AffineConfig(_Section):
    enabled: bool = True
    degrees: float = Field(0.0, ge=0)
    translate: tuple[float, float] = (0.05, 0.05)
    scale: tuple[float, float] = (0.9, 1.1)
    shear: float = Field(5.0, ge=0)


class ColorJitterConfig(_Section):
    enabled: bool = True
    brightness: float = Field(0.2, ge=0)
    contrast: float = Field(0.2, ge=0)
    saturation: float = Field(0.2, ge=0)
    hue: float = Field(0.02, ge=0, le=0.5)


class RandomErasingConfig(_Section):
    enabled: bool = False
    p: float = Field(0.25, ge=0, le=1)
    scale: tuple[float, float] = (0.02, 0.15)


class TrainTransformConfig(_Section):
    resize: tuple[int, int] = (256, 256)
    crop_size: tuple[int, int] = (224, 224)
    random_resized_crop: RandomResizedCropConfig = Field(default_factory=RandomResizedCropConfig)
    random_crop: RandomCropConfig = Field(default_factory=RandomCropConfig)
    horizontal_flip: FlipConfig = Field(default_factory=FlipConfig)
    vertical_flip: FlipConfig = Field(default_factory=lambda: FlipConfig(enabled=False, p=0.0))
    rotation: RotationConfig = Field(default_factory=RotationConfig)
    affine: AffineConfig = Field(default_factory=AffineConfig)
    color_jitter: ColorJitterConfig = Field(default_factory=ColorJitterConfig)
    random_erasing: RandomErasingConfig = Field(default_factory=RandomErasingConfig)


class EvalTransformConfig(_Section):
    resize: tuple[int, int] = (256, 256)
    center_crop: tuple[int, int] | None = (224, 224)


class TransformsConfig(_Section):
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bilinear"
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    train: TrainTransformConfig = Field(default_factory=TrainTransformConfig)
    validation: EvalTransformConfig = Field(default_factory=EvalTransformConfig)
    inference: EvalTransformConfig = Field(default_factory=EvalTransformConfig)


# --------------------------------------------------------------------------- #
# Dataloader, packaging, reports
# --------------------------------------------------------------------------- #


class DataLoaderConfig(_Section):
    batch_size: int = Field(32, ge=1)
    num_workers: int = Field(0, ge=0)
    shuffle_train: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = Field(2, ge=1)
    drop_last: bool = False
    sampler: Literal["random", "weighted", "sequential"] = "random"
    loader_format: Literal["state", "pickle"] = "state"

    @property
    def resolved_workers(self) -> int:
        return self.num_workers

    @model_validator(mode="after")
    def _worker_dependent_flags(self) -> "DataLoaderConfig":
        if self.num_workers == 0 and (self.persistent_workers or self.prefetch_factor is not None):
            # torch rejects these combinations outright; normalise instead of failing the run.
            object.__setattr__(self, "persistent_workers", False)
            object.__setattr__(self, "prefetch_factor", None)
        return self


class ImageOutputConfig(_Section):
    size: tuple[int, int] = Field((224, 224), description="(height, width) of packaged images")
    format: Literal["jpeg", "png", "webp"] = "jpeg"
    quality: int = Field(95, ge=1, le=100)
    resize_mode: Literal["stretch", "shortest_side", "pad"] = "shortest_side"
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bicubic"


class PackagingConfig(_Section):
    materialize: Literal["resize", "copy", "link", "none"] = "resize"
    overwrite: bool = True
    image: ImageOutputConfig = Field(default_factory=ImageOutputConfig)
    manifest_parquet: bool = True
    write_dataloaders: bool = True


class ReportsConfig(_Section):
    html: bool = True
    pdf: bool = True
    dpi: int = Field(120, ge=50, le=600)
    figure_format: Literal["png"] = "png"
    max_classes_in_plots: int = Field(40, ge=5)
    sample_images_per_class: int = Field(0, ge=0)


# --------------------------------------------------------------------------- #
# Root document
# --------------------------------------------------------------------------- #


class Config(_Section):
    """Fully resolved configuration for one preprocessing run."""

    project: str = "crop-disease-benchmark"
    seed: int = 42
    project_root: Path = Path(".")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    sources: tuple[SourceConfig, ...] = ()
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    transforms: TransformsConfig = Field(default_factory=TransformsConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    packaging: PackagingConfig = Field(default_factory=PackagingConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)

    @property
    def enabled_sources(self) -> tuple[SourceConfig, ...]:
        return tuple(source for source in self.sources if source.enabled)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        """Hash of the effective configuration, recorded in the run manifest."""
        payload = self.as_dict()
        payload.pop("project_root", None)  # machine-specific, must not change the hash
        return hash_mapping(payload)

    def semantic_fingerprint(self) -> str:
        """Hash of the configuration with every location stripped out.

        Feeds the dataset fingerprint: moving the same data to a different
        directory, or writing the output elsewhere, must not change the identity
        of the resulting corpus. Anything that changes *what* is produced
        (thresholds, ratios, filters, caps) is still covered.
        """
        payload = self.as_dict()
        for key in ("project_root", "paths", "logging"):
            payload.pop(key, None)
        for source in payload.get("sources", []):
            source.pop("path", None)
        return hash_mapping(payload)

    def dump(self, path: Path) -> Path:
        """Write the fully resolved configuration (preprocessing_config.yaml)."""
        return write_yaml(path, self.as_dict())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_config(
    path: Path | str | None = None,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    project_root: Path | str | None = None,
) -> Config:
    """Load ``pipeline.yaml``, merge includes, apply overrides, and validate.

    Precedence, lowest to highest: included files, ``pipeline.yaml`` inline
    values, ``CDP__``-prefixed environment variables, explicit overrides.
    """
    root = Path(project_root).resolve() if project_root else _detect_project_root()
    config_path = Path(path) if path else root / "config" / "pipeline.yaml"
    config_path = config_path if config_path.is_absolute() else root / config_path

    try:
        raw = read_yaml(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc

    raw = _merge_includes(raw, config_path.parent)
    _apply_overrides(raw, _env_overrides())
    _apply_overrides(raw, _normalize_overrides(overrides))

    # Path resolution rewrites the document in place. Deep-copy first so that
    # nested structures supplied by the caller (a reused `sources` list, say)
    # are never mutated and repeated loads stay independent.
    raw = copy.deepcopy(raw)
    raw.setdefault("project_root", str(root))
    _resolve_source_paths(raw, root)  # must precede _resolve_paths: sources anchor on raw_root
    _resolve_paths(raw, root)

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration in {config_path}:\n{_format_errors(exc)}") from exc


def _detect_project_root() -> Path:
    """The repository root: two levels above this file (preprocessing/core/)."""
    return Path(__file__).resolve().parents[2]


def _merge_includes(raw: MutableMapping[str, Any], config_dir: Path) -> dict[str, Any]:
    """Resolve the ``include`` section: ``{section: file}`` merged under ``section``."""
    merged = dict(raw)
    includes = merged.pop("include", {}) or {}
    if not isinstance(includes, Mapping):
        raise ConfigurationError("'include' must be a mapping of section name to file path")

    for section, filename in includes.items():
        include_path = Path(filename)
        include_path = include_path if include_path.is_absolute() else config_dir / include_path
        try:
            content = read_yaml(include_path)
        except (FileNotFoundError, ValueError) as exc:
            raise ConfigurationError(f"include '{section}' -> {exc}") from exc
        merged[section] = _deep_merge(content, merged.get(section) or {})
    return merged


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _env_overrides() -> dict[str, Any]:
    """``CDP__QUALITY__BLUR__MIN_SCORE=120`` -> ``quality.blur.min_score = 120``."""
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        dotted = key[len(ENV_PREFIX) :].lower().replace("__", ".")
        overrides[dotted] = _parse_scalar(value)
    return overrides


def _normalize_overrides(overrides: Mapping[str, Any] | Sequence[str] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    if isinstance(overrides, Mapping):
        return dict(overrides)
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ConfigurationError(f"override '{item}' must use the form key.path=value")
        key, _, value = item.partition("=")
        parsed[key.strip()] = _parse_scalar(value.strip())
    return parsed


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_overrides(raw: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor: MutableMapping[str, Any] = raw
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, MutableMapping):
                existing = {}
                cursor[part] = existing
            cursor = existing
        cursor[parts[-1]] = value


def _resolve_paths(node: Any, root: Path, key: str | None = None) -> None:
    """Make every known path key absolute against the project root, in place."""
    if isinstance(node, MutableMapping):
        for child_key, child in node.items():
            if child_key in _OPAQUE_KEYS:
                continue
            if child_key in _PATH_KEYS and isinstance(child, str):
                node[child_key] = str(_absolute(child, root))
            else:
                _resolve_paths(child, root, child_key)
    elif isinstance(node, list) and key not in _OPAQUE_KEYS:
        for item in node:
            _resolve_paths(item, root, key)


def _resolve_source_paths(raw: MutableMapping[str, Any], root: Path) -> None:
    """Source paths are relative to ``paths.raw_root`` rather than the project root."""
    sources = raw.get("sources")
    if not isinstance(sources, list):
        return
    raw_root = _absolute(str((raw.get("paths") or {}).get("raw_root", "data/raw")), root)
    for source in sources:
        if isinstance(source, MutableMapping) and isinstance(source.get("path"), str):
            source["path"] = str(_absolute(source["path"], raw_root))


def _absolute(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  - {location or '<root>'}: {error['msg']}")
    return "\n".join(lines)


__all__ = [
    "Config",
    "DiscoveryConfig",
    "PathsConfig",
    "SourceConfig",
    "ExecutionConfig",
    "LoggingConfig",
    "StagesConfig",
    "CorpusConfig",
    "VersioningConfig",
    "ValidationConfig",
    "LabelValidationConfig",
    "MetadataValidationConfig",
    "IntegrityValidationConfig",
    "QualityConfig",
    "BlurConfig",
    "BrightnessConfig",
    "ContrastConfig",
    "ResolutionConfig",
    "DuplicateConfig",
    "QualityScoringConfig",
    "AnalysisConfig",
    "SplitConfig",
    "SplitRatios",
    "TransformsConfig",
    "NormalizationConfig",
    "TrainTransformConfig",
    "EvalTransformConfig",
    "DataLoaderConfig",
    "PackagingConfig",
    "ImageOutputConfig",
    "ReportsConfig",
    "load_config",
]
