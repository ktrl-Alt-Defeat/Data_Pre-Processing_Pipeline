"""Shared infrastructure: configuration, logging, records, IO and run context.

This package has no dependencies on the pipeline stages, only the other way
around, which keeps the dependency graph acyclic and the stages independently
testable.
"""

from .config import Config, DiscoveryConfig, SourceConfig, load_config
from .context import (
    OutputLayout,
    RunContext,
    collect_environment,
    git_commit,
    seed_torch,
    set_global_seed,
)
from .errors import (
    ConfigurationError,
    ImageDecodeError,
    ImageError,
    ImageReadError,
    PackagingError,
    PipelineError,
    PreprocessingError,
    SourceError,
    StageError,
    UnsupportedFormatError,
)
from .logging import StageTracker, StructuredLogger, configure_logging, get_logger, stage_scope
from .records import (
    Corpus,
    ImageRecord,
    LabelMapping,
    Operation,
    PipelineStage,
    Provenance,
    QualityMetrics,
    RecordStatus,
    Rejection,
    RejectionCode,
    RunManifest,
    Severity,
    SourceSummary,
    Split,
    StageReport,
    ValidationIssue,
)

__all__ = [
    "Config",
    "ConfigurationError",
    "Corpus",
    "DiscoveryConfig",
    "ImageDecodeError",
    "ImageError",
    "ImageReadError",
    "ImageRecord",
    "LabelMapping",
    "Operation",
    "OutputLayout",
    "PackagingError",
    "PipelineError",
    "PipelineStage",
    "PreprocessingError",
    "Provenance",
    "QualityMetrics",
    "RecordStatus",
    "Rejection",
    "RejectionCode",
    "RunContext",
    "RunManifest",
    "Severity",
    "SourceConfig",
    "SourceError",
    "SourceSummary",
    "Split",
    "StageError",
    "StageReport",
    "StageTracker",
    "StructuredLogger",
    "UnsupportedFormatError",
    "ValidationIssue",
    "collect_environment",
    "configure_logging",
    "get_logger",
    "git_commit",
    "load_config",
    "seed_torch",
    "set_global_seed",
    "stage_scope",
]
