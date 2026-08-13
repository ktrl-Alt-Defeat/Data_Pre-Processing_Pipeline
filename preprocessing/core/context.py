"""Run context: identity, determinism, output layout and environment capture.

A :class:`RunContext` is created once per pipeline invocation and threaded
through every stage. It owns the two things reproducibility depends on — the
seeded RNG state and the exact configuration fingerprint — plus the canonical
output layout, so no stage ever composes an output path by hand.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as metadata
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import Config
from .io import configure_pillow_limits, ensure_dir, stable_id
from .logging import StructuredLogger, configure_logging, get_logger
from .records import PipelineStage, RunManifest, Split

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "pillow",
    "opencv-python-headless",
    "scikit-learn",
    "scipy",
    "torch",
    "torchvision",
    "matplotlib",
    "pydantic",
)


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """The mandated ``preprocessed_dataset/`` package layout.

    Every artefact path in the framework is derived from this object, which is
    why the delivered package structure is guaranteed rather than incidental.
    """

    root: Path

    # --- directories -------------------------------------------------------- #
    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def analytics_dir(self) -> Path:
        return self.root / "analytics"

    @property
    def visualizations_dir(self) -> Path:
        return self.root / "visualizations"

    @property
    def dataloaders_dir(self) -> Path:
        return self.root / "dataloaders"

    def split_dir(self, split: Split) -> Path:
        return self.dataset_dir / split.value

    def class_dir(self, split: Split, label: str) -> Path:
        return self.split_dir(split) / label

    # --- metadata ----------------------------------------------------------- #
    @property
    def metadata_csv(self) -> Path:
        return self.metadata_dir / "metadata.csv"

    @property
    def image_manifest(self) -> Path:
        return self.metadata_dir / "image_manifest.parquet"

    @property
    def label_mapping(self) -> Path:
        return self.metadata_dir / "label_mapping.json"

    @property
    def dataset_fingerprint(self) -> Path:
        return self.metadata_dir / "dataset_fingerprint.json"

    @property
    def preprocessing_config(self) -> Path:
        return self.metadata_dir / "preprocessing_config.yaml"

    # --- reports ------------------------------------------------------------ #
    @property
    def quality_report(self) -> Path:
        return self.reports_dir / "quality_report.html"

    @property
    def profiling_report(self) -> Path:
        return self.reports_dir / "profiling_report.html"

    @property
    def preprocessing_report_pdf(self) -> Path:
        return self.reports_dir / "preprocessing_report.pdf"

    @property
    def analysis_report(self) -> Path:
        return self.reports_dir / "analysis_report.html"

    @property
    def preprocessing_summary(self) -> Path:
        return self.reports_dir / "preprocessing_summary.json"

    # --- analytics ---------------------------------------------------------- #
    @property
    def class_distribution_csv(self) -> Path:
        return self.analytics_dir / "class_distribution.csv"

    @property
    def image_statistics_csv(self) -> Path:
        return self.analytics_dir / "image_statistics.csv"

    @property
    def duplicate_report_csv(self) -> Path:
        return self.analytics_dir / "duplicate_report.csv"

    @property
    def quality_metrics_csv(self) -> Path:
        return self.analytics_dir / "quality_metrics.csv"

    @property
    def leakage_report_csv(self) -> Path:
        return self.analytics_dir / "leakage_report.csv"

    @property
    def dataset_score_json(self) -> Path:
        return self.analytics_dir / "dataset_score.json"

    # --- visualizations ----------------------------------------------------- #
    def visualization(self, name: str) -> Path:
        return self.visualizations_dir / name

    # --- dataloaders -------------------------------------------------------- #
    def dataloader(self, split: Split) -> Path:
        return self.dataloaders_dir / f"{split.value}_loader.pt"

    def create(self) -> "OutputLayout":
        """Materialise the full directory tree."""
        for directory in (
            self.dataset_dir,
            self.metadata_dir,
            self.reports_dir,
            self.analytics_dir,
            self.visualizations_dir,
            self.dataloaders_dir,
        ):
            ensure_dir(directory)
        for split in (Split.TRAIN, Split.VAL, Split.TEST):
            ensure_dir(self.split_dir(split))
        return self


@dataclass(slots=True)
class RunContext:
    """Identity and shared services for a single pipeline run.

    The context is mutable only in one direction: it swaps its immutable
    :class:`RunManifest` for an updated copy as the run learns its dataset
    version and completion time. Manifest instances themselves are frozen.
    """

    run_id: str
    started_at: dt.datetime
    config: Config
    config_fingerprint: str
    layout: OutputLayout
    manifest: RunManifest
    log_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, config: Config, run_id: str | None = None) -> "RunContext":
        """Bootstrap a run: seed RNGs, configure logging, create the output tree."""
        started = dt.datetime.now(dt.timezone.utc)
        fingerprint = config.fingerprint()
        # The suffix must be unique per process, not per configuration: two runs of
        # the same config in the same second would otherwise share a log file and a
        # run id, which breaks first-seen tracking.
        identifier = run_id or "{:%Y%m%dT%H%M%SZ}-{}".format(
            started, stable_id(fingerprint, str(time.time_ns()), str(os.getpid()), length=8)
        )

        set_global_seed(config.seed)
        configure_pillow_limits(config.validation.max_pixels)
        log_path = configure_logging(config.logging, identifier, config.project_root)
        layout = OutputLayout(config.paths.output_dir).create()

        environment = collect_environment()
        manifest = RunManifest(
            run_id=identifier,
            pipeline_version=_pipeline_version(),
            config_hash=fingerprint,
            python_version=environment["python"],
            platform=environment["platform"],
            started_at=started.isoformat(timespec="seconds"),
            git_commit=git_commit(config.project_root),
            environment=environment,
        )

        context = cls(
            run_id=identifier,
            started_at=started,
            config=config,
            config_fingerprint=fingerprint,
            layout=layout,
            manifest=manifest,
            log_path=log_path,
        )
        get_logger("preprocessing.run").info(
            "run.started",
            run_id=identifier,
            project=config.project,
            pipeline_version=manifest.pipeline_version,
            config_hash=fingerprint[:16],
            git_commit=manifest.git_commit,
            output_dir=str(layout.root),
            seed=config.seed,
            workers=config.execution.resolved_workers,
        )
        return context

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.manifest.environment

    @property
    def elapsed_seconds(self) -> float:
        return (dt.datetime.now(dt.timezone.utc) - self.started_at).total_seconds()

    def logger(self, name: str | PipelineStage) -> StructuredLogger:
        """Logger pre-bound to this run's identity."""
        resolved = name.logger_name if isinstance(name, PipelineStage) else name
        return get_logger(resolved).bind(run_id=self.run_id)

    def set_dataset_version(self, version: str) -> RunManifest:
        """Record the corpus version resolved during ingestion."""
        self.manifest = self.manifest.with_dataset_version(version)
        return self.manifest

    def finish(self) -> RunManifest:
        """Stamp the manifest with a completion time and log the run summary."""
        self.manifest = self.manifest.finished()
        get_logger("preprocessing.run").info(
            "run.completed",
            run_id=self.run_id,
            dataset_version=self.manifest.dataset_version,
            duration_seconds=round(self.elapsed_seconds, 3),
        )
        return self.manifest

    def summary(self) -> dict[str, Any]:
        return {
            "project": self.config.project,
            "manifest": self.manifest.as_dict(),
            "seed": self.config.seed,
            "output_dir": str(self.layout.root),
            "log_file": str(self.log_path) if self.log_path else None,
        }


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline can reach, including torch when installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    if "torch" in sys.modules:  # avoid paying the torch import cost during light runs
        sys.modules["torch"].manual_seed(seed)


def seed_torch(seed: int, deterministic: bool = True) -> None:
    """Seed torch explicitly; called by the dataloader stage once torch is loaded."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit(project_root: Path) -> str | None:
    """Current commit hash, or ``None`` when the tree is not a git checkout."""
    if not (project_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _pipeline_version() -> str:
    import preprocessing  # deferred: the package imports this module during init

    return preprocessing.__version__


def collect_environment() -> dict[str, Any]:
    """Interpreter, platform and dependency versions, recorded in every report."""
    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }


__all__ = ["OutputLayout", "RunContext", "collect_environment", "seed_torch", "set_global_seed"]
