"""Master preprocessing pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .analysis import analyze_dataset, render_analysis_report
from .core.config import Config, load_config
from .core.context import RunContext, set_global_seed
from .core.logging import get_logger
from .corpus import build_corpus
from .dataloader import build_dataloaders
from .profiling import profile_dataset, render_profiling_report
from .quality import assess_quality
from .splitting import split_corpus
from .validation import validate_corpus
from .packaging import export_dataset
from reports.visualizations import VisualizationRenderer

logger = get_logger("preprocessing.pipeline")


def run_pipeline(config_or_path: str | Path | Config = "config/pipeline.yaml") -> RunContext:
    """Execute the full data pre-processing pipeline."""
    if isinstance(config_or_path, Config):
        config = config_or_path
    else:
        config = load_config(config_or_path)

    set_global_seed(config.seed)
    context = RunContext.start(config)

    logger.info("pipeline.starting", run_id=context.run_id, project=config.project)

    # 1. Ingest & Build Corpus
    corpus = build_corpus(config, context)

    # 2. Structural & Image Validation
    if config.stages.validation:
        validate_corpus(config, context, corpus)

    # 3. Quality Gate (Blur, Exposure, Contrast, Duplicates)
    if config.stages.quality:
        assess_quality(config, context, corpus)

    # 4. Statistical Profiling & RGB metrics
    profile = None
    if config.stages.profiling:
        profiling_res = profile_dataset(config, context, corpus)
        profile = profiling_res.profile

    # 5. Dataset Analysis & Scoring
    analysis = None
    if config.stages.analysis and profile is not None:
        analysis = analyze_dataset(config, context, corpus, profile)

    # 6. Dataset Splitting (Train / Val / Test)
    split_result = None
    if config.stages.splitting:
        split_result = split_corpus(config, context, corpus)

    # 7. Materialize Dataset Folders (train, val, test) & Summary JSON
    if config.stages.packaging:
        export_dataset(config, context, corpus, split_result, analysis_report=analysis)

    # 8. PyTorch DataLoaders Initialization
    if config.stages.dataloaders and split_result is not None:
        build_dataloaders(split_result.splits, config)

    # 9. Visualizations & Analytical HTML/PDF Reports
    if config.stages.visualizations and profile is not None:
        renderer = VisualizationRenderer(config.reports, context.layout)
        figures = renderer.render(profile, analysis)

        if config.stages.reports:
            render_profiling_report(context, profile, figures.figures)
            if analysis is not None:
                render_analysis_report(context, analysis, profile)

    logger.info("pipeline.completed", run_id=context.run_id, output_dir=str(context.layout.root))
    return context
