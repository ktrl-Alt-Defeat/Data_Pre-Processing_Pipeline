"""Dataset Packaging and Export module.

Materializes the processed images into train/val/test split directories under
preprocessed_dataset/dataset/ and writes the master preprocessing summary JSON.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from ..core.config import Config
from ..core.context import RunContext
from ..core.io import ensure_dir, write_json
from ..core.logging import get_logger, stage_scope
from ..core.records import Corpus, ImageRecord, Split, PipelineStage
from ..splitting.splitter import SplitResult

logger = get_logger("preprocessing.packaging")


def _process_record(
    record: ImageRecord,
    layout: Any,
    materialize_mode: str,
    target_size: tuple[int, int],
    quality: int,
    image_format: str,
) -> tuple[str, bool]:
    split = record.split if record.split else Split.TRAIN
    if split is Split.UNASSIGNED:
        split = Split.TRAIN

    out_dir = layout.class_dir(split, record.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / record.source_path.name

    src_path = record.source_path
    if not src_path.exists():
        return split.value, False

    try:
        if materialize_mode == "copy":
            shutil.copy2(src_path, out_path)
        elif materialize_mode == "link":
            try:
                if out_path.exists():
                    out_path.unlink()
                os.link(src_path, out_path)
            except Exception:
                shutil.copy2(src_path, out_path)
        else:
            # "resize" or default fallback
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                h, w = target_size
                # Resize according to mode
                img_resized = img.resize((w, h), Image.Resampling.BICUBIC)
                fmt = "JPEG" if image_format.lower() in ("jpg", "jpeg") else "PNG"
                img_resized.save(out_path, format=fmt, quality=quality)
        return split.value, True
    except Exception as exc:
        logger.warning("packaging.image_failed", image_id=record.image_id, error=str(exc))
        return split.value, False


def export_dataset(
    config: Config,
    context: RunContext,
    corpus: Corpus,
    split_result: SplitResult | None = None,
    quality_report: Any | None = None,
    analysis_report: Any | None = None,
) -> dict[str, Any]:
    """Materialize images into dataset/train, dataset/val, dataset/test and write summary."""
    with stage_scope(PipelineStage.PACKAGING, context.logger(PipelineStage.PACKAGING)) as tracker:
        layout = context.layout
        ensure_dir(layout.dataset_dir)
        ensure_dir(layout.reports_dir)

        accepted_records = corpus.accepted()
        total_accepted = len(accepted_records)

        materialize_mode = config.packaging.materialize
        target_size = tuple(config.packaging.image.size)  # (height, width)
        quality = config.packaging.image.quality
        image_format = config.packaging.image.format

        tracker.info(
            "packaging.starting",
            total_images=total_accepted,
            materialize=materialize_mode,
            target_size=target_size,
        )

        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

        if materialize_mode != "none" and total_accepted > 0:
            workers = config.execution.resolved_workers or (os.cpu_count() or 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _process_record,
                        rec,
                        layout,
                        materialize_mode,
                        target_size,
                        quality,
                        image_format,
                    )
                    for rec in accepted_records
                ]
                for future in concurrent.futures.as_completed(futures):
                    split_name, success = future.result()
                    if success:
                        split_counts[split_name] = split_counts.get(split_name, 0) + 1

        tracker.info("packaging.materialized", **split_counts)

        # Build master preprocessing_summary.json
        summary_payload = {
            "run_id": context.run_id,
            "project": config.project,
            "pipeline_version": context.manifest.pipeline_version,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "corpus": {
                "raw_total": len(corpus.records),
                "accepted": total_accepted,
                "rejected": len(corpus.rejected()),
                "classes": corpus.labels.num_classes,
                "class_names": corpus.labels.classes,
            },
            "split": {
                "strategy": split_result.statistics.get("strategy") if split_result else config.split.strategy,
                "counts": split_counts,
            },
            "output": {
                "dataset_dir": str(layout.dataset_dir),
                "metadata_dir": str(layout.metadata_dir),
                "reports_dir": str(layout.reports_dir),
                "visualizations_dir": str(layout.visualizations_dir),
            },
        }

        if analysis_report and hasattr(analysis_report, "score"):
            summary_payload["analysis"] = {
                "score": analysis_report.score.value,
                "grade": analysis_report.score.grade,
            }

        # Write preprocessing_summary.json
        summary_file = layout.preprocessing_summary
        write_json(summary_file, summary_payload)
        tracker.info("packaging.summary_written", path=str(summary_file))

        tracker.metrics(
            materialized=sum(split_counts.values()),
            train_count=split_counts.get("train", 0),
            val_count=split_counts.get("val", 0),
            test_count=split_counts.get("test", 0),
        )
        return summary_payload
