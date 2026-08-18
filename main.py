"""CLI entrypoint for running the Crop-Disease Preprocessing Pipeline."""

import sys
from pathlib import Path

from preprocessing.pipeline import run_pipeline


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
    print(f"Starting Preprocessing Pipeline with config: {config_path}")
    context = run_pipeline(config_path)
    print("\n" + "=" * 60)
    print(" PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 60)
    print(f" Run ID:         {context.run_id}")
    print(f" Output Dir:     {context.layout.root}")
    print(f" Reports Dir:    {context.layout.reports_dir}")
    print(f" Log File:       {context.log_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
