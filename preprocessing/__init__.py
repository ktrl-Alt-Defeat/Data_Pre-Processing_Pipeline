"""Crop-disease dataset preprocessing framework.

Merges heterogeneous folder-per-class image datasets into a single, validated,
quality-gated and fully traceable benchmark corpus.

The public entry points are :func:`preprocessing.core.load_config` and the
pipeline in :mod:`preprocessing.pipeline`; stage packages can also be used
standalone.
"""

from .core import Config, RunContext, get_logger, load_config

__version__ = "1.0.0"

__all__ = ["Config", "RunContext", "__version__", "get_logger", "load_config"]
