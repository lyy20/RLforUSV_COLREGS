"""Shared output paths for training and evaluation artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def usv_log_root() -> Path:
    """Return the shared root for all logs and checkpoints.

    The default is ``D:\\USV\\logs``. Set ``USV_LOG_ROOT`` to override it
    without changing code.
    """
    return Path(os.environ.get("USV_LOG_ROOT", r"D:\USV\logs"))


def training_run_root(config_stem: str) -> Path:
    return usv_log_root() / str(config_stem)


def training_log_dir(config_stem: str) -> Path:
    return training_run_root(config_stem) / "log"


def training_model_dir(config_stem: str) -> Path:
    return training_run_root(config_stem) / "model_dir"


def training_before_dir(config_stem: str) -> Path:
    return training_model_dir(config_stem) / "before_dir"


def training_benchmark_dir(config_stem: str) -> Path:
    return training_run_root(config_stem) / "benchmark_dir"


def evaluation_output_dir(config_stem: str) -> Path:
    return training_run_root(config_stem) / "evaluation"


def colregs_visualization_dir() -> Path:
    return usv_log_root() / "colregs_visualization"


def scenario_generation_dir() -> Path:
    return usv_log_root() / "scenario_generation"
