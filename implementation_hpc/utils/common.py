import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def setup_logger(name: str = "implementation_hpc", level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads master YAML configuration with environment variable override support."""
    if config_path is None:
        env_cfg = os.environ.get("CAPSTONE_CONFIG")
        if env_cfg:
            config_path = Path(env_cfg)
        else:
            config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow runtime override of dataset root via environment variable
    env_root = os.environ.get("CAPSTONE_DATA_ROOT")
    if env_root:
        cfg["data"]["root"] = env_root

    return cfg


def get_dataset_root(config: Dict[str, Any]) -> Path:
    """Resolves data root path, gracefully falling back to demo root if external disk is detached."""
    primary_root = Path(config["data"]["root"])
    if primary_root.exists():
        return primary_root

    # Check fallback demo root
    fallback = Path(__file__).resolve().parent.parent / config["data"].get("fallback_demo_root", "datasets")
    if fallback.exists():
        return fallback

    # Check project-level datasets folder
    project_datasets = Path(__file__).resolve().parent.parent.parent / "implementation" / "datasets"
    if project_datasets.exists():
        return project_datasets

    return primary_root


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolves computational device (CUDA on RTX 4000 Ada with fallback)."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensures that a directory exists, creating all parent directories if necessary."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_gpu_cache() -> None:
    """Clears PyTorch CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
