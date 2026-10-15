from typing import Any, Dict, Optional

from .base_dataset import BaseVideoDataset
from .data_utils import VideoDecodeError, collate_video_batch, frames_to_tensor, normalize_frames, sample_frames
from .query_templates import get_queries_for_dataset
from .rlvs_dataset import RLVSDataset
from .tinyvirat_dataset import TinyVIRATDataset
from .ucf_arg_dataset import UCFArgDataset
from .ucf_crime_dataset import UCFCrimeDataset

DATASET_REGISTRY = {
    "rlvs": RLVSDataset,
    "ucf_arg": UCFArgDataset,
    "tinyvirat": TinyVIRATDataset,
    "ucf_crime": UCFCrimeDataset,
}


def build_dataset(
    name: str,
    config: Dict[str, Any],
    split: str = "all",
    viewpoint_filter: Optional[str] = None,
) -> BaseVideoDataset:
    """Factory creating appropriate video dataset loader."""
    key = name.lower().replace("-", "_")
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")

    cls = DATASET_REGISTRY[key]
    if key == "ucf_arg":
        return cls(config=config, split=split, viewpoint_filter=viewpoint_filter)
    return cls(config=config, split=split)


__all__ = [
    "BaseVideoDataset",
    "RLVSDataset",
    "UCFArgDataset",
    "TinyVIRATDataset",
    "UCFCrimeDataset",
    "build_dataset",
    "sample_frames",
    "frames_to_tensor",
    "normalize_frames",
    "collate_video_batch",
    "get_queries_for_dataset",
    "DATASET_REGISTRY",
]
