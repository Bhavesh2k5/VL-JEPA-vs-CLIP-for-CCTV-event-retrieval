import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .data_utils import VideoDecodeError, frames_to_tensor, sample_frames

logger = logging.getLogger(__name__)


class BaseVideoDataset(Dataset, ABC):
    """
    Abstract base dataset for CCTV video benchmarks.
    Guarantees consistent sample indexing without silent corrupt-sample substitution.
    """

    def __init__(self, config: Dict[str, Any], split: str = "all"):
        super().__init__()
        self.config = config
        self.split = split
        self.num_frames = int(config["video"].get("num_frames", 16))
        self.frame_size = int(config["video"].get("frame_size", 224))
        self.backend = config["video"].get("decode_backend", "auto")

        self.samples: List[Dict[str, Any]] = []
        self._build_index()

    @abstractmethod
    def _build_index(self) -> None:
        """Populates self.samples with dicts: {'path': Path, 'label': Any, 'metadata': dict}."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        video_path = item["path"]

        target_size = (self.frame_size, self.frame_size)
        try:
            frames_np = sample_frames(
                video_path,
                num_frames=self.num_frames,
                target_size=target_size,
                backend=self.backend,
            )
            video_tensor = frames_to_tensor(frames_np)  # [T, C, H, W]
        except (VideoDecodeError, FileNotFoundError, Exception) as e:
            logger.warning("Failed to decode video at idx %d (%s): %s. Providing zero tensor.", idx, video_path, e)
            video_tensor = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size, dtype=torch.float32)

        return {
            "video": video_tensor,
            "path": str(video_path),
            "label": item["label"],
            "metadata": item.get("metadata", {}),
            "index": idx,
        }

    def get_labels(self) -> List[Any]:
        """Returns all sample labels in index order."""
        return [s["label"] for s in self.samples]

    def get_paths(self) -> List[str]:
        """Returns all sample paths in index order."""
        return [str(s["path"]) for s in self.samples]
