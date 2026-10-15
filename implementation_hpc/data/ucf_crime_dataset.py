import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_dataset import BaseVideoDataset

logger = logging.getLogger(__name__)

ANOMALY_CLASSES = [
    "abuse",
    "arrest",
    "arson",
    "assault",
    "burglary",
    "explosion",
    "fighting",
    "normal",
    "roadaccidents",
    "robbery",
    "shooting",
    "shoplifting",
    "stealing",
    "vandalism",
]

VIDEO_EXTS = (".mp4", ".avi", ".MP4", ".AVI")


class UCFCrimeDataset(BaseVideoDataset):
    """
    UCF-Crime Surveillance Anomaly Benchmark loader.
    Indexes all 14 anomaly categories and normal footage across extracted folders.
    """

    def _infer_label(self, path: Path) -> str:
        """Infers anomaly class from filename and folder hierarchy."""
        name_lower = path.stem.lower()
        parent_lower = path.parent.name.lower()

        # Check direct class matches
        for cls in ANOMALY_CLASSES:
            if cls in name_lower or cls in parent_lower:
                return cls

        # Normal videos prefix pattern
        if "normal" in name_lower or "normal" in parent_lower:
            return "normal"

        return "unknown"

    def _build_index(self) -> None:
        ucf_cfg = self.config["data"].get("ucf_crime", {})
        root_path = Path(ucf_cfg.get("path", "C:/Users/HPCLAB/Desktop/capstone-ucf dataset"))

        if not root_path.exists():
            logger.warning("UCF-Crime directory not found at: %s", root_path)
            return

        max_clips = ucf_cfg.get("max_clips", None)
        samples = []

        # Find all MP4 and AVI video files across all extracted subdirectories
        for video_path in sorted(root_path.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTS:
                # Exclude splits or temporary files
                if "split" in str(video_path).lower():
                    continue

                label = self._infer_label(video_path)
                samples.append({
                    "path": video_path,
                    "label": label,
                    "metadata": {
                        "dataset": "ucf_crime",
                        "filename": video_path.name,
                        "class_raw": label,
                        "category": "normal" if label == "normal" else "anomaly",
                    },
                })

        # Deterministic ordering
        samples = sorted(samples, key=lambda s: str(s["path"]))
        if max_clips and len(samples) > max_clips:
            samples = samples[:max_clips]

        self.samples = samples
        logger.info("Indexed UCF-Crime Dataset: %d clips loaded from %s.", len(self.samples), root_path)
