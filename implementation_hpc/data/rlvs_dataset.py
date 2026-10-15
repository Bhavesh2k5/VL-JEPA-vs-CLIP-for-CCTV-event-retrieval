import glob
import logging
from pathlib import Path
from typing import Any, Dict

from utils.common import get_dataset_root
from .base_dataset import BaseVideoDataset

logger = logging.getLogger(__name__)


class RLVSDataset(BaseVideoDataset):
    """
    Real Life Violence Situations (RLVS) Dataset loader.
    Captures both *.mp4 and *.avi files (including the 49 AVI files in NonViolence).
    """

    def _build_index(self) -> None:
        data_root = get_dataset_root(self.config)
        rlvs_cfg = self.config["data"]["rlvs"]
        rel_path = rlvs_cfg["path"]

        dataset_dir = data_root / rel_path
        if not dataset_dir.exists():
            # Check direct under data_root or demo structure
            alt_dir = data_root / "real life voilence data"
            if alt_dir.exists():
                dataset_dir = alt_dir
            else:
                logger.warning("RLVS directory not found at %s. Initializing empty dataset.", dataset_dir)
                return

        classes = rlvs_cfg.get("classes", ["Violence", "NonViolence"])
        max_clips = rlvs_cfg.get("max_clips", None)

        samples = []
        for cls_name in classes:
            # Case-insensitive folder search
            cls_folder = dataset_dir / cls_name
            if not cls_folder.exists():
                candidates = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.lower() == cls_name.lower()]
                if candidates:
                    cls_folder = candidates[0]
                else:
                    logger.warning("RLVS class folder %s not found in %s", cls_name, dataset_dir)
                    continue

            # Glob both MP4 and AVI files to prevent dropping the 49 AVI samples
            video_files = []
            for ext in ("*.mp4", "*.avi", "*.MP4", "*.AVI"):
                video_files.extend(list(cls_folder.glob(ext)))

            # Sort for deterministic order
            video_files = sorted(set(video_files), key=lambda p: p.name)

            for vpath in video_files:
                samples.append({
                    "path": vpath,
                    "label": "violence" if cls_name.lower() == "violence" else "non_violence",
                    "metadata": {
                        "dataset": "rlvs",
                        "class_raw": cls_name,
                        "filename": vpath.name,
                    },
                })

        if max_clips and len(samples) > max_clips:
            samples = samples[:max_clips]

        self.samples = samples
        logger.info("Indexed RLVS Dataset: %d clips loaded (%s).", len(self.samples), dataset_dir)
