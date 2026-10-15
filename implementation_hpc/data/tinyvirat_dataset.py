import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.common import get_dataset_root
from .base_dataset import BaseVideoDataset

logger = logging.getLogger(__name__)


class TinyVIRATDataset(BaseVideoDataset):
    """
    TinyVIRAT-v2 Low-Resolution Multi-Label Action Benchmark loader.
    Preserves fine surveillance dynamics and handles multi-label metadata.
    """

    def _build_index(self) -> None:
        data_root = get_dataset_root(self.config)
        tiny_cfg = self.config["data"]["tinyvirat"]
        rel_path = tiny_cfg["path"]

        dataset_dir = data_root / rel_path
        if not dataset_dir.exists():
            logger.warning("TinyVIRAT directory not found at %s. Initializing empty dataset.", dataset_dir)
            return

        splits_cfg = tiny_cfg.get("splits", {})
        split_file_name = splits_cfg.get(self.split)
        if not split_file_name:
            if self.split in ("train", "val", "test"):
                split_file_name = f"tiny_{self.split}_v2.json"
            else:
                split_file_name = splits_cfg.get("train", "tiny_train_v2.json")

        split_path = dataset_dir / split_file_name
        class_map_path = dataset_dir / tiny_cfg.get("class_map", "class_map.json")

        class_map = {}
        if class_map_path.exists():
            try:
                with open(class_map_path, "r", encoding="utf-8") as f:
                    class_map = json.load(f)
            except Exception as e:
                logger.warning("Failed to load class_map.json: %e", e)

        samples = []
        if split_path.exists():
            try:
                with open(split_path, "r", encoding="utf-8") as f:
                    data_entries = json.load(f)

                videos_dir = dataset_dir / "videos"
                for item in data_entries:
                    rel_vpath = item.get("path", "")
                    cand_path = videos_dir / rel_vpath
                    if not cand_path.exists():
                        split_key = self.split if self.split in ("train", "val", "test") else "train"
                        cand_path = videos_dir / split_key / rel_vpath
                    if not cand_path.exists():
                        for s_name in ("train", "val", "test"):
                            sub_p = videos_dir / s_name / rel_vpath
                            if sub_p.exists():
                                cand_path = sub_p
                                break
                    if not cand_path.exists():
                        cand_path = dataset_dir / rel_vpath
                    if not cand_path.exists():
                        continue

                    labels_raw = item.get("label", [])
                    if isinstance(labels_raw, str):
                        labels_list = [labels_raw]
                    else:
                        labels_list = list(labels_raw)

                    # Primary class is first non-empty label, or joined
                    primary_label = labels_list[0] if labels_list else "unknown"

                    samples.append({
                        "path": cand_path,
                        "label": primary_label,
                        "metadata": {
                            "dataset": "tinyvirat",
                            "labels": labels_list,
                            "dim": item.get("dim", []),
                            "video_id": item.get("video_id", ""),
                            "filename": cand_path.name,
                        },
                    })
            except Exception as e:
                logger.error("Error reading TinyVIRAT split JSON %s: %s", split_path, e)
        else:
            # Fallback: glob video directory directly
            videos_dir = dataset_dir / "videos"
            if videos_dir.exists():
                for mp4 in videos_dir.rglob("*.mp4"):
                    samples.append({
                        "path": mp4,
                        "label": mp4.parent.name,
                        "metadata": {
                            "dataset": "tinyvirat",
                            "labels": [mp4.parent.name],
                            "filename": mp4.name,
                        },
                    })

        max_clips = tiny_cfg.get("max_clips", None)
        if max_clips and len(samples) > max_clips:
            samples = samples[:max_clips]

        self.samples = samples
        logger.info("Indexed TinyVIRAT-v2 Dataset: %d clips loaded (split=%s).", len(self.samples), self.split)
