import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from utils.common import get_dataset_root
from .base_dataset import BaseVideoDataset

logger = logging.getLogger(__name__)


class UCFArgDataset(BaseVideoDataset):
    """
    UCF-ARG Multi-View Action Dataset loader.
    Handles nested viewpoint folders (aerial, ground, rooftop) and supports viewpoint filtering for H2.
    """

    def __init__(self, config: Dict[str, Any], split: str = "all", viewpoint_filter: Optional[str] = None):
        self.viewpoint_filter = viewpoint_filter.lower() if viewpoint_filter else None
        super().__init__(config, split=split)

    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """Parses person<ActorID>_<RunID>_<Viewpoint>_<Action>.avi."""
        stem = Path(filename).stem
        parts = stem.split("_")
        meta = {"actor_id": "unknown", "run_id": "unknown", "viewpoint": "unknown", "action": "unknown"}
        if len(parts) >= 4:
            meta["actor_id"] = parts[0]
            meta["run_id"] = parts[1]
            meta["viewpoint"] = parts[2]
            meta["action"] = parts[3]
        return meta

    def _build_index(self) -> None:
        data_root = get_dataset_root(self.config)
        ucf_cfg = self.config["data"]["ucf_arg"]
        rel_path = ucf_cfg["path"]

        dataset_dir = data_root / rel_path
        if not dataset_dir.exists():
            logger.warning("UCF-ARG directory not found at %s. Initializing empty dataset.", dataset_dir)
            return

        target_viewpoints = ucf_cfg.get("viewpoints", ["aerial_clips", "ground_clips", "rooftop_clips"])
        if self.viewpoint_filter:
            target_viewpoints = [vp for vp in target_viewpoints if self.viewpoint_filter in vp.lower()]

        classes = ucf_cfg.get("classes", [])
        max_clips = ucf_cfg.get("max_clips", None)

        samples = []
        for vp in target_viewpoints:
            vp_dir = dataset_dir / vp
            if not vp_dir.exists():
                continue

            # Support nested viewpoint folder structure: e.g. aerial_clips/aerial_clips/...
            nested_candidate = vp_dir / vp
            search_dir = nested_candidate if nested_candidate.exists() else vp_dir

            # Search across all avi files
            for avi_file in search_dir.rglob("*.avi"):
                file_meta = self._parse_filename(avi_file.name)
                # Determine action label from folder name or filename
                action = file_meta.get("action", "").lower()
                if not action or action == "unknown":
                    action = avi_file.parent.name.lower()

                # Infer viewpoint
                vp_clean = "aerial" if "aerial" in vp else ("rooftop" if "rooftop" in vp else "ground")

                samples.append({
                    "path": avi_file,
                    "label": action,
                    "metadata": {
                        "dataset": "ucf_arg",
                        "viewpoint": vp_clean,
                        "raw_viewpoint": vp,
                        "action": action,
                        "actor_id": file_meta.get("actor_id", "unknown"),
                        "filename": avi_file.name,
                    },
                })

        # Deterministic sorting
        samples = sorted(samples, key=lambda s: str(s["path"]))
        if max_clips and len(samples) > max_clips:
            samples = samples[:max_clips]

        self.samples = samples
        logger.info(
            "Indexed UCF-ARG Dataset: %d clips loaded (viewpoint_filter=%s, path=%s).",
            len(self.samples), self.viewpoint_filter, dataset_dir,
        )
