import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import cv2

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def test_config(tmp_path):
    """Generates self-contained test configuration pointing to temporary fixtures."""
    return {
        "system": {
            "device": "cpu",
            "fp16": False,
            "seed": 42,
            "num_workers": 0,
        },
        "data": {
            "root": str(tmp_path),
            "rlvs": {
                "path": "rlvs_mock",
                "classes": ["Violence", "NonViolence"],
                "extensions": ["*.mp4", "*.avi"],
            },
            "ucf_arg": {
                "path": "ucf_arg_mock",
                "viewpoints": ["aerial_clips", "ground_clips", "rooftop_clips"],
                "classes": ["boxing", "running"],
            },
            "tinyvirat": {
                "path": "tinyvirat_mock",
                "class_map": "class_map.json",
                "splits": {"train": "tiny_train_v2.json"},
            },
        },
        "video": {
            "num_frames": 4,
            "frame_size": 64,
            "fps": 30,
            "decode_backend": "opencv",
        },
        "models": {
            "clip": {
                "primary": "openai/clip-vit-base-patch32",
                "batch_size": 2,
            },
            "vljepa": {
                "visual_backbone": "qubvel-hf/vjepa2-vitl-fpc16-256-ssv2",
                "visual_dim": 1024,
                "text_dim": 768,
                "input_size": 64,
                "num_frames": 4,
            },
            "predictor": {
                "d_model": 64,
                "shared_dim": 32,
                "num_layers": 1,
                "num_heads": 2,
                "dim_feedforward": 128,
                "dropout": 0.0,
                "temperature": 0.07,
            },
        },
        "retrieval": {
            "index_type": "FlatIP",
            "top_k": 3,
            "embedding_save_dir": str(tmp_path / "embeddings"),
            "index_save_dir": str(tmp_path / "indices"),
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "save_dir": str(tmp_path / "checkpoints"),
        },
        "evaluation": {
            "latency_runs": 2,
            "output_dir": str(tmp_path / "metrics"),
        },
    }


def create_dummy_video(path: Path, num_frames: int = 10, width: int = 64, height: int = 64):
    """Creates a small valid video file using OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if path.suffix.lower() == ".mp4" else cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
