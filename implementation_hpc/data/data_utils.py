import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoDecodeError(Exception):
    """Raised when video cannot be decoded by any available backend."""
    pass


def _uniform_indices(total_frames: int, num_frames: int) -> List[int]:
    """Generates uniformly distributed frame indices across total length."""
    if total_frames <= 0:
        return [0] * num_frames
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return indices
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def _read_with_opencv(video_path: str, num_frames: int, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Decodes video with OpenCV using direct seeking for high performance."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoDecodeError(f"OpenCV failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_indices(total_frames, num_frames)

    frames = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback: re-read current frame or last valid
            if frames:
                frame = frames[-1].copy()
            else:
                cap.release()
                raise VideoDecodeError(f"Could not read frame {idx} from {video_path}")
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None and (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
        prev_idx = idx

    cap.release()
    if len(frames) != num_frames:
        raise VideoDecodeError(f"Expected {num_frames} frames, obtained {len(frames)}")
    return np.stack(frames, axis=0)  # [T, H, W, C]


def sample_frames(
    video_path: Union[str, Path],
    num_frames: int = 16,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    backend: str = "auto",
) -> np.ndarray:
    """Extracts uniformly sampled frames from a video file."""
    vpath = str(video_path)
    if not Path(vpath).exists():
        raise FileNotFoundError(f"Video file does not exist: {vpath}")

    # Attempt decord if installed and requested
    if backend in ("auto", "decord"):
        try:
            import decord
            decord.bridge.set_bridge("numpy")
            vr = decord.VideoReader(vpath)
            total = len(vr)
            idxs = _uniform_indices(total, num_frames)
            raw_frames = vr.get_batch(idxs).asnumpy()
            if target_size is not None:
                resized = []
                for f in raw_frames:
                    if (f.shape[1], f.shape[0]) != target_size:
                        f = cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR)
                    resized.append(f)
                return np.stack(resized, axis=0)
            return raw_frames
        except Exception as e:
            if backend == "decord":
                raise VideoDecodeError(f"Decord failed on {vpath}: {e}")

    # Robust OpenCV fallback
    try:
        return _read_with_opencv(vpath, num_frames=num_frames, target_size=target_size)
    except Exception as e:
        raise VideoDecodeError(f"All backends failed for {vpath}: {e}")


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Converts numpy array [T, H, W, C] in [0, 255] to PyTorch tensor [T, C, H, W] in [0.0, 1.0]."""
    tensor = torch.from_numpy(frames).float() / 255.0
    return tensor.permute(0, 3, 1, 2)


def normalize_frames(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Normalizes PyTorch video tensor [..., 3, H, W] using mean and std."""
    device = tensor.device
    dtype = tensor.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def collate_video_batch(batch: List[dict]) -> dict:
    """
    Robust collate function for video batches.
    Handles variable spatial dimensions by center-padding or standardizing,
    preventing torch.stack() crashes on TinyVIRAT-v2.
    """
    valid_items = [item for item in batch if item is not None and "video" in item]
    if not valid_items:
        return {}

    # Standardize spatial dimensions to max in batch if they differ
    sample_shape = valid_items[0]["video"].shape  # [T, C, H, W]
    all_same = all(item["video"].shape == sample_shape for item in valid_items)

    if not all_same:
        max_h = max(item["video"].shape[-2] for item in valid_items)
        max_w = max(item["video"].shape[-1] for item in valid_items)
        standardized_videos = []
        for item in valid_items:
            v = item["video"]  # [T, C, H, W]
            h, w = v.shape[-2], v.shape[-1]
            if (h, w) != (max_h, max_w):
                pad_top = (max_h - h) // 2
                pad_bottom = max_h - h - pad_top
                pad_left = (max_w - w) // 2
                pad_right = max_w - w - pad_left
                v = F.pad(v, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            standardized_videos.append(v)
        videos = torch.stack(standardized_videos, dim=0)
    else:
        videos = torch.stack([item["video"] for item in valid_items], dim=0)

    paths = [item["path"] for item in valid_items]
    labels = [item["label"] for item in valid_items]
    metadata = [item.get("metadata", {}) for item in valid_items]

    return {
        "video": videos,      # [B, T, C, H, W]
        "path": paths,
        "label": labels,
        "metadata": metadata,
    }
