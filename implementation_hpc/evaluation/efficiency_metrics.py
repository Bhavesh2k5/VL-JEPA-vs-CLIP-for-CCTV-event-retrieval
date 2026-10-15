import time
from typing import Any, Dict, List

import numpy as np
import torch

from utils.common import clear_gpu_cache


def measure_encoding_latency(encoder: Any, sample_videos: torch.Tensor, n_runs: int = 50) -> Dict[str, float]:
    """
    Measures per-clip video encoding latency over n_runs on the target hardware.
    """
    clear_gpu_cache()
    # Warmup
    for _ in range(5):
        _ = encoder.encode_video(sample_videos[:1])

    latencies = []
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(n_runs):
            start_event.record()
            _ = encoder.encode_video(sample_videos[:1])
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))  # ms
    else:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = encoder.encode_video(sample_videos[:1])
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    return {
        "mean_latency_ms": round(float(np.mean(lat_arr)), 2),
        "std_latency_ms": round(float(np.std(lat_arr)), 2),
        "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
    }


def measure_memory_usage(encoder: Any, sample_video: torch.Tensor) -> Dict[str, float]:
    """Measures peak GPU memory allocated during video encoding."""
    if not torch.cuda.is_available():
        return {"peak_gpu_mem_mb": 0.0, "reserved_gpu_mem_mb": 0.0}

    clear_gpu_cache()
    torch.cuda.reset_peak_memory_stats()

    _ = encoder.encode_video(sample_video)
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        "peak_gpu_mem_mb": round(float(peak_alloc), 2),
        "reserved_gpu_mem_mb": round(float(peak_res), 2),
    }


def measure_throughput(encoder: Any, video_batch: torch.Tensor, n_repeats: int = 10) -> float:
    """Measures processing throughput in clips per second."""
    batch_size = video_batch.shape[0]
    clear_gpu_cache()

    # Warmup
    _ = encoder.encode_video(video_batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = encoder.encode_video(video_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    throughput = (batch_size * n_repeats) / max(1e-6, elapsed)
    return round(throughput, 2)
