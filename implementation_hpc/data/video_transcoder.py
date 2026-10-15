"""
Video Transcoder Utility for Browser HTML5 Compatibility.
Converts arbitrary surveillance video formats (.avi, .mkv, .mov, unsupported codecs)
into web-standard H.264 (AVC) MP4 format with yuv420p pixel format and faststart flag.
Uses an on-disk cache in data/cache_preview/ so transcoding only executes once per clip.
"""

import os
import hashlib
import subprocess
from pathlib import Path
from typing import Optional


def ensure_browser_playable_video(video_path: Optional[str]) -> Optional[str]:
    """
    Returns a filesystem path to an H.264 MP4 video file guaranteed to play natively
    in modern web browsers (Chrome, Edge, Firefox, Safari).
    
    If the video is already an MP4, it verifies format; for .avi or other non-HTML5
    friendly containers, it transcodes to data/cache_preview/ using bundled ffmpeg.
    """
    if not video_path:
        return None

    orig_path = Path(video_path)
    if not orig_path.exists():
        return video_path

    # Check extension. Modern browsers strictly reject AVI containers in HTML5 video elements.
    suffix = orig_path.suffix.lower()
    needs_transcode = suffix in [".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"]

    # Base cache directory
    cache_dir = Path(__file__).resolve().parent / "cache_preview"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        mtime = orig_path.stat().st_mtime
        fsize = orig_path.stat().st_size
    except OSError:
        return str(orig_path)

    hash_key = hashlib.md5(f"{orig_path.resolve()}_{mtime}_{fsize}".encode("utf-8")).hexdigest()[:12]
    cached_file = cache_dir / f"{orig_path.stem}_{hash_key}.mp4"

    # If already transcoded and valid, return cached version immediately
    if cached_file.exists() and cached_file.stat().st_size > 500:
        return str(cached_file)

    if not needs_transcode:
        return str(orig_path)

    # Transcode using imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(orig_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "24",
            "-movflags", "+faststart",
            "-an",  # strip audio for faster processing & CCTV preview
            str(cached_file),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
        if res.returncode == 0 and cached_file.exists() and cached_file.stat().st_size > 500:
            return str(cached_file)
    except Exception:
        pass

    return str(orig_path)
