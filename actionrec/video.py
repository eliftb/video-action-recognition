"""Video I/O: read frames, pick clip indices, fetch videos from URLs."""

import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def read_video(path) -> tuple[np.ndarray, float]:
    """Decode a whole video into a uint8 RGB array of shape (T, H, W, 3).

    Returns the frames and the video's FPS. Demo clips are a few seconds long,
    so keeping everything in memory is simpler than streaming.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames), fps


def clip_indices(total_frames: int, num_frames: int, sampling_rate: int) -> np.ndarray:
    """Indices of one clip of `num_frames` frames spaced `sampling_rate` apart,
    centred in the video.

    If the video is shorter than the clip span, the frames are spread evenly
    over the whole video instead (some may repeat).
    """
    span = num_frames * sampling_rate
    if total_frames >= span:
        start = (total_frames - span) // 2
        return start + np.arange(num_frames) * sampling_rate
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int)


def fetch_video(url: str, dest_dir: Path) -> Path:
    """Download a video from a direct link, or from YouTube if yt-dlp is installed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "youtube.com" in url or "youtu.be" in url:
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("YouTube links need yt-dlp: pip install yt-dlp")
        out = dest_dir / "youtube_%(id)s.%(ext)s"
        # A short low-res mp4 is enough for recognition and fast to download.
        subprocess.run(
            ["yt-dlp", "-f", "mp4[height<=480]/best[height<=480]", "--download-sections",
             "*0-15", "-o", str(out), "--print", "after_move:filepath", url],
            check=True, capture_output=True, text=True,
        )
        return max(dest_dir.glob("youtube_*.mp4"), key=lambda p: p.stat().st_mtime)
    target = dest_dir / Path(url.split("?")[0]).name
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target
