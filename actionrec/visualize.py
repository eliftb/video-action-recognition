"""Result visualisation: a summary figure (PNG) and an annotated video (MP4)."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # render to file; works over SSH and without a display
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import BenchmarkResult
from .models import Prediction

# One fixed colour per model, used in every panel so identity never shifts.
MODEL_COLORS = {
    "SlowFast R50": "#2a78d6",
    "VideoMAE-B": "#eb6834",
    "TimeSformer-B": "#1baf7a",
    "I3D R50": "#eda100",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _hbar(ax, labels, values, colors, fmt, xmax=None):
    """Horizontal bars, largest on top, value printed at the end of each bar."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    xmax = xmax or max(values) * 1.25
    ax.set_xlim(0, xmax)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.015, yi, fmt(v), va="center", fontsize=9, color=INK)
    _style_axis(ax)


def summary_figure(frames: np.ndarray, predictions: list[Prediction],
                   benchmarks: list[BenchmarkResult] | None, video_name: str,
                   out_path: Path) -> Path:
    """Sampled frames on top, each model's top-5 in the middle, speed/accuracy
    comparison at the bottom (skipped if no benchmarks were run)."""
    n_models = len(predictions)
    rows = 3 if benchmarks else 2
    fig = plt.figure(figsize=(16, 2.6 + 3.4 * (rows - 1)), facecolor=SURFACE)
    grid = fig.add_gridspec(rows, 1, height_ratios=[0.8, 1.3, 1.2][:rows], hspace=0.4)
    fig.suptitle(f"Action recognition: {video_name}", fontsize=15, color=INK,
                 x=0.06, ha="left", y=0.995)

    # Row 1: evenly spaced frames from the video.
    n_thumbs = 6
    thumbs = grid[0].subgridspec(1, n_thumbs, wspace=0.04)
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, n_thumbs).astype(int)):
        ax = fig.add_subplot(thumbs[i])
        ax.imshow(frames[idx])
        ax.set_title(f"frame {idx}", fontsize=8, color=INK_SECONDARY)
        ax.axis("off")

    # Row 2: top-5 classes per model (small multiples, same x scale).
    panels = grid[1].subgridspec(1, n_models, wspace=0.9)
    for i, p in enumerate(predictions):
        ax = fig.add_subplot(panels[i])
        color = MODEL_COLORS.get(p.model, "#2a78d6")
        labels = [_shorten(lbl) for lbl in p.top_labels]
        _hbar(ax, labels, [v * 100 for v in p.top_probs], color,
              lambda v: f"{v:.1f}%", xmax=125)
        ax.set_title(p.model, fontsize=11, color=INK, loc="left")
        ax.set_xticks([0, 50, 100])

    # Row 3: speed, memory and published accuracy, one axis per measure.
    if benchmarks:
        names = [b.model for b in benchmarks]
        colors = [MODEL_COLORS.get(n, "#2a78d6") for n in names]
        cmp = grid[2].subgridspec(1, 3, wspace=0.55)
        measures = [
            ("Clips per second (higher is faster)", [b.clips_per_s for b in benchmarks],
             lambda v: f"{v:.1f}"),
            ("Weights + activation memory, MB", [b.weights_mb + b.peak_mem_mb for b in benchmarks],
             lambda v: f"{v:.0f}"),
            ("Published Kinetics-400 top-1, %", [b.k400_top1 for b in benchmarks],
             lambda v: f"{v:.1f}"),
        ]
        for i, (title, values, fmt) in enumerate(measures):
            ax = fig.add_subplot(cmp[i])
            _hbar(ax, names, values, colors, fmt,
                  xmax=100 if "top-1" in title else None)
            ax.set_title(title, fontsize=10, color=INK, loc="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def annotated_video(frames: np.ndarray, fps: float, predictions: list[Prediction],
                    out_path: Path) -> Path:
    """Write the input video with every model's top-1 prediction overlaid."""
    h, w = frames.shape[1:3]
    scale = max(0.45, w / 1100)
    line_h = int(30 * scale)
    pad = int(10 * scale)
    box_h = pad * 2 + line_h * len(predictions)
    box_w = int(w * 0.62)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (20, 20, 20), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        for i, p in enumerate(predictions):
            y = pad + line_h * (i + 1) - int(8 * scale)
            r, g, b = _hex_to_rgb(MODEL_COLORS.get(p.model, "#2a78d6"))
            cv2.circle(img, (pad + int(6 * scale), y - int(7 * scale)), int(6 * scale),
                       (b, g, r), -1)
            text = f"{p.model}: {p.label} ({p.confidence * 100:.0f}%)"
            cv2.putText(img, text, (pad + int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, (255, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)
        writer.write(img)
    writer.release()
    return out_path


def _shorten(label: str, n: int = 22) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
