"""Real-time recognition: a sliding window over a live camera (or a looping file).

Three pieces, kept separate so the logic can be tested without a camera or
model weights:

- FrameBuffer holds the last few seconds of frames with their timestamps.
- A worker thread keeps taking the newest window from the buffer, runs each
  model on it and smooths the class probabilities over time.
- The main thread only captures, draws the latest result and shows the frame,
  so the picture stays smooth even when a model takes 100 ms per clip.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .config import ModelSpec
from .models import ActionRecognizer
from .visualize import MODEL_COLORS, _hex_to_rgb

# The checkpoints were trained on ~30 fps video, so "stride 2" means 1/15 s.
TRAINING_FPS = 30.0
BUFFER_SHORT_SIDE = 320   # frames are stored at this size; models crop 224-256 px
LOW_CONFIDENCE = 0.30     # below this the label is drawn dimmed


def window_seconds(spec: ModelSpec, max_window: float) -> float:
    """How much recent video a model looks at.

    The native span is num_frames * sampling_rate at 30 fps (about 2.1 s for
    three of the models, 8.5 s for TimeSformer). It is capped so the prediction
    can follow a change of action within a few seconds.
    """
    return min(spec.num_frames * spec.sampling_rate / TRAINING_FPS, max_window)


class FrameBuffer:
    """Thread-safe store of the most recent `seconds` of frames (BGR, downscaled)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._items: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def push(self, timestamp: float, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        scale = BUFFER_SHORT_SIDE / min(h, w)
        if scale < 1:
            frame_bgr = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        with self._lock:
            self._items.append((timestamp, frame_bgr))
            while self._items[0][0] < timestamp - self.seconds:
                self._items.popleft()

    def window(self, seconds: float, num_frames: int) -> np.ndarray | None:
        """`num_frames` RGB frames evenly spaced in time over the last `seconds`.

        Sampling by timestamp rather than by index keeps the clip's speed right
        whatever frame rate the camera actually delivers. Returns None until
        the buffer covers (nearly) the whole window.
        """
        with self._lock:
            if not self._items:
                return None
            times = np.array([t for t, _ in self._items])
            newest = times[-1]
            if newest - times[0] < seconds * 0.9:
                return None
            targets = np.linspace(newest - seconds, newest, num_frames)
            right = np.searchsorted(times, targets).clip(1, len(times) - 1)
            nearest = np.where(targets - times[right - 1] < times[right] - targets,
                               right - 1, right)
            frames = [self._items[i][1] for i in nearest]
        return np.stack(frames)[..., ::-1].copy()   # BGR -> RGB


class ProbabilitySmoother:
    """Exponential moving average over class probabilities.

    Consecutive windows overlap heavily, yet the top class can still flicker
    between similar actions; averaging trades a little delay for a steady label.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha   # weight of the newest window; 1.0 disables smoothing
        self._probs: torch.Tensor | None = None

    def update(self, probs: torch.Tensor) -> torch.Tensor:
        if self._probs is None:
            self._probs = probs
        else:
            self._probs = self.alpha * probs + (1 - self.alpha) * self._probs
        return self._probs


@dataclass
class LiveResult:
    model: str
    labels: list[str] = field(default_factory=list)   # best first; empty while warming up
    probs: list[float] = field(default_factory=list)
    latency_ms: float = 0.0      # preprocessing + forward for one window
    updates_per_s: float = 0.0


class LiveRecognizer:
    """Runs the models on the newest window in a background thread."""

    def __init__(self, models: dict[str, ActionRecognizer], buffer: FrameBuffer,
                 max_window: float = 3.0, smoothing: float = 0.5, top_k: int = 3):
        self.models = models
        self.buffer = buffer
        self.max_window = max_window
        self.top_k = top_k
        self.error: BaseException | None = None
        self._smoothers = {k: ProbabilitySmoother(smoothing) for k in models}
        self._results = {k: LiveResult(rec.spec.display_name) for k, rec in models.items()}
        self._last_update: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def results(self) -> list[LiveResult]:
        with self._lock:
            return list(self._results.values())

    def step(self) -> bool:
        """Run every model once on its newest window. False if no window was ready."""
        ran = False
        for key, rec in self.models.items():
            frames = self.buffer.window(window_seconds(rec.spec, self.max_window),
                                        rec.spec.num_frames)
            if frames is None:
                continue
            t0 = time.perf_counter()
            # Exactly num_frames frames go in, so preprocess() uses all of them.
            logits = rec.forward(rec.preprocess(frames))
            rec._sync()
            t1 = time.perf_counter()
            probs = self._smoothers[key].update(logits.float().softmax(-1))
            values, indices = probs.topk(self.top_k)
            previous = self._last_update.get(key)
            self._last_update[key] = t1
            with self._lock:
                self._results[key] = LiveResult(
                    model=rec.spec.display_name,
                    labels=[rec.labels[i] for i in indices.tolist()],
                    probs=values.tolist(),
                    latency_ms=(t1 - t0) * 1000,
                    updates_per_s=1 / (t1 - previous) if previous else 0.0,
                )
            ran = True
        return ran

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self.step():
                    time.sleep(0.05)   # buffer still filling
        except BaseException as exc:   # surfaced by the main loop
            self.error = exc


def draw_overlay(img: np.ndarray, results: list[LiveResult], status: str = "") -> np.ndarray:
    """Draw every model's current prediction onto a BGR frame (in place).

    One model: its top-k classes with confidence bars. Several models: one line
    each with the top-1 class, so they can be compared at a glance.
    """
    h, w = img.shape[:2]
    scale = max(0.6, w / 1000)
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(1.5 * scale))
    line_h, pad = int(30 * scale), int(10 * scale)
    single = len(results) == 1
    n_lines = 1 + len(results[0].probs or [0]) if single else len(results)
    box_w, box_h = min(w, int(600 * scale)), pad * 2 + line_h * n_lines

    panel = img[:box_h, :box_w]
    panel[:] = (panel * 0.35 + 20 * 0.65).astype(np.uint8)   # darken behind the text

    def put(text, x, y, color=(255, 255, 255), size=0.6):
        cv2.putText(img, text, (x, y), font, size * scale, color, thick, cv2.LINE_AA)

    y = pad + line_h - int(8 * scale)
    for r in results:
        rc, gc, bc = _hex_to_rgb(MODEL_COLORS.get(r.model, "#2a78d6"))
        color = (bc, gc, rc)
        cv2.circle(img, (pad + int(6 * scale), y - int(7 * scale)), int(6 * scale), color, -1)
        x = pad + int(20 * scale)
        if not r.labels:
            put(f"{r.model}: collecting frames...", x, y, (170, 170, 170))
            y += line_h
            continue
        speed = f"{r.latency_ms:.0f} ms"
        if single:
            put(f"{r.model}  ({speed}, {r.updates_per_s:.0f} updates/s)", x, y)
            y += line_h
            bar_x, bar_w = pad, int(110 * scale)
            for label, prob in zip(r.labels, r.probs):
                top = y - int(16 * scale)
                cv2.rectangle(img, (bar_x, top), (bar_x + bar_w, top + int(16 * scale)),
                              (70, 70, 70), -1)
                cv2.rectangle(img, (bar_x, top),
                              (bar_x + int(bar_w * prob), top + int(16 * scale)), color, -1)
                ink = (255, 255, 255) if prob >= LOW_CONFIDENCE else (170, 170, 170)
                put(f"{prob * 100:3.0f}%  {label}", bar_x + bar_w + pad, y, ink)
                y += line_h
        else:
            ink = (255, 255, 255) if r.probs[0] >= LOW_CONFIDENCE else (170, 170, 170)
            put(f"{r.model}: {r.labels[0]} {r.probs[0] * 100:.0f}%  [{speed}]", x, y, ink)
            y += line_h

    if status:
        (tw, th), _ = cv2.getTextSize(status, font, 0.5 * scale, thick)
        cv2.rectangle(img, (0, h - th - 2 * pad), (tw + 2 * pad, h), (20, 20, 20), -1)
        put(status, pad, h - pad, (220, 220, 220), size=0.5)
    return img


class FrameSource:
    """A camera, or a video file played in a loop at its own frame rate.

    The file mode exists so the live pipeline can be tried (and demoed) on a
    machine without a camera.
    """

    def __init__(self, source: str, width: int = 640, height: int = 480):
        self.is_camera = source.isdigit()
        self.cap = cv2.VideoCapture(int(source) if self.is_camera else source)
        if not self.cap.isOpened():
            if self.is_camera:
                raise RuntimeError(
                    f"Cannot open camera {source}. Check that a camera is connected and "
                    "enabled (ls /dev/video*), that no other program is using it, or "
                    "pick another one with --source 1.")
            raise RuntimeError(f"Cannot open video: {source}")
        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._next_due = time.perf_counter()

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if self.is_camera:
            return frame if ok else None
        if not ok:   # end of file: rewind
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        # A file decodes faster than real time; wait so it plays at its own fps.
        self._next_due = max(self._next_due + 1 / self.fps, time.perf_counter() - 0.1)
        time.sleep(max(0.0, self._next_due - time.perf_counter()))
        return frame

    def release(self):
        self.cap.release()
