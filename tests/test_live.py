"""Checks for the live sliding-window logic; no camera or model weights needed."""

import numpy as np
import torch

from actionrec.config import MODELS
from actionrec.live import (FrameBuffer, LiveResult, ProbabilitySmoother, draw_overlay,
                            window_seconds)


def numbered_frame(i, h=24, w=32):
    """A frame whose every pixel holds its own index, so sampling can be checked."""
    return np.full((h, w, 3), i, dtype=np.uint8)


def filled_buffer(seconds=4.0, fps=30.0, duration=4.0):
    buf = FrameBuffer(seconds)
    for i in range(int(duration * fps) + 1):
        buf.push(i / fps, numbered_frame(i))
    return buf


def test_window_is_capped_for_long_native_spans():
    assert window_seconds(MODELS["timesformer"], max_window=3.0) == 3.0   # native 8.5 s
    assert abs(window_seconds(MODELS["slowfast"], max_window=3.0) - 64 / 30) < 1e-9


def test_buffer_returns_nothing_until_window_is_covered():
    buf = FrameBuffer(seconds=4.0)
    assert buf.window(2.0, 8) is None
    for i in range(30):                      # one second of frames
        buf.push(i / 30, numbered_frame(i))
    assert buf.window(2.0, 8) is None
    for i in range(30, 70):
        buf.push(i / 30, numbered_frame(i))
    assert buf.window(2.0, 8).shape == (8, 24, 32, 3)


def test_window_samples_evenly_in_time_and_ends_at_newest_frame():
    frames = filled_buffer().window(2.0, 9)  # frames 0..120 at 30 fps; last 2 s = 60..120
    ids = frames[:, 0, 0, 0].astype(int)
    assert ids[0] == 60 and ids[-1] == 120
    assert np.all(np.abs(np.diff(ids) - 7.5) <= 1)


def test_window_follows_time_not_frame_count():
    """A 10 fps camera must still yield a 2 s window, not 2 s worth of 30 fps indices."""
    buf = FrameBuffer(seconds=4.0)
    for i in range(41):
        buf.push(i / 10, numbered_frame(i))
    ids = buf.window(2.0, 5)[:, 0, 0, 0].astype(int)
    assert list(ids) == [20, 25, 30, 35, 40]


def test_buffer_drops_old_frames_and_downscales():
    buf = FrameBuffer(seconds=1.0)
    for i in range(100):
        buf.push(i / 10, np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(buf._items) == 11
    assert buf._items[0][1].shape == (320, 427, 3)


def test_window_converts_bgr_to_rgb():
    buf = FrameBuffer(seconds=2.0)
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    bgr[..., 0] = 255                        # pure blue in BGR
    for i in range(40):
        buf.push(i / 30, bgr)
    assert tuple(buf.window(1.0, 4)[0, 0, 0]) == (0, 0, 255)


def test_smoother_averages_and_can_be_disabled():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    s = ProbabilitySmoother(alpha=0.5)
    assert torch.equal(s.update(a), a)
    assert torch.allclose(s.update(b), torch.tensor([0.5, 0.5]))
    off = ProbabilitySmoother(alpha=1.0)
    off.update(a)
    assert torch.equal(off.update(b), b)


def test_overlay_draws_for_one_model_several_models_and_warm_up():
    ready = LiveResult("VideoMAE-B", ["archery", "golf", "bowling"], [0.7, 0.2, 0.05], 95, 9)
    warming = LiveResult("I3D R50")
    for results in ([ready], [ready, warming], [warming]):
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        out = draw_overlay(img, results, "30 fps")
        assert out.shape == (480, 640, 3)
        assert (out != 128).any()
