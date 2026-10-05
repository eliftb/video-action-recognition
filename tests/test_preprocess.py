"""Fast checks for the preprocessing pipeline; no model weights needed."""

import numpy as np
import pytest
import torch

from actionrec.config import MODELS
from actionrec.models import load_kinetics_labels
from actionrec.preprocess import frames_to_clip, to_model_input
from actionrec.video import clip_indices


def fake_video(t=300, h=240, w=320):
    return np.random.default_rng(0).integers(0, 256, (t, h, w, 3), dtype=np.uint8)


def test_clip_indices_centred_with_stride():
    idx = clip_indices(total_frames=100, num_frames=8, sampling_rate=8)
    assert len(idx) == 8
    assert np.all(np.diff(idx) == 8)
    assert idx[0] == (100 - 64) // 2


def test_clip_indices_short_video_spreads_over_whole_video():
    idx = clip_indices(total_frames=20, num_frames=16, sampling_rate=4)
    assert len(idx) == 16
    assert idx[0] == 0 and idx[-1] == 19
    assert np.all(np.diff(idx) >= 0)


@pytest.mark.parametrize("key", list(MODELS))
def test_clip_shape_and_normalisation(key):
    spec = MODELS[key]
    clip = frames_to_clip(fake_video(), spec)
    assert clip.shape == (3, spec.num_frames, spec.crop_size, spec.crop_size)
    # Uniform noise in [0, 1] normalised with mean ~0.45 should be centred near 0.
    assert abs(clip.mean().item()) < 0.3


def test_slowfast_input_has_two_pathways():
    spec = MODELS["slowfast"]
    slow, fast = to_model_input(frames_to_clip(fake_video(), spec), spec)
    assert fast.shape[2] == spec.num_frames
    assert slow.shape[2] == spec.num_frames // spec.slowfast_alpha


def test_huggingface_input_is_time_major():
    spec = MODELS["videomae"]
    inputs = to_model_input(frames_to_clip(fake_video(), spec), spec)
    assert inputs["pixel_values"].shape == (1, spec.num_frames, 3, 224, 224)


def test_i3d_input_is_channel_major():
    spec = MODELS["i3d"]
    x = to_model_input(frames_to_clip(fake_video(), spec), spec)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 3, spec.num_frames, 224, 224)


def test_kinetics_labels_complete():
    labels = load_kinetics_labels()
    assert len(labels) == 400
    assert all(labels)
    assert "archery" in labels
