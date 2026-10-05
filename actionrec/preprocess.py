"""Shared preprocessing: frames -> normalised tensor in the layout each model expects.

Every model gets the same steps (sample clip, resize shorter side, centre crop,
normalise); only the numbers in its ModelSpec differ. Doing it in torch on the
target device keeps preprocessing cheap and identical across models.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelSpec
from .video import clip_indices


def frames_to_clip(frames: np.ndarray, spec: ModelSpec, device="cpu") -> torch.Tensor:
    """(T, H, W, 3) uint8 video -> float tensor (C, T, crop, crop), normalised."""
    idx = clip_indices(len(frames), spec.num_frames, spec.sampling_rate)
    clip = torch.from_numpy(frames[idx]).to(device)            # T,H,W,C uint8
    clip = clip.permute(0, 3, 1, 2).float() / 255.0           # T,C,H,W in [0,1]

    # Resize so the shorter side matches spec.short_side, keeping aspect ratio.
    h, w = clip.shape[-2:]
    scale = spec.short_side / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    clip = F.interpolate(clip, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Centre crop.
    top = (new_h - spec.crop_size) // 2
    left = (new_w - spec.crop_size) // 2
    clip = clip[..., top:top + spec.crop_size, left:left + spec.crop_size]

    mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    return clip.permute(1, 0, 2, 3).contiguous()               # C,T,H,W


def to_model_input(clip: torch.Tensor, spec: ModelSpec):
    """Wrap a (C, T, H, W) clip into the batch format each architecture wants."""
    if spec.source == "huggingface":
        # transformers video models take pixel_values as (B, T, C, H, W).
        return {"pixel_values": clip.permute(1, 0, 2, 3).unsqueeze(0)}
    if spec.slowfast_alpha:
        # SlowFast: the fast pathway sees every sampled frame, the slow one
        # every alpha-th frame.
        fast = clip
        slow_idx = torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // spec.slowfast_alpha)
        slow = clip.index_select(1, slow_idx.long().to(clip.device))
        return [slow.unsqueeze(0), fast.unsqueeze(0)]
    return clip.unsqueeze(0)                                   # B,C,T,H,W
