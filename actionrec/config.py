"""Model registry: input specs and published reference accuracy for each model.

All four checkpoints are trained on Kinetics-400, so they share one label space
and can be compared directly on the same clip.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"


def use_project_cache() -> None:
    """Keep model weights inside the project folder instead of ~/.cache.

    Must run before torch.hub / transformers download anything. Variables the
    user has already set are respected.
    """
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # short CLI name
    display_name: str
    source: str              # "pytorchvideo" (torch.hub) or "huggingface"
    checkpoint: str          # hub entry point or HF repo id
    num_frames: int          # frames fed to the network
    sampling_rate: int       # stride between sampled frames in the source video
    short_side: int          # resize shorter side to this before cropping
    crop_size: int
    mean: tuple
    std: tuple
    # Published Kinetics-400 accuracy. These come from multi-view testing
    # (several temporal clips x spatial crops), whereas this demo uses one
    # centre clip, so expect slightly lower numbers if you measure yourself.
    k400_top1: float
    k400_top5: float
    accuracy_source: str
    slowfast_alpha: int = 0  # >0 only for SlowFast: fast/slow frame ratio


KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELS = {
    "slowfast": ModelSpec(
        key="slowfast",
        display_name="SlowFast R50",
        source="pytorchvideo",
        checkpoint="slowfast_r50",
        num_frames=32, sampling_rate=2,  # fast pathway 32 frames, slow = 32/alpha
        short_side=256, crop_size=256,
        mean=KINETICS_MEAN, std=KINETICS_STD,
        k400_top1=76.94, k400_top5=92.69,
        accuracy_source="PyTorchVideo model zoo, 8x8, 3x10 views",
        slowfast_alpha=4,
    ),
    "i3d": ModelSpec(
        key="i3d",
        display_name="I3D R50",
        source="pytorchvideo",
        checkpoint="i3d_r50",
        num_frames=8, sampling_rate=8,
        short_side=256, crop_size=224,
        mean=KINETICS_MEAN, std=KINETICS_STD,
        k400_top1=73.27, k400_top5=90.70,
        accuracy_source="PyTorchVideo model zoo, 8x8, 3x10 views",
    ),
    "videomae": ModelSpec(
        key="videomae",
        display_name="VideoMAE-B",
        source="huggingface",
        checkpoint="MCG-NJU/videomae-base-finetuned-kinetics",
        num_frames=16, sampling_rate=4,
        short_side=224, crop_size=224,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
        k400_top1=80.9, k400_top5=94.7,
        accuracy_source="Hugging Face model card (MCG-NJU)",
    ),
    "timesformer": ModelSpec(
        key="timesformer",
        display_name="TimeSformer-B",
        source="huggingface",
        checkpoint="facebook/timesformer-base-finetuned-k400",
        num_frames=8, sampling_rate=32,
        short_side=224, crop_size=224,
        mean=KINETICS_MEAN, std=KINETICS_STD,
        k400_top1=77.9, k400_top5=93.2,
        accuracy_source="facebookresearch/TimeSformer model zoo, 8x32, 1x3 views",
    ),
}

DEFAULT_ORDER = ["slowfast", "videomae", "timesformer", "i3d"]

# Public sample clips, downloaded on first use instead of shipped in the repo.
SAMPLE_VIDEOS = {
    "archery.mp4": "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4",
    "eating_spaghetti.mp4":
        "https://huggingface.co/datasets/nielsr/video-demo/resolve/main/eating_spaghetti.mp4",
}
