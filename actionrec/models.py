"""Load the four pretrained models behind one common interface."""

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from .config import ASSETS_DIR, MODELS, ModelSpec, use_project_cache
from .preprocess import frames_to_clip, to_model_input

use_project_cache()


def load_kinetics_labels() -> list[str]:
    """Kinetics-400 class names indexed by the PyTorchVideo label ids."""
    with open(ASSETS_DIR / "kinetics_classnames.json") as f:
        name_to_id = json.load(f)
    labels = [""] * len(name_to_id)
    for name, idx in name_to_id.items():
        labels[idx] = name.strip('"').replace("_", " ")
    return labels


@dataclass
class Prediction:
    model: str
    top_labels: list[str]      # best first
    top_probs: list[float]
    preprocess_ms: float
    inference_ms: float

    @property
    def label(self) -> str:
        return self.top_labels[0]

    @property
    def confidence(self) -> float:
        return self.top_probs[0]


class ActionRecognizer:
    """One pretrained Kinetics-400 model plus its preprocessing."""

    def __init__(self, spec: ModelSpec, device: str = "cuda", half: bool = False):
        self.spec = spec
        self.device = device
        self.dtype = torch.float16 if half and device.startswith("cuda") else torch.float32

        if spec.source == "pytorchvideo":
            self.model = torch.hub.load("facebookresearch/pytorchvideo", spec.checkpoint,
                                        pretrained=True, verbose=False)
            self.labels = load_kinetics_labels()
        else:
            from transformers import AutoModelForVideoClassification
            self.model = AutoModelForVideoClassification.from_pretrained(spec.checkpoint)
            id2label = self.model.config.id2label
            self.labels = [id2label[i] for i in range(len(id2label))]

        self.model = self.model.to(device=device, dtype=self.dtype).eval()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def preprocess(self, frames: np.ndarray):
        clip = frames_to_clip(frames, self.spec, self.device).to(self.dtype)
        return to_model_input(clip, self.spec)

    @torch.inference_mode()
    def forward(self, inputs) -> torch.Tensor:
        """Raw logits for one preprocessed clip, shape (num_classes,)."""
        if isinstance(inputs, dict):
            return self.model(**inputs).logits[0]
        return self.model(inputs)[0]

    def predict(self, frames: np.ndarray, top_k: int = 5) -> Prediction:
        """Video frames -> top-k Kinetics-400 actions with probabilities."""
        t0 = time.perf_counter()
        inputs = self.preprocess(frames)
        self._sync()
        t1 = time.perf_counter()
        logits = self.forward(inputs)
        self._sync()
        t2 = time.perf_counter()

        probs = logits.float().softmax(-1)
        values, indices = probs.topk(top_k)
        return Prediction(
            model=self.spec.display_name,
            top_labels=[self.labels[i] for i in indices.tolist()],
            top_probs=values.tolist(),
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
        )

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


def load_models(keys, device="cuda", half=False, log=print) -> dict[str, ActionRecognizer]:
    models = {}
    for key in keys:
        spec = MODELS[key]
        t0 = time.perf_counter()
        models[key] = ActionRecognizer(spec, device=device, half=half)
        log(f"  loaded {spec.display_name:<14} {models[key].num_params / 1e6:6.1f}M params "
            f"in {time.perf_counter() - t0:4.1f}s")
    return models
