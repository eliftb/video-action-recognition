"""Speed and memory measurements for a loaded model."""

import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch

from .models import ActionRecognizer


@dataclass
class BenchmarkResult:
    model: str
    params_m: float
    latency_ms: float          # mean model forward time for one clip
    latency_std_ms: float
    end_to_end_ms: float       # preprocessing + forward
    clips_per_s: float         # 1000 / end_to_end_ms
    frames_per_s: float        # frames the model consumes per second of compute
    weights_mb: float
    peak_mem_mb: float         # extra memory used while running one clip
    k400_top1: float           # published reference, not measured here
    k400_top5: float


def benchmark(rec: ActionRecognizer, frames: np.ndarray,
              warmup: int = 3, runs: int = 10) -> BenchmarkResult:
    """Time preprocessing + inference on one clip; report mean over `runs`.

    Warm-up runs are discarded so cuDNN autotuning and lazy CUDA init
    don't skew the numbers.
    """
    cuda = rec.device.startswith("cuda")
    weights_mb = sum(p.numel() * p.element_size() for p in rec.model.parameters()) / 2**20

    for _ in range(warmup):
        rec.forward(rec.preprocess(frames))

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    else:
        base_mem = psutil.Process().memory_info().rss

    fwd_times, e2e_times = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        inputs = rec.preprocess(frames)
        rec._sync()
        t1 = time.perf_counter()
        rec.forward(inputs)
        rec._sync()
        t2 = time.perf_counter()
        fwd_times.append((t2 - t1) * 1000)
        e2e_times.append((t2 - t0) * 1000)

    if cuda:
        peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / 2**20
    else:
        # RSS is coarse on CPU; it only catches growth beyond the warm-up peak.
        peak_mb = max(0, psutil.Process().memory_info().rss - base_mem) / 2**20

    latency = float(np.mean(fwd_times))
    e2e = float(np.mean(e2e_times))
    return BenchmarkResult(
        model=rec.spec.display_name,
        params_m=rec.num_params / 1e6,
        latency_ms=latency,
        latency_std_ms=float(np.std(fwd_times)),
        end_to_end_ms=e2e,
        clips_per_s=1000 / e2e,
        frames_per_s=rec.spec.num_frames * 1000 / e2e,
        weights_mb=weights_mb,
        peak_mem_mb=peak_mb,
        k400_top1=rec.spec.k400_top1,
        k400_top5=rec.spec.k400_top5,
    )


def format_table(results: list[BenchmarkResult]) -> str:
    header = (f"{'Model':<15}{'Params':>8}{'Fwd ms':>10}{'E2E ms':>9}{'Clips/s':>9}"
              f"{'Frames/s':>10}{'Weights':>10}{'Act. mem':>10}{'K400 top1*':>12}")
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.model:<15}{r.params_m:>7.1f}M{r.latency_ms:>7.1f}±{r.latency_std_ms:<2.0f}"
            f"{r.end_to_end_ms:>9.1f}{r.clips_per_s:>9.1f}{r.frames_per_s:>10.0f}"
            f"{r.weights_mb:>8.0f}MB{r.peak_mem_mb:>8.0f}MB{r.k400_top1:>11.1f}%"
        )
    lines.append("* published Kinetics-400 accuracy (multi-view testing), see README")
    return "\n".join(lines)
