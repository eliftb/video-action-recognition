#!/usr/bin/env python3
"""Video action recognition demo: SlowFast, VideoMAE, TimeSformer and I3D.

Runs one video through all four Kinetics-400 models, prints each model's
predicted action and confidence, benchmarks speed and memory, and saves a
summary figure plus an annotated copy of the video.

    python demo.py                                  # bundled archery clip
    python demo.py --video assets/eating_spaghetti.mp4
    python demo.py --url https://www.youtube.com/watch?v=...   # needs yt-dlp
    python demo.py --models videomae i3d --no-benchmark

For a live camera, see webcam.py.
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch

from actionrec.benchmark import benchmark, format_table
from actionrec.config import ASSETS_DIR, DEFAULT_ORDER, MODELS, OUTPUTS_DIR, SAMPLE_VIDEOS
from actionrec.models import load_models
from actionrec.video import fetch_video, read_video
from actionrec.visualize import annotated_video, summary_figure

# Library deprecation notices add noise to a live demo without telling us anything.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def section(title: str):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def run_demo(video_path, model_keys=DEFAULT_ORDER, device=None, half=False,
             run_benchmark=True, runs=10, models=None, out_dir=OUTPUTS_DIR):
    """Full pipeline for one video: load -> predict -> benchmark -> visualise.

    Pass already-loaded `models` to skip loading when demoing several videos.
    Returns (predictions, benchmarks, models) so a notebook can reuse them.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    video_path = Path(video_path)

    section("Video")
    frames, fps = read_video(video_path)
    h, w = frames.shape[1:3]
    print(f"  {video_path.name}: {len(frames)} frames, {w}x{h}, {fps:.1f} fps, "
          f"{len(frames) / fps:.1f}s")

    if models is None:
        section(f"Loading models on {device}"
                + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))
        models = load_models(model_keys, device=device, half=half)

    section("Predictions (top-1 of 400 Kinetics classes)")
    predictions = []
    for key in model_keys:
        rec = models[key]
        rec.predict(frames)  # first call includes CUDA warm-up; don't show it
        p = rec.predict(frames)
        predictions.append(p)
        runner_up = f"{p.top_labels[1]} {p.top_probs[1] * 100:.0f}%"
        print(f"  {p.model:<14} -> {p.label:<28} {p.confidence * 100:5.1f}%   "
              f"(next: {runner_up})")

    benchmarks = None
    if run_benchmark:
        section(f"Performance (mean of {runs} runs, batch size 1)")
        benchmarks = [benchmark(models[k], frames, runs=runs) for k in model_keys]
        print(format_table(benchmarks))

    section("Outputs")
    stem = video_path.stem
    fig = summary_figure(frames, predictions, benchmarks, video_path.name,
                         out_dir / f"{stem}_results.png")
    vid = annotated_video(frames, fps, predictions, out_dir / f"{stem}_annotated.mp4")
    report = out_dir / f"{stem}_report.json"
    report.write_text(json.dumps({
        "video": str(video_path),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "predictions": [asdict(p) for p in predictions],
        "benchmarks": [asdict(b) for b in benchmarks] if benchmarks else None,
    }, indent=2))
    for path in (fig, vid, report):
        print(f"  {path}")
    return predictions, benchmarks, models


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", nargs="+", type=Path, help="local video file(s)")
    src.add_argument("--url", help="direct video link or YouTube URL (needs yt-dlp)")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_ORDER, default=DEFAULT_ORDER)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--half", action="store_true", help="fp16 inference on GPU")
    parser.add_argument("--no-benchmark", action="store_true", help="skip timing runs")
    parser.add_argument("--runs", type=int, default=10, help="timed runs per model")
    args = parser.parse_args()

    if args.url:
        videos = [fetch_video(args.url, ASSETS_DIR / "downloads")]
    else:
        videos = args.video or [ASSETS_DIR / "archery.mp4"]
    for v in videos:
        if not Path(v).exists():
            if Path(v).name in SAMPLE_VIDEOS:
                print(f"Downloading sample video {Path(v).name} ...")
                fetch_video(SAMPLE_VIDEOS[Path(v).name], Path(v).parent)
            else:
                sys.exit(f"Video not found: {v}")

    start = time.perf_counter()
    models = None
    for v in videos:
        _, _, models = run_demo(v, args.models, device=args.device, half=args.half,
                                run_benchmark=not args.no_benchmark, runs=args.runs,
                                models=models)
    print(f"\nDone in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
