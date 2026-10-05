#!/usr/bin/env python3
"""Measure top-1 / top-5 accuracy of each model on your own labelled videos.

The demo's accuracy chart shows *published* Kinetics-400 numbers. Use this
script to get numbers you measured yourself. Expected layout, one folder per
Kinetics-400 class name (e.g. from the Kinetics validation set):

    data/eval/
        archery/clip1.mp4
        playing guitar/clip2.mp4
        ...

    python evaluate.py data/eval
"""

import argparse
import json
from pathlib import Path

import torch

from actionrec.config import DEFAULT_ORDER, MODELS, OUTPUTS_DIR
from actionrec.models import load_kinetics_labels, load_models
from actionrec.video import read_video

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def normalise(label: str) -> str:
    return label.lower().replace("_", " ").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=DEFAULT_ORDER)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    known = {normalise(lbl) for lbl in load_kinetics_labels()}
    samples = []
    for class_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        label = normalise(class_dir.name)
        if label not in known:
            print(f"skipping '{class_dir.name}': not a Kinetics-400 class")
            continue
        samples += [(f, label) for f in sorted(class_dir.iterdir())
                    if f.suffix.lower() in VIDEO_EXTS]
    if not samples:
        raise SystemExit(f"No labelled videos found under {args.root}")
    print(f"{len(samples)} videos, {len({lbl for _, lbl in samples})} classes")

    models = load_models(args.models, device=args.device)
    hits = {k: {"top1": 0, "top5": 0} for k in args.models}
    for i, (path, label) in enumerate(samples, 1):
        frames, _ = read_video(path)
        for key in args.models:
            top = [normalise(lbl) for lbl in models[key].predict(frames).top_labels]
            hits[key]["top1"] += top[0] == label
            hits[key]["top5"] += label in top
        print(f"\r  {i}/{len(samples)}", end="", flush=True)
    print()

    results = {}
    print(f"\n{'Model':<15}{'Top-1':>8}{'Top-5':>8}")
    for key in args.models:
        top1 = 100 * hits[key]["top1"] / len(samples)
        top5 = 100 * hits[key]["top5"] / len(samples)
        results[key] = {"top1": top1, "top5": top5, "n": len(samples)}
        print(f"{MODELS[key].display_name:<15}{top1:>7.1f}%{top5:>7.1f}%")
    out = OUTPUTS_DIR / "evaluation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
