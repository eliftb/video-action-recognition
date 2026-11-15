#!/usr/bin/env python3
"""Real-time action recognition from a webcam.

Keeps the last few seconds of camera frames, runs the models on that sliding
window over and over, and draws the current prediction on the live picture.

    python webcam.py                          # camera 0, all five models
    python webcam.py --models videomae        # one model: top-3 with bars
    python webcam.py --source 1               # another camera
    python webcam.py --source assets/archery.mp4   # no camera: loop a file
    python webcam.py --record outputs/live.mp4

Press q or Esc to quit.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import torch

from actionrec.config import LIVE_ORDER, MODELS
from actionrec.live import FrameBuffer, FrameSource, LiveRecognizer, draw_overlay
from actionrec.models import load_models

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

WINDOW_TITLE = "Action recognition (q to quit)"
MIN_DISPLAY_WIDTH = 640


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="camera index (default 0) or a video file to loop")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None,
                        help="default: all five on GPU, i3d only on CPU")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--half", action="store_true", help="fp16 inference on GPU")
    parser.add_argument("--max-window", type=float, default=3.0,
                        help="longest stretch of recent video a model sees, seconds")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="weight of the newest window, 0-1; 1 disables smoothing")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not flip the camera picture horizontally")
    parser.add_argument("--record", type=Path, help="also save the annotated picture as mp4")
    parser.add_argument("--no-display", action="store_true",
                        help="no window; print predictions once a second")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = args.models or (LIVE_ORDER if device == "cuda" else ["i3d"])

    try:
        source = FrameSource(args.source)
    except RuntimeError as exc:
        sys.exit(str(exc))

    print(f"Loading models on {device} ...")
    models = load_models(model_keys, device=device, half=args.half)

    buffer = FrameBuffer(seconds=args.max_window + 1)
    recognizer = LiveRecognizer(models, buffer, max_window=args.max_window,
                                smoothing=args.smoothing,
                                top_k=3 if len(models) == 1 else 1)
    recognizer.start()
    mirror = source.is_camera and not args.no_mirror
    writer = None
    start = last_print = last_frame = time.perf_counter()
    frame_dt = 1 / source.fps
    print("Running. Press q or Esc in the window to quit."
          if not args.no_display else "Running. Ctrl+C to stop.")

    try:
        while True:
            frame = source.read()
            if frame is None:
                sys.exit("The camera stopped delivering frames.")
            if recognizer.error:
                raise recognizer.error
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            buffer.push(now, frame)
            # Average the interval, not its inverse: one late frame followed by a
            # quick one would otherwise read as a burst of several hundred fps.
            frame_dt = 0.9 * frame_dt + 0.1 * (now - last_frame)
            last_frame = now
            if frame.shape[1] < MIN_DISPLAY_WIDTH:   # keep the overlay readable
                k = MIN_DISPLAY_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)

            results = recognizer.results()
            draw_overlay(frame, results, f"{1 / frame_dt:.0f} fps")

            if args.record:
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source.fps, (w, h))
                writer.write(frame)

            if args.no_display:
                if now - last_print >= 1:
                    last_print = now
                    print("  " + " | ".join(
                        f"{r.model}: {r.labels[0]} {r.probs[0] * 100:.0f}%" if r.labels
                        else f"{r.model}: ..." for r in results))
            else:
                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break   # window closed with the mouse
            if args.duration and now - start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        recognizer.stop()
        source.release()
        if writer is not None:
            writer.release()
            print(f"Saved {args.record}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
