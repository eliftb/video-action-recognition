#!/usr/bin/env python3
"""Download one part of the Kinetics-400 validation set and sort it by class.

The validation set is published as 20 archives of about 1000 videos (1.5 GB)
each. This fetches one and builds the layout evaluate.py expects:

    python tools/k400_val_subset.py            # part 0 -> data/eval/<class>/<video>.mp4
    python evaluate.py data/eval
"""

import argparse
import csv
import os
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://s3.amazonaws.com/kinetics/400"
DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", type=int, default=0, choices=range(20), metavar="0-19")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    annotations = DATA / "val.csv"
    if not annotations.exists():
        urllib.request.urlretrieve(f"{BASE}/annotations/val.csv", annotations)

    videos = DATA / f"k400_val_part{args.part}"
    if not videos.exists():
        archive = DATA / f"part_{args.part}.tar.gz"
        print(f"Downloading {archive.name} (about 1.5 GB) ...")
        urllib.request.urlretrieve(f"{BASE}/val/{archive.name}", archive)
        videos.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(videos, filter="data")
        archive.unlink()

    # File names are <youtube id>_<start>_<end>.mp4; the class is only in val.csv.
    files = {p.stem: p for p in videos.rglob("*.mp4")}
    out = DATA / "eval"
    linked, classes = 0, set()
    with open(annotations) as f:
        for row in csv.DictReader(f):
            stem = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}"
            if stem not in files:
                continue
            link = out / row["label"] / files[stem].name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.symlink(files[stem].resolve(), link)
            linked += 1
            classes.add(row["label"])
    print(f"{linked} of {len(files)} videos sorted into {len(classes)} classes under {out}")


if __name__ == "__main__":
    main()
