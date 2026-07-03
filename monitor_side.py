#!/usr/bin/env python3
"""
Infer which side the stimulus monitor is on (top/bottom) and write it into each clip's
annotation as ``monitor_side`` (milestone M3).

The stim ROI sits on the monitor, so its vertical position tells us where the monitor is: if
the ROI's center is in the top half of the frame the monitor is at the top, otherwise the
bottom. Those are the only two orientations.

Usage:
    uv run monitor_side.py videos/Sculpin_SloMo/flapping        # a whole condition
    uv run monitor_side.py videos/Sculpin_SloMo/flapping/48.MP4 # one clip
"""

import argparse
import sys
from pathlib import Path

import cv2

import annotations as anno


def frame_height(video: Path):
    """Frame height in pixels, or None if the clip can't be read."""
    cap = cv2.VideoCapture(str(video))
    try:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if h > 0:
            return h
        ok, frame = cap.read()          # fallback if the header doesn't report it
        return frame.shape[0] if ok else None
    finally:
        cap.release()


def monitor_side(stim_roi, height: int) -> str:
    """'top' if the stim ROI's vertical center is in the top half of the frame, else 'bottom'."""
    _, y, _, h = stim_roi
    return "top" if (y + h / 2) < height / 2 else "bottom"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Set each annotation's `monitor_side` (top/bottom) from its stim ROI."
    )
    p.add_argument("path", type=Path, help="Folder of videos, or a single video file")
    return p.parse_args()


def gather_targets(path: Path):
    if path.is_dir():
        return anno.list_videos(path)
    if path.is_file():
        return [path]
    print(f"Not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = parse_args()
    updated = 0
    for video in gather_targets(args.path):
        ann = anno.load_annotation(video)
        if ann is None:
            print(f"  [skip] no annotation: {video.name}", file=sys.stderr)
            continue
        if ann.stim_roi is None:
            print(f"  [skip] no stim_roi: {video.name}", file=sys.stderr)
            continue
        height = frame_height(video)
        if not height:
            print(f"  [skip] cannot read frame size: {video.name}", file=sys.stderr)
            continue

        side = monitor_side(ann.stim_roi, height)
        ann.annotation["monitor_side"] = side
        anno.save_annotation(video, ann)
        cy = ann.stim_roi[1] + ann.stim_roi[3] / 2
        print(f"  {video.name}: monitor_side={side}  (stim y-center {cy:.0f} of {height})")
        updated += 1

    print(f"\nUpdated {updated} annotation(s).")


if __name__ == "__main__":
    main()
