#!/usr/bin/env python3
"""
Manually correct the first-movement ("start of the fish movement") frame for clips where
auto-detection is off.

Opens the video at the annotation's current movement-onset frame and lets you scrub to the
right one. On accept it (a) records `annotation.manual_det`, (b) updates `results.det_refined`,
and (c) regenerates the geometry frame stack around the new frame. `batch.py` respects
`manual_det`, so future re-runs won't clobber the correction.

Scrub keys (the frame number is always shown):
    a / d   : -1 / +1 frame          (also  <-  /  ->)
    , / .   : -30 / +30 frames
    ; / '   : -240 / +240 frames
    Space / Enter : accept this frame as the start
    q / Esc : cancel (leave the clip unchanged)

Usage:
    uv run fix_start.py videos/Sculpin_SloMo/flapping/48.MP4   # one clip
    uv run fix_start.py videos/Sculpin_SloMo/flapping          # step through a folder
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2

import annotations as anno

_LEFT = {63234, 65361, 2424832, 81}       # left-arrow codes across backends
_RIGHT = {63235, 65363, 2555904, 83}      # right-arrow codes


def auto_det(ann: anno.Annotation) -> Optional[int]:
    """The auto-detected movement frame (ignoring any manual override), for reference display."""
    r = ann.results or {}
    for k in ("det_refined", "det_coarse"):
        if r.get(k) is not None:
            return int(r[k])
    return None


def current_start(ann: anno.Annotation) -> Optional[int]:
    """Where to start scrubbing: the manual override if present, else the detected frame."""
    manual = ann.annotation.get("manual_det")
    if manual is not None:
        return int(manual)
    r = ann.results or {}
    for k in ("det_refined", "det_coarse", "stim_idx_detected"):
        if r.get(k) is not None:
            return int(r[k])
    return None


def scrub(video: Path, start: int, auto: Optional[int]) -> Optional[int]:
    """Interactive scrub over the video. Returns the chosen frame index, or None if cancelled."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [skip] cannot open {video.name}", file=sys.stderr)
        return None
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = nframes - 1 if nframes > 0 else None
    idx = max(0, min(start, last) if last is not None else start)
    win = f"{video.name} - pick start"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                if idx <= 0:
                    print(f"  [skip] cannot read frames: {video.name}", file=sys.stderr)
                    return None
                idx -= 1
                continue
            disp = frame.copy()
            total = f" / {last}" if last is not None else ""
            cv2.putText(disp, f"frame {idx}{total}", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            if auto is not None:
                delta = "unchanged" if idx == auto else f"{idx - auto:+d}"
                cv2.putText(disp, f"auto-detected: {auto}  ({delta})", (10, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, "a/d +/-1   ,/. +/-30   ;/' +/-240   Space=accept   q=cancel",
                        (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)

            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            k = key & 0xFF
            if k in (ord("q"), 27):
                return None
            if k in (13, 32):
                return idx
            step = 0
            if k == ord("d") or key in _RIGHT:
                step = 1
            elif k == ord("a") or key in _LEFT:
                step = -1
            elif k == ord("."):
                step = 30
            elif k == ord(","):
                step = -30
            elif k == ord("'"):
                step = 240
            elif k == ord(";"):
                step = -240
            idx = max(0, idx + step)
            if last is not None:
                idx = min(idx, last)
    finally:
        cap.release()
        cv2.destroyWindow(win)


def fix_video(video: Path) -> bool:
    ann = anno.load_annotation(video)
    if ann is None:
        print(f"  [skip] no annotation: {video.name}", file=sys.stderr)
        return False

    auto = auto_det(ann)
    start = current_start(ann)
    picked = scrub(video, start if start is not None else 0, auto)
    if picked is None:
        print(f"  [skip] cancelled: {video.name}", file=sys.stderr)
        return False

    ann.annotation["manual_det"] = picked
    if ann.results is not None:
        ann.results["det_refined"] = picked
    info = anno.export_frames(video, picked)
    if info:
        ann.annotation["frame_dir"] = info["frame_dir"]
        ann.annotation["frame_start"] = info["frame_start"]
        ann.annotation["n_frames"] = info["n_frames"]
    anno.save_annotation(video, ann)
    extra = f", {info['n_frames']} frames regenerated" if info else ""
    print(f"  {video.name}: start set to {picked}{extra}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manually pick the first-movement frame by scrubbing the video."
    )
    p.add_argument("path", type=Path, help="A video file, or a folder of videos")
    return p.parse_args()


def gather_targets(path: Path) -> List[Path]:
    if path.is_dir():
        return anno.list_videos(path)
    if path.is_file():
        return [path]
    print(f"Not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    for video in gather_targets(parse_args().path):
        print(f"\n=== {video.name} ===")
        fix_video(video)


if __name__ == "__main__":
    main()
