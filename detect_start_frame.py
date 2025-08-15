#!/usr/bin/env python3
"""
Detect the first frame where a colored rectangle disappears inside a
user-selected ROI.

Usage:
  python detect_start_frame.py /path/to/video.mp4 --max-frames 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path, help="Input video path")
    p.add_argument("--max-frames", type=int, default=150, help="Max frames to scan")
    p.add_argument(
        "--baseline-frames",
        type=int,
        default=8,
        help="Frames to average while rectangle is present",
    )
    p.add_argument(
        "--sat-drop",
        type=float,
        default=25.0,
        help="Min drop in mean saturation to declare change",
    )
    p.add_argument(
        "--diff-thresh",
        type=float,
        default=18.0,
        help="Alt: min grayscale L2 diff per pixel to trigger",
    )
    p.add_argument(
        "--show", action="store_true", help="Visualize detection as it scans"
    )
    return p.parse_args()


def read_first_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def select_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Returns (x, y, w, h). Exits if user cancels.
    """
    disp = frame.copy()
    win = "Select ROI then press ENTER (or SPACE). Press c to cancel."
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    roi = cv2.selectROI(win, disp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("ROI selection cancelled.", file=sys.stderr)
        sys.exit(2)
    return x, y, w, h


def roi_slice(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def mean_saturation(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return float(np.mean(sat))


def gray_l2_per_pixel(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = ga - gb
    l2 = np.sqrt(np.mean(diff * diff))
    return float(l2)


def build_baseline(
    cap: cv2.VideoCapture,
    first_frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    n_frames: int,
) -> Tuple[float, np.ndarray, List[int]]:
    """
    Build a baseline using up to n_frames after the first. Returns:
      (baseline_mean_sat, baseline_mean_bgr, frame_indices_used)
    """
    sats: List[float] = []
    rois: List[np.ndarray] = []
    idxs: List[int] = []

    base_roi = roi_slice(first_frame, roi)
    sats.append(mean_saturation(base_roi))
    rois.append(base_roi.copy())
    idxs.append(0)

    # Peek ahead without losing position: we already consumed frame 0.
    for i in range(1, n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        r = roi_slice(frame, roi)
        sats.append(mean_saturation(r))
        rois.append(r.copy())
        idxs.append(i)

    # Compute robust central tendency
    baseline_sat = float(np.median(np.array(sats)))
    baseline_bgr = np.mean(np.stack(rois, axis=0).astype(np.float32), axis=0).astype(
        np.uint8
    )

    # Rewind stream to after the first frame to continue scanning
    cap.set(cv2.CAP_PROP_POS_FRAMES, len(idxs))
    return baseline_sat, baseline_bgr, idxs


def scan_for_change(
    cap: cv2.VideoCapture,
    roi: Tuple[int, int, int, int],
    baseline_sat: float,
    baseline_roi_bgr: np.ndarray,
    max_frames: int,
    sat_drop: float,
    diff_thresh: float,
    show: bool,
) -> Optional[int]:
    """
    Returns the first frame index where change is detected
    (0-based index relative to start of video), or None.
    Assumes we start scanning at current cap position.
    """
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        r = roi_slice(frame, roi)
        ms = mean_saturation(r)
        l2 = gray_l2_per_pixel(r, baseline_roi_bgr)

        sat_changed = (baseline_sat - ms) >= sat_drop
        pix_changed = l2 >= diff_thresh
        changed = sat_changed or pix_changed

        if show:
            vis = frame.copy()
            x, y, w, h = roi
            color = (0, 255, 0) if not changed else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            txt = (
                f"idx={frame_idx}  sat={ms:.1f} "
                f"Δsat={baseline_sat - ms:.1f}  L2={l2:.1f}"
            )
            cv2.putText(
                vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
            )
            cv2.imshow("Scanning (q to quit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow("Scanning (q to quit)")
                return None

        if changed:
            if show:
                cv2.destroyWindow("Scanning (q to quit)")
            return frame_idx

        frame_idx += 1

    if show:
        cv2.destroyWindow("Scanning (q to quit)")
    return None


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Failed to open video.", file=sys.stderr)
        sys.exit(1)

    first = read_first_frame(cap)
    if first is None:
        print("Empty video.", file=sys.stderr)
        sys.exit(1)

    roi = select_roi(first)

    baseline_sat, baseline_bgr, used = build_baseline(
        cap, first, roi, n_frames=args.baseline_frames
    )

    start_idx = scan_for_change(
        cap=cap,
        roi=roi,
        baseline_sat=baseline_sat,
        baseline_roi_bgr=baseline_bgr,
        max_frames=args.max_frames,
        sat_drop=args.sat_drop,
        diff_thresh=args.diff_thresh,
        show=args.show,
    )

    cap.release()

    if start_idx is None:
        print("No disappearance detected in the scan window.")
        sys.exit(3)

    print(f"First disappearance frame index: {start_idx}")


if __name__ == "__main__":
    main()
