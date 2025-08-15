#!/usr/bin/env python3
"""
Stimulus + motion energy tracker with plotting and subsampling.

- ROI selection by two clicks (top-left, bottom-right).
- Stimulus onset via colored-rectangle disappearance (ROI #1).
- Fish motion energy inside ROI #2:
    E_t = mean( (g_t - g_{t-1})^2 ) over ROI (grayscale, optional norm)
- Noise model from pre-stim frames: mean+sigma*std
- Subsampling: compute diffs every frame, record/evaluate every N frames
- Plot energy, threshold, stim and detected movement.

Usage:
  python track_energy.py videos/trial01.mp4 --show \
      --stride 5 --motion-baseline-n 40 --energy-sigma 5 \
      --plot out/trial01_energy.png --csv out/trial01_energy.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------- CLI ------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path, help="Input video path")

    # Stimulus detection
    p.add_argument(
        "--max-frames", type=int, default=150, help="Max frames to scan for stimulus"
    )
    p.add_argument(
        "--baseline-frames",
        type=int,
        default=8,
        help="Frames to avg while rectangle is present",
    )
    p.add_argument(
        "--sat-drop",
        type=float,
        default=25.0,
        help="Min drop in mean saturation to trigger",
    )
    p.add_argument(
        "--diff-thresh",
        type=float,
        default=18.0,
        help="Alt: min grayscale L2 diff per pixel",
    )
    p.add_argument(
        "--show", action="store_true", help="Show diagnostic windows during detection"
    )

    # Energy tracking
    p.add_argument(
        "--motion-baseline-n",
        type=int,
        default=40,
        help="Frames before stimulus to model noise",
    )
    p.add_argument(
        "--energy-sigma",
        type=float,
        default=5.0,
        help="Threshold = mean + sigma*std (baseline)",
    )
    p.add_argument(
        "--min-run", type=int, default=2, help="Consecutive eval points over threshold"
    )
    p.add_argument(
        "--motion-max-frames",
        type=int,
        default=600,
        help="Max frames to scan after stimulus",
    )
    p.add_argument(
        "--norm",
        choices=["none", "zscore", "clahe"],
        default="zscore",
        help="Preprocess ROI grayscale before differencing",
    )
    p.add_argument(
        "--stride", type=int, default=5, help="Record/evaluate every N frames (>=1)"
    )

    # Outputs
    p.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Path to save plot (PNG). If omitted, just show.",
    )
    p.add_argument(
        "--csv", type=Path, default=None, help="Path to save CSV of (frame, energy)."
    )
    return p.parse_args()


# ----------------------- ROI selection (2 clicks) ---------------------


def select_roi_click2(frame: np.ndarray, title: str) -> Tuple[int, int, int, int]:
    msg = (
        f"{title} — click top-left then bottom-right; "
        f"[r]=reset, [q]=cancel, [Enter]=accept"
    )
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    pts: List[Tuple[int, int]] = []
    base = frame.copy()
    disp = frame.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param: Optional[int]):
        nonlocal pts, disp
        if event == cv2.EVENT_LBUTTONUP:
            if len(pts) < 2:
                pts.append((int(x), int(y)))
            disp = base.copy()
            for pt in pts:
                cv2.circle(disp, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                xa, xb = sorted([x1, x2])
                ya, yb = sorted([y1, y2])
                cv2.rectangle(disp, (xa, ya), (xb, yb), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.setMouseCallback(msg, on_mouse)
    while True:
        cv2.imshow(msg, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter/Space
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                xa, xb = sorted([x1, x2])
                ya, yb = sorted([y1, y2])
                w, h = xb - xa, yb - ya
                if w > 0 and h > 0:
                    cv2.destroyWindow(msg)
                    return xa, ya, w, h
        elif key == ord("r"):
            pts.clear()
            disp = base.copy()
        elif key == ord("q"):
            cv2.destroyWindow(msg)
            print("ROI selection cancelled.", file=sys.stderr)
            sys.exit(2)


# ----------------------- Small helpers --------------------------------


def crop(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def mean_sat(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1].astype(np.float32)))


def l2_gray(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = ga - gb
    return float(np.sqrt(np.mean(d * d)))


def preprocess_gray(bgr: np.ndarray, mode: str) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if mode == "none":
        return g
    if mode == "zscore":
        mu = float(np.mean(g))
        sd = float(np.std(g)) or 1.0
        return (g - mu) / sd
    if mode == "clahe":
        g8 = g.clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        return clahe.apply(g8).astype(np.float32)
    return g


# ----------------------- Stimulus detection ---------------------------


def build_stim_baseline(
    cap: cv2.VideoCapture, first: np.ndarray, roi: Tuple[int, int, int, int], n: int
) -> Tuple[float, np.ndarray]:
    sats, rois = [], []
    r0 = crop(first, roi)
    sats.append(mean_sat(r0))
    rois.append(r0.copy())
    for _ in range(1, n):
        ok, frame = cap.read()
        if not ok:
            break
        r = crop(frame, roi)
        sats.append(mean_sat(r))
        rois.append(r.copy())
    base_sat = float(np.median(np.array(sats)))
    base_bgr = np.mean(np.stack(rois, 0).astype(np.float32), 0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, len(rois))
    return base_sat, base_bgr


def find_stimulus(
    cap: cv2.VideoCapture,
    roi: Tuple[int, int, int, int],
    base_sat: float,
    base_bgr: np.ndarray,
    max_frames: int,
    sat_drop: float,
    diff_thresh: float,
    show: bool,
) -> Optional[int]:
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    win = "Stimulus scan (q quits)"
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        r = crop(frame, roi)
        ms = mean_sat(r)
        l2 = l2_gray(r, base_bgr)
        changed = (base_sat - ms) >= sat_drop or l2 >= diff_thresh
        if show:
            vis = frame.copy()
            x, y, w, h = roi
            cv2.rectangle(
                vis, (x, y), (x + w, y + h), (0, 0, 255) if changed else (0, 255, 0), 2
            )
            txt = f"stim idx={idx} sat={ms:.1f} Δsat={base_sat - ms:.1f} L2={l2:.1f}"
            cv2.putText(
                vis,
                txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyWindow(win)
                return None
        if changed:
            if show:
                cv2.destroyWindow(win)
            return idx
        idx += 1
    if show:
        cv2.destroyWindow(win)
    return None


# ----------------------- Energy tracking w/ subsampling ---------------


def track_energy(
    cap: cv2.VideoCapture,
    roi: Tuple[int, int, int, int],
    stim_idx: int,
    baseline_n: int,
    sigma: float,
    min_run: int,
    max_scan: int,
    norm: str,
    stride: int,
) -> Tuple[List[int], List[float], float, Optional[int]]:
    """
    Compute energy every frame, but record/evaluate only every `stride`.
    Baseline from last `baseline_n` pre-stim frames. Returns:
      (eval_frame_idxs, eval_energies, threshold, detected_idx)
    """
    stride = max(1, int(stride))

    # ---- Baseline (need baseline_n+1 frames to get baseline_n diffs)
    base_start = max(0, stim_idx - (baseline_n + 1))
    pos0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, base_start)

    ok, prev = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
        return [], [], 0.0, None
    prev_g = preprocess_gray(crop(prev, roi), norm)

    base_vals: List[float] = []
    for i in range(base_start + 1, stim_idx):
        ok, frame = cap.read()
        if not ok:
            break
        g = preprocess_gray(crop(frame, roi), norm)
        d = g - prev_g
        base_vals.append(float(np.mean(d * d)))
        prev_g = g

    if len(base_vals) < max(5, min_run + 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
        print("Baseline too short for noise model.", file=sys.stderr)
        return [], [], 0.0, None

    mu = float(np.mean(base_vals))
    sd = float(np.std(base_vals)) or 1e-6
    thr = mu + sigma * sd

    # ---- Scan after stimulus; start one frame before to form first diff
    scan_start = max(0, stim_idx - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, scan_start)

    idxs: List[int] = []
    vals: List[float] = []
    run = 0
    detected: Optional[int] = None

    ok, prev = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
        return [], [], thr, None
    prev_g = preprocess_gray(crop(prev, roi), norm)

    # We still compute energy every frame to keep consecutive diffs,
    # but only RECORD/EVALUATE every `stride` step.
    for i in range(1, max_scan + 2):
        idx = scan_start + i
        ok, frame = cap.read()
        if not ok:
            break
        g = preprocess_gray(crop(frame, roi), norm)
        e = float(np.mean((g - prev_g) * (g - prev_g)))
        prev_g = g

        if i % stride == 0:
            idxs.append(idx)
            vals.append(e)
            over = e > thr
            run = run + 1 if over else 0
            if detected is None and run >= min_run:
                detected = idx

    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    return idxs, vals, thr, detected


# ----------------------- Plot / CSV -----------------------------------


def save_csv(path: Path, idxs: List[int], vals: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "energy"])
        for i, v in zip(idxs, vals):
            w.writerow([i, v])


def make_plot(
    idxs: List[int],
    vals: List[float],
    thr: float,
    stim_idx: int,
    det_idx: Optional[int],
    out: Optional[Path],
) -> None:
    plt.figure()
    plt.plot(idxs, vals, label="energy (subsampled)")
    plt.axhline(thr, linestyle="--", label="threshold")
    plt.axvline(stim_idx, linestyle=":", label="stimulus")
    if det_idx is not None:
        plt.axvline(det_idx, linestyle="-.", label="first movement")
    plt.xlabel("frame")
    plt.ylabel("energy")
    plt.legend()
    plt.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        plt.close()
    else:
        plt.show()


# ----------------------- Main -----------------------------------------


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Failed to open video.", file=sys.stderr)
        sys.exit(1)

    ok, first = cap.read()
    if not ok:
        print("Empty video.", file=sys.stderr)
        sys.exit(1)

    # Stimulus ROI & detection
    stim_roi = select_roi_click2(first, "Stimulus ROI")
    base_sat, base_bgr = build_stim_baseline(
        cap, first, stim_roi, n=args.baseline_frames
    )
    stim_idx = find_stimulus(
        cap=cap,
        roi=stim_roi,
        base_sat=base_sat,
        base_bgr=base_bgr,
        max_frames=args.max_frames,
        sat_drop=args.sat_drop,
        diff_thresh=args.diff_thresh,
        show=args.show,
    )
    if stim_idx is None:
        cap.release()
        print("Stimulus not detected.", file=sys.stderr)
        sys.exit(3)
    print(f"Stimulus frame index: {stim_idx}")

    # Fish ROI & energy tracking (with subsampling)
    fish_roi = select_roi_click2(first, "Fish ROI")
    idxs, vals, thr, det_idx = track_energy(
        cap=cap,
        roi=fish_roi,
        stim_idx=stim_idx,
        baseline_n=args.motion_baseline_n,
        sigma=args.energy_sigma,
        min_run=args.min_run,
        max_scan=args.motion_max_frames,
        norm=args.norm,
        stride=args.stride,
    )
    cap.release()

    if not idxs:
        print("No energy values computed.", file=sys.stderr)
        sys.exit(4)

    if det_idx is None:
        print("No movement detected within scan window.")
    else:
        print(f"Fish first-movement frame index (subsampled): {det_idx}")

    if args.csv:
        save_csv(args.csv, idxs, vals)
        print(f"Saved CSV: {args.csv}")

    make_plot(idxs, vals, thr, stim_idx, det_idx, args.plot)
    if args.plot:
        print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
