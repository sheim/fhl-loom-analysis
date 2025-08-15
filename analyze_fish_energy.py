#!/usr/bin/env python3
"""
Analyze high-speed videos:
  1) Detect stimulus onset via colored rectangle disappearance (ROI #1).
  2) Detect fish first movement via temporal-kernel energy (ROI #2).

Features:
- Two-click ROI selection.
- Temporal derivative-like kernel (odd, zero-sum), e.g., diff3/diff5/diff7.
- Pre-stim baseline to set threshold = mean + sigma*std.
- Subsampling: compute per-frame but record/evaluate every N frames.
- Optional live viz: processed ROI, |response|, energy bar vs threshold.
- Plot & CSV export of energy vs frame.
- Save debug frames with prefix naming.

Quick start:
  python analyze_fish_energy.py videos/trial01.mp4 --show \
    --kernel diff7 --stride 5 --viz-roi --viz-every 5 \
    --plot out/energy.png --csv out/energy.csv \
    --save-frames stim,det
"""

import argparse
import csv
import re
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
        "--show", action="store_true", help="Show stimulus scan visualization"
    )

    # Energy tracking
    p.add_argument(
        "--motion-baseline-n",
        type=int,
        default=40,
        help="Centers for pre-stim baseline (>=5 recommended)",
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
        help="Max centers to scan after stimulus",
    )
    p.add_argument(
        "--norm",
        choices=["none", "zscore", "clahe"],
        default="zscore",
        help="ROI grayscale preprocessing before filtering",
    )
    p.add_argument(
        "--stride", type=int, default=5, help="Record/evaluate every N centers (>=1)"
    )
    p.add_argument(
        "--kernel",
        default="diff5",
        help=(
            "Temporal kernel: diff3|diff5|diff7|custom:a,b,c (odd length, zero-sum)."
        ),
    )

    # Live visualization
    p.add_argument(
        "--viz-roi",
        action="store_true",
        help="Show processed ROI + |response| + energy bar",
    )
    p.add_argument(
        "--viz-scale", type=int, default=2, help="Upscale factor for ROI viz"
    )
    p.add_argument(
        "--viz-every", type=int, default=1, help="Show viz every k centers (>=1)"
    )

    # Outputs
    p.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save energy plot PNG; if omitted, just show",
    )
    p.add_argument(
        "--csv", type=Path, default=None, help="Save CSV of (center_frame, energy)"
    )

    # Debug frames
    p.add_argument(
        "--save-frames",
        type=str,
        default="",
        help=(
            "Comma list of centers to save: integers and/or keywords "
            "'stim','det'. Example: 'stim,det,120,135'"
        ),
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("out/debug"),
        help="Directory for saved debug frames",
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


# ----------------------- Helpers --------------------------------------


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


def to_u8(img: np.ndarray) -> np.ndarray:
    """Robust per-frame normalization to 0..255 uint8 for display."""
    a, b = np.percentile(img, (1, 99))
    if b <= a:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - a) * (255.0 / (b - a))
    return np.clip(out, 0, 255).astype(np.uint8)


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
            color = (0, 0, 255) if changed else (0, 255, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            txt = f"stim idx={idx} sat={ms:.1f} Δsat={base_sat - ms:.1f} L2={l2:.1f}"
            cv2.putText(
                vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
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


# ----------------------- Live viz -------------------------------------


def draw_viz(
    roi_g: np.ndarray, resp_abs: np.ndarray, e: float, thr: float, idx: int, scale: int
) -> np.ndarray:
    roi_u8 = to_u8(roi_g)
    resp_u8 = to_u8(resp_abs)
    roi_u8 = cv2.resize(
        roi_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    resp_u8 = cv2.resize(
        resp_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    if roi_u8.ndim == 2:
        roi_u8 = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    if resp_u8.ndim == 2:
        resp_u8 = cv2.cvtColor(resp_u8, cv2.COLOR_GRAY2BGR)

    side = cv2.hconcat([roi_u8, resp_u8])

    h, w, _ = side.shape
    bar_h = 50
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h, :, :] = side

    max_ref = max(thr * 1.5, 1e-6)
    frac = float(min(e / max_ref, 1.0))
    bar_w = int(frac * (w - 20))
    color = (0, 220, 0) if e <= thr else (0, 0, 220)
    cv2.rectangle(canvas, (10, h + 15), (10 + bar_w, h + 35), color, -1, cv2.LINE_AA)
    thr_x = 10 + int(min(thr / max_ref, 1.0) * (w - 20))
    cv2.line(canvas, (thr_x, h + 12), (thr_x, h + 38), (255, 255, 255), 2, cv2.LINE_AA)

    txt = f"idx={idx}  E={e:.3g}  thr={thr:.3g}"
    cv2.putText(
        canvas,
        txt,
        (10, h + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


# ----------------------- Temporal kernel & energy ----------------------


def get_kernel(spec: str) -> np.ndarray:
    """
    Return odd-length kernel (approx zero-sum).
      diff3 -> [-1, 0, 1]
      diff5 -> [-2, -1, 0, 1, 2]
      diff7 -> [-3, -2, -1, 0, 1, 2, 3]
      custom:a,b,c,... -> parsed list (odd length >=3)
    Enforce zero-sum by subtracting the mean.
    """
    spec = spec.strip().lower()
    if spec == "diff3":
        k = np.array([-1, 0, 1], dtype=np.float32)
    elif spec == "diff5":
        k = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    elif spec == "diff7":
        k = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
    elif spec.startswith("custom:"):
        nums = re.split(r"[,\s]+", spec.split("custom:", 1)[1].strip())
        vals = [float(x) for x in nums if x]
        if len(vals) < 3 or len(vals) % 2 == 0:
            raise ValueError("custom kernel must be odd-length >=3")
        k = np.array(vals, dtype=np.float32)
    else:
        raise ValueError(f"Unknown kernel spec: {spec}")
    return (k - float(np.mean(k))).astype(np.float32)


def energy_temporal_series(
    cap: cv2.VideoCapture,
    roi: Tuple[int, int, int, int],
    center_start: int,
    center_end: int,
    kernel: np.ndarray,
    norm: str,
    viz: bool = False,
    viz_scale: int = 2,
    viz_every: int = 1,
    thr: float = 0.0,
) -> Tuple[List[int], List[float]]:
    """
    Compute temporal response:
      R_t = sum_i k[i] * g_{t+i-H},  E_t = mean(R_t^2)
    for center indices t in [center_start, center_end].
    Uses a ring buffer of size K = 2H+1.
    """
    H = len(kernel) // 2
    if center_end < center_start:
        return [], []

    first_needed = max(0, center_start - H)
    pos0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_needed)

    buf: deque[np.ndarray] = deque(maxlen=2 * H + 1)

    def read_gray() -> Optional[np.ndarray]:
        ok, fr = cap.read()
        if not ok:
            return None
        return preprocess_gray(crop(fr, roi), norm)

    while len(buf) < (2 * H + 1):
        g = read_gray()
        if g is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
            return [], []
        buf.append(g)

    centers: List[int] = []
    energies: List[float] = []
    win = "ROI viz (q quits)"

    cur_right = first_needed + len(buf) - 1
    cur_center = cur_right - H
    step = 0

    while cur_center <= center_end:
        resp = np.zeros_like(buf[0], dtype=np.float32)
        for w, img in zip(kernel, buf):
            resp += float(w) * img
        e = float(np.mean(resp * resp))

        centers.append(cur_center)
        energies.append(e)

        if viz and (step % max(1, viz_every) == 0):
            disp = draw_viz(buf[H], np.abs(resp), e, thr, cur_center, viz_scale)
            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                try:
                    cv2.destroyWindow(win)
                except cv2.error:
                    pass
                break

        nxt = read_gray()
        if nxt is None:
            break
        buf.append(nxt)
        cur_right += 1
        cur_center = cur_right - H
        step += 1

    try:
        if viz:
            cv2.destroyWindow(win)
    except cv2.error:
        pass

    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    return centers, energies


def track_energy_temporal(
    cap: cv2.VideoCapture,
    roi: Tuple[int, int, int, int],
    stim_idx: int,
    baseline_n: int,
    sigma: float,
    min_run: int,
    max_scan: int,
    norm: str,
    stride: int,
    kernel: np.ndarray,
    viz: bool,
    viz_scale: int,
    viz_every: int,
) -> Tuple[List[int], List[float], float, Optional[int]]:
    """
    Baseline + scan using a zero-sum temporal kernel.
    Returns (eval_centers, eval_energies, threshold, detected_center).
    """
    H = len(kernel) // 2
    stride = max(1, int(stride))

    # ---- Baseline centers entirely pre-stim (keep H margin)
    base_end = stim_idx - 1 - H
    base_start = base_end - (baseline_n - 1)
    base_start = max(base_start, H)
    if base_start > base_end:
        print("Baseline window too short for temporal kernel.", file=sys.stderr)
        return [], [], 0.0, None

    b_centers, b_vals = energy_temporal_series(
        cap=cap,
        roi=roi,
        center_start=base_start,
        center_end=base_end,
        kernel=kernel,
        norm=norm,
        viz=False,
    )
    if len(b_vals) < max(5, min_run + 2):
        print("Not enough baseline centers.", file=sys.stderr)
        return [], [], 0.0, None

    mu = float(np.mean(b_vals))
    sd = float(np.std(b_vals)) or 1e-6
    thr = mu + sigma * sd

    # ---- Scan centers after stim (avoid pre-stim leakage by H)
    scan_start = stim_idx + H
    scan_end = scan_start + max_scan - 1

    s_centers, s_vals = energy_temporal_series(
        cap=cap,
        roi=roi,
        center_start=scan_start,
        center_end=scan_end,
        kernel=kernel,
        norm=norm,
        viz=viz,
        viz_scale=viz_scale,
        viz_every=viz_every,
        thr=thr,
    )
    if not s_centers:
        return [], [], thr, None

    eval_centers = s_centers[::stride]
    eval_vals = s_vals[::stride]

    run = 0
    detected = None
    for idx, e in zip(eval_centers, eval_vals):
        over = e > thr
        run = run + 1 if over else 0
        if run >= min_run:
            detected = idx
            break

    return eval_centers, eval_vals, thr, detected


# ----------------------- Plot / CSV -----------------------------------


def save_csv(path: Path, idxs: List[int], vals: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center_frame", "energy"])
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
    plt.xlabel("frame (center)")
    plt.ylabel("energy")
    plt.legend()
    plt.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        plt.close()
    else:
        plt.show()


# ----------------------- Debug frame saving ---------------------------


def save_debug_frames_temporal(
    video_path: Path,
    roi: Tuple[int, int, int, int],
    centers: List[int],
    kernel: np.ndarray,
    norm: str,
    out_dir: Path,
) -> None:
    """
    Save for each center t:
      - full_frame_{t:06d}.png (with ROI box, energy label)
      - roi_frame_{t:06d}.png  (processed center frame g_t)
      - resp_frame_{t:06d}.png (|temporal response| = |R_t|)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    H = len(kernel) // 2

    def read_gray_at(idx: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            return None
        return preprocess_gray(crop(fr, roi), norm)

    x, y, w, h = roi
    for t in sorted(set(c for c in centers if c >= 0)):
        # Build window g_{t-H}..g_{t+H}
        stack: List[np.ndarray] = []
        valid = True
        for j in range(t - H, t + H + 1):
            g = read_gray_at(j)
            if g is None:
                valid = False
                break
            stack.append(g)
        if not valid:
            continue

        resp = np.zeros_like(stack[0], dtype=np.float32)
        for wgt, img in zip(kernel, stack):
            resp += float(wgt) * img
        e = float(np.mean(resp * resp))

        # Full frame with ROI
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, full = cap.read()
        if not ok:
            continue
        cv2.rectangle(full, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            full,
            f"center={t} E={e:.3g}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"full_frame_{t:06d}.png"), full)

        # Center ROI and |response|
        cv2.imwrite(str(out_dir / f"roi_frame_{t:06d}.png"), to_u8(stack[H]))
        cv2.imwrite(str(out_dir / f"resp_frame_{t:06d}.png"), to_u8(np.abs(resp)))

    cap.release()


# ----------------------- Main -----------------------------------------


def main() -> None:
    args = parse_args()

    # parameters
    debug = True
    max_frames = 150
    video = "videos/Trial_2_circle.MP4"
    baseline_frames = 8
    sat_drop = 25.0
    diff_thresh = 18.0
    show = True
    motion_baseline_n = 40
    energy_sigma = 5.0
    min_run = 2
    motion_max_frames = 600
    norm = "zscore"
    stride = 5
    kernel = "diff5"
    viz_roi = True
    viz_scale = 2
    viz_every = 5
    # plot_path = "out/trial01_energy.png"
    # csv_path = "out/trial01_energy.csv"
    save_frames = ""

    # -----

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
    base_sat, base_bgr = build_stim_baseline(cap, first, stim_roi, n=baseline_frames)
    stim_idx = find_stimulus(
        cap=cap,
        roi=stim_roi,
        base_sat=base_sat,
        base_bgr=base_bgr,
        max_frames=max_frames,
        sat_drop=sat_drop,
        diff_thresh=diff_thresh,
        show=show,
    )
    if stim_idx is None:
        cap.release()
        print("Stimulus not detected.", file=sys.stderr)
        sys.exit(3)
    print(f"Stimulus frame index: {stim_idx}")

    # Fish ROI & temporal energy
    fish_roi = select_roi_click2(first, "Fish ROI")
    kernel = get_kernel(kernel)

    idxs, vals, thr, det_idx = track_energy_temporal(
        cap=cap,
        roi=fish_roi,
        stim_idx=stim_idx,
        baseline_n=motion_baseline_n,
        sigma=energy_sigma,
        min_run=min_run,
        max_scan=motion_max_frames,
        norm=norm,
        stride=stride,
        kernel=kernel,
        viz=viz_roi,
        viz_scale=viz_scale,
        viz_every=viz_every,
    )
    cap.release()

    if not idxs:
        print("No energy values computed.", file=sys.stderr)
        sys.exit(4)

    if det_idx is None:
        print("No movement detected within scan window.")
    else:
        print(f"Fish first-movement (center frame): {det_idx}")

    if args.csv:
        save_csv(args.csv, idxs, vals)
        print(f"Saved CSV: {args.csv}")

    make_plot(idxs, vals, thr, stim_idx, det_idx, args.plot)
    if args.plot:
        print(f"Saved plot: {args.plot}")

    # Debug frame saving
    if debug:
        targets: List[int] = []
        parts = [s.strip() for s in args.save_frames.split(",") if s]
        for p in parts:
            if p.lower() == "stim":
                targets.append(stim_idx)
            elif p.lower() == "det" and det_idx is not None:
                targets.append(det_idx)
            else:
                try:
                    targets.append(int(p))
                except ValueError:
                    pass
        if targets:
            save_debug_frames_temporal(
                video_path=args.video,
                roi=fish_roi,
                centers=targets,
                kernel=kernel,
                norm=norm,
                out_dir="out/debug",
            )
            print("Saved debug frames to: out/debug")


if __name__ == "__main__":
    main()
