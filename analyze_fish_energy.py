#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Known true capture rate of the high-speed footage (Hz). File FPS metadata is unreliable for
# these clips (e.g. 240 fps footage tagged 30 fps), so playback uses this instead. Matches
# analysis.py's FPS.
CAPTURE_FPS = 240.0


# ----------------------- Config & result types -----------------------


class VideoOpenError(Exception):
    """Raised when a video cannot be opened or is empty."""


class ROISelectionCancelled(Exception):
    """Raised when the user cancels interactive ROI selection."""


@dataclass
class AnalysisParams:
    """Tunable parameters for stimulus + first-movement detection.

    Defaults reproduce the previous hard-coded single-video behaviour.
    """

    # Stimulus detection
    baseline_frames: int = 8
    saturation_drop: float = 25.0
    diff_thresh: float = 18.0
    stim_max_frames: int = 5000
    # Motion energy & detection
    motion_baseline_n: int = 40
    energy_sigma: float = 5.0
    min_run: int = 2
    motion_max_frames: int = 600
    norm: str = "zscore"  # 'none' | 'zscore' | 'clahe'
    stride: int = 5
    kernel: str = "diff5"
    # Spatial smoothing
    smooth: str = "none"  # 'none' | 'gaussian' | 'box'
    smooth_ksize: int = 5
    smooth_sigma: float = 1.0
    # Stride-1 refinement window (± centers); None -> max(2*stride, 50)
    refine_halfwin: Optional[int] = None


@dataclass
class AnalysisResult:
    """Outcome of :func:`analyze_video` for one clip."""

    stim_idx: Optional[int]
    det_coarse: Optional[int]
    det_refined: Optional[int]
    threshold: float
    centers: List[int]
    energies: List[float]

    @property
    def final_det_idx(self) -> Optional[int]:
        """Refined detection if available, else coarse."""
        return self.det_refined if self.det_refined is not None else self.det_coarse


# ----------------------- CLI ------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path, help="Input video path")
    return p.parse_args()


# ----------------------- ROI selection (2 clicks) ---------------------


def select_roi_click(frame: np.ndarray, title: str) -> Tuple[int, int, int, int]:
    msg = (
        f"{title} — click top-left then bottom-right; "
        f"[r]=reset, [q]=cancel, [Space/Enter]=accept"
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
            raise ROISelectionCancelled()


def choice_popup(
    title: str,
    options: List[Tuple[str, str]],
    default: Optional[str] = None,
    window_name: Optional[str] = None,
) -> str:
    """Small clickable button popup — pick an option with the mouse.

    ``options`` is a list of ``(label, value)``. Click a button, press a label's first-letter
    hotkey, or press Space/Enter to take ``default`` (highlighted). ``q``/Esc raises
    :class:`ROISelectionCancelled`. Returns the chosen value.
    """
    win = window_name or title
    pad, bw, bh, gap, top = 12, 336, 46, 10, 44
    width = bw + 2 * pad
    height = top + len(options) * (bh + gap) + pad
    rects = [(pad, top + i * (bh + gap), bw, bh) for i in range(len(options))]

    hotkeys: dict = {}
    for label, value in options:
        for ch in label.lower():
            if ch.isalpha() and ch not in hotkeys:
                hotkeys[ch] = value
                break

    state = {"hover": -1, "value": None}

    def on_mouse(event, x, y, flags, param):
        idx = -1
        for i, (rx, ry, rw, rh) in enumerate(rects):
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                idx = i
                break
        if event == cv2.EVENT_MOUSEMOVE:
            state["hover"] = idx
        elif event == cv2.EVENT_LBUTTONUP and idx >= 0:
            state["value"] = options[idx][1]

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            canvas = np.full((height, width, 3), 40, np.uint8)
            cv2.putText(
                canvas, title, (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA,
            )
            for i, (label, value) in enumerate(options):
                rx, ry, rw, rh = rects[i]
                fill = (95, 95, 95) if i == state["hover"] else (70, 70, 70)
                border = (0, 210, 0) if value == default else (120, 120, 120)
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), fill, -1)
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border, 2)
                cv2.putText(
                    canvas, label, (rx + 14, ry + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (245, 245, 245), 2, cv2.LINE_AA,
                )
            cv2.imshow(win, canvas)
            if state["value"] is not None:
                return state["value"]
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):  # q / Esc
                raise ROISelectionCancelled()
            if key in (13, 32) and default is not None:  # Space/Enter -> default
                return default
            ch = chr(key).lower() if 32 <= key < 127 else ""
            if ch in hotkeys:
                return hotkeys[ch]
    finally:
        cv2.destroyWindow(win)


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


def apply_spatial_smoothing(
    g: np.ndarray, kind: str, ksize: int, sigma: float
) -> np.ndarray:
    """
    Spatial smoothing on a float32 grayscale image g.
    kind: 'none' | 'gaussian' | 'box'
    ksize: odd >= 1
    sigma: stddev for gaussian (ignored for box)
    """
    if kind == "none":
        return g
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    if kind == "gaussian":
        return cv2.GaussianBlur(g, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    if kind == "box":
        return cv2.blur(g, (k, k))
    # Fallback: no smoothing
    return g


def preprocess_gray_smooth(
    bgr: np.ndarray, mode: str, smooth: str, smooth_ksize: int, smooth_sigma: float
) -> np.ndarray:
    """
    Convert BGR->gray float32, apply per-frame normalization (mode),
    then optional spatial smoothing.
    """
    g = preprocess_gray(bgr, mode)
    if smooth == "none":
        return g
    else:
        return apply_spatial_smoothing(g, smooth, smooth_ksize, smooth_sigma)


def _playback_step(source_fps: float, speed: float, display_fps: float) -> int:
    """Source frames to advance per displayed frame to hit ``speed``x real-time."""
    display_fps = max(1.0, float(display_fps))
    return max(1, int(round(float(source_fps) * float(speed) / display_fps)))


def playback_video(
    cap,
    source_fps: Optional[float] = None,
    speed: float = 1.0,
    display_fps: float = 30.0,
    window_name: str = "Playback",
):
    """Play the clip in a window at ``speed``x real-time (1.0 = real-time, never slow-mo).

    High-speed clips often carry unreliable FPS metadata (e.g. 240 fps footage tagged 30 fps),
    which made playback speed inconsistent. Pass ``source_fps`` (the true capture rate, e.g.
    ``CAPTURE_FPS``) to set the speed deterministically; the file's metadata is only a fallback.
    """
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps and source_fps > 0:
        src = float(source_fps)
    elif meta_fps and meta_fps > 0:
        src = float(meta_fps)
    else:
        src = 30.0

    step = _playback_step(src, speed, display_fps)
    delay = max(1, int(round(1000.0 / max(1.0, display_fps))))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            cv2.imshow(window_name, frame)
            if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                break

        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
    cv2.destroyWindow(window_name)


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
    saturation_drop: float,
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
        changed = (base_sat - ms) >= saturation_drop or l2 >= diff_thresh
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
    smooth: str = "none",
    smooth_ksize: int = 3,
    smooth_sigma: float = 1.0,
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
        roi_bgr = crop(fr, roi)
        return preprocess_gray_smooth(roi_bgr, norm, smooth, smooth_ksize, smooth_sigma)

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


def first_sustained(
    centers: List[int],
    vals: List[float],
    thr: float,
    min_run: int,
    stride: int = 1,
) -> Optional[int]:
    """First center whose energy stays above ``thr`` for ``min_run`` consecutive
    (strided) samples, else ``None``.

    Pure: operates on a precomputed energy series, so coarse (stride>1) and refined
    (stride=1) detection reuse the same in-memory arrays with no video re-reads.
    """
    stride = max(1, int(stride))
    run = 0
    for c, e in zip(centers[::stride], vals[::stride]):
        run = run + 1 if e > thr else 0
        if run >= min_run:
            return c
    return None


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
    show: bool = False,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(idxs, vals, label="energy (subsampled)")
    ax.axhline(thr, linestyle="--", label="threshold")
    ax.axvline(stim_idx, linestyle=":", label="stimulus")
    if det_idx is not None:
        ax.axvline(det_idx, linestyle="-.", label="first movement")

    ax.set_ylabel("energy")
    ax.set_xlabel("frame (center)", labelpad=26)  # leave room for extra row
    ax.legend()

    # --- Special labels on a separate bottom axis (no overlap) ---
    special_ticks = [stim_idx]
    special_labels = [f"{stim_idx} (stim)"]
    if det_idx is not None:
        special_ticks.append(det_idx)
        special_labels.append(f"{det_idx} (det)")

    # If stim/det are extremely close, stack into one two-line label
    locs = list(ax.get_xticks())
    steps = [b - a for a, b in zip(locs, locs[1:])] or [1]
    avg_step = sum(steps) / len(steps)
    if det_idx is not None and abs(det_idx - stim_idx) < 0.35 * avg_step:
        special_ticks = [stim_idx]
        special_labels = [f"{stim_idx} (stim)\n{det_idx} (det)"]

    secax = ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    secax.set_xticks(special_ticks)
    secax.set_xticklabels(special_labels)
    secax.tick_params(axis="x", pad=16, length=0)

    fig.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
    if show or not out:
        plt.show()
    plt.close(fig)


def read_gray_at(idx: int, cap, roi) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return None
    roi_bgr = crop(fr, roi)
    return roi_bgr


def save_debug_grid(
    video_path: Path,
    roi: Tuple[int, int, int, int],
    frame_indices: List[int],
    kernel: np.ndarray,
    norm: str,
    out_path: Path,
    smooth: str = "none",
    smooth_ksize: int = 3,
    smooth_sigma: float = 0.8,
    dpi: int = 180,
    annotate: bool = True,
) -> None:
    """
    Create a single figure:
      Row 1: ROI center frames g_t for each requested t (left->right).
      Row 2: |temporal response| = |R_t| for the same t's.

    Saves to out_path (PNG/PDF/SVG based on extension).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if len(kernel) % 2 == 0:
        raise ValueError("kernel length must be odd")
    H = len(kernel) // 2

    xs: List[np.ndarray] = []  # ROI center frames (uint8)
    rs: List[np.ndarray] = []  # |response| images (uint8)
    ts: List[int] = []  # kept centers

    for t in frame_indices:
        stack: List[np.ndarray] = []
        valid = True
        for j in range(t - H, t + H + 1):
            g = preprocess_gray_smooth(
                read_gray_at(j, cap, roi),
                norm,
                smooth,
                smooth_ksize,
                smooth_sigma,
            )
            if g is None:
                valid = False
                break
            stack.append(g)
        if not valid:
            continue

        resp = np.zeros_like(stack[0], dtype=np.float32)
        for wgt, img in zip(kernel, stack):
            resp += float(wgt) * img

        xs.append(to_u8(stack[H]))
        rs.append(to_u8(np.abs(resp)))
        ts.append(t)

    cap.release()

    if not xs:
        raise RuntimeError("No valid centers to display (all skipped).")

    n = len(xs)
    figsize = (2.6 * n, 5.2)  # width scales with number of columns
    fig, axes = plt.subplots(2, n, figsize=figsize)

    # axes shape normalization for n == 1
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]], dtype=object)

    for i in range(n):
        ax_roi = axes[0, i]
        ax_rsp = axes[1, i]

        ax_roi.imshow(xs[i], cmap="gray", vmin=0, vmax=255)
        ax_rsp.imshow(rs[i], cmap="gray", vmin=0, vmax=255)

        if annotate:
            ax_roi.set_title(f"t={ts[i]}", fontsize=10)

        ax_roi.axis("off")
        ax_rsp.axis("off")

    axes[0, 0].set_ylabel("ROI", fontsize=10)
    axes[1, 0].set_ylabel("|R_t|", fontsize=10)

    plt.tight_layout(w_pad=0.2, h_pad=0.4)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_debug_frames(
    video_path: Path,
    roi: Tuple[int, int, int, int],
    frame_indices: List[int],
    kernel: np.ndarray,
    norm: str,
    out_dir: Path,
    smooth: str = "none",
    smooth_ksize: int = 3,
    smooth_sigma: float = 0.8,
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

    x, y, w, h = roi
    for t in frame_indices:
        # Build window g_{t-H}..g_{t+H}
        stack: List[np.ndarray] = []
        valid = True
        for j in range(t - H, t + H + 1):
            g = preprocess_gray_smooth(
                read_gray_at(j, cap, roi), norm, smooth, smooth_ksize, smooth_sigma
            )
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

        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, full = cap.read()
        if not ok:
            continue

        fig, axes = plt.subplots(nrows=2, ncols=len(stack))
        for k, g in enumerate(stack):
            ax = axes[0, k]
            idx_abs = t - H + k
            ax.imshow(g, cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"idx={idx_abs}",
                fontsize=9,
                color=("crimson" if idx_abs == t else "black"),
            )
            ax.axis("off")
        # cv2.rectangle(full, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # cv2.putText(
        #     full,
        #     f"center={t} E={e:.3g}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )
        # cv2.imwrite(str(out_dir / f"full_frame_{t:06d}.png"), full)

        cv2.imwrite(str(out_dir / f"roi_frame_{t:06d}.png"), to_u8(stack[H]))
        cv2.imwrite(str(out_dir / f"resp_frame_{t:06d}.png"), to_u8(np.abs(resp)))

    cap.release()


# ----------------------- Core pipeline (importable) -------------------


def analyze_video(
    video_path: Path,
    stim_roi: Tuple[int, int, int, int],
    fish_roi: Tuple[int, int, int, int],
    params: Optional[AnalysisParams] = None,
    cap: Optional[cv2.VideoCapture] = None,
) -> AnalysisResult:
    """Analyse a single clip given pre-selected ROIs. Side-effect free.

    Detects the stimulus onset (saturation/grayscale change in ``stim_roi``), then the
    first sustained motion in ``fish_roi`` via temporal-kernel energy. The energy series
    is computed **once**; coarse (strided) and refined (stride-1) detection run on the
    in-memory arrays with no video re-reads.

    No printing, GUI, or ``sys.exit``: raises :class:`VideoOpenError` for unreadable
    input, and returns ``None`` fields when a stage yields nothing (stimulus not found,
    baseline too short, no movement).
    """
    if params is None:
        params = AnalysisParams()

    owns_cap = cap is None
    if owns_cap:
        cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoOpenError(f"Failed to open video: {video_path}")
        ok, first = cap.read()
        if not ok:
            raise VideoOpenError(f"Empty video: {video_path}")

        kernel = get_kernel(params.kernel)
        H = len(kernel) // 2

        # --- Stimulus onset ------------------------------------------------
        base_sat, base_bgr = build_stim_baseline(
            cap, first, stim_roi, n=params.baseline_frames
        )
        stim_idx = find_stimulus(
            cap=cap,
            roi=stim_roi,
            base_sat=base_sat,
            base_bgr=base_bgr,
            max_frames=params.stim_max_frames,
            saturation_drop=params.saturation_drop,
            diff_thresh=params.diff_thresh,
            show=False,
        )
        if stim_idx is None:
            return AnalysisResult(None, None, None, float("nan"), [], [])

        # --- Baseline energy -> threshold (computed once) ------------------
        base_end = stim_idx - 1 - H
        base_start = max(base_end - (params.motion_baseline_n - 1), H)
        if base_start > base_end:
            return AnalysisResult(stim_idx, None, None, float("nan"), [], [])

        _, base_vals = energy_temporal_series(
            cap=cap,
            roi=fish_roi,
            center_start=base_start,
            center_end=base_end,
            kernel=kernel,
            norm=params.norm,
            smooth=params.smooth,
            smooth_ksize=params.smooth_ksize,
            smooth_sigma=params.smooth_sigma,
            viz=False,
        )
        if len(base_vals) < max(5, params.min_run + 2):
            return AnalysisResult(stim_idx, None, None, float("nan"), [], [])

        mu = float(np.mean(base_vals))
        sd = float(np.std(base_vals)) or 1e-6
        thr = mu + params.energy_sigma * sd

        # --- Scan energy after stim (computed once, stride-1) --------------
        scan_start = stim_idx + H
        scan_end = scan_start + params.motion_max_frames - 1
        centers, energies = energy_temporal_series(
            cap=cap,
            roi=fish_roi,
            center_start=scan_start,
            center_end=scan_end,
            kernel=kernel,
            norm=params.norm,
            smooth=params.smooth,
            smooth_ksize=params.smooth_ksize,
            smooth_sigma=params.smooth_sigma,
            viz=False,
            thr=thr,
        )

        # --- Detection on in-memory arrays (coarse then refine) ------------
        stride = max(1, int(params.stride))
        det_coarse = first_sustained(centers, energies, thr, params.min_run, stride)
        det_refined = det_coarse
        if det_coarse is not None:
            halfwin = (
                params.refine_halfwin
                if params.refine_halfwin is not None
                else max(2 * stride, 50)
            )
            r_start = max(scan_start, det_coarse - halfwin)
            r_end = min(scan_end, det_coarse + halfwin)
            sub = [(c, e) for c, e in zip(centers, energies) if r_start <= c <= r_end]
            if sub:
                det_ref = first_sustained(
                    [c for c, _ in sub], [e for _, e in sub], thr, params.min_run, 1
                )
                if det_ref is not None:
                    det_refined = det_ref

        return AnalysisResult(stim_idx, det_coarse, det_refined, thr, centers, energies)
    finally:
        if owns_cap:
            cap.release()


# ----------------------- CLI ------------------------------------------


def default_case_name(video: Path) -> str:
    """``<species>_<condition>_<stem>`` — includes the species dir to avoid the old
    ``out/`` name collision (sculpin & shiner both wrote ``circle34.png``)."""
    parts = video.resolve().parts
    cond = parts[-2] if len(parts) >= 2 else ""
    species = parts[-3] if len(parts) >= 3 else ""
    prefix = "_".join(p for p in (species, cond) if p)
    return f"{prefix}_{video.stem}" if prefix else video.stem


def main() -> None:
    args = parse_args()
    params = AnalysisParams()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # --- Interactive ROI acquisition (GUI) -----------------------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Failed to open video.", file=sys.stderr)
        sys.exit(1)
    ok, first = cap.read()
    if not ok:
        print("Empty video.", file=sys.stderr)
        sys.exit(1)

    playback_video(cap, source_fps=CAPTURE_FPS, window_name="Video " + args.video.stem)
    try:
        stim_roi = select_roi_click(first, "Stimulus ROI")
        fish_roi = select_roi_click(first, "Fish ROI")
    except ROISelectionCancelled:
        cap.release()
        print("ROI selection cancelled.", file=sys.stderr)
        sys.exit(2)
    cap.release()

    # --- Analysis (no GUI) ---------------------------------------------
    try:
        result = analyze_video(args.video, stim_roi, fish_roi, params)
    except VideoOpenError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if result.stim_idx is None:
        print("Stimulus not detected.", file=sys.stderr)
        sys.exit(3)
    print(f"Stimulus frame index: {result.stim_idx}")

    if result.det_coarse is None:
        print("No movement detected within scan window.")
    else:
        print(f"Coarse first-movement frame: {result.det_coarse}")
        print(f"Refined first-movement frame: {result.final_det_idx}")

    if result.centers:
        out_path = Path("out") / (default_case_name(args.video) + ".png")
        make_plot(
            result.centers,
            result.energies,
            result.threshold,
            result.stim_idx,
            result.final_det_idx,
            out_path,
        )


if __name__ == "__main__":
    main()
