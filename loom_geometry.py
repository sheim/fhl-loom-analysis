#!/usr/bin/env python3
"""
Loom geometry (milestone M3.3).

For each clip, from the marked geometry (`tank_corners`, `fish`) and the detection results, compute:
  - the loom origin = midpoint of the two monitor-side tank corners (where the loom expands from);
  - the pixel->metre scale from the tank width (the two corners span TANK_WIDTH_M = 0.59 m);
  - the distance from the origin to the first-responding fish's head;
  - the visual angle the silhouette subtends on that fish's retina at movement onset:
    look up the silhouette width at the elapsed time, then theta = 2*atan((W/2) / distance).

Timing note: the detection frames (`det_refined`, `stim_idx`) are CAMERA frames at 240 fps, but the
diameter lookup table is at the MONITOR rate of 60 fps, so the elapsed lookup frame is
(det - stim) * 60/240 = (det - stim) / 4.

Caches the result in each clip's annotation JSON (`geometry` block, alongside `results`); pass
`-o FILE.csv` to also export an aggregated table for stats. `--show` draws the triangle.

Usage:
    uv run loom_geometry.py videos/Sculpin_SloMo/circle
    uv run loom_geometry.py videos/Sculpin_SloMo/circle/8.MP4 --show
"""

import argparse
import csv
import math
import sys
from bisect import bisect_left
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

import annotations as anno

TANK_WIDTH_M = 0.59       # the two marked corners span the tank width
CAMERA_FPS = 240.0
MONITOR_FPS = 60.0
LOOKUP_CSV = anno.REPO_ROOT / "diameter_lookup_table.csv"


# ----------------------- silhouette lookup ----------------------------


def load_lookup(path: Path = LOOKUP_CSV) -> Tuple[List[float], List[float]]:
    """(monitor_frame, silhouette diameter in metres) columns from the lookup table."""
    frames, diam = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frames.append(float(row["frame"]))
            diam.append(float(row["diameter_m"]))
    return frames, diam


def silhouette_m(lookup, monitor_frame: float) -> Tuple[float, bool]:
    """Linearly-interpolated silhouette diameter (m) at a fractional monitor frame.
    Returns (diameter_m, in_range)."""
    frames, diam = lookup
    if monitor_frame <= frames[0]:
        return diam[0], monitor_frame >= frames[0]
    if monitor_frame >= frames[-1]:
        return diam[-1], False               # past the end of the loom table
    i = bisect_left(frames, monitor_frame)
    f0, f1, d0, d1 = frames[i - 1], frames[i], diam[i - 1], diam[i]
    t = (monitor_frame - f0) / (f1 - f0)
    return d0 + t * (d1 - d0), True


# ----------------------- geometry -------------------------------------


def compute(ann: anno.Annotation, lookup) -> Optional[dict]:
    """Compute loom geometry for a clip, or None if the required marks/results are missing."""
    corners = ann.annotation.get("tank_corners")
    fish = ann.annotation.get("fish") or []
    r = ann.results or {}
    if not corners or len(corners) != 2 or not fish:
        return None
    stim = r.get("stim_idx")
    det = ann.annotation.get("manual_det")
    if det is None:
        det = r.get("det_refined")
    if stim is None or det is None:
        return None

    (x1, y1), (x2, y2) = corners
    origin = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    px_dist = math.hypot(x2 - x1, y2 - y1)
    if px_dist == 0:
        return None
    m_per_px = TANK_WIDTH_M / px_dist

    head = fish[0]["head"]
    dist_px = math.hypot(head[0] - origin[0], head[1] - origin[1])
    dist_m = dist_px * m_per_px

    latency_frames = det - stim
    monitor_frame = latency_frames * (MONITOR_FPS / CAMERA_FPS)   # 240 fps -> 60 fps
    sil_m, in_range = silhouette_m(lookup, monitor_frame)
    angle_deg = math.degrees(2 * math.atan((sil_m / 2) / dist_m)) if dist_m > 0 else float("nan")

    return {
        "origin": origin,
        "m_per_px": m_per_px,
        "head": tuple(head),
        "dist_m": dist_m,
        "stim": int(stim),
        "det": int(det),
        "latency_s": latency_frames / CAMERA_FPS,
        "monitor_frame": monitor_frame,
        "silhouette_m": sil_m,
        "angle_deg": angle_deg,
        "in_range": in_range,
    }


def load_frames(video: Path) -> List:
    d = anno.frames_dir(video)
    frames = []
    if d.is_dir():
        for p in sorted(d.glob("*.png")):
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
    return frames


def _text(img, lines) -> None:
    """Stacked white text with a dark outline for readability."""
    for i, txt in enumerate(lines):
        org = (10, 28 + i * 28)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def show_triangle(video: Path, ann: anno.Annotation, g: dict) -> None:
    """Overlay the full geometry on the exported frame: the screen (tank corners), all fish
    (head->tail, first responder highlighted), the loom triangle, and the frames/elapsed time."""
    frames = load_frames(video)
    if not frames:
        print(f"  [skip --show] no exported frames: {video.name}", file=sys.stderr)
        return
    img = frames[len(frames) // 2].copy()
    ox, oy = int(g["origin"][0]), int(g["origin"][1])
    hx, hy = int(g["head"][0]), int(g["head"][1])

    # --- screen: the monitor-side edge between the two tank corners ---
    corners = ann.annotation.get("tank_corners") or []
    if len(corners) == 2:
        c1 = (int(corners[0][0]), int(corners[0][1]))
        c2 = (int(corners[1][0]), int(corners[1][1]))
        cv2.line(img, c1, c2, (0, 255, 0), 2, cv2.LINE_AA)
        for c in (c1, c2):
            cv2.circle(img, c, 5, (0, 255, 0), 2, cv2.LINE_AA)
        mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
        cv2.putText(img, "screen 59cm", (mid[0] - 45, mid[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # --- all fish: head->tail arrows (index 0 = first responder, highlighted) ---
    for i, f in enumerate(ann.annotation.get("fish") or []):
        head = (int(f["head"][0]), int(f["head"][1]))
        tail = (int(f["tail"][0]), int(f["tail"][1]))
        color = (0, 255, 255) if i == 0 else (0, 165, 255)
        cv2.arrowedLine(img, tail, head, color, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(img, str(i + 1), (head[0] + 6, head[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # --- loom triangle for the first responder ---
    dx, dy = ox - hx, oy - hy
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L, dx / L                        # unit perpendicular to head->origin
    half_px = (g["silhouette_m"] / 2) / g["m_per_px"]
    b1 = (int(ox + px * half_px), int(oy + py * half_px))
    b2 = (int(ox - px * half_px), int(oy - py * half_px))
    cv2.line(img, (hx, hy), b1, (255, 255, 0), 1, cv2.LINE_AA)   # triangle sides
    cv2.line(img, (hx, hy), b2, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, b1, b2, (255, 0, 0), 2, cv2.LINE_AA)          # silhouette base
    cv2.line(img, (ox, oy), (hx, hy), (0, 255, 255), 1, cv2.LINE_AA)  # distance
    cv2.circle(img, (ox, oy), 6, (0, 255, 0), -1, cv2.LINE_AA)  # loom origin
    cv2.circle(img, (hx, hy), 6, (0, 0, 255), -1, cv2.LINE_AA)  # first responder head

    _text(img, [
        f"stim frame {g['stim']}   movement frame {g['det']}   "
        f"elapsed {g['latency_s']:.3f}s ({g['det'] - g['stim']} cam frames)",
        f"dist={g['dist_m'] * 100:.1f}cm   silhouette W={g['silhouette_m'] * 100:.1f}cm   "
        f"retina angle={g['angle_deg']:.1f}deg",
    ])
    cv2.imshow(f"{video.name} - loom geometry (any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------- CLI ------------------------------------------


def geometry_record(g: dict) -> dict:
    """The per-clip geometry cached into the annotation JSON's `geometry` block."""
    return {
        "loom_origin": [round(g["origin"][0], 1), round(g["origin"][1], 1)],
        "distance_cm": round(g["dist_m"] * 100, 2),
        "silhouette_cm": round(g["silhouette_m"] * 100, 2),
        "angle_deg": round(g["angle_deg"], 2),
        "monitor_frame": round(g["monitor_frame"], 2),
        "latency_s": round(g["latency_s"], 4),
        "in_range": g["in_range"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute loom geometry (distance + retinal angle).")
    p.add_argument("path", type=Path, help="Folder of videos, or a single video file")
    p.add_argument("-o", "--out-csv", type=Path, default=None,
                   help="Also export an aggregated CSV (per-clip geometry always goes into the JSON)")
    p.add_argument("--show", action="store_true", help="Draw the triangle on the exported frame")
    return p.parse_args()


def gather_targets(path: Path) -> List[Path]:
    if path.is_dir():
        return anno.list_videos(path)
    if path.is_file():
        return [path]
    print(f"Not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = parse_args()
    lookup = load_lookup()
    videos = gather_targets(args.path)

    rows = []
    for video in videos:
        ann = anno.load_annotation(video)
        if ann is None:
            print(f"  [skip] no annotation: {video.name}", file=sys.stderr)
            continue
        g = compute(ann, lookup)
        if g is None:
            print(f"  [skip] missing geometry/results: {video.name}", file=sys.stderr)
            continue

        ann.geometry = geometry_record(g)          # cache into the clip's JSON
        anno.save_annotation(video, ann)

        flag = "" if g["in_range"] else "  [latency past loom table!]"
        print(
            f"  {video.name}: dist={g['dist_m'] * 100:.1f}cm  "
            f"W={g['silhouette_m'] * 100:.1f}cm  angle={g['angle_deg']:.1f}deg  "
            f"(latency={g['latency_s']:.3f}s, monitor_frame={g['monitor_frame']:.1f}){flag}"
        )
        rows.append((video.name, g))
        if args.show:
            show_triangle(video, ann, g)

    if args.out_csv:                               # optional aggregation for stats
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "filename", "stim_idx", "det_idx", "latency_s",
                "monitor_frame", "distance_cm", "silhouette_cm", "angle_deg",
            ])
            for name, g in rows:
                w.writerow([
                    name, g["stim"], g["det"], f"{g['latency_s']:.4f}",
                    f"{g['monitor_frame']:.2f}", f"{g['dist_m'] * 100:.2f}",
                    f"{g['silhouette_m'] * 100:.2f}", f"{g['angle_deg']:.2f}",
                ])
        print(f"\nUpdated {len(rows)} annotation(s); exported table to {args.out_csv}")
    else:
        print(f"\nUpdated {len(rows)} annotation(s) (pass -o FILE.csv to also export a table).")


if __name__ == "__main__":
    main()
