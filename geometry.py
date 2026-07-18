#!/usr/bin/env python3
"""
Interactive geometry marking (milestone M3.2).

Works on the reference-frame stack `batch.py` exported to the git-ignored `frames/` mirror — no
video access needed. Per clip, on the stack (step through with a/d or ←/→):

  1. Click the TWO tank corners on the monitor side  -> `annotation.tank_corners`
     (they span the tank width = 59 cm; baseline for the pixel<->cm scale).
  2. Click the OTHER TWO tank corners (the far edge)  -> `annotation.tank_far_corners`.
     Together the four corners give the full tank quad for a perspective-correct mapping.
     If a far corner is OFF-FRAME, pick "reconstruct": click the visible far corner + a point on
     each of its two edges, and the off-frame corner is their line intersection (may be off-frame).
  3. Pick the tank depth (2nd edge): 44 or 30 cm (Space=44 default) -> `annotation.tank_depth_cm`.
     The 44 cm tank is used for all shiner experiments.
  4. For up to FOUR fish, **starting with the first to move**, click HEAD then TAIL
     -> `annotation.fish = [{head, tail}, ...]` — index 0 is the first responder (encoded by
     marking order); the other three can be marked afterwards.

Marks are written into the same per-video annotation JSON (extensible `annotation` block).

Re-marking (`--redo`/`--redo-all`) pre-loads and draws any existing marks: press Enter to keep
each, or `r` to reset and re-pick. Handy after `fix_start.py` moves the onset — keep the tank
corners (Enter) and re-mark just the fish (r).

Usage:
    uv run geometry.py videos/Sculpin_SloMo/flapping          # only un-marked clips (default)
    uv run geometry.py videos/Sculpin_SloMo/flapping --redo-all
    uv run geometry.py videos/Sculpin_SloMo/flapping/48.MP4   # one clip
    uv run geometry.py videos/Sculpin_SloMo/flapping --show   # display saved marks
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

import cv2

import analyze_fish_energy as afe
import annotations as anno

MAX_FISH = 4


def load_frames(video: Path) -> List:
    """Load the exported reference-frame stack for a clip (sorted), or [] if none."""
    d = anno.frames_dir(video)
    if not d.is_dir():
        return []
    frames = [cv2.imread(str(p)) for p in sorted(d.glob("*.png"))]
    return [f for f in frames if f is not None]


def is_marked(ann: Optional[anno.Annotation]) -> bool:
    return bool(ann and (ann.annotation.get("tank_corners") or ann.annotation.get("fish")))


def _line_intersection(a1, a2, b1, b2):
    """Intersection point of line a1–a2 with line b1–b2, or None if they're ~parallel. The point may
    lie outside the frame (that's the whole point — it recovers an off-frame corner)."""
    (x1, y1), (x2, y2) = a1, a2
    (x3, y3), (x4, y4) = b1, b2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    pa, pb = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return ((pa * (x3 - x4) - (x1 - x2) * pb) / den,
            (pa * (y3 - y4) - (y1 - y2) * pb) / den)


def _reference_image(frames: List, near=None, far=None):
    """Middle frame with any already-known corners drawn — the companion image shown beside the
    dialogs so the user can visually double-check while answering."""
    img = frames[len(frames) // 2].copy()
    for c in near or []:
        cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 255, 0), 2, cv2.LINE_AA)
    if near and len(near) == 2:
        cv2.line(img, (int(near[0][0]), int(near[0][1])), (int(near[1][0]), int(near[1][1])),
                 (0, 255, 0), 2, cv2.LINE_AA)
    for c in far or []:
        cv2.circle(img, (int(c[0]), int(c[1])), 5, (255, 200, 0), 2, cv2.LINE_AA)
    return img


def _popup_with_frame(frames: List, near, far, title, options, **kw) -> str:
    """``afe.choice_popup`` with the reference frame kept up in a companion window for the duration."""
    win = "Reference frame"
    ref = _reference_image(frames, near, far)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, ref)
    cv2.waitKey(1)
    try:
        return afe.choice_popup(title, options, companion=(win, ref), **kw)
    finally:
        cv2.destroyWindow(win)


def pick_far_corners(frames: List, near_corners, existing_far):
    """Return ``(far_corners, reconstructed)`` — the two far tank corners as ``[[x,y],[x,y]]``.

    If both are visible, click them (2 points). If one is **out of frame**, choose "reconstruct":
    click the visible far corner, then one point along each of its two tank edges (the far wall and
    the side wall); the off-frame corner is their **line intersection** (may lie outside the frame).
    ``reconstructed`` is True in that case (provenance for the perspective step, M3.7)."""
    prompt = "Far tank corners — both visible? (Space = both)"
    if existing_far:
        prompt += "  [redo]"
    mode = _popup_with_frame(
        frames, near_corners, None,
        prompt,
        [("Both visible", "both"), ("One off-frame (reconstruct)", "one")],
        default="both",
        note=["If a far corner is off-frame, reconstruct it by line intersection:",
              "click the visible far corner, then a point on each of its two edges."],
        window_name="Far corners",
    )
    if mode == "both":
        far = afe.pick_points(
            frames, 2, "The two far tank corners (click 2)",
            point_labels=["corner 3", "corner 4"], window_name="Far tank corners",
            initial=existing_far,
        )
        return [list(c) for c in far], False

    vis = afe.pick_points(frames, 1, "The VISIBLE far corner (click 1)",
                          point_labels=["far corner"], window_name="Visible far corner")[0]
    n0, n1 = near_corners
    # the visible far corner pairs with its nearest near corner; the missing one adjoins the other
    if math.dist(vis, n0) <= math.dist(vis, n1):
        n_near, n_adj = n0, n1
    else:
        n_near, n_adj = n1, n0
    far_pt = afe.pick_points(
        frames, 1, "A point on the FAR wall (far tank edge, toward the off-frame corner)",
        point_labels=["far-edge pt"], window_name="Far edge")[0]
    side_pt = afe.pick_points(
        frames, 1, "A point on the SIDE wall (from that near corner, toward the off-frame corner)",
        point_labels=["side-edge pt"], window_name="Side edge")[0]
    inter = _line_intersection(vis, far_pt, n_adj, side_pt)
    if inter is None:                                # edges ~parallel → parallelogram completion
        inter = (n_adj[0] + (vis[0] - n_near[0]), n_adj[1] + (vis[1] - n_near[1]))
    return [list(vis), [int(round(inter[0])), int(round(inter[1]))]], True


def mark_clip(frames: List, existing: Optional[dict] = None) -> dict:
    """Collect tank corners + up to 4 fish (head/tail each; the first marked is the first
    responder). If ``existing`` marks are supplied they are **pre-loaded and shown**, so the user
    can accept each with Enter or press ``r`` to reset and re-pick (e.g. keep the tank corners but
    re-mark the fish after the onset frame changed). Raises ``ROISelectionCancelled`` if aborted."""
    existing = existing or {}

    prev_corners = existing.get("tank_corners")
    corner_prompt = "Tank corners on the MONITOR side (click 2)"
    if prev_corners:
        corner_prompt += "  [existing: Enter=keep, r=redo]"
    corners = afe.pick_points(
        frames, 2, corner_prompt,
        point_labels=["corner 1", "corner 2"], window_name="Tank corners",
        initial=prev_corners,
    )

    far_corners, far_reconstructed = pick_far_corners(
        frames, corners, existing.get("tank_far_corners"))

    prev_depth = existing.get("tank_depth_cm")
    default_depth = str(prev_depth) if prev_depth in (44, 30) else "44"
    depth_cm = int(_popup_with_frame(
        frames, corners, far_corners,
        "Tank depth (2nd edge) — Space = 44 (default)",
        [("44 cm", "44"), ("30 cm", "30")],
        default=default_depth,
        note=["The 44 cm tank is used for ALL shiner experiments,",
              "and has a 1 cm grid plexiglass lying somewhere."],
        window_name="Tank depth",
    ))

    prev_fish = existing.get("fish") or []
    fish = []
    for i in range(MAX_FISH):
        prompt = f"Fish {i + 1}: click HEAD then TAIL"
        if i == 0:
            prompt += "  (the FIRST fish to respond)"
        init = None
        if i < len(prev_fish):
            init = [prev_fish[i]["head"], prev_fish[i]["tail"]]
            prompt += "  [existing: Enter=keep, r=redo]"
        pts = afe.pick_points(
            frames, 2, prompt, point_labels=["head", "tail"],
            window_name=f"Fish {i + 1}", allow_finish=(i > 0), initial=init,
        )
        if pts is None:  # user pressed 'f' -> no more fish
            break
        fish.append({"head": list(pts[0]), "tail": list(pts[1])})

    return {
        "tank_corners": [list(c) for c in corners],
        "tank_far_corners": far_corners,
        "tank_far_reconstructed": far_reconstructed,
        "tank_depth_cm": depth_cm,
        "fish": fish,
    }


def mark_video(video: Path) -> bool:
    """Mark one clip and save. Returns True if written, False if skipped."""
    ann = anno.load_annotation(video)
    if ann is None:
        print(f"  [skip] no annotation: {video.name}", file=sys.stderr)
        return False
    frames = load_frames(video)
    if not frames:
        print(f"  [skip] no exported frames (run batch first): {video.name}", file=sys.stderr)
        return False
    try:
        marks = mark_clip(frames, ann.annotation)
    except afe.ROISelectionCancelled:
        print(f"  [skip] cancelled: {video.name}", file=sys.stderr)
        return False

    ann.annotation["tank_corners"] = marks["tank_corners"]
    ann.annotation["tank_far_corners"] = marks["tank_far_corners"]
    ann.annotation["tank_far_reconstructed"] = marks["tank_far_reconstructed"]
    ann.annotation["tank_depth_cm"] = marks["tank_depth_cm"]
    ann.annotation["fish"] = marks["fish"]
    anno.save_annotation(video, ann)
    recon = "  (1 far corner reconstructed)" if marks["tank_far_reconstructed"] else ""
    print(f"  {video.name}: 4 corners set (depth {marks['tank_depth_cm']}cm), "
          f"{len(marks['fish'])} fish marked{recon}")
    return True


def show_marks(video: Path) -> None:
    ann = anno.load_annotation(video)
    if not is_marked(ann):
        print(f"  [none] {video.name}")
        return
    frames = load_frames(video)
    fishes = ann.annotation.get("fish") or []
    depth = ann.annotation.get("tank_depth_cm")
    recon = "  [far corner reconstructed]" if ann.annotation.get("tank_far_reconstructed") else ""
    print(f"  {video.name}: {len(fishes)} fish (index 0 = first responder)"
          f"{f', tank depth {depth}cm' if depth else ''}{recon}")
    if not frames:
        return
    img = frames[len(frames) // 2].copy()
    corners = ann.annotation.get("tank_corners") or []
    far = ann.annotation.get("tank_far_corners") or []
    for c in corners:
        cv2.circle(img, tuple(c), 5, (0, 255, 0), 2, cv2.LINE_AA)
    if len(corners) == 2:
        cv2.line(img, tuple(corners[0]), tuple(corners[1]), (0, 255, 0), 2, cv2.LINE_AA)
    for c in far:                                            # far edge (2nd edge)
        cv2.circle(img, tuple(c), 5, (255, 200, 0), 2, cv2.LINE_AA)
    if len(corners) == 2 and len(far) == 2:
        # close the quad: pair each far corner to its nearest near corner
        f0 = far[0] if (math.dist(far[0], corners[0]) <= math.dist(far[1], corners[0])) else far[1]
        f1 = far[1] if f0 is far[0] else far[0]
        cv2.line(img, tuple(corners[0]), tuple(f0), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(img, tuple(corners[1]), tuple(f1), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(img, tuple(f0), tuple(f1), (255, 200, 0), 1, cv2.LINE_AA)
    for i, f in enumerate(fishes):
        head, tail = tuple(f["head"]), tuple(f["tail"])
        color = (0, 255, 255) if i == 0 else (0, 165, 255)   # first responder highlighted
        cv2.arrowedLine(img, tail, head, color, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(img, str(i + 1), (head[0] + 6, head[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imshow(f"{video.name} - geometry (any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mark tank corners + fish head/tail on the frame stack.")
    p.add_argument("path", type=Path, help="Folder of videos, or a single video file")
    p.add_argument("--redo-all", action="store_true", help="Re-mark every clip (ignore existing)")
    p.add_argument("--redo", metavar="VIDEO", default=None, help="Re-mark one clip (by name/stem)")
    p.add_argument("--show", "--review", nargs="?", const="", default=None, metavar="VIDEO",
                   dest="show", help="Display saved marks; optionally name one clip")
    return p.parse_args()


def matches_name(video: Path, name: str) -> bool:
    r = Path(name)
    return video.name == name or video.stem == r.stem or video.name == r.name


def gather_targets(path: Path) -> List[Path]:
    if path.is_dir():
        return anno.list_videos(path)
    if path.is_file():
        return [path]
    print(f"Not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = parse_args()
    targets = gather_targets(args.path)
    if not targets:
        print(f"No videos found in {args.path}", file=sys.stderr)
        sys.exit(2)

    if args.show is not None:
        chosen = [v for v in targets if matches_name(v, args.show)] if args.show else targets
        if args.show and not chosen:
            print(f"No video matching --show {args.show!r}", file=sys.stderr)
            sys.exit(2)
        for v in chosen:
            show_marks(v)
        return

    if args.redo:
        targets = [v for v in targets if matches_name(v, args.redo)]
        if not targets:
            print(f"No video matching --redo {args.redo!r}", file=sys.stderr)
            sys.exit(2)
    elif not args.redo_all:
        targets = [v for v in targets if not is_marked(anno.load_annotation(v))]

    if not targets:
        print("Nothing to mark (all already marked; use --redo-all or --redo).")
        return

    for video in targets:
        print(f"\n=== {video.name} ===")
        mark_video(video)


if __name__ == "__main__":
    main()
