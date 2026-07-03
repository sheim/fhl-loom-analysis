#!/usr/bin/env python3
"""
Interactive geometry marking (milestone M3.2).

Works on the reference-frame stack `batch.py` exported to the git-ignored `frames/` mirror — no
video access needed. Per clip, on the stack (step through with a/d or ←/→):

  1. Click the TWO tank corners on the monitor side  -> `annotation.tank_corners`
     (they span the tank width = 59 cm; baseline for the pixel<->cm scale).
  2. For up to FOUR fish, **starting with the first to move**, click HEAD then TAIL
     -> `annotation.fish = [{head, tail}, ...]` — index 0 is the first responder (encoded by
     marking order); the other three can be marked afterwards.

Marks are written into the same per-video annotation JSON (extensible `annotation` block).

Usage:
    uv run geometry.py videos/Sculpin_SloMo/flapping          # only un-marked clips (default)
    uv run geometry.py videos/Sculpin_SloMo/flapping --redo-all
    uv run geometry.py videos/Sculpin_SloMo/flapping/48.MP4   # one clip
    uv run geometry.py videos/Sculpin_SloMo/flapping --show   # display saved marks
"""

import argparse
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


def mark_clip(frames: List) -> dict:
    """Collect tank corners + up to 4 fish (head/tail each; the first marked is the first
    responder). Raises ``ROISelectionCancelled`` if the user aborts."""
    corners = afe.pick_points(
        frames, 2, "Tank corners on the MONITOR side (click 2)",
        point_labels=["corner 1", "corner 2"], window_name="Tank corners",
    )

    fish = []
    for i in range(MAX_FISH):
        prompt = f"Fish {i + 1}: click HEAD then TAIL"
        if i == 0:
            prompt += "  (the FIRST fish to respond)"
        pts = afe.pick_points(
            frames, 2, prompt, point_labels=["head", "tail"],
            window_name=f"Fish {i + 1}", allow_finish=(i > 0),
        )
        if pts is None:  # user pressed 'f' -> no more fish
            break
        fish.append({"head": list(pts[0]), "tail": list(pts[1])})

    return {"tank_corners": [list(c) for c in corners], "fish": fish}


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
        marks = mark_clip(frames)
    except afe.ROISelectionCancelled:
        print(f"  [skip] cancelled: {video.name}", file=sys.stderr)
        return False

    ann.annotation["tank_corners"] = marks["tank_corners"]
    ann.annotation["fish"] = marks["fish"]
    anno.save_annotation(video, ann)
    print(f"  {video.name}: corners set, {len(marks['fish'])} fish marked")
    return True


def show_marks(video: Path) -> None:
    ann = anno.load_annotation(video)
    if not is_marked(ann):
        print(f"  [none] {video.name}")
        return
    frames = load_frames(video)
    fishes = ann.annotation.get("fish") or []
    print(f"  {video.name}: {len(fishes)} fish (index 0 = first responder)")
    if not frames:
        return
    img = frames[len(frames) // 2].copy()
    corners = ann.annotation.get("tank_corners") or []
    for c in corners:
        cv2.circle(img, tuple(c), 5, (0, 255, 0), 2, cv2.LINE_AA)
    if len(corners) == 2:
        cv2.line(img, tuple(corners[0]), tuple(corners[1]), (0, 255, 0), 2, cv2.LINE_AA)
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
