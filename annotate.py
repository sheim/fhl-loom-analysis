#!/usr/bin/env python3
"""
Interactive annotation sweep (milestone M2).

Walk a folder (or a single video), and for each clip: play it once for orientation, record a
disposition, then click the stimulus and fish (first-responder) ROIs. The analyzer runs
immediately so the detected ``stim_idx`` / ``det_refined`` are cached alongside the ROIs in a
per-video JSON (see ``annotations.py``). Do this once; downstream analysis then runs headless
(``uv run batch.py <folder> --from-annotations``).

Modes:
    uv run annotate.py <folder>              # annotate only un-annotated videos (default)
    uv run annotate.py <folder> --redo-all   # re-annotate everything
    uv run annotate.py <folder> --redo 34.MP4  # re-annotate one clip
    uv run annotate.py <folder> --show       # display saved ROIs/results, no editing
    uv run annotate.py <video.MP4>           # a single clip also works
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2

import analyze_fish_energy as afe
import annotations as anno


def default_disposition(video: Path) -> str:
    """Guess a disposition from the path (``no_response``/``bad_video`` subfolders)."""
    parents = {p.name.lower() for p in video.resolve().parents}
    if "no_response" in parents:
        return "no_response"
    if "bad_video" in parents:
        return "bad_video"
    return "usable"


def prompt_disposition(default: str) -> str:
    opts = "[u]sable / [n]o_response / [b]ad_video"
    ans = input(f"  disposition {opts} (default {default}): ").strip().lower()
    mapping = {
        "u": "usable",
        "n": "no_response",
        "b": "bad_video",
        "usable": "usable",
        "no_response": "no_response",
        "bad_video": "bad_video",
        "": default,
    }
    return mapping.get(ans, default)


def annotate_video(
    video: Path, params: afe.AnalysisParams
) -> Optional[anno.Annotation]:
    """Interactively annotate one clip. Returns the Annotation, or None if skipped."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [skip] cannot open {video.name}", file=sys.stderr)
        return None
    ok, first = cap.read()
    if not ok:
        cap.release()
        print(f"  [skip] empty video {video.name}", file=sys.stderr)
        return None

    afe.playback_video(cap, fps=30, window_name=f"{video.name} (q to stop)")
    disp = prompt_disposition(default_disposition(video))
    ann = anno.Annotation(video=anno.video_rel(video), disposition=disp)

    if disp != "usable":
        cap.release()
        print(f"  disposition={disp} (no ROIs recorded)")
        return ann

    try:
        stim_roi = afe.select_roi_click(first, f"{video.name} - Stimulus ROI")
        fish_roi = afe.select_roi_click(
            first, f"{video.name} - Fish ROI (first responder)"
        )
    except afe.ROISelectionCancelled:
        cap.release()
        print(f"  [skip] ROI selection cancelled: {video.name}", file=sys.stderr)
        return None
    cap.release()

    ann.set_rois(stim_roi, fish_roi)
    try:
        result = afe.analyze_video(video, stim_roi, fish_roi, params)
        ann.results = anno.results_dict(result, params)
        print(
            f"  stim_idx={result.stim_idx}  det_refined={result.final_det_idx}"
        )
    except afe.VideoOpenError as e:
        print(f"  [warn] analysis failed, saving ROIs only: {e}", file=sys.stderr)
    return ann


def show_annotation(video: Path, params: afe.AnalysisParams) -> None:
    """Display the saved annotation for one clip (ROIs drawn on the first frame)."""
    ann = anno.load_annotation(video)
    if ann is None:
        print(f"  [none] {video.name}")
        return

    line = f"  {video.name}: disposition={ann.disposition}"
    if ann.results:
        stale = " [STALE: params differ from defaults]" if anno.results_stale(ann, params) else ""
        line += (
            f"  stim_idx={ann.results.get('stim_idx')}"
            f"  det_refined={ann.results.get('det_refined')}{stale}"
        )
    print(line)

    if not ann.has_rois:
        return
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    cap.release()
    if not ok:
        return
    vis = first.copy()
    for roi, label, color in (
        (ann.stim_roi, "stim", (0, 255, 0)),
        (ann.fish_roi, "fish", (0, 255, 255)),
    ):
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis, label, (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    cv2.imshow(f"{video.name} - saved ROIs (any key)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive per-video annotation sweep.")
    p.add_argument("path", type=Path, help="Folder of videos, or a single video file")
    p.add_argument(
        "--redo-all", action="store_true", help="Re-annotate every video (ignore existing)"
    )
    p.add_argument(
        "--redo", metavar="VIDEO", default=None,
        help="Re-annotate one clip (match by file name or stem)",
    )
    p.add_argument(
        "--show", "--review", action="store_true", dest="show",
        help="Display existing annotations (no editing)",
    )
    return p.parse_args()


def gather_targets(path: Path) -> List[Path]:
    if path.is_dir():
        return anno.list_videos(path)
    if path.is_file():
        return [path]
    print(f"Not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def matches_redo(video: Path, redo: str) -> bool:
    r = Path(redo)
    return video.name == redo or video.stem == r.stem or video.name == r.name


def main() -> None:
    args = parse_args()
    params = afe.AnalysisParams()
    targets = gather_targets(args.path)
    if not targets:
        print(f"No videos found in {args.path}", file=sys.stderr)
        sys.exit(2)

    if args.show:
        for v in targets:
            show_annotation(v, params)
        return

    if args.redo:
        targets = [v for v in targets if matches_redo(v, args.redo)]
        if not targets:
            print(f"No video matching --redo {args.redo!r}", file=sys.stderr)
            sys.exit(2)
    elif not args.redo_all:
        targets = [v for v in targets if anno.load_annotation(v) is None]

    if not targets:
        print("Nothing to annotate (all already annotated; use --redo-all or --redo).")
        return

    for v in targets:
        print(f"\n=== {v.name} ===")
        ann = annotate_video(v, params)
        if ann is None:
            continue
        p = anno.save_annotation(v, ann)
        try:
            shown = p.relative_to(anno.REPO_ROOT)
        except ValueError:
            shown = p
        print(f"  saved {shown}")


if __name__ == "__main__":
    main()
