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
    uv run annotate.py <folder> --show       # display saved ROIs/results for all, no editing
    uv run annotate.py <folder> --show 34.MP4  # display just one clip's annotation
    uv run annotate.py <folder> --show-results  # text-only results, no ROI window (headless)
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
    """Clickable popup for the video disposition.

    Click a button, or use the [u]/[n]/[b] hotkeys; Space/Enter takes the highlighted default.
    Raises ``ROISelectionCancelled`` on q/Esc.
    """
    return afe.choice_popup(
        "Disposition — click, or [u]/[n]/[b], Space = default",
        [("Usable", "usable"), ("No response", "no_response"), ("Bad video", "bad_video")],
        default=default,
        window_name="Disposition",
    )


def prompt_detected_blip(default: str = "first") -> str:
    """Popup: which of the 3 on-screen blips did auto-detection catch?

    The stimulus blip appears 3x per clip, 1 s apart (start, +240 frames, +480 at 240 FPS).
    Normally the detector locks onto the first; but if recording started late the first blip is
    gone, so it catches the middle (or last) one — which shifts the stimulus reference. Click,
    or [f]/[m]/[l]; Space/Enter takes the default (first).
    """
    return afe.choice_popup(
        "Detected blip is the... (first = normal)",
        [("First (default)", "first"), ("Middle", "middle"), ("Last", "last")],
        default=default,
        window_name="Detected blip",
    )


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

    afe.playback_video(
        cap, source_fps=afe.CAPTURE_FPS, window_name=f"{video.name} (q to stop)"
    )
    try:
        disp = prompt_disposition(default_disposition(video))
    except afe.ROISelectionCancelled:
        cap.release()
        print(f"  [skip] cancelled: {video.name}", file=sys.stderr)
        return None
    ann = anno.Annotation(video=anno.video_rel(video), disposition=disp)

    if disp != "usable":
        cap.release()
        print(f"  disposition={disp} (no ROIs recorded)")
        return ann

    try:
        ann.annotation["detected_blip"] = prompt_detected_blip()
    except afe.ROISelectionCancelled:
        cap.release()
        print(f"  [skip] cancelled: {video.name}", file=sys.stderr)
        return None

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

    blip = ann.annotation.get("detected_blip", "first")
    ann.set_rois(stim_roi, fish_roi)
    try:
        result = afe.analyze_video(video, stim_roi, fish_roi, params)
        ann.results = anno.results_dict(result, params, blip)
        ref = ann.results["stim_idx"]
        note = f" (detected {result.stim_idx}, blip={blip})" if blip != "first" else ""
        print(f"  stim_ref={ref}{note}  det_refined={result.final_det_idx}")
    except afe.VideoOpenError as e:
        print(f"  [warn] analysis failed, saving ROIs only: {e}", file=sys.stderr)
    return ann


def show_annotation(
    video: Path, params: afe.AnalysisParams, visual: bool = True
) -> None:
    """Print the saved annotation for one clip; if ``visual``, also draw its ROIs in a window."""
    ann = anno.load_annotation(video)
    if ann is None:
        print(f"  [none] {video.name}")
        return

    line = f"  {video.name}: disposition={ann.disposition}"
    blip = ann.annotation.get("detected_blip")
    if blip:
        line += f"  blip={blip}"
    side = ann.annotation.get("monitor_side")
    if side:
        line += f"  monitor={side}"
    if ann.results:
        stale = " [STALE: params differ from defaults]" if anno.results_stale(ann, params) else ""
        stim = ann.results.get("stim_idx")
        detected = ann.results.get("stim_idx_detected")
        note = f" (detected {detected})" if detected is not None and detected != stim else ""
        line += (
            f"  stim_idx={stim}{note}"
            f"  det_refined={ann.results.get('det_refined')}{stale}"
        )
    print(line)

    if not visual or not ann.has_rois:
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
        "--show", "--review", nargs="?", const="", default=None, metavar="VIDEO", dest="show",
        help="Display existing annotations (ROI window); optionally name one clip to show just it",
    )
    p.add_argument(
        "--show-results", "--show_results", nargs="?", const="", default=None, metavar="VIDEO",
        dest="show_results",
        help="Like --show but text only (no ROI window); optionally name one clip",
    )
    return p.parse_args()


def gather_targets(path: Path) -> List[Path]:
    if path.is_dir():
        return anno.list_videos(path)
    if path.is_file():
        return [path]
    print(f"Not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def matches_name(video: Path, name: str) -> bool:
    r = Path(name)
    return video.name == name or video.stem == r.stem or video.name == r.name


def main() -> None:
    args = parse_args()
    params = afe.AnalysisParams()
    targets = gather_targets(args.path)
    if not targets:
        print(f"No videos found in {args.path}", file=sys.stderr)
        sys.exit(2)

    show_arg = args.show if args.show is not None else args.show_results
    if show_arg is not None:
        visual = args.show is not None  # --show -> ROI window; --show-results -> text only
        if show_arg:  # a specific video name was passed
            show_targets = [v for v in targets if matches_name(v, show_arg)]
            if not show_targets:
                print(f"No video matching {show_arg!r}", file=sys.stderr)
                sys.exit(2)
        else:
            show_targets = targets
        for v in show_targets:
            show_annotation(v, params, visual=visual)
        return

    if args.redo:
        targets = [v for v in targets if matches_name(v, args.redo)]
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
