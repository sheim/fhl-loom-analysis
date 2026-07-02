#!/usr/bin/env python3
"""
Batch fish-escape analyzer.

Imports the single-video pipeline from ``analyze_fish_energy`` and runs it over every
video in a folder, writing ``<folder>_results.csv`` (schema: ``filename,stim_idx,
final_det_idx``) plus an optional per-video energy plot under ``out/``.

Replaces the old stdout-scraping ``batch_analyze.sh``: results come straight from
``analyze_video()``'s return value, so there is no fragile text parsing. ROIs are still
selected interactively per video for now (saved/reused ROIs arrive in milestone M2).

Usage:
    uv run batch.py videos/Shiner_SloMo/circle
    uv run batch.py videos/Shiner_SloMo/circle --energy-sigma 5 --stride 5
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

import analyze_fish_energy as afe
import annotations as anno

Roi = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    d = afe.AnalysisParams()
    p = argparse.ArgumentParser(description="Batch fish-escape analyzer.")
    p.add_argument("folder", type=Path, help="Folder of videos to process")
    p.add_argument(
        "-o",
        "--out-csv",
        type=Path,
        default=None,
        help="Results CSV (default: <folder-name>_results.csv in CWD)",
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Skip per-video energy plots"
    )
    p.add_argument(
        "--from-annotations",
        action="store_true",
        help="Read ROIs from saved annotations and run headless (no ROI clicking)",
    )
    p.add_argument(
        "--update-annotations",
        action="store_true",
        help="Write the computed stim_idx/det back into each video's annotation cache",
    )
    # Tunable parameters (defaults mirror AnalysisParams)
    p.add_argument("--baseline-frames", type=int, default=d.baseline_frames)
    p.add_argument("--sat-drop", type=float, default=d.saturation_drop)
    p.add_argument("--diff-thresh", type=float, default=d.diff_thresh)
    p.add_argument("--motion-baseline-n", type=int, default=d.motion_baseline_n)
    p.add_argument("--energy-sigma", type=float, default=d.energy_sigma)
    p.add_argument("--min-run", type=int, default=d.min_run)
    p.add_argument("--motion-max-frames", type=int, default=d.motion_max_frames)
    p.add_argument("--norm", choices=["none", "zscore", "clahe"], default=d.norm)
    p.add_argument("--stride", type=int, default=d.stride)
    p.add_argument("--kernel", default=d.kernel)
    p.add_argument("--smooth", choices=["none", "gaussian", "box"], default=d.smooth)
    p.add_argument("--smooth-ksize", type=int, default=d.smooth_ksize)
    p.add_argument("--smooth-sigma", type=float, default=d.smooth_sigma)
    p.add_argument("--refine-halfwin", type=int, default=d.refine_halfwin)
    return p.parse_args()


def params_from_args(args: argparse.Namespace) -> afe.AnalysisParams:
    return afe.AnalysisParams(
        baseline_frames=args.baseline_frames,
        saturation_drop=args.sat_drop,
        diff_thresh=args.diff_thresh,
        motion_baseline_n=args.motion_baseline_n,
        energy_sigma=args.energy_sigma,
        min_run=args.min_run,
        motion_max_frames=args.motion_max_frames,
        norm=args.norm,
        stride=args.stride,
        kernel=args.kernel,
        smooth=args.smooth,
        smooth_ksize=args.smooth_ksize,
        smooth_sigma=args.smooth_sigma,
        refine_halfwin=args.refine_halfwin,
    )


def select_rois(video: Path) -> Optional[Tuple[Roi, Roi]]:
    """Interactive per-video ROI selection.

    Returns ``(stim_roi, fish_roi)``, or ``None`` if the clip is unreadable or the user
    cancels selection (``q``) — in which case the video is skipped, not the whole batch.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            print(f"[skip] cannot open {video.name}", file=sys.stderr)
            return None
        ok, first = cap.read()
        if not ok:
            print(f"[skip] empty video {video.name}", file=sys.stderr)
            return None
        try:
            stim_roi = afe.select_roi_click(first, f"{video.name} - Stimulus ROI")
            fish_roi = afe.select_roi_click(first, f"{video.name} - Fish ROI")
        except afe.ROISelectionCancelled:
            print(f"[skip] ROI selection cancelled: {video.name}", file=sys.stderr)
            return None
        return stim_roi, fish_roi
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    folder = args.folder
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    videos = anno.list_videos(folder)
    if not videos:
        print(f"No videos found in {folder}", file=sys.stderr)
        sys.exit(2)

    params = params_from_args(args)
    out_csv = args.out_csv or Path(f"{folder.name}_results.csv")

    rows: List[Tuple[str, Optional[int], Optional[int]]] = []
    for video in videos:
        print(f"\n=== {video.name} ===")

        if args.from_annotations:
            ann = anno.load_annotation(video)
            if ann is None:
                print("  [skip] no annotation", file=sys.stderr)
                continue
            if ann.disposition != "usable":
                print(f"  [skip] disposition={ann.disposition}", file=sys.stderr)
                continue
            if not ann.has_rois:
                print("  [skip] annotation has no ROIs", file=sys.stderr)
                continue
            stim_roi, fish_roi = ann.stim_roi, ann.fish_roi
        else:
            rois = select_rois(video)
            if rois is None:
                continue
            stim_roi, fish_roi = rois

        try:
            result = afe.analyze_video(video, stim_roi, fish_roi, params)
        except afe.VideoOpenError as e:
            print(f"  [skip] {e}", file=sys.stderr)
            continue

        print(f"  stim_idx={result.stim_idx}  first_movement={result.final_det_idx}")
        rows.append((video.name, result.stim_idx, result.final_det_idx))

        if args.update_annotations:
            ann = anno.load_annotation(video) or anno.Annotation(
                video=anno.video_rel(video)
            )
            ann.set_rois(stim_roi, fish_roi)
            ann.results = anno.results_dict(result, params)
            anno.save_annotation(video, ann)

        if not args.no_plots and result.centers:
            plot_path = Path("out") / (afe.default_case_name(video) + ".png")
            afe.make_plot(
                result.centers,
                result.energies,
                result.threshold,
                result.stim_idx,
                result.final_det_idx,
                plot_path,
            )

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "stim_idx", "final_det_idx"])
        for name, stim, det in rows:
            w.writerow(
                [name, "" if stim is None else stim, "" if det is None else det]
            )

    print(f"\nWrote {len(rows)} row(s) to {out_csv}")


if __name__ == "__main__":
    main()
