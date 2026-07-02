#!/usr/bin/env python3
"""
Batch fish-escape analyzer.

Imports the single-video pipeline from ``analyze_fish_energy`` and runs it over every
video in a folder, writing ``<species>_<condition>_results.csv`` (e.g.
``sculpin_flapping_results.csv``; schema ``filename,stim_idx,final_det_idx``) plus an
optional per-video energy plot under ``out/``.

Replaces the old stdout-scraping ``batch_analyze.sh``: results come straight from
``analyze_video()``'s return value, so there is no fragile text parsing. The target may be a
folder (whole condition) or a single video file (re-run one clip). With ``--from-annotations``
it reads saved ROIs and runs headless; combine with the tuning flags to sweep parameters.

Usage:
    uv run batch.py videos/Shiner_SloMo/circle --from-annotations
    # re-run ONE clip from its annotation with a higher detection threshold:
    uv run batch.py videos/Sculpin_SloMo/flapping/48.MP4 --from-annotations --energy-sigma 8
"""

import argparse
import csv
import dataclasses
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
    p.add_argument(
        "folder", type=Path, help="Folder of videos, or a single video file (re-run one clip)"
    )
    p.add_argument(
        "-o",
        "--out-csv",
        type=Path,
        default=None,
        help="Results CSV (default: <folder-name>_results.csv in CWD)",
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Skip saving per-video energy plots"
    )
    p.add_argument(
        "--show", action="store_true",
        help="Pop up the energy plot interactively (blocks until closed) - handy for tuning",
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
    # Tunable parameters. Default None so we can tell which flags you actually passed:
    # effective params = AnalysisParams defaults <- the clip's saved params (annotation) <-
    # these flags. A flag you pass always wins; unset flags fall back to the annotation/defaults.
    p.add_argument("--baseline-frames", type=int, default=None, help=f"(default {d.baseline_frames})")
    p.add_argument("--sat-drop", type=float, default=None, help=f"(default {d.saturation_drop})")
    p.add_argument("--diff-thresh", type=float, default=None, help=f"(default {d.diff_thresh})")
    p.add_argument("--motion-baseline-n", type=int, default=None, help=f"(default {d.motion_baseline_n})")
    p.add_argument("--energy-sigma", type=float, default=None, help=f"(default {d.energy_sigma})")
    p.add_argument("--min-run", type=int, default=None, help=f"(default {d.min_run})")
    p.add_argument("--motion-max-frames", type=int, default=None, help=f"(default {d.motion_max_frames})")
    p.add_argument("--norm", choices=["none", "zscore", "clahe"], default=None, help=f"(default {d.norm})")
    p.add_argument("--stride", type=int, default=None, help=f"(default {d.stride})")
    p.add_argument("--kernel", default=None, help=f"(default {d.kernel})")
    p.add_argument("--smooth", choices=["none", "gaussian", "box"], default=None, help=f"(default {d.smooth})")
    p.add_argument("--smooth-ksize", type=int, default=None, help=f"(default {d.smooth_ksize})")
    p.add_argument("--smooth-sigma", type=float, default=None, help=f"(default {d.smooth_sigma})")
    p.add_argument("--refine-halfwin", type=int, default=None, help=f"(default {d.refine_halfwin})")
    return p.parse_args()


# CLI dest -> AnalysisParams field name
_CLI_TO_FIELD = {
    "baseline_frames": "baseline_frames",
    "sat_drop": "saturation_drop",
    "diff_thresh": "diff_thresh",
    "motion_baseline_n": "motion_baseline_n",
    "energy_sigma": "energy_sigma",
    "min_run": "min_run",
    "motion_max_frames": "motion_max_frames",
    "norm": "norm",
    "stride": "stride",
    "kernel": "kernel",
    "smooth": "smooth",
    "smooth_ksize": "smooth_ksize",
    "smooth_sigma": "smooth_sigma",
    "refine_halfwin": "refine_halfwin",
}
_PARAM_FIELDS = {f.name for f in dataclasses.fields(afe.AnalysisParams)}


def resolve_params(
    args: argparse.Namespace, ann: Optional[anno.Annotation]
) -> afe.AnalysisParams:
    """Effective params: ``AnalysisParams`` defaults <- the clip's saved params (its
    annotation's ``results.params``) <- explicitly-passed CLI flags (a flag you pass wins).

    So per-clip tuning persists in the annotation (write it with ``--update-annotations``),
    and a one-off ``--energy-sigma`` on the command line still overrides it.
    """
    values = dataclasses.asdict(afe.AnalysisParams())
    if ann is not None and ann.results:
        stored = ann.results.get("params") or {}
        values.update({k: v for k, v in stored.items() if k in _PARAM_FIELDS})
    for dest, field in _CLI_TO_FIELD.items():
        v = getattr(args, dest)
        if v is not None:
            values[field] = v
    return afe.AnalysisParams(**values)


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


def results_csv_name(target: Path) -> str:
    """``<species>_<condition>_results.csv`` (e.g. ``sculpin_flapping_results.csv``).

    species = first token of the folder *above* the condition folder, lowercased
    (``Sculpin_SloMo`` -> ``sculpin``); condition = the containing folder. This matches the
    names ``analysis.py`` reads, so the CSV needs no manual rename.
    """
    r = target.resolve()
    if target.is_dir():
        condition, species_dir = r.name, r.parent.name
    else:
        condition, species_dir = r.parent.name, r.parent.parent.name
    species = species_dir.split("_")[0].lower()
    stem = f"{species}_{condition}" if species else condition
    return f"{stem}_results.csv"


def main() -> None:
    args = parse_args()
    target = args.folder
    if target.is_dir():
        videos = anno.list_videos(target)
    elif target.is_file():
        videos = [target]                      # single-clip re-run (e.g. to tune thresholds)
    else:
        print(f"Not a file or directory: {target}", file=sys.stderr)
        sys.exit(1)

    if not videos:
        print(f"No videos found in {target}", file=sys.stderr)
        sys.exit(2)

    out_csv = args.out_csv or Path(results_csv_name(target))

    rows: List[Tuple[str, Optional[int], Optional[int]]] = []
    for video in videos:
        print(f"\n=== {video.name} ===")
        ann = anno.load_annotation(video)

        if args.from_annotations:
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
            detected_blip = ann.annotation.get("detected_blip", "first")
        else:
            rois = select_rois(video)
            if rois is None:
                continue
            stim_roi, fish_roi = rois
            detected_blip = ann.annotation.get("detected_blip", "first") if ann else "first"

        params = resolve_params(args, ann)
        try:
            result = afe.analyze_video(video, stim_roi, fish_roi, params)
        except afe.VideoOpenError as e:
            print(f"  [skip] {e}", file=sys.stderr)
            continue

        # Report the first-blip reference (corrected for a middle/last detection).
        stim_ref = anno.corrected_stim_idx(result.stim_idx, detected_blip)
        note = f" (detected {result.stim_idx}, blip={detected_blip})" if detected_blip != "first" else ""
        emax = max(result.energies) if result.energies else float("nan")
        print(
            f"  stim_idx={stim_ref}{note}  first_movement={result.final_det_idx}"
            f"  thr={result.threshold:.3g}  Emax={emax:.3g}  sigma={params.energy_sigma:g}"
        )
        rows.append((video.name, stim_ref, result.final_det_idx))

        if args.update_annotations:
            a = ann or anno.Annotation(video=anno.video_rel(video))
            a.set_rois(stim_roi, fish_roi)
            a.annotation.setdefault("detected_blip", detected_blip)
            a.results = anno.results_dict(
                result, params, a.annotation.get("detected_blip", "first")
            )
            anno.save_annotation(video, a)

        if (not args.no_plots or args.show) and result.centers:
            plot_path = (
                None if args.no_plots
                else Path("out") / (afe.default_case_name(video) + ".png")
            )
            afe.make_plot(
                result.centers,
                result.energies,
                result.threshold,
                result.stim_idx,
                result.final_det_idx,
                plot_path,
                show=args.show,
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
