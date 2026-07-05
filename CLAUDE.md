# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A computer-vision pipeline that measures **fish escape-response latency** from high-speed
(~240 fps) video: detect when a **visual stimulus** appears, detect when the **fish first
moves**, and report the gap as reaction time, aggregated across species (sculpin, shiner) and
conditions (circle, fixed, flapping, ghost). Local NumPy/OpenCV/Matplotlib only — no network,
no API keys. See `README.md` for full usage and `MILESTONES.md` for the refactor plan.

## Environment & commands

- Managed with **`uv`** (Python ≥ 3.13, pinned in `.python-version`; `uv.lock` committed).
- Install: `uv sync`. Run scripts with `uv run <script.py> ...`.
- Runtime deps are `numpy`, `opencv-python`, `matplotlib` (`scipy`/`ipython` are listed in
  `pyproject.toml` but unused).

```bash
uv run analyze_fish_energy.py <video.MP4>        # single video (interactive)
bash  sort_videos.sh videos/Shiner_SloMo         # footage prep (one-time)
uv run batch.py videos/Shiner_SloMo/circle       # batch → <folder>_results.csv
uv run analysis.py                               # aggregate CSVs → latency PDFs
```

## Code map

- `analyze_fish_energy.py` — **the canonical library + interactive single-video CLI**. The
  importable entry point is `analyze_video(video, stim_roi, fish_roi, params) -> AnalysisResult`
  (side-effect-free: no print/GUI/`sys.exit`). Underneath: `find_stimulus` (stimulus onset),
  the temporal motion-energy engine (`get_kernel`, `energy_temporal_series`), `first_sustained`
  (pure coarse/refine detector on precomputed arrays), and plotting/QA (`make_plot`,
  `save_debug_grid`). Parameters live in the `AnalysisParams` dataclass; `main()` is a thin CLI.
- `annotations.py` — per-video annotation metadata (M2): `Annotation` model + `load/save`,
  `annotation_path()` (mirrors `videos/` → git-tracked `annotations/`), `results_stale()`,
  `frames_dir()`/`export_frames()` (the `frames/` mirror). `from_dict`/`to_dict` preserve
  unknown keys (forward-compatible with M3 geometry fields).
- `annotate.py` — one-time interactive sweep: disposition + ROI clicks per video → JSON, caching
  `stim_idx`/`det_refined`. This is where the GUI is concentrated (`--redo-all/--redo/--show`).
- `batch.py` — batch runner (folder or single file) → `<species>_<condition>_results.csv`
  (`filename,stim_idx,final_det_idx`). Interactive ROIs by default, or `--from-annotations`
  (headless) reading saved ROIs; `--update-annotations` writes results back. Effective params =
  defaults ← annotation's `results.params` ← CLI flags. Also exports a 10-frame stack around
  the movement onset to the git-ignored `frames/` mirror (`--no-frames` to skip; M3 canvas).
- `monitor_side.py` — infers `annotation.monitor_side` (top/bottom) from the stim ROI's
  vertical position vs frame height (M3 geometry; no clicking).
- `geometry.py` — interactive M3 marking on the exported `frames/` stack: 2 tank corners +
  per-fish head/tail (first marked = first responder) → `annotation.tank_corners` /
  `annotation.fish`. Uses the `afe.pick_points` helper (frame-stepping point picker). Tank width = 59 cm.
- `fix_start.py` — manual movement-onset ("start") correction: scrub the video (±1/±30/±240),
  set `annotation.manual_det` + `results.det_refined`, and regenerate the `frames/` stack.
  `batch.py` respects `manual_det` (won't recompute over it).
- `loom_geometry.py` — M3.3: loom origin = midpoint of `tank_corners`; distance to `fish[0].head`
  (scaled by the 0.59 m tank width); retinal angle via `diameter_lookup_table.csv` (monitor 60 fps,
  elapsed lookup frame = `(det-stim)/4`), whose base lies ON the screen (tank-corner line) centred
  at the origin → `θ` = angle subtended at the head (general triangle, not isosceles). Plus **dθ/dt**
  at onset (M3.5): θ as a function of the fractional monitor frame, weighted-centred on the exact
  onset, finite-differenced in seconds with 1st/2nd/4th-order stencils (`--deriv-step-frames`).
  Caches into each clip's JSON `geometry` block (`-o` also exports an aggregated CSV);
  `--show`/`--save` render the annotated overlay.
  `diameter_lookup_table.csv` is a git-tracked input (`.gitignore` has `!diameter_lookup_table.csv`).
  The `Annotation` dataclass has a first-class `geometry` field (parallel to `results`; not wiped by
  `batch`).
- `analysis.py` — reads `{species}_{cond}_results.csv` from cwd, computes latency at
  `FPS = 240`, writes `*_latency_hist.pdf`.
- `sort_videos.sh` — sorts/renames raw `Trial_*.MP4` into condition folders.
- `annotations/` — git-tracked JSON sidecars (hand-made; keep in version control unlike `videos/`).

## Important gotchas

- **GUI only for annotation.** `annotate.py` and the interactive `batch.py`/single-video CLI
  use `cv2.imshow` + mouse callbacks and block on clicks — never run them to "verify" changes.
  Once a folder is annotated, `batch.py --from-annotations` is fully headless.
- **Verify headlessly.** Exercise `analyze_video` / `first_sustained` on a synthetic clip or
  arrays, the annotations round-trip, and `batch.py --from-annotations` (set `FISH_VIDEOS_DIR` /
  `FISH_ANNOTATIONS_DIR` to a scratch dir so tests don't touch the repo), plus `py_compile`.
- **Keep the library side-effect-free.** `analyze_video` and its helpers must not `print`,
  `sys.exit`, or open GUI windows — those belong in the CLI/`batch.py` layer. `select_roi_click`
  raises `ROISelectionCancelled`; unreadable input raises `VideoOpenError`.
- **Parameters live in `AnalysisParams`** (defaults reproduce the old hard-coded values). The
  single-video CLI uses defaults; `batch.py` exposes them as flags. Change defaults in one place.
- **Data is git-ignored.** `videos/` (~16 GB), `out/`, `*.png`, `*.csv`, `*.pdf` are all
  ignored. Result CSVs feed `analysis.py` and may contain dirty rows (negative `stim_idx`,
  stray whitespace) — validate before trusting (M4).
- `analysis.py` expects species-prefixed CSV names (e.g. `shiner_circle_results.csv`), but
  `batch.py` emits folder-named ones (`circle_results.csv`) — a manual rename bridges them today.

## Conventions

- Match the existing style: type hints, small pure helper functions, plots saved under `out/`.
- Latency uses `FPS = 240.0`; frame indices are ints; latency = `(final_det_idx − stim_idx)/FPS`.
- Prefer extending the canonical functions in `analyze_fish_energy.py` over re-implementing
  detection logic elsewhere.
