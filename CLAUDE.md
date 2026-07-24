# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A computer-vision pipeline that measures **fish escape-response latency** from high-speed
(~240 fps) video: detect when a **visual stimulus** appears, detect when the **fish first
moves**, and report the gap as reaction time, aggregated across species (sculpin, shiner) and
conditions (circle, fixed, flapping, ghost). Local-only (no network / API keys). See `README.md`
for full usage and `MILESTONES.md` for the refactor plan + status.

## Environment & commands

- Managed with **`uv`** (Python ≥ 3.13, pinned in `.python-version`; `uv.lock` committed).
- Install: `uv sync`. Run scripts with `uv run <script.py> ...`.
- Deps: `numpy`, `opencv-python`, `matplotlib` (core pipeline); `scipy`, `pandas`, `marimo`,
  `plotly`, `altair` (M4 analysis). See `pyproject.toml`.

```bash
uv run analyze_fish_energy.py <video.MP4>            # single video (interactive)
bash  sort_videos.sh videos/Shiner_SloMo             # footage prep (one-time)
uv run batch.py videos/Shiner_SloMo/circle           # detect → <folder>_results.csv (+ frames/)
uv run geometry.py videos/Sculpin_SloMo/circle       # mark 4 tank corners + depth + fish (GUI)
uv run loom_geometry.py videos/Sculpin_SloMo/circle  # loom geometry → JSON geometry block
uv run python plots.py                               # M4 static plots → out/analysis/
uv run marimo edit notebooks/responses.py            # M4 interactive notebook
```

## Code map

- `analyze_fish_energy.py` — **the canonical library + interactive single-video CLI**. The
  importable entry point is `analyze_video(video, stim_roi, fish_roi, params) -> AnalysisResult`
  (side-effect-free: no print/GUI/`sys.exit`). Underneath: `find_stimulus` (stimulus onset),
  the temporal motion-energy engine (`get_kernel`, `energy_temporal_series`), `first_sustained`
  (pure coarse/refine detector on precomputed arrays), and plotting/QA (`make_plot`), plus the GUI
  helpers `pick_points`/`choice_popup`. Parameters live in the `AnalysisParams` dataclass; `main()`
  is a thin CLI.
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
- `geometry.py` — interactive M3.2 marking on the exported `frames/` stack: the **4 tank corners**
  (2 monitor-side `tank_corners` + 2 `tank_far_corners`, with off-frame reconstruction via edge
  intersection → `tank_far_reconstructed`), the **tank depth** (`tank_depth_cm` = 44/30, picked in a
  popup), and per-fish head/tail (first marked = first responder → `annotation.fish`). Uses
  `afe.pick_points` + `afe.choice_popup`. Tank width = 59 cm; depth 44 (shiner) or 44/30 (sculpin).
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
  `--show`/`--save` render the annotated overlay (full tank quad + a cached `tank` record of the
  sides & lengths). Also the **analytic closed-form dθ/dt** (M3.6, the headline value) and a
  **perspective-correct** mode (M4.2): `tank_frame` + `tank.py` build a homography from the 4 corners
  → true tank-cm, so distance/angle/dθ/dt are computed in cm (`perspective_corrected` flag; linear
  fallback for a degenerate quad).
  `diameter_lookup_table.csv` is a git-tracked input (`.gitignore` has `!diameter_lookup_table.csv`).
  The `Annotation` dataclass has a first-class `geometry` field (parallel to `results`; not wiped by
  `batch`).
- **M4 analysis stack** (aggregate the per-clip JSONs → stats + visualisations; marimo notebooks that
  also run as scripts to save plots — supersedes the old `analysis.py`):
  - `dataset.py` — JSONs → tidy per-clip DataFrame (`load_dataframe`); `clip_key`, `exclude`,
    `outlier_scores` (robust-z outlier candidates).
  - `plots.py` — all figures: `response_histograms` (+ distribution-fit overlay), `sensitivity_*`,
    `fish_overlay`/`fish_tankmap`/`fish_interactive` (positions); `save_all` → `out/analysis/`.
  - `fits.py` (distribution fitting, AICc), `sensitivity.py` (dθ/dt timing sensitivity),
    `positions.py` (fish registered into loom-centred cm, perspective via `tank`).
  - `tank.py` — pure homography `image px → tank cm` (`homography`/`to_cm`/`tank_record`), shared by
    `loom_geometry` + `positions`.
  - `notebooks/responses.py`, `notebooks/positions.py` — marimo (interactive `marimo edit`, or run as
    a plain script for saved plots).
- `sort_videos.sh` — sorts/renames raw `Trial_*.MP4` into condition folders.
- `annotations/` — hand-made JSON sidecars (one per clip). **Currently git-untracked** (a pending
  decision — expensive to redo, so consider re-tracking).

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
- **Data is git-ignored.** `videos/` (~16 GB), `out/`, `frames/`, and `*.png`/`*.csv`/`*.pdf` are
  all ignored (except `!diameter_lookup_table.csv`). The M4 analysis reads the annotation JSONs
  directly (not the batch CSVs).
- **Re-running `loom_geometry.py` rewrites every clip's `geometry` block** (idempotent). It's the
  canonical way to repopulate derived metrics after a code change — back up `annotations/` first
  (they're untracked), then diff to confirm a surgical change.

## Conventions

- Match the existing style: type hints, small pure helper functions, plots saved under `out/`.
- Latency uses `FPS = 240.0`; frame indices are ints; latency = `(final_det_idx − stim_idx)/FPS`.
- Prefer extending the canonical functions in `analyze_fish_energy.py` over re-implementing
  detection logic elsewhere.
