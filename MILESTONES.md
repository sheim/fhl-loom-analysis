# Milestones — Refactor & Unification Plan

Status legend: ☐ todo · ◐ in progress · ☑ done.

**Experiment description**: overhead high-speed video of 4 fish at rest, stimulated with a loom (expanding silhouette shown from a monitor) to trigger an escape response. In each video, we detect the start of the experiment from a colored-blip on the screen. We then need to detect which fish is the first to move, and find the timing. This is the current state of the code. Note, while the general set-up is the same across experiments, the camera is often moved, so the exact orientation or positions can shift, requiring per-video adjustments.

Milestone is in 2 phases.

First we're going to refactor and unify the code, and improve it.

Second, we're going to develop new code for more advanced analysis.

---

## M0 — Analyze & identify refactoring opportunities  ☑
Done — findings feed M1. `analyze_fish_energy.py` is one 1051-line file mixing pure core
(kernel/energy/threshold), video IO, GUI, plotting, and CLI. Key issues:
- Logic lives in `main()`; library fns call `sys.exit()`/`print()` (e.g. `select_roi_click`
  exits on cancel) → nothing is safely importable.
- ~18 tuning params are hard-coded as `main()` locals; batch can't tune them.
- Redundant compute: baseline is computed twice (`compute_threshold_temporal`'s result is
  discarded and recomputed inside `track_energy_temporal`); `stride` only subsamples the
  detection check but energy is computed for every frame anyway; the stride-1 refine pass
  re-reads frames already processed.
- Single fish ROI is intentional — it targets the first fish to respond; not a limitation.
- Cruft: doubled `cap.release()`, unused `import math`, commented-out `save_debug_panels`
  blocks, empty `out/plot_latency_hist.py`, `out/` name collision (`parent.name+stem`),
  fragile stdout-scraping in `batch_analyze.sh`.
- `batch_analyze.py` deleted (was untracked/broken) — removes the duplicate batch runner.

## M1 — One canonical module + working batch  ◐
Goal: one side-effect-free library that is the sole source of detection logic; the interactive
CLI and a new Python batch runner both call it.
- ☑ Extract `analyze_video(video, stim_roi, fish_roi, params) -> AnalysisResult` from `main()`;
  no `print`/`sys.exit`/GUI in library code (ROI acquisition & viz become separate steps).
- ☑ `AnalysisParams` dataclass (defaults = today's hard-coded values), shared by CLI + batch.
- ☑ Compute the energy series once; run coarse + refine detection on in-memory arrays (drops
  the double-baseline and the refine re-reads).
- ☑ New Python batch runner (`batch.py`) that imports the library and writes results directly
  (keeps the `filename,stim_idx,final_det_idx` schema); retired `batch_analyze.sh`.
- ☑ Remove dead code (doubled `cap.release()`, unused `import math`, commented debug blocks,
  empty `out/plot_latency_hist.py`) and fix the `out/` name collision (include species).

_Single fish ROI = the first responder._
_Batch = new Python runner (recommended over shell — see table in discussion); reversible._
_Verified headlessly (unit tests + synthetic clip: stim@60, refine 89 < coarse 97). Remaining:
a real-video spot-check by a human, since ROI selection is interactive (◐ until then)._

## M2 — Reusable ROIs (per-video annotations)  ◐
Goal: a one-time manual sweep that records, per video, what a human must eyeball — so all
downstream analysis loads it and runs **headless** (annotate once, re-run freely). Builds on
M1's seam (`analyze_video` already takes ROIs as inputs).

**Storage:** one **per-video JSON** in a git-tracked `annotations/` mirror
(e.g. `annotations/Shiner_SloMo/circle/34.json`) — the videos are git-ignored, but the
hand-made annotations must be version-controlled. Linked to the clip by **relative path**.

**JSON schema (keep flexible / versioned):**
- `disposition`: `usable | no_response | bad_video`
- `annotation`: `stim_roi`, `fish_roi` (first responder), `detected_blip`
  (`first`/`middle`/`last` — the stimulus blip appears 3x, 1 s apart; late-started recordings
  can miss the first, so the detector may catch the middle/last) — extensible (M3 adds screen
  edges, etc.)
- `results` (stored cache, regenerable): `stim_idx` (first-blip **reference**, corrected from
  `detected_blip` by 0/240/480 frames = 0/1/2 s at 240 FPS — can be negative for late starts),
  `stim_idx_detected` (raw), `det_refined` (fine-grained), plus the `AnalysisParams` used
- provenance: `schema_version`, relative `video` path

**Tasks:**
- ☑ JSON schema + `load/save_annotation()` loader in `annotations.py` (forward-compatible:
  unknown keys preserved for M3; `results_stale()` flags param drift).
- ☑ `annotate.py` — interactive sweep (reuses `playback_video` + `select_roi_click`), runs
  `analyze_video` after ROI selection and caches `stim_idx`/`det_refined` into the JSON.
  - ☑ default: process only **un-annotated** videos in a folder
  - ☑ `--redo-all`: re-annotate everything
  - ☑ `--redo <video>`: re-annotate a specific clip
  - ☑ `--show`/`--review`: load & display existing annotations (draw saved ROIs on the frame,
    print `stim_idx`/`det`, warn if stale), no editing
- ☑ Headless consumption: `batch.py --from-annotations` reads ROIs from annotations and runs
  with no GUI; `--update-annotations` refreshes the cached results.

_Verified headlessly (annotation round-trip, forward-compat, stale check, and
`batch --from-annotations` reproducing `analyze_video` on a synthetic clip). Remaining: a real
interactive annotation sweep by a human (the ROI-clicking part can't be auto-tested) — ◐ until then._

## M3 — Identify Orientation and Geometries
Goal: detect the fish position and orientation heading orientation in relation to the loom direction. This will be used to calculate the angle of the loom w.r.t. to the fish, and the effective rate-of-expansion of the silhouette.
Note the loom screen is sometimes on opposite sides of the tank, this info will need to be save. We'll also need to somehow detect and save the edges of the monitor to compute the loom axis center (or determine it some other way – in a pinch by heuristic hard-coding, but that's not ideal).
Details to be fleshed-out later.
- ☑ `monitor_side.py` — infer monitor side (`top`/`bottom`) from the stim ROI's vertical
  position → `annotation.monitor_side` (no clicking needed; the stim ROI is on the monitor).
- ☐ Extend `annotate.py` with a clicking option to pick the finer geometry — monitor
  edges/corners for the loom axis center — written into the same per-video JSON.

## M4 — Detection accuracy  ☐
Goal: fix known detection quality issues.
- ☐ Investigate the flagged "gaussian smoothing gives premature detection".
- ☐ Add a quick way to spot-check detections (debug grids/frames) as a first-class feature.

## M5 — Packaging, tests & repo hygiene  ☐
Goal: reproducible on a fresh machine.
- ☐ Drop unused deps (`scipy`, `ipython`); write a real `pyproject` description.
- ☐ Remove cruft (`.venv`, `.venv_win`, `profile.cprof`, `.DS_Store`); tidy `.gitignore`.
- ☐ Track the files that should be tracked (`README.md`, `MILESTONES.md`, chosen batch script).
- ☐ Add a small test suite (unit tests on a tiny sample clip / synthetic frames).

---

_Notes / open questions_
- Which species/conditions still need to be (re)processed?
- Is `sculpin_NR` ("no reuse") a permanent comparison or one-off?
