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

## M1 — One canonical module + working batch  ☐
Goal: one side-effect-free library that is the sole source of detection logic; the interactive
CLI and a new Python batch runner both call it.
- ☐ Extract `analyze_video(video, stim_roi, fish_roi, params) -> AnalysisResult` from `main()`;
  no `print`/`sys.exit`/GUI in library code (ROI acquisition & viz become separate steps).
- ☐ `AnalysisParams` dataclass (defaults = today's hard-coded values), shared by CLI + batch.
- ☐ Compute the energy series once; run coarse + refine detection on in-memory arrays (drops
  the double-baseline and the refine re-reads).
- ☐ New Python batch runner that imports the library and writes results directly (keep the
  `filename,stim_idx,final_det_idx` schema); retire the stdout-scraping `batch_analyze.sh`.
- ☐ Remove dead code (doubled `cap.release()`, unused `import math`, commented debug blocks,
  empty `out/plot_latency_hist.py`) and fix the `out/` name collision (include species).

_Single fish ROI = the first responder (placed manually now; auto-detected in M2)._
_Batch = new Python runner (recommended over shell — see table in discussion); reversible._

## M2 — Reusable ROIs  ☐
Goal: run a first batch sweep that identifies the first fish to move, and defines the ROI. This should be saved in a metadata linked to that video, so further analysis scripts can load the right frames/ROIs without reprocessing everything from scratch. Details to be discussed.

## M3 — Identify Orientation and Geometries
Goal: detect the fish position and orientation heading orientation in relation to the loom direction. This will be used to calculate the angle of the loom w.r.t. to the fish, and the effective rate-of-expansion of the silhouette.
Note the loom screen is sometimes on opposite sides of the tank, this info will need to be save. We'll also need to somehow detect and save the edges of the monitor to compute the loom axis center (or determine it some other way – in a pinch by heuristic hard-coding, but that's not ideal).
Details to be fleshed-out later.

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
