# Milestones — Refactor & Unification Plan

Status legend: ☐ todo · ◐ in progress · ☑ done.

**Experiment description**: overhead high-speed video of 4 fish at rest, stimulated with a loom (expanding silhouette shown from a monitor) to trigger an escape response. In each video, we detect the start of the experiment from a colored-blip on the screen. We then need to detect which fish is the first to move, and find the timing. This is the current state of the code. Note, while the general set-up is the same across experiments, the camera is often moved, so the exact orientation or positions can shift, requiring per-video adjustments.

Milestone is in 2 phases.

First we're going to refactor and unify the code, and improve it.

Second, we're going to develop new code for more advanced analysis.

---

## M0 - Analyze and identify refactoring opportunities, revisit milestones.

## M1 — One canonical module + working batch  ☐
Goal: a single source of truth for detection logic that all entry points call.
- ☐ Promote `analyze_fish_energy.py` into a clean importable API (no logic in `main()`).
- ☐ Remove duplicated `compute_threshold` / `scan_window` from `batch_analyze.py`.
- ☐ One batch entry point that works end-to-end.

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
