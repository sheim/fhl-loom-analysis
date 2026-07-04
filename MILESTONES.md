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

A **new interactive script** does the geometry marking (annotate.py stays as-is — we don't redo
the ROI sweep). Marks extend the same per-video annotation JSON (extensible `annotation` block).
**Every piece must also work on `no_response` clips** (for complete statistics).

- ☑ `monitor_side.py` — infer monitor side (`top`/`bottom`) from the stim ROI's vertical
  position → `annotation.monitor_side` (no clicking needed; the stim ROI is on the monitor).

### M3.1 — Export reference frames (batch.py)  ☑
- ☑ At the end of `batch.py`, save a **10-frame window `[det-3 … det+6]`** around the
  first-movement frame (`final_det_idx`) as PNGs — a stack, not one frame, so the marker can
  step through them to disambiguate the moving fish. On by default; skip with `--no-frames`.
- ☑ Location: git-ignored `frames/` mirror of `videos/`, one subfolder per clip
  (`frames/Sculpin_SloMo/flapping/48/000.png … 009.png`) via `annotations.frames_dir()`.
- ☑ With `--update-annotations`, records `frame_dir` / `frame_start` / `n_frames` in the JSON.

_Verified: 10 PNGs written for onset 89 (window 86–95) and the provenance cached in the JSON._

### M3.1b — `fix_start.py` — manual movement-onset correction  ☑
For clips where the first-movement auto-detection is off:
- ☑ Scrub the video (±1 / ±30 / ±240 frames, frame number always shown) to the correct onset.
- ☑ On accept: set `annotation.manual_det` + `results.det_refined`, and regenerate the `frames/`
  stack around the new frame (shared `annotations.export_frames`).
- ☑ `batch.py` respects `manual_det` (overrides the recomputed `det_refined`), so re-runs don't
  clobber the fix.

_Verified: `manual_det`=120 → `det_refined`=120, 10 frames regenerated (start 117), and `batch
--from-annotations` reports `first_movement=120` (CSV `…,120`). Real scrub UI needs a human._

### M3.2 — `geometry.py` — interactive marking (works on the exported frames, no video needed)  ◐
Per clip, load the ~10-frame stack (step through with a/d or ←/→ to disambiguate the moving
fish; mark on the current frame) and collect into the annotation:
- ☑ **Tank corners (monitor side):** 2 clicked points → `tank_corners = [[x,y],[x,y]]`. Baseline
  for pixel scale + defines the monitor-side edge (also confirms `monitor_side`).
- ☑ **Fish (up to 4), starting with the first to move:** per fish click **head then tail**
  (2 points) → `fish = [{head, tail}, …]` — index 0 is the first responder (encoded by marking
  order; no separate reaction label). Head→tail gives position + heading in one go.
- ☑ New GUI helper `afe.pick_points` (click N labelled points on a frame stack, step/undo/finish).
- ☑ Skips already-marked clips by default; `--redo`, `--redo-all`, `--show` (draw saved marks).

_Point-collection logic verified headlessly (clicks/step/undo/cancel/finish); the real marking
pass needs a human (GUI can't be auto-tested) — ◐ until then._

### M3.3 — Loom geometry (`loom_geometry.py`)  ☑
From `tank_corners` + `fish[0].head` + results:
- ☑ **Loom origin** = midpoint of the two monitor-side corners (center of the monitor edge — where
  the loom expands from); pixel→metre scale from the tank width (0.59 m spans the two corners).
- ☑ **Distance** from the origin to the first responder's head (cm).
- ☑ **Retinal angle**: elapsed monitor frame = `(det-stim)/4` (camera 240 fps → monitor 60 fps) →
  look up silhouette width (`diameter_lookup_table.csv` `diameter_m`) → `θ = 2·atan((W/2)/dist)`.
- ☑ Caches the per-clip result in the annotation JSON (`geometry` block — a first-class field
  parallel to `results`, not wiped by `batch`); `-o FILE.csv` also exports an aggregated table for
  stats. `--show` draws the triangle on the frame.

_Verified on real clips (clip 8: dist 30.8 cm, W 17.0 cm, angle 30.9° at latency 1.746 s; whole
circle folder computes). `diameter_lookup_table.csv` is now git-tracked (input, not a result)._

_Deferred: fish **heading** (head→tail) is captured but not yet used — loom-angle-relative-to-fish
and rate-of-expansion are future work. "Loom origin = corner midpoint" is an interpretation of
"tank center"; revisit if the true tank centre is wanted._

### M3.4 — No-response clips (for complete statistics)  ☐
`no_response`/`bad_video` clips have no ROIs and no movement detection, yet we still want their
geometry:
- ☐ Reference frame: no `final_det_idx`, so export a fallback frame (a fixed index, or a quick
  manual pick). [decision]
- ☐ `monitor_side` comes from the marked tank corners (not the stim ROI), so `geometry.py`
  needs no ROIs — it runs the same on these clips.
- ☐ All fish get `reaction = 3`.
- ☐ Needs a lightweight path to export a reference frame for clips that don't run `analyze_video`. [decision]

### Decisions
1. ☑ Exported frames: git-ignored `frames/` mirror; a **~10-frame window** around onset (exact offsets TBD).
2. ◻ `no_response` reference frame — deferred.
3. ◻ Fish position (head vs midpoint) — store **both head & tail** now; derive position later (M3.3).
4. ☑ Pixel↔cm scale: the two tank corners span the **tank width = 59 cm**. (Camera height above
   the tank bottom = **69 cm** — recorded for later perspective work.)

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
