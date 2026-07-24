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

## M1 — One canonical module + working batch  ☑
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

## M2 — Reusable ROIs (per-video annotations)  ☑
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

### M3.2 — `geometry.py` — interactive marking (works on the exported frames, no video needed)  ☑
Per clip, load the ~10-frame stack (step through with a/d or ←/→ to disambiguate the moving
fish; mark on the current frame) and collect into the annotation:
- ☑ **Tank corners (monitor side):** 2 clicked points → `tank_corners = [[x,y],[x,y]]`. Baseline
  for pixel scale + defines the monitor-side edge (also confirms `monitor_side`).
- ☑ **Far tank corners (2nd edge):** 2 more clicked points → `tank_far_corners = [[x,y],[x,y]]`.
  The four corners give the full tank quad for a perspective-correct pixel→cm map (M3.7).
  **Off-frame corner handling:** if one far corner is out of frame, pick "reconstruct" — click the
  visible far corner + a point on each of its two edges (far wall, side wall); the off-frame corner
  is their **line intersection** (`_line_intersection`, may fall outside the frame; parallel-edge
  fallback = parallelogram completion). `tank_far_reconstructed` flags such clips (provenance).
- ☑ **Tank depth (2nd edge):** `choice_popup` 44 / 30 cm, **Space = 44 default** → `tank_depth_cm`
  (with a reminder note: 44 cm tank = all shiner experiments, has a 1 cm-grid plexiglass somewhere).
  `choice_popup` gained a tinted multi-line `note` (auto-widened window).
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
  look up silhouette width W (`diameter_lookup_table.csv` `diameter_m`); its base sits ON the screen
  (tank-corner line) centred at the origin → θ = angle subtended at the head (general triangle).
- ☑ Caches the per-clip result in the annotation JSON (`geometry` block — a first-class field
  parallel to `results`, not wiped by `batch`); `-o FILE.csv` also exports an aggregated table for
  stats. `--show` draws the triangle on the frame.

_Verified on real clips (clip 8: dist 30.8 cm, W 17.0 cm, angle 30.9° at latency 1.746 s; whole
circle folder computes). `diameter_lookup_table.csv` is now git-tracked (input, not a result)._

_Deferred: fish **heading** (head→tail) is captured but not yet used (loom-angle-relative-to-fish
is future work). "Loom origin = corner midpoint" is an interpretation of "tank center"; revisit if
the true tank centre is wanted._

### M3.5 — Rate of change of the retinal angle dθ/dt (`loom_geometry.py`)  ☑
Goal: the angular **expansion rate** of the retinal angle at movement onset — a key looming cue —
computed numerically from the silhouette schedule.

Plan:
- Make the retinal angle a **function of time**: `theta(t)` = at time `t` (s since stimulus), look up
  the silhouette width `W = diameter_m(monitor_frame = t·60)` (interpolated), put its base ON the
  screen (tank-corner line) centred at the origin, and take the angle it subtends at the **fixed**
  first-responder head. Only W varies with t; distance/origin/head are held at their onset values.
- Onset `t0 = latency_s = (det − stim)/240` s. **Frame-rate handling:** do everything in **seconds**
  — the silhouette lookup is indexed by monitor frame (60 fps, `mf = t·60`) while detection is in
  camera frames (240 fps, `/240`); differencing in seconds makes dθ/dt correctly per-second.
- **Numerical derivative, selectable accuracy order** (to compare sensitivity):
  - 1st-order (one-sided): `(theta(t0+h) − theta(t0)) / h`
  - 2nd-order (central): `(theta(t0+h) − theta(t0−h)) / (2h)`   ← the "before & after" scheme
  - (optional 4th-order 5-point central)
  - step `h` default = **1 monitor frame = 1/60 s** (the silhouette's native resolution); configurable.
- Output: `dtheta_dt_deg_per_s` in the `geometry` block (store each order's estimate so 1st vs 2nd
  can be compared for sensitivity); add it to the `--show`/`--save` overlay text.

**Decisions:** order = finite-difference **accuracy** — compute 1st (one-sided), 2nd (central),
and 4th (5-point) estimates and store all three to compare sensitivity. Step = **weighted center**:
anchor the stencil at the exact fractional onset `mf0 = (det−stim)/4` and evaluate θ by linearly
interpolating the lookup (weighting by proximity to the bracketing monitor frames — which is what
`silhouette_m` already does); default step `h = 1 monitor frame`, configurable via `--deriv-step-frames`.

_Done. Stored in the `geometry` block as `dtheta_dt_deg_per_s` (2nd-order headline) +
`dtheta_dt_by_order` (1st/2nd/4th). Verified: clip 8 → 116 deg/s, with 2nd (116.08) and 4th (115.65)
agreeing to 0.4% and 1st ~6% high (sensitivity visible); a linear-θ test gives exactly 180 deg/s
(3 deg/frame × 60 fps), confirming the seconds conversion. Also on the `--show`/`--save` overlay._

### M3.6 — Analytical dθ/dt (closed form) (`loom_geometry.py`)  ☑
Goal: an exact closed form for the expansion rate, to cross-check the numerical stencils and to be
robust near contact (where finite differences blow up).

**Loom schedule.** The lookup table is a virtual object approaching at constant speed: distance is
linear in time and on-screen diameter ∝ 1/distance, so `W(t) = C / (D0 − v·t)`. `loom_params()` fits
`(C, D0, v)` from the table by least effort (all `diameter·distance` products equal C = 0.0432 m·m,
D0 = 2.0 m, v = 1.0 m/s). Then **`dW/dt = v·W² / C`** exactly.

**Chain rule.** dθ/dt = (dθ/da)·(da/dt), where `a` = on-screen half-width in pixels.
- Geometry: with the base centred on the screen at the origin, θ = `atan2(2a·d⊥, R² − a²)` (apex at
  the head; `R` = |head−origin|, `d⊥` = perpendicular head→screen distance, both in px). Differentiating:
  **`dθ/da = 2·d⊥·(R² + a²) / (4a²d⊥² + (R² − a²)²)`** (rad per px).
- Rate: `da/dt = ½·(dW/dt)/m_per_px = ½·(v·W²/C)/m_per_px` (px/s).
- `analytical_dtheta_dt()` multiplies them; stored as `dtheta_dt_analytic_deg_per_s` in the `geometry`
  block (headline numerical `dtheta_dt_deg_per_s` kept alongside).

_Done + validated (71 clips, read-only comparison scripts, no annotations rewritten):_
- **Closed form is exact.** Finite-differencing θ built from the *smooth* schedule (bypassing the
  table) converges to the analytical value to ~6 digits as h→0 (clip circle/13, onset on integer
  frame 112: 1613.03 → analytic 1613.01). The tiny residual for mid-frame onsets (clip 8: 0.14%) is
  the table's piecewise-linear interpolation of W, not a formula error.
- **Agreement in the bulk.** For onsets in the first ~85% of the loom, analytic ≈ numerical-2nd/4th
  (h=1) to <1% (median 0.33% / 0.04%). 1st-order is systematically ~6% high (one-sided bias).
- **Numerical fails near contact.** In the last ~10% of the table θ(t) curves so sharply that a
  1-frame stencil has huge truncation error (clip circle/13: num-4th 2079 vs analytic 1613, +29%).
  The analytical has no truncation or step-size error there → **more trustworthy near contact.**
- Comparison figure: `out/analysis/dtheta_analytic_vs_numeric.png`.
- **Headline switched to analytic.** `geometry_record()` now writes `dtheta_dt_deg_per_s` =
  analytic closed form (numeric 1st/2nd/4th retained in `dtheta_dt_by_order`; explicit
  `dtheta_dt_analytic_deg_per_s` alongside). Re-ran `loom_geometry.py` over all clips to repopulate
  (71 usable updated; surgical diff — only the dθ/dt fields changed). `dataset.py` sources the M4
  "expansion rate" variable from `dtheta_dt_analytic_deg_per_s`; histogram PNGs regenerated.

### M3.7 — Perspective-correct positions (homography)  ☑  → built under **M4.2**
Marking side **done**: `geometry.py` captures `tank_far_corners` + `tank_depth_cm` (+ off-frame
reconstruction). The compute side (homography `image px → tank cm`, recompute of
distance/angle/dθ/dt/positions, fallback + validation) is **done under M4.2**.

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

## M4 — Analysis  ◐
Goal: Analyze the data that we've processed into the json files in aggregation, producing both visualizations and the raw numbers. Let's make this part interactive with marimo notebooks (which should be made to be able to run them as scripts as well, and just generate saved plots).
_Status: response histograms + distribution fits + outlier exclusion + dθ/dt sensitivity + position
overlays/tank-map/interactive map + perspective homography (M4.2) all done; **circular-stats plots
are the remaining item**._

- ☑ Start by looking at the data, do some general statistics and see if you see anything of particular interest, and also check for anomalies. Then we'll discuss and populate this list.
- ◐ Plot histograms of responses vs {latency, distance, loom-size, expansion-rate of disc, retinal angle}. Have two plots, one for sculpin and one for shiners, and plot all three stimuli-types (circle, fixed, flapping).
  - ☑ v1 tooling: `dataset.py` (JSON → tidy DataFrame), `plots.py` (`response_histograms` + `save_all`),
    `notebooks/responses.py` (marimo: variable dropdown + condition/bins/density toggles). Runs
    interactive (`marimo edit`), headless (`marimo export html`), or as a script (`python …` → `out/analysis/`).
    Variable map: loom-size = silhouette Ø at onset; expansion-rate = dθ/dt.
  - ☑ Second stage — distribution fitting (`fits.py`). Candidates: normal (baseline), lognormal,
    gamma, weibull (last three loc=0, positive support; weibull's shape also captures latency's
    *left* skew). All 2-param → fair model selection by **AICc** (small-sample-corrected — the right
    criterion at n≈9-17). KS reported as a descriptive distance only (fitted-param p-values are
    optimistic). `fit_table()` gives the per-group AICc comparison (+ a `pool`-conditions option to
    borrow strength); `response_histograms(..., fit=...)` overlays the PDF (`"auto"`=best, or a named
    dist) annotated with ΔAICc-vs-normal; `save_all` writes `_fit` PNGs + `distribution_fits.csv`;
    notebook gains a Fit dropdown + pool switch + the AICc table.
    _Finding: normal wins only 2 / 30 groups. **distance** ≈ normal (ΔAICc<2); **latency** is
    left-skewed (ceiling at loom contact ~1.9 s) → weibull; **loom size** mild skew; **expansion
    rate** and **retinal angle** are heavily right-skewed → lognormal/gamma decisively beat normal
    (ΔAICc up to 36). So use lognormal/gamma for rate-like vars, not normal._
  - ☑ Reversible outlier exclusion (`dataset.clip_key`, `dataset.exclude`, `dataset.outlier_scores`).
    Robust modified z-score (0.6745·(x−med)/MAD) within each species×condition group flags candidates
    (|z|≥3.5) without removing anything; the notebook's "Exclude clips" multiselect filters the plots
    + fits live (dataset on disk untouched). _E.g. shiner/circle/34 (latency 0.29 s, z=−13, suspected
    spurious): excluding it leaves the median ~unchanged (1.735→1.742 s) but drops the mean
    1.636→1.726 s and the SD 0.38→0.12 — a high-leverage point. Note the near-contact high-dθ/dt
    clips are also flagged but are genuine, so candidates need judgement, not auto-removal._
  - ☑ dθ/dt **timing sensitivity** (`sensitivity.py`). Per clip: how much the analytic dθ/dt shifts
    if the movement-onset frame is off by 1 camera frame (central difference in onset time; reported
    absolute deg/s·frame⁻¹ and relative %·frame⁻¹). Reuses `loom_geometry.screen_frame` (factored out
    of `compute`) + `analytical_dtheta_dt` — recomputes nothing on disk. `plots.sensitivity_scatter`
    plots it vs a selectable x-axis with a marginal histogram of where the first-responders sit;
    `save_all` writes `dtheta_sensitivity_vs_{latency,distance}.png`; notebook gets an x-axis dropdown
    + relative/absolute toggle (respects the exclusion filter). The notebook uses an interactive
    **Plotly** version (`sensitivity_scatter_interactive`) so hovering a point shows its `clip_key`
    (+ latency/phase/distance/dθ-dt); the matplotlib version stays for the static PNGs. (Added `plotly`.)
    Plus a **2D bubble view** (`sensitivity_bubble` / `_interactive`): both axes chosen from `XVARS`
    (now incl. **loom diameter** = silhouette Ø, monotonic with phase), **bubble area ∝ |sensitivity|**,
    open markers (border = condition, shape = species), and a |sensitivity| threshold keeping only
    clips at/above it (default 0 = all; magnitude, so the negative-sensitivity clip stays visible).
    Default axes loom-phase × distance; saved `dtheta_sensitivity_bubble.png`; notebook adds x/y
    dropdowns + threshold slider with clip_key hover.
    _Finding: sensitivity tracks **loom phase / latency** (ρ≈+0.94), **not distance** (ρ≈−0.16) — so
    the milestone's "distance?" guess was wrong. It's ~0 until latency ≈1.5 s then climbs steeply to
    contact (shiner/circle/13: **34 %/frame**; a 1-frame onset error moves dθ/dt by a third). Because
    most first-responders respond late (1.7–1.9 s), many sit in this steep regime → the upper tail of
    the dθ/dt distribution is the least timing-robust. The one large negative (shiner/flapping/4,
    θ≈160°) is a near-θ=180° geometric edge case (base half-width approaching R)._
- ☑ Overlay of all fish positions (`positions.py` + `plots.fish_overlay` + `notebooks/positions.py`).
  Each fish head/tail is registered into a **loom-centred, screen-aligned metric frame** (cm):
  along-screen (x) from the loom origin, depth-from-screen (y), oriented so fish are at depth>0 —
  comparable across clips/species. `fish_frame(df)` → one row per fish (`is_responder` = index 0).
  Overlay: first responders + other fish, with independently toggleable KDE heatmaps for responders
  (reds), others (grays), **and all fish combined (purples)** — **boundary-corrected** against the
  tank walls by the reflection (mirror) method (`_kde_reflected`: interior points mirrored across each
  wall + corners, original Scott bandwidth), so density fills to the walls without leaking past them
  or fading at them; the folded tank map reflects on the half-tank walls. First responders are marked **by
  condition** — colour + shape (circle ○ / fixed ■ / flapping ◆); `by_condition` / `others_by_condition`
  switches (else uniform red / gray). New marimo notebook with species/condition filters + point,
  by-condition, and heatmap switches; legends sit **below** the plot (never occlude). Plus a custom
  folded **tank map** (`fish_tankmap`): left/right mirrored into one half (fixes the arbitrary along-sign),
  the tank drawn as its physical rectangle (59 cm × depth), loom origin at the bottom-left corner, with
  radiating angle rays (0°=along screen, 90°=perpendicular) to the tank edge + faint radial distance arcs.
  Tank depth is a control (**shiner 44 cm; sculpin 44 or 30**); `save_all` writes `fish_positions.png` /
  `_heat.png` / `_tankmap.png`. Uses `scipy`. Plus an **interactive Altair map** (`fish_interactive`):
  hovering a fish shows a rich tooltip (fish + trial details) and — linked by `clip_key` — keeps every
  fish from the *same clip* highlighted while the rest fade (head→tail sticks show heading); `fish_frame`
  now carries the per-clip experiment fields for the tooltip. (Added `altair`.)
  _Finding: first responders sit **farther from the screen** (mean depth 27.7 cm) than other fish
  (21.6 cm) — a lobe of non-responders hugs the screen (~8 cm) while responders cluster deeper
  (~30–40 cm). Tank dims: width 59 cm confirmed by the along-axis extent (±29.5 cm); **registered
  depths run ~10–15 % past the physical far wall (44 cm) — perspective over-projection to correct
  later** (camera 69 cm above the tank). The folded map removes the earlier along-sign mirror artifact._
- ◻ Evaluate what plots would make sense to plot using circular stats? Fill out below ideas that I'm missing.
  - ◻ centered around loom-direction (orthogonal to the tank-corners-line, and centered in its middle), the location of the fish on the polar coordinate, and {response-time, dθ/dt} as the radial coordinate.
  - ◻ using the difference between fish orientation and loom-direction as polar coordinate, and {response-time, dθ/dt} as the radial coordinate.

### M4.2 — Perspective-correct geometry & distances (homography from the 4 tank corners)  ☑
Every usable clip now carries 4 corners (`tank_corners` near + `tank_far_corners`) + `tank_depth_cm`
(**56×44 cm, 15 sculpin×30 cm, 4 with a reconstructed far corner**). Use them to map image pixels →
**true tank-cm** with a per-clip homography and recompute distance / retinal angle / dθ/dt / positions
perspective-correctly — fixing the ~10–15 % **depth over-projection** (registered depths currently
reach ~50 cm inside a 44 cm tank; the along-screen axis is already exact from the 59 cm edge).

**Streamlined idea — one canonical cm frame, reuse all the existing math.** A homography
`H: image px → tank cm` (near edge → `(0,0)–(59,0)`, far edge at `y = depth`) lets us drop the fish
head into a *canonical* cm frame where the loom origin is `(29.5, 0)`, the screen direction is `(1,0)`,
and "metres per unit" = `0.01`. Feeding those into the **existing** helpers (`_base_points`,
`_subtended_angle_deg`, `_retinal_angle_deg`, `analytical_dtheta_dt`) makes them perspective-correct
with essentially no new geometry — the silhouette base already lies on the near edge (where the linear
scale was exact), so only the **head** moves to its true position, which is precisely the error we're
removing. `compute()` stays the same downstream; only how we obtain `(origin, screen_u, m_per_px, head)`
changes.

Where to build (each a small, local change):
- ☑ **`tank.py` (new, pure — numpy/cv2 only):** `homography(near, far, depth_cm) -> H | None`
  (`cv2.getPerspectiveTransform`; far corners paired to near by proximity → `(0,d)/(59,d)`; None if
  data missing/degenerate) and `to_cm(H, pts)`. Single source of truth, unit-testable on synthetic quads.
- ☑ **`loom_geometry.py`:** add `tank_frame(ann)` → `(origin, screen_u, m_per_px, head)` in the cm
  frame when `H` exists, else today's pixel `screen_frame` path. `compute()` calls it instead of the
  inline `screen_frame`/head lines — everything after is unchanged. Add `perspective_corrected: bool`
  to the `geometry` block; keep the old values as `distance_cm_linear` / `angle_deg_linear` for one
  validation pass, then drop.
- ☑ **`positions.fish_frame`:** map each fish head/tail through `H` → cm (`along = x−29.5`, `depth = y`),
  a drop-in for the current linear `_project`; also carry `tank_depth_cm` per clip so the folded tank
  map's far wall is per-clip (shiner 44; sculpin 44/30) rather than one global slider default.
- ☑ **Fallback + provenance:** clips without 4 corners/depth fall back to the linear scale (all 71 have
  them now, but keep it robust); `tank_far_reconstructed` rides along as a QA flag.
- ☑ **Recompute + validate:** back up `annotations/` (untracked), re-run `loom_geometry.py` over all
  folders to repopulate the `geometry` block (same idempotent pattern as the analytic-dθ/dt re-run),
  then a **read-only** linear-vs-perspective compare of distance/angle/dθ/dt (expect the deepest fish
  to shrink most; **all depths should now land within `[0, tank_depth_cm]`**). `dataset.py`/plots read
  the refreshed block unchanged; positions recompute live from the annotations.

_Supersedes the M3.7 stub — the marking side (4 corners + depth + off-frame reconstruction) is done;
this is the compute side._

_Built + validated:_ `tank.py` (`homography`/`to_cm`, exact on a synthetic quad to 1e-6; rejects a
degenerate quad where the far edge collapses onto the near edge → caller falls back to linear).
`loom_geometry.tank_frame` feeds the cm frame into `compute` (metrics in cm; **px drawing coords kept
separate** so the `--show` overlay still works); `perspective_corrected` in the `geometry` block.
`positions.fish_frame` + `dataset` carry `tank_far_corners`/`tank_depth_cm`. Re-ran over all folders
(backup first): **70 perspective-corrected, 1 linear**. Effect: distances shrink (median 0.5 cm, max
7 cm on corrected clips) and retinal angles grow slightly as the head moves to its true (closer)
position; **30 cm sculpin clips now bounded to ~[0,30] cm depth** (were over-scaled as 44), 44 cm
clips to ~[0,44] (all fish ≤5 cm past their own far wall). Also recomputed the previously-empty
geometry for `Shiner_SloMo/fixed/40`.
_QA finding (resolved):_ `Sculpin_SloMo/circle/30` originally had its far corners clicked on top of
the near ones (degenerate quad → auto-fell-back to linear); the degeneracy guard caught it and it was
re-marked. **All 71 usable clips are now `perspective_corrected` (4 with a reconstructed far corner).**

_Saved tank record:_ `geometry_record` now embeds a **`tank`** block (`tank.tank_record`) documenting
the quad used — `width_cm` (59) + `depth_cm` (44/30), the near/far corner pixel coords, each edge's
**pixel length** (`sides_px`: near/far ≈ width, left/right = depth — the near-vs-far gap shows the
perspective), and `far_reconstructed`. Re-ran all folders to populate it (surgical diff: only the
`tank` key added). The **`--save`/`--show` overlay** (`_draw_overlay`) now draws the **full tank quad**
(far + side edges + far corners, cyan) with the side lengths labelled (near/far 59 cm, sides 44/30 cm)
and a `tank W x D cm [perspective-corrected]` text line — not just the monitor edge.

_Follow-up (minor):_ the folded tank map still uses one `tank_length` slider; fish now carry
`tank_depth_cm`, so the map could default its far wall per selection (shiner 44; sculpin 44/30).


## M5 — Packaging, tests & repo hygiene  ☐
Goal: reproducible on a fresh machine.
- ☐ Drop unused deps ; write a real `pyproject` description.
- ☐ Remove cruft (`.venv`, `.venv_win`, `profile.cprof`, `.DS_Store`); tidy `.gitignore`.
- ☐ Track the files that should be tracked (e.g. `README.md`, `MILESTONES.md`, chosen batch script).
- ☐ Add a small test suite (unit tests on a tiny sample clip / synthetic frames).

---

_Notes / open questions_
- Which species/conditions still need to be (re)processed?
- Is `sculpin_NR` ("no reuse") a permanent comparison or one-off?
