# Loom & Doom — Preliminary Analysis

A small computer-vision pipeline for measuring **fish escape-response latency** (reaction
time) from high-speed video. For each clip it:

1. Detects the frame where a **visual stimulus** appears (saturation / grayscale change in a
   stimulus ROI).
2. Detects the frame where the **fish first moves** (temporal motion-energy in a fish ROI).
3. Reports the gap between the two as latency (frames → seconds at **240 fps**), aggregated
   across species (**sculpin**, **shiner**) and stimulus conditions (**circle, fixed,
   flapping, ghost**).

Pure local processing — NumPy + OpenCV + Matplotlib. No network, no API keys.

---

## Requirements

- Python **≥ 3.13** (pinned via `.python-version`)
- A **display / GUI** — the analyzer uses `cv2.imshow` + mouse callbacks for ROI selection
  and cannot currently run headless (see [Known issues](#known-issues--scripts-to-unify)).
- Managed with [`uv`](https://docs.astral.sh/uv/) (an `uv.lock` is committed).

## Install

```bash
# Recommended: uv (matches the committed lockfile)
uv sync

# Or plain pip (into your own venv)
pip install numpy opencv-python matplotlib
```

> Note: `pyproject.toml` also lists `scipy` and `ipython`, but no source file imports them —
> the runtime dependencies are just `numpy`, `opencv-python`, and `matplotlib`.

## Expected data layout

Raw footage lives under `videos/` (git-ignored, ~16 GB), organized by species and condition:

```
videos/
├── Sculpin_SloMo/{circle,fixed,flapping,ghost}/*.MP4
└── Shiner_SloMo/{circle,fixed,flapping,ghost}/*.MP4
```

Clips are ~240 fps `.MP4` (the analyzer also accepts `.mov/.avi/.mkv`).

---

## The pipeline (scripts & commands)

| Script | Language | Role | Status |
|---|---|---|---|
| `sort_videos.sh` | bash | Sort/rename raw `Trial_*.MP4` into condition folders | ✅ active (one-time prep) |
| `analyze_fish_energy.py` | python | Interactive single-video analyzer (core logic) | ✅ active |
| `batch_analyze.sh` | bash | Batch a folder by scraping analyzer stdout → CSV | ✅ active (git-tracked) |
| `batch_analyze.py` | python | Batch a folder → per-video outputs + `summary.json` | ⚠️ **broken / untracked** — see below |
| `analysis.py` | python | Aggregate result CSVs → latency histogram PDFs | ✅ active |
| `out/plot_latency_hist.py` | python | — | ❌ **dead** (empty 0-byte stub) |

### 1. `sort_videos.sh` — footage prep (run once)

Creates `circle/ghost/flapping/fixed` subfolders inside the target dir, then moves each
`Trial_<num>_<...>.MP4` into the folder whose name it contains (case-insensitive), renaming
it to `<num>.MP4` (collisions get a `_i` suffix).

```bash
bash sort_videos.sh videos/Shiner_SloMo
bash sort_videos.sh videos/Sculpin_SloMo
```

### 2. `analyze_fish_energy.py` — interactive single-video analyzer

The core of the project. Opens one clip, plays it once for orientation, then prompts you to
draw two ROIs with two mouse clicks each (**stimulus area**, then **fish body**).

```bash
uv run analyze_fish_energy.py videos/Shiner_SloMo/circle/34.MP4
```

- **Input:** one video path (only positional arg).
- **Stdout:** `Stimulus frame index: N`, `Coarse first-movement frame: N`, `Refined
  first-movement frame: N` (these lines are what `batch_analyze.sh` scrapes).
- **Output file:** an energy-vs-frame plot at `out/<condition><stem>.png`
  (e.g. `out/circle34.png`).
- **Tuning:** all parameters (thresholds, kernel, stride, smoothing, sigma…) are **hard-coded
  as locals in `main()`** near the top of the function — edit the source to change them.

### 3. `batch_analyze.sh` — shell batch runner (working batch path)

Loops over every `*.MP4` in a folder, runs the analyzer on each (you still select ROIs per
video), scrapes the printed indices, and writes one CSV named after the folder.

```bash
bash batch_analyze.sh videos/Shiner_SloMo/circle
#  → circle_results.csv   with header:  filename,stim_idx,final_det_idx
```

To feed `analysis.py`, rename the output to the species-prefixed name it expects, e.g.
`circle_results.csv` → `shiner_circle_results.csv`.

### 4. `batch_analyze.py` — python batch runner (⚠️ currently broken)

Intended as a richer batch runner: same per-video pipeline but with ~15 CLI flags, per-video
output subfolders (`energy.csv`, `energy.png`, `debug/`), and a top-level `summary.json`.

```bash
# Intended usage (see Known issues — does not run as-is):
uv run batch_analyze.py videos/Shiner_SloMo/circle --energy-sigma 5 --stride 50
```

It **crashes on the first video** today (parameter/name drift against
`analyze_fish_energy.py`; details in [Known issues](#known-issues--scripts-to-unify)). It is
also **untracked in git**, whereas `batch_analyze.sh` is committed.

### 5. `analysis.py` — latency aggregation & histograms

Reads per-condition result CSVs from the **current directory**, computes latency
`= (final_det_idx − stim_idx) / 240`, prints mean/std per condition, and writes grouped-bar
histogram PDFs.

```bash
uv run analysis.py        # or: python analysis.py
```

- **Inputs (from cwd):** `{species}_{cond}_results.csv` for `species ∈ {sculpin, sculpin_NR,
  shiner}` and `cond ∈ {circle, fixed, flapping}`. Missing files are skipped with a `[skip]`
  message.
- **Outputs:** `sculpin_latency_hist.pdf`, `shiner_latency_hist.pdf`,
  `sculpin_NR_latency_hist.pdf`.

---

## End-to-end example

```bash
uv sync                                               # 1. install
bash sort_videos.sh videos/Shiner_SloMo               # 2. prep footage (once)
bash batch_analyze.sh videos/Shiner_SloMo/circle      # 3. analyze a condition (interactive)
mv circle_results.csv shiner_circle_results.csv       # 4. rename for the aggregator
uv run analysis.py                                    # 5. produce latency histograms
```

---

## Known issues & scripts to unify

This section is deliberately explicit so we can decide what to deprecate before unifying.

**Definitely dead**
- `out/plot_latency_hist.py` — empty 0-byte file, fully superseded by `analysis.py`. Safe to
  delete.

**Two competing batch runners — pick one to keep**
- `batch_analyze.sh` (bash) — **git-tracked, works today**. Crude: re-selects ROIs per video
  and parses the analyzer's stdout with `grep`/`awk`.
- `batch_analyze.py` (python) — **untracked, currently broken**. Richer design (CLI flags,
  per-video folders, `summary.json`) but has drifted out of sync with the library:
  - passes `sat_drop=` to `find_stimulus`, which expects `saturation_drop=`
    (`batch_analyze.py:273` vs `analyze_fish_energy.py:210`) → `TypeError` on the first video.
  - calls `afe.save_debug_frames_temporal`, which does not exist (the module defines
    `save_debug_frames` / `save_debug_grid`) → `AttributeError`.
  - reimplements `compute_threshold` / `scan_window`, duplicating
    `analyze_fish_energy.py`'s `compute_threshold_temporal` / `scan_temporal`.

**To empirically confirm status, try each (needs a display + a small test folder):**

```bash
uv run batch_analyze.py --help          # inspect the intended CLI without running
uv run batch_analyze.py videos/Shiner_SloMo/circle    # expect it to crash (see above)
bash  batch_analyze.sh  videos/Shiner_SloMo/circle    # expect a *_results.csv to appear
```

**Other rough edges**
- Config is inconsistent: the single-video CLI hard-codes parameters in `main()`; the batch
  script exposes them as flags.
- No headless mode (ROI selection requires a GUI).
- Some committed result CSVs contain negative `stim_idx` and stray whitespace; `analysis.py`
  does not validate rows.
- Minor dead code: doubled `cap.release()`, commented-out debug blocks referencing an
  undefined `save_debug_panels`, unused `import math`.

See `MILESTONES.md` for the plan to address these.
