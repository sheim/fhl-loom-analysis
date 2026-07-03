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
- A **display / GUI** for the one-time **annotation** step (ROI selection uses `cv2.imshow` +
  mouse callbacks). Once a folder is annotated, analysis re-runs **headless** via
  `batch.py --from-annotations`.
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

| Script | Language | Role |
|---|---|---|
| `sort_videos.sh` | bash | Sort/rename raw `Trial_*.MP4` into condition folders (one-time prep) |
| `analyze_fish_energy.py` | python | Core importable library **and** interactive single-video CLI |
| `annotate.py` | python | One-time interactive sweep: record ROIs + disposition per video → `annotations/*.json` |
| `annotations.py` | python | Annotation schema + load/save (imported by `annotate.py` and `batch.py`) |
| `monitor_side.py` | python | Infer monitor side (top/bottom) from the stim ROI → `annotation.monitor_side` |
| `geometry.py` | python | Mark tank corners + fish head/tail (first = first responder) on the exported frame stack (M3) |
| `fix_start.py` | python | Scrub to manually correct the movement-onset frame → updates annotation + regenerates frames |
| `batch.py` | python | Batch a folder or one file → `<species>_<condition>_results.csv`; `--from-annotations` (headless) |
| `analysis.py` | python | Aggregate result CSVs → latency histogram PDFs |

`analyze_fish_energy.py` is the single source of detection logic. `analyze_video(video,
stim_roi, fish_roi, params) -> AnalysisResult` is side-effect-free (no printing/GUI/exit);
both the CLI and `batch.py` call it, and parameters live in one shared `AnalysisParams`
dataclass.

### 1. `sort_videos.sh` — footage prep (run once)

Creates `circle/ghost/flapping/fixed` subfolders inside the target dir, then moves each
`Trial_<num>_<...>.MP4` into the folder whose name it contains (case-insensitive), renaming
it to `<num>.MP4` (collisions get a `_i` suffix).

```bash
bash sort_videos.sh videos/Shiner_SloMo
bash sort_videos.sh videos/Sculpin_SloMo
```

### 2. `analyze_fish_energy.py` — interactive single-video analyzer

Opens one clip, plays it once for orientation, then prompts you to draw two ROIs with two
mouse clicks each (**stimulus area**, then **fish body** — place it on the first fish to
respond).

```bash
uv run analyze_fish_energy.py videos/Shiner_SloMo/circle/34.MP4
```

- **Input:** one video path (only positional arg).
- **Stdout:** `Stimulus frame index: N`, `Coarse first-movement frame: N`, `Refined
  first-movement frame: N`.
- **Output file:** an energy-vs-frame plot at `out/<species>_<condition>_<stem>.png`
  (e.g. `out/Shiner_SloMo_circle_34.png`).
- **Tuning:** parameters live in the `AnalysisParams` dataclass (defaults reproduce the
  previous behaviour). The single-video CLI uses the defaults; `batch.py` exposes them as flags.

### 3. `annotate.py` — annotation sweep (do this once per folder)

Records, per video, what a human must eyeball — so downstream analysis runs headless. For each
clip it plays the video, asks a **disposition** (`usable` / `no_response` / `bad_video`) and
which **blip** the detector caught (`first` / `middle` / `last` — the stimulus blip appears 3×,
and late-started recordings can miss the first), then lets you click the **stimulus** and
**fish** (first responder) ROIs. It runs the analyzer and caches `stim_idx` / `det_refined`
alongside the ROIs in a per-video JSON under a git-tracked `annotations/` mirror of `videos/`
(e.g. `annotations/Shiner_SloMo/circle/34.json`).

The reported/CSV `stim_idx` is the **first-blip reference**: a `middle`/`last` detection is
shifted back 240/480 frames (the blips are 1 s = 240 frames apart at 240 FPS; `analysis.py`'s
latency then comes out right). This can be **negative** for late-started clips — that's
expected, not a bad row.

```bash
uv run annotate.py videos/Shiner_SloMo/circle              # only un-annotated clips (default)
uv run annotate.py videos/Shiner_SloMo/circle --redo-all   # re-annotate everything
uv run annotate.py videos/Shiner_SloMo/circle --redo 34.MP4  # re-annotate one clip
uv run annotate.py videos/Shiner_SloMo/circle --show       # display saved ROIs/results (all)
uv run annotate.py videos/Shiner_SloMo/circle --show 34.MP4  # ...or just one clip
uv run annotate.py videos/Shiner_SloMo/circle --show-results  # text-only, no window (headless)
```

The `annotations/*.json` files are hand-made and **version-controlled** (unlike the videos).

### 4. `batch.py` — batch runner

Runs the analyzer over a **folder or a single video file** and writes one CSV. Results come
from `analyze_video()`'s return value — no stdout scraping — and unreadable/cancelled clips are
skipped, not fatal. Each line prints the threshold `thr`, the peak scan energy `Emax`, and the
effective `sigma`, so you can tune from the terminal (add `--show` to pop the energy plot). It
also exports a 10-frame stack around the movement onset to a git-ignored `frames/` mirror (the
canvas for M3 geometry marking; disable with `--no-frames`).

```bash
# Headless once annotated (recommended):
uv run batch.py videos/Shiner_SloMo/circle --from-annotations
#  → shiner_circle_results.csv   (species_condition; header filename,stim_idx,final_det_idx)

# Re-run ONE clip to tune detection (annotation untouched until --update-annotations):
uv run batch.py videos/Sculpin_SloMo/flapping/48.MP4 --from-annotations --energy-sigma 8 --show
uv run batch.py videos/Sculpin_SloMo/flapping/48.MP4 --from-annotations --energy-sigma 8 --update-annotations
uv run batch.py --help                                                              # all flags
```

**Parameter precedence.** With `--from-annotations`, effective params =
`AnalysisParams` defaults ← the clip's **saved params** (its annotation's `results.params`) ←
any flag you pass (a passed flag always wins). So per-clip tuning persists in the JSON once you
save it with `--update-annotations`. Note the annotation's `annotation` block (ROIs, `detected_blip`)
and `results.params` are **inputs**; `results.stim_idx`/`det_*` are recomputed **outputs** —
editing those by hand has no effect.

The CSV is named `<species>_<condition>_results.csv` (species = first token of the folder
above the condition folder, lowercased: `Sculpin_SloMo` → `sculpin`) — exactly what
`analysis.py` reads, so **no manual rename** is needed.

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
uv sync                                                   # 1. install
bash sort_videos.sh videos/Shiner_SloMo                   # 2. prep footage (once)
uv run annotate.py videos/Shiner_SloMo/circle             # 3. annotate ROIs once (interactive)
uv run batch.py videos/Shiner_SloMo/circle --from-annotations  # 4. → shiner_circle_results.csv
uv run analysis.py                                        # 5. produce latency histograms
```

---

## Status & known limitations

M1 unified detection logic into `analyze_fish_energy.py` (replacing the stdout-scraping
`batch_analyze.sh` with `batch.py`); M2 added per-video annotations so analysis runs headless
via `batch.py --from-annotations`. Remaining limitations, tracked in `MILESTONES.md`:

- **Annotation is still interactive** — the one-time sweep needs a GUI to click ROIs (by
  design); only the re-runs are headless.
- **No data validation** — result CSVs can contain stray whitespace and `analysis.py` doesn't
  reject malformed rows (M4). Note: negative `stim_idx` is **legitimate** (first-blip reference
  for late-started clips), so validation must not reject it.
- **`analysis.py` expects renamed CSVs** — `batch.py` emits `<folder>_results.csv`; you must
  rename to the species-prefixed form (e.g. `shiner_circle_results.csv`) it reads.
- **Gaussian smoothing can trigger premature detection** (noted in git history) — M4.

See `MILESTONES.md` for the full plan.
