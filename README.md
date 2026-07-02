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
| `batch.py` | python | Batch a folder → `<folder>_results.csv`; interactive **or** `--from-annotations` (headless) |
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
clip it plays the video, asks a **disposition** (`usable` / `no_response` / `bad_video`), then
lets you click the **stimulus** and **fish** (first responder) ROIs; it runs the analyzer and
caches `stim_idx` / `det_refined` alongside the ROIs in a per-video JSON under a git-tracked
`annotations/` mirror of `videos/` (e.g. `annotations/Shiner_SloMo/circle/34.json`).

```bash
uv run annotate.py videos/Shiner_SloMo/circle              # only un-annotated clips (default)
uv run annotate.py videos/Shiner_SloMo/circle --redo-all   # re-annotate everything
uv run annotate.py videos/Shiner_SloMo/circle --redo 34.MP4  # re-annotate one clip
uv run annotate.py videos/Shiner_SloMo/circle --show       # display saved ROIs/results
```

The `annotations/*.json` files are hand-made and **version-controlled** (unlike the videos).

### 4. `batch.py` — batch runner

Runs the analyzer over every video in a folder and writes one CSV named after the folder.
Results come from `analyze_video()`'s return value — no stdout scraping — and
unreadable/cancelled clips are skipped, not fatal.

```bash
# Headless once annotated (recommended):
uv run batch.py videos/Shiner_SloMo/circle --from-annotations
#  → circle_results.csv   with header:  filename,stim_idx,final_det_idx

# Interactive (selects ROIs per video), with tuning:
uv run batch.py videos/Shiner_SloMo/circle --energy-sigma 5 --stride 5
uv run batch.py videos/Shiner_SloMo/circle --from-annotations --update-annotations  # refresh cache
uv run batch.py --help                                                              # all flags
```

To feed `analysis.py`, rename the output to the species-prefixed name it expects, e.g.
`circle_results.csv` → `shiner_circle_results.csv`.

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
uv run batch.py videos/Shiner_SloMo/circle --from-annotations  # 4. analyze headless
mv circle_results.csv shiner_circle_results.csv           # 5. rename for the aggregator
uv run analysis.py                                        # 6. produce latency histograms
```

---

## Status & known limitations

M1 unified detection logic into `analyze_fish_energy.py` (replacing the stdout-scraping
`batch_analyze.sh` with `batch.py`); M2 added per-video annotations so analysis runs headless
via `batch.py --from-annotations`. Remaining limitations, tracked in `MILESTONES.md`:

- **Annotation is still interactive** — the one-time sweep needs a GUI to click ROIs (by
  design); only the re-runs are headless.
- **No data validation** — some existing result CSVs contain negative `stim_idx` / stray
  whitespace, and `analysis.py` doesn't reject bad rows (M4).
- **`analysis.py` expects renamed CSVs** — `batch.py` emits `<folder>_results.csv`; you must
  rename to the species-prefixed form (e.g. `shiner_circle_results.csv`) it reads.
- **Gaussian smoothing can trigger premature detection** (noted in git history) — M4.

See `MILESTONES.md` for the full plan.
