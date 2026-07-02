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
- `batch.py` — batch runner; imports `analyze_fish_energy` and calls `analyze_video()` per
  video, writing `<folder>_results.csv` (`filename,stim_idx,final_det_idx`). No stdout scraping.
- `analysis.py` — reads `{species}_{cond}_results.csv` from cwd, computes latency at
  `FPS = 240`, writes `*_latency_hist.pdf`.
- `sort_videos.sh` — sorts/renames raw `Trial_*.MP4` into condition folders.

## Important gotchas

- **Requires a display.** ROI selection uses `cv2.imshow` + mouse callbacks; there is no
  headless mode yet (saved/reused ROIs are M2). Do not assume `batch.py` runs in CI/sandbox.
- **Do not run the interactive analyzer/`batch.py` to "verify" changes** — they block on ROI
  clicks. To verify logic, exercise `analyze_video` / `first_sustained` on a synthetic clip or
  arrays (no GUI), plus `py_compile`/import checks.
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
