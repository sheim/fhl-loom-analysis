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
bash  batch_analyze.sh videos/Shiner_SloMo/circle # batch → <folder>_results.csv
uv run analysis.py                               # aggregate CSVs → latency PDFs
```

## Code map

- `analyze_fish_energy.py` (~1050 lines) — **core logic + interactive single-video CLI**. This
  is the source of truth for stimulus detection (`find_stimulus`), the temporal motion-energy
  engine (`get_kernel`, `energy_temporal_series`), thresholding/scanning
  (`compute_threshold_temporal`, `scan_temporal`, `track_energy_temporal`), and plotting/QA.
- `batch_analyze.sh` — git-tracked batch runner; runs the analyzer per video and scrapes its
  stdout (`Stimulus frame index:`, `Coarse/Refined first-movement frame:`) into a CSV. If you
  change those print strings, update this script's `grep`/`awk`.
- `batch_analyze.py` — untracked, **currently broken** python batch runner that duplicates core
  logic. Do not treat as working; see README "Known issues".
- `analysis.py` — reads `{species}_{cond}_results.csv` from cwd, computes latency at
  `FPS = 240`, writes `*_latency_hist.pdf`.
- `sort_videos.sh` — sorts/renames raw `Trial_*.MP4` into condition folders.

## Important gotchas

- **Requires a display.** ROI selection and viz use `cv2.imshow` + mouse callbacks; nothing
  runs headless yet. Do not assume batch scripts can run in CI/sandbox.
- **Do not run the interactive analyzer to "verify" changes** unless a human is present to
  click ROIs — it will block. Prefer reasoning, targeted reads, and (future) unit tests.
- **Two batch implementations exist** (`.sh` working, `.py` broken). Don't unify or delete
  either without confirming the intended direction (tracked in `MILESTONES.md` M0/M1).
- **Single-video parameters are hard-coded** as locals in `analyze_fish_energy.py:main()`
  (~lines 849–869); the batch python script exposes the same knobs as CLI flags. Keep this
  inconsistency in mind when changing defaults.
- **Data is git-ignored.** `videos/` (~16 GB), `out/`, `*.png`, `*.csv`, `*.pdf` are all
  ignored. Result CSVs are inputs to `analysis.py` and may contain dirty rows (negative
  `stim_idx`, stray whitespace) — validate before trusting.
- `analysis.py` expects species-prefixed CSV names (e.g. `shiner_circle_results.csv`), but
  `batch_analyze.sh` emits folder-named ones (`circle_results.csv`) — a manual rename bridges
  them today.

## Conventions

- Match the existing style: type hints, small pure helper functions, plots saved under `out/`.
- Latency uses `FPS = 240.0`; frame indices are ints; latency = `(final_det_idx − stim_idx)/FPS`.
- Prefer extending the canonical functions in `analyze_fish_energy.py` over re-implementing
  detection logic elsewhere.
