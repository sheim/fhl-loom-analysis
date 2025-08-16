# Re-run after environment reset: reload libraries and files, then produce side-by-side histogram

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FPS = 240.0

CIRCLE_CSV = Path("batch_results_circle.csv")
FIXED_FINS_CSV = Path("batch_results_fixed_fins.csv")
FLAPPING_CSV = Path("batch_results_flapping.csv")
OUT_FIG = Path("latency_histograms_side_by_side.png")


def load_csv_np(path: Path):
    data = np.genfromtxt(
        str(path), delimiter=",", skip_header=1, dtype=None, encoding=None
    )
    stim = data["f1"].astype(np.int64)
    final = data["f2"].astype(np.int64)
    latency_frames = final - stim
    latency_seconds = latency_frames / FPS
    return latency_seconds


circle_secs = load_csv_np(CIRCLE_CSV)
fixed_secs = load_csv_np(FIXED_FINS_CSV)
flapping_secs = load_csv_np(FLAPPING_CSV)

# Common binning
all_secs = np.concatenate([circle_secs, fixed_secs, flapping_secs])
lo = float(np.min(all_secs))
hi = float(np.max(all_secs))
bins = np.linspace(lo, hi + 1e-9, 25)

# Compute counts
circle_counts, _ = np.histogram(circle_secs, bins=bins)
fixed_counts, _ = np.histogram(fixed_secs, bins=bins)
flapping_counts, _ = np.histogram(flapping_secs, bins=bins)

# Bar centers and width
centers = (bins[:-1] + bins[1:]) / 2
width = (bins[1] - bins[0]) / 4.0

# Plot side-by-side bars
plt.figure(figsize=(9, 5))
plt.bar(
    centers - width,
    circle_counts,
    width=width,
    label="circle",
    color="tab:blue",
    align="center",
)
plt.bar(
    centers,
    fixed_counts,
    width=width,
    label="fixed_fins",
    color="tab:orange",
    align="center",
)
plt.bar(
    centers + width,
    flapping_counts,
    width=width,
    label="flapping",
    color="tab:green",
    align="center",
)

plt.xlabel("Latency (s)")
plt.ylabel("Count")
plt.title("Response Latency by Stimulus Type (side-by-side)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
plt.show()

OUT_FIG
