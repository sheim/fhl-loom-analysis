# Re-run after environment reset: reload libraries and files, then produce side-by-side histogram

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update(
    {
        "font.size": 24,  # default text size
        "axes.titlesize": 18,  # title size
        "axes.labelsize": 18,  # x/y label size
        "xtick.labelsize": 18,  # x tick labels
        "ytick.labelsize": 18,  # y tick labels
        "legend.fontsize": 18,  # legend
    }
)

FPS = 240.0

CIRCLE_CSV = Path("sculpin_circle_results.csv")
FIXED_FINS_CSV = Path("sculpin_fixed_results.csv")
FLAPPING_CSV = Path("sculpin_flapping_results.csv")
OUT_FIG = Path("sculpin_latency_histograms_side_by_side.png")


def load_csv_np(path: Path):
    data = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=int)

    stim = data[:, 1]
    final = data[:, 2]
    latency_frames = final - stim
    latency_seconds = latency_frames / FPS
    return latency_seconds, data[:, 0]


circle_secs, circle_trials = load_csv_np(CIRCLE_CSV)
fixed_secs, fixed_trials = load_csv_np(FIXED_FINS_CSV)
flapping_secs, flapping_trials = load_csv_np(FLAPPING_CSV)

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
    align="center",
)
plt.bar(
    centers,
    fixed_counts,
    width=width,
    label="fixed_fins",
    align="center",
)
plt.bar(
    centers + width,
    flapping_counts,
    width=width,
    label="flapping",
    align="center",
)

plt.xlabel("Timing (s)")
plt.ylabel("Count")
plt.title("Sculpin Response Timing by Stimulus Type")
plt.legend()
plt.tight_layout()
# plt.savefig(OUT_FIG, dpi=150)
plt.show()


# nice print of trial number and timing, split by stimulus type
def print_timings(label, timings, trial_number):
    print(f"{label}:")
    for i, t in enumerate(timings):
        print(f"  Trial {trial_number[i]}: {t:.3f} s")
    mean_t = np.mean(timings)
    std_t = np.std(timings)
    print(f"  Mean: {mean_t:.3f} s, Std: {std_t:.3f} s")
    print()


print_timings("Circle", circle_secs, circle_trials)
print_timings("Fixed Fins", fixed_secs, fixed_trials)
print_timings("Flapping", flapping_secs, flapping_trials)
