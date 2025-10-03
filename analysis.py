# Save separate histograms for sculpin and shiner with shared axes

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
)

FPS = 240.0
CONDITIONS = ("circle", "fixed", "flapping")
CSV_TEMPL = "{species}_{cond}_results.csv"


def summarize_stats(
    species: str, data: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std of latencies for each condition."""
    stats = {}
    for cond, (secs, _) in data.items():
        if secs.size:
            mean = float(np.mean(secs))
            std = float(np.std(secs, ddof=1))
            stats[cond] = (mean, std)
            print(f"{species:8s} - {cond:9s}: mean={mean:.3f} s, std={std:.3f} s")
        else:
            stats[cond] = (np.nan, np.nan)
            print(f"{species:8s} - {cond:9s}: no data")
    return stats


def load_csv_np(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    stim = data[:, 1].astype(int)
    final = data[:, 2].astype(int)
    latency_frames = final - stim
    latency_seconds = latency_frames / FPS
    trial_ids = data[:, 0].astype(int)
    return latency_seconds, trial_ids


def load_species(species: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for cond in CONDITIONS:
        fname = (
            f"{species}_{cond}_results.csv"
            if cond != "fixed"
            else f"{species}_fixed_results.csv"
        )
        path = Path(fname)
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue
        secs, trials = load_csv_np(path)
        out[cond] = (secs, trials)
    return out


def all_latencies(groups):
    arrs = []
    for g in groups:
        for secs, _ in g.values():
            if secs.size:
                arrs.append(secs)
    if not arrs:
        raise RuntimeError("No data found across all species/conditions.")
    return np.concatenate(arrs)


def make_bins(values: np.ndarray, nbins: int = 25) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        eps = 1e-6 if lo == 0 else abs(lo) * 1e-6
        lo -= eps
        hi += eps
    return np.linspace(lo, hi + 1e-12, nbins)


def plot_species(
    species: str,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bins: np.ndarray,
    ymax: int,
    out_path: Path,
    stats: Dict[str, Tuple[float, float]],
) -> None:
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) / 4.0

    offsets = {"circle": -width, "fixed": 0.0, "flapping": +width}
    labels = {"circle": "circle", "fixed": "fixed_fins", "flapping": "flapping"}

    plt.figure(figsize=(9, 5))
    for cond in CONDITIONS:
        if cond not in data:
            continue
        secs = data[cond][0]
        counts, _ = np.histogram(secs, bins=bins)

        mean, std = stats.get(cond, (np.nan, np.nan))
        label = f"{labels[cond]} (μ={mean:.3f}, σ={std:.3f})"

        plt.bar(
            centers + offsets[cond],
            counts,
            width=width,
            label=label,
            align="center",
        )

    plt.xlabel("Timing (s)")
    plt.ylabel("Count")
    plt.title(f"{species.capitalize()} response timing")
    plt.ylim(0, ymax * 1.1 if ymax > 0 else 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    sculpin = load_species("sculpin")
    sculpin_NR = load_species("sculpin_NR")
    shiner = load_species("shiner")

    groups = [g for g in (sculpin, shiner) if g]
    if not groups:
        raise SystemExit("No data files found for sculpin or shiner.")

    # Common bins and global ymax for consistent axes
    all_secs = all_latencies(groups)
    bins = make_bins(all_secs, nbins=25)

    ymax = 0
    for g in groups:
        for cond, (secs, _) in g.items():
            counts, _ = np.histogram(secs, bins=bins)
            ymax = max(ymax, int(np.max(counts)) if counts.size else 0)

    if sculpin:
        stats = summarize_stats("sculpin", sculpin)
        plot_species(
            "sculpin", sculpin, bins, ymax, Path("sculpin_latency_hist.pdf"), stats
        )
    if shiner:
        stats = summarize_stats("shiner", shiner)
        plot_species(
            "shiner", shiner, bins, ymax, Path("shiner_latency_hist.pdf"), stats
        )

    if sculpin_NR:
        stats = summarize_stats("sculpin no-reuse", sculpin_NR)
        plot_species(
            "sculpin no-reuse",
            sculpin_NR,
            bins,
            ymax,
            Path("sculpin_NR_latency_hist.pdf"),
            stats,
        )


if __name__ == "__main__":
    main()
