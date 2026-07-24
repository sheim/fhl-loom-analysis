"""Aggregate the per-clip annotation JSONs into a tidy table for M4 analysis.

Reusable by the marimo notebooks and by plain scripts — one row per clip. Reads from
``annotations.ANNOTATIONS_DIR`` so it works regardless of the current directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

import annotations as anno

CAMERA_FPS = 240.0
SPECIES = ["sculpin", "shiner"]
CONDITIONS = ["circle", "fixed", "flapping"]

# Response variables the histograms can plot:  label -> (dataframe column, unit)
RESPONSE_VARS = {
    "latency": ("latency_s", "s"),
    "distance": ("distance_cm", "cm"),
    "loom size": ("silhouette_cm", "cm"),
    "expansion rate (dθ/dt)": ("dtheta_dt_deg_per_s", "deg/s"),
    "retinal angle": ("angle_deg", "deg"),
}


def _record(path: Path) -> Dict:
    d = json.loads(path.read_text())
    ann = d.get("annotation") or {}
    res = d.get("results") or {}
    geo = d.get("geometry") or {}
    stim = res.get("stim_idx")
    det = ann.get("manual_det")
    if det is None:
        det = res.get("det_refined")
    lat_frames = (det - stim) if (det is not None and stim is not None) else None
    fish = ann.get("fish") or []
    species = path.parts[-3].split("_")[0].lower()          # Sculpin_SloMo -> sculpin
    condition = path.parts[-2]
    return {
        "species": species,
        "condition": condition,
        "clip": path.stem,
        "clip_key": f"{species}/{condition}/{path.stem}",   # stable id for exclusion/what-if
        "disposition": d.get("disposition"),
        "monitor_side": ann.get("monitor_side"),
        "detected_blip": ann.get("detected_blip"),
        "stim_idx": stim,
        "det_idx": det,
        "latency_frames": lat_frames,
        "latency_s": (lat_frames / CAMERA_FPS) if lat_frames is not None else None,
        "distance_cm": geo.get("distance_cm"),
        "silhouette_cm": geo.get("silhouette_cm"),
        "angle_deg": geo.get("angle_deg"),
        # expansion rate: the analytic closed form (M3.6) — exact, robust near contact
        "dtheta_dt_deg_per_s": geo.get("dtheta_dt_analytic_deg_per_s"),
        "monitor_frame": geo.get("monitor_frame"),
        "in_range": geo.get("in_range"),
        "n_fish": len(fish),
        "tank_corners": ann.get("tank_corners"),
        "tank_far_corners": ann.get("tank_far_corners"),   # M4.2 perspective homography
        "tank_depth_cm": ann.get("tank_depth_cm"),
        "fish": fish,                                     # kept for the position/orientation plots
    }


def load_records() -> List[Dict]:
    return [_record(p) for p in sorted(anno.ANNOTATIONS_DIR.glob("*/*/*.json"))]


def load_dataframe(usable_only: bool = True) -> pd.DataFrame:
    """One row per clip. ``species``/``condition`` are ordered categoricals for stable plots."""
    df = pd.DataFrame(load_records())
    if usable_only:
        df = df[df["disposition"] == "usable"].reset_index(drop=True)
    df["species"] = pd.Categorical(df["species"], categories=SPECIES, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=True)
    return df


def exclude(df: pd.DataFrame, clip_keys) -> pd.DataFrame:
    """Return ``df`` without the given ``clip_key``s — the reversible 'what-if' filter for outliers."""
    keys = set(clip_keys or [])
    return df[~df["clip_key"].isin(keys)].reset_index(drop=True) if keys else df


def outlier_scores(df: pd.DataFrame, thresh: float = 3.5, var_label: str | None = None) -> pd.DataFrame:
    """Robust outlier candidates via the modified z-score (Iglewicz–Hoaglin: 0.6745·(x−median)/MAD)
    computed *within* each species×condition group. Returns one row per clip with its variable, value,
    and |z| (descending), flagging |z| ≥ ``thresh``. With ``var_label`` set, scores that one response
    variable; otherwise keeps each clip's single most-extreme variable across all of them. Helps decide
    what to exclude — it never drops anything itself."""
    import numpy as np

    labels = [var_label] if var_label else list(RESPONSE_VARS)
    best: Dict[str, dict] = {}
    for (sp, cond), g in df.groupby(["species", "condition"], observed=True):
        for label in labels:
            col = RESPONSE_VARS[label][0]
            x = g[col].to_numpy(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            if not mad:                                     # degenerate group → no spread to judge
                continue
            z = 0.6745 * (x - med) / mad
            for key, zi, xi in zip(g["clip_key"], z, x):
                if np.isnan(zi):
                    continue
                if key not in best or abs(zi) > abs(best[key]["z"]):
                    best[key] = {"clip_key": key, "species": sp, "condition": cond,
                                 "variable": label, "value": round(float(xi), 3),
                                 "z": round(float(zi), 2)}
    out = pd.DataFrame(best.values())
    if out.empty:
        return out
    out["outlier"] = out["z"].abs() >= thresh
    return out.sort_values("z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = load_dataframe()
    print("clips per group:")
    print(df.groupby(["species", "condition"], observed=True).size().to_string())
    cols = [c for _, (c, _) in RESPONSE_VARS.items()]
    print("\nresponse variables (usable):")
    print(df[cols].describe().round(2).to_string())
