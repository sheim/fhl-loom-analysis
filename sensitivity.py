"""dθ/dt timing-sensitivity analysis (M4.1).

dθ/dt is a steep function of loom phase near contact, so the *estimated* dθ/dt at movement onset is
sensitive to error in the onset frame. For each clip we quantify that: how much does the analytic
dθ/dt shift if the onset is off by one camera frame? Reported both absolutely (deg/s per frame) and
relatively (% per frame). Everything needed — tank corners, first-responder head, onset monitor
frame — already lives in the aggregated DataFrame, so this recomputes nothing on disk.

Finding (see MILESTONES M4.1): sensitivity tracks loom **phase / latency** (ρ≈+0.86), *not* distance
(ρ≈−0.08) — and since most first-responders respond late (near contact), many sit in the steep,
high-sensitivity regime, a caveat when interpreting the dθ/dt distribution.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import dataset
import loom_geometry as lg

# camera frame → monitor frames (240 fps camera vs 60 fps monitor lookup)
_CAM_TO_MON = lg.MONITOR_FPS / lg.CAMERA_FPS

# axes worth plotting sensitivity against:  label -> (column, unit)
XVARS = {
    "latency": ("latency_s", "s"),
    "loom phase": ("monitor_frame", "monitor frame"),
    "loom diameter": ("silhouette_cm", "cm"),      # monotonic with loom phase
    "distance": ("distance_cm", "cm"),
    "expansion rate (dθ/dt)": ("dtheta_dt_deg_per_s", "deg/s"),
    "retinal angle": ("angle_deg", "deg"),
}

SENS_ABS = "dtheta_dt_sens_abs"     # deg/s per camera frame
SENS_REL = "dtheta_dt_sens_rel"     # % per camera frame


def _dtheta_at(origin, screen_u, head, m_per_px, monitor_frame, lookup, loom) -> float:
    """Analytic dθ/dt with the silhouette taken at ``monitor_frame`` (fish geometry held fixed)."""
    w, _ = lg.silhouette_m(lookup, monitor_frame)
    return lg.analytical_dtheta_dt(origin, screen_u, head, w, m_per_px, loom)


def clip_sensitivity(corners, head, monitor_frame, lookup, loom, dframes_cam: float = 1.0):
    """(abs, rel) sensitivity of dθ/dt to a ``dframes_cam``-camera-frame onset error, via a central
    difference in onset time. ``abs`` is deg/s per camera frame, ``rel`` is % of the onset dθ/dt per
    frame. None if the geometry is missing."""
    frame = lg.screen_frame(corners) if corners else None
    if frame is None or head is None or monitor_frame is None:
        return None
    origin, screen_u, m_per_px = frame
    d = dframes_cam * _CAM_TO_MON
    up = _dtheta_at(origin, screen_u, head, m_per_px, monitor_frame + d, lookup, loom)
    dn = _dtheta_at(origin, screen_u, head, m_per_px, monitor_frame - d, lookup, loom)
    base = _dtheta_at(origin, screen_u, head, m_per_px, monitor_frame, lookup, loom)
    sens_abs = (up - dn) / (2.0 * dframes_cam)
    sens_rel = 100.0 * sens_abs / base if base else float("nan")
    return sens_abs, sens_rel


def sensitivity_frame(df: pd.DataFrame, dframes_cam: float = 1.0) -> pd.DataFrame:
    """Return a copy of ``df`` with the two sensitivity columns added (one row per clip)."""
    lookup = lg.load_lookup()
    loom = lg.loom_params()
    abs_, rel_ = [], []
    for row in df.itertuples():
        head = tuple(row.fish[0]["head"]) if getattr(row, "fish", None) else None
        mf = row.monitor_frame
        if mf is None or (isinstance(mf, float) and math.isnan(mf)):
            abs_.append(np.nan); rel_.append(np.nan); continue
        res = clip_sensitivity(row.tank_corners, head, mf, lookup, loom, dframes_cam)
        if res is None:
            abs_.append(np.nan); rel_.append(np.nan)
        else:
            abs_.append(res[0]); rel_.append(res[1])
    out = df.copy()
    out[SENS_ABS] = abs_
    out[SENS_REL] = rel_
    return out


def ensure_sensitivity(df: pd.DataFrame, dframes_cam: float = 1.0) -> pd.DataFrame:
    """Add sensitivity columns only if absent (so plots can be handed either kind of frame)."""
    if SENS_ABS in df.columns and SENS_REL in df.columns:
        return df
    return sensitivity_frame(df, dframes_cam)


if __name__ == "__main__":
    df = ensure_sensitivity(dataset.load_dataframe())
    cols = ["clip_key", "latency_s", "monitor_frame", "distance_cm",
            "dtheta_dt_deg_per_s", SENS_ABS, SENS_REL]
    print("most timing-sensitive dθ/dt estimates:")
    print(df.sort_values(SENS_REL, ascending=False)[cols].head(8).round(2).to_string(index=False))
    print("\nSpearman ρ of |sensitivity| vs candidate x-axes:")
    r = df[SENS_REL].abs()
    for label, (col, _) in XVARS.items():
        rho = df[col].rank().corr(r.rank())
        print(f"  {label:22s} {rho:+.3f}")
