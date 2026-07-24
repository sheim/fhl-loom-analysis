"""Fish-position registration for the M4 spatial overlay.

Raw head/tail pixels aren't comparable across clips (camera framing differs), so each fish is mapped
into a **loom-centred, screen-aligned metric frame** using the clip's tank corners:

  - origin = loom origin (midpoint of the two tank corners);
  - **along** = signed position parallel to the screen, in cm from the loom centre;
  - **depth** = perpendicular distance from the screen, in cm, oriented so the fish are at depth > 0.

This makes every fish comparable across clips and species. One row per fish per clip; ``is_responder``
marks the first-marked fish (index 0). ``along`` sign (left/right of the loom centre) is arbitrary per
clip — the two tank corners have no canonical order — so read the along-axis as symmetric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dataset
import loom_geometry as lg
import tank

M_TO_CM = 100.0
HALF_SCREEN_CM = lg.TANK_WIDTH_M * M_TO_CM / 2.0     # 29.5 cm each side of the loom origin


def _project(pt, origin, u, n, m_per_px):
    dx, dy = pt[0] - origin[0], pt[1] - origin[1]
    along = (dx * u[0] + dy * u[1]) * m_per_px * M_TO_CM
    depth = (dx * n[0] + dy * n[1]) * m_per_px * M_TO_CM
    return along, depth


def _clip_mapper(corners, far, depth, fish):
    """A per-clip ``pt -> (along, depth)`` in loom-centred cm. Uses the tank homography (perspective-
    correct) when the 4 corners + depth are marked, else the linear 2-corner scale. None if unusable."""
    H = tank.homography(corners, far, depth)
    if H is not None:
        return lambda pt: (float(tank.to_cm(H, pt)[0]) - HALF_SCREEN_CM, float(tank.to_cm(H, pt)[1]))
    frame = lg.screen_frame(corners) if corners else None
    if frame is None:
        return None
    origin, u, m_per_px = frame
    n = (-u[1], u[0])                                 # screen normal
    heads = [f["head"] for f in fish if "head" in f]
    if np.mean([(h[0] - origin[0]) * n[0] + (h[1] - origin[1]) * n[1] for h in heads]) < 0:
        n = (u[1], -u[0])                             # orient so the fish sit at depth > 0
    return lambda pt: _project(pt, origin, u, n, m_per_px)


def fish_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the clip-level table into one row per fish, in loom-centred cm coordinates
    (perspective-correct via the tank homography where the 4 corners + depth exist)."""
    rows = []
    for r in df.itertuples():
        fish = r.fish or []
        depth = getattr(r, "tank_depth_cm", None)
        mapper = _clip_mapper(r.tank_corners, getattr(r, "tank_far_corners", None), depth, fish)
        if mapper is None or not fish:
            continue
        # per-clip experiment details (same for every fish in the clip) — carried for tooltips
        trial = {c: getattr(r, c, None) for c in
                 ("latency_s", "distance_cm", "angle_deg", "dtheta_dt_deg_per_s",
                  "monitor_side", "detected_blip", "in_range")}
        for i, f in enumerate(fish):
            ha, hd = mapper(f["head"])
            ta, td = mapper(f["tail"])
            rows.append({
                "clip_key": r.clip_key,
                "species": str(r.species),
                "condition": str(r.condition),
                "fish_index": i,
                "is_responder": i == 0,
                "head_along": ha, "head_depth": hd,
                "tail_along": ta, "tail_depth": td,
                "tank_depth_cm": depth,
                **trial,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fdf = fish_frame(dataset.load_dataframe())
    print(f"{len(fdf)} fish  ({fdf['is_responder'].sum()} responders, "
          f"{(~fdf['is_responder']).sum()} others)")
    for lab, g in (("responders", fdf[fdf.is_responder]), ("others", fdf[~fdf.is_responder])):
        print(f"  {lab:11s}: along {g.head_along.min():6.1f}..{g.head_along.max():5.1f} cm   "
              f"depth {g.head_depth.min():5.1f}..{g.head_depth.max():5.1f} cm   "
              f"(mean depth {g.head_depth.mean():.1f})")
