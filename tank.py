"""Tank coordinate system: a perspective-correct map from image pixels to physical tank centimetres.

The four marked corners (2 `tank_corners` on the monitor edge + 2 `tank_far_corners`) plus the tank
depth define a homography `H: image px → tank cm`, where the physical tank is the rectangle
``(0,0)–(59,0)`` (monitor/near edge) by ``(0,d)–(59,d)`` (far edge, ``d = tank_depth_cm``). Mapping the
fish head through ``H`` removes the ~10–15 % depth over-projection of the old single-linear-scale
registration (M4.2). Pure numpy/cv2 — takes raw corner lists, so it's independent and unit-testable.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

TANK_WIDTH_CM = 59.0                                   # the two monitor-edge corners span this


def homography(near, far, depth_cm) -> Optional[np.ndarray]:
    """3×3 homography mapping **image px → tank cm**, or None if the marks are missing/degenerate.

    ``near``/``far`` are ``[[x,y],[x,y]]``. The near corners map to ``(0,0),(59,0)``; each far corner
    is paired to the nearest near corner → ``(0,d)/(59,d)`` (so corner-click order doesn't matter)."""
    if not near or not far or not depth_cm or len(near) != 2 or len(far) != 2:
        return None
    n0, n1 = np.asarray(near[0], float), np.asarray(near[1], float)
    f0, f1 = np.asarray(far[0], float), np.asarray(far[1], float)
    # pair far corners to near by proximity: fA adjoins n0 → (0,d); fB adjoins n1 → (59,d)
    fA, fB = (f0, f1) if np.hypot(*(f0 - n0)) <= np.hypot(*(f1 - n0)) else (f1, f0)
    # reject a degenerate quad: the far edge collapsed onto the near edge (near-collinear corners —
    # a marking slip). Both far corners lie ~on the near-edge line → the homography would be wildly
    # ill-conditioned. Fall back to the linear scale (caller flags perspective_corrected=False).
    span = np.hypot(*(n1 - n0))
    if span == 0:
        return None
    nrm = np.array([-(n1[1] - n0[1]), n1[0] - n0[0]]) / span   # unit normal to the near edge
    if min(abs(np.dot(fA - n0, nrm)), abs(np.dot(fB - n0, nrm))) < 0.05 * span:
        return None
    d = float(depth_cm)
    src = np.float32([n0, n1, fB, fA])
    dst = np.float32([[0, 0], [TANK_WIDTH_CM, 0], [TANK_WIDTH_CM, d], [0, d]])
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    return H if np.all(np.isfinite(H)) else None


def tank_record(near, far, depth_cm) -> Optional[dict]:
    """A self-describing record of the marked tank quad for the saved `geometry` block: the four
    sides (near/monitor + far edge corners) and their lengths — physical (``width_cm`` 59, ``depth_cm``)
    and as measured in pixels for each of the four edges (near/far ≈ width, left/right = depth; the
    near-vs-far pixel gap shows the perspective foreshortening). None if the 4 corners aren't marked."""
    if not near or not far or len(near) != 2 or len(far) != 2:
        return None
    n0, n1 = np.asarray(near[0], float), np.asarray(near[1], float)
    f0, f1 = np.asarray(far[0], float), np.asarray(far[1], float)
    fA, fB = (f0, f1) if np.hypot(*(f0 - n0)) <= np.hypot(*(f1 - n0)) else (f1, f0)  # fA↔n0, fB↔n1

    def seg(a, b):
        return round(float(np.hypot(*(b - a))), 1)

    return {
        "width_cm": TANK_WIDTH_CM,                     # near + far edges
        "depth_cm": float(depth_cm) if depth_cm else None,   # left + right edges
        "near_corners": [[float(near[0][0]), float(near[0][1])], [float(near[1][0]), float(near[1][1])]],
        "far_corners": [[float(far[0][0]), float(far[0][1])], [float(far[1][0]), float(far[1][1])]],
        "sides_px": {
            "near": seg(n0, n1),                       # monitor edge  (≈ width_cm)
            "far": seg(fA, fB),                        # far edge      (≈ width_cm)
            "left": seg(n0, fA),                       # side edge     (= depth_cm)
            "right": seg(n1, fB),                      # side edge     (= depth_cm)
        },
    }


def to_cm(H: np.ndarray, pts):
    """Map image-px point(s) → tank cm through ``H``. ``pts`` is ``(x,y)`` or an ``N×2`` array-like;
    returns the same shape."""
    arr = np.asarray(pts, dtype=float)
    single = arr.ndim == 1
    p = arr.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(p, H).reshape(-1, 2)
    return out[0] if single else out
