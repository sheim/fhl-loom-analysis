"""Plot functions for the M4 analysis (pure: take a DataFrame, return a matplotlib Figure).

Used by the marimo notebook for interactive display and by the ``__main__`` here to save PNGs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

import dataset
import fits
import positions as pos_mod
import sensitivity as sens_mod

_COND_COLORS = {"circle": "#4C72B0", "fixed": "#DD8452", "flapping": "#55A868"}
_COND_MARKERS = {"circle": "o", "fixed": "s", "flapping": "D"}
_SPECIES_MARKERS = {"sculpin": "o", "shiner": "^"}


def _overlay_fit(ax, vals, fit: str, edges, log: bool) -> None:
    """Fit the requested distribution(s) to ``vals`` and draw the PDF over the panel's x-range,
    annotating the chosen distribution and its ΔAICc vs normal."""
    ranked = fits.fit_all(vals)
    if not ranked:
        return
    chosen = ranked[0] if fit == "auto" else next((f for f in ranked if f.name == fit), None)
    if chosen is None:
        return
    lo, hi = edges[0], edges[-1]
    xs = (np.logspace(np.log10(max(lo, 1e-9)), np.log10(hi), 200) if log
          else np.linspace(lo, hi, 200))
    ax.plot(xs, chosen.pdf(xs), color="k", lw=1.6, zorder=5)
    aicc_norm = next((f.aicc for f in ranked if f.name == "normal"), None)
    delta = f"  ΔAICc {aicc_norm - chosen.aicc:+.0f}" if aicc_norm is not None else ""
    ax.text(0.96, 0.80, f"{chosen.name}{delta}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="k")


def response_histograms(
    df,
    var_label: str,
    conditions: Optional[List[str]] = None,
    bins: int = 12,
    density: bool = False,
    log: bool = False,
    fit: Optional[str] = None,
):
    """Grid of histograms — rows = species (sculpin, shiner), columns = the selected conditions,
    each in its own panel on shared axes. ``log`` uses log-spaced bins + a log x-axis (for skewed
    variables like dθ/dt); ``density`` normalises each panel to unit area.

    ``fit`` overlays a fitted distribution PDF per panel: ``"auto"`` picks the best by AICc, or name a
    candidate (``"normal"``/``"lognormal"``/``"gamma"``/``"weibull"``). Fitting forces the density
    scale; each panel is annotated with the distribution and its ΔAICc vs normal (>0 ⇒ beats normal)."""
    col, unit = dataset.RESPONSE_VARS[var_label]
    conditions = list(conditions) if conditions else list(dataset.CONDITIONS)
    ncol = max(1, len(conditions))
    density = density or bool(fit)                  # a PDF overlay only makes sense on a density axis

    allvals = df[df["condition"].isin(conditions)][col].dropna().to_numpy(dtype=float)
    if log:
        pos = allvals[allvals > 0]
        lo, hi = (pos.min(), pos.max()) if pos.size else (1.0, 10.0)
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    else:
        lo, hi = (allvals.min(), allvals.max()) if allvals.size else (0.0, 1.0)
        edges = np.linspace(lo, hi, bins + 1)

    fig, axes = plt.subplots(
        len(dataset.SPECIES),
        ncol,
        figsize=(3.6 * ncol, 5.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for r, sp in enumerate(dataset.SPECIES):
        for c, cond in enumerate(conditions):
            ax = axes[r][c]
            vals = (
                df[(df["species"] == sp) & (df["condition"] == cond)][col]
                .dropna()
                .to_numpy(dtype=float)
            )
            ax.hist(vals, bins=edges, color=_COND_COLORS.get(cond), density=density)
            if log:
                ax.set_xscale("log")
            if fit:
                _overlay_fit(ax, vals, fit, edges, log)
            ax.text(
                0.96,
                0.94,
                f"n={vals.size}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            if r == 0:
                ax.set_title(cond)
            if r == len(dataset.SPECIES) - 1:
                ax.set_xlabel(f"{var_label} ({unit})")
            if c == 0:
                ax.set_ylabel(f"{sp.capitalize()}\n{'density' if density else 'count'}")
    fig.suptitle(f"Response distribution — {var_label}")
    fig.tight_layout()
    return fig


def sensitivity_scatter(df, xvar: str = "latency", rel: bool = True, dframes_cam: float = 1.0):
    """Per-clip dθ/dt timing sensitivity (change per 1-camera-frame onset error) vs a chosen x-axis,
    with a light marginal histogram of where the first-responders sit. ``rel`` → % per frame (else
    deg/s per frame). Markers = species, colours = condition."""
    from matplotlib.lines import Line2D

    df = sens_mod.ensure_sensitivity(df, dframes_cam)
    xcol, xunit = sens_mod.XVARS[xvar]
    ycol = sens_mod.SENS_REL if rel else sens_mod.SENS_ABS
    yunit = "% per camera frame" if rel else "deg/s per camera frame"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax_h = ax.twinx()                                   # responder marginal, drawn behind the points
    xall = df[xcol].dropna().to_numpy(dtype=float)
    if xall.size:
        ax_h.hist(xall, bins=15, color="0.88", zorder=0)
    ax_h.set_ylabel("first-responder count", color="0.55")
    ax_h.tick_params(axis="y", colors="0.55")

    for sp in dataset.SPECIES:
        for cond in dataset.CONDITIONS:
            g = df[(df["species"] == sp) & (df["condition"] == cond)]
            ax.scatter(g[xcol], g[ycol], marker=_SPECIES_MARKERS[sp], color=_COND_COLORS[cond],
                       edgecolor="k", linewidth=0.3, s=42, zorder=3)
    ax.axhline(0, color="k", lw=0.6, zorder=2)
    ax.set_xlabel(f"{xvar} ({xunit})")
    ax.set_ylabel(f"dθ/dt sensitivity ({yunit})")
    ax.set_zorder(ax_h.get_zorder() + 1)                # keep the scatter above the histogram
    ax.patch.set_visible(False)

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=_COND_COLORS[c],
                      markersize=9, label=c) for c in dataset.CONDITIONS]
    handles += [Line2D([0], [0], marker=_SPECIES_MARKERS[s], color="k", linestyle="",
                       markerfacecolor="0.7", markersize=8, label=s) for s in dataset.SPECIES]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    fig.suptitle(f"dθ/dt timing sensitivity vs {xvar}")
    fig.tight_layout()
    return fig


def sensitivity_scatter_interactive(df, xvar: str = "latency", rel: bool = True, dframes_cam: float = 1.0):
    """Plotly version of :func:`sensitivity_scatter` — same content but with **hover tooltips**
    (clip_key + latency/phase/distance/dθ-dt) and a marginal responder histogram. For the marimo
    notebook; the matplotlib version above is what ``save_all`` writes as a static PNG."""
    import plotly.express as px

    df = sens_mod.ensure_sensitivity(df, dframes_cam)
    xcol, xunit = sens_mod.XVARS[xvar]
    ycol = sens_mod.SENS_REL if rel else sens_mod.SENS_ABS
    yunit = "% per camera frame" if rel else "deg/s per camera frame"

    keep = list(dict.fromkeys([
        xcol, ycol, "condition", "species", "clip_key",
        "latency_s", "monitor_frame", "distance_cm", "dtheta_dt_deg_per_s",
    ]))
    d = df[keep].copy()
    d["species"] = d["species"].astype(str)
    d["condition"] = d["condition"].astype(str)

    fig = px.scatter(
        d, x=xcol, y=ycol, color="condition", symbol="species",
        marginal_x="histogram", color_discrete_map=_COND_COLORS,
        category_orders={"condition": list(dataset.CONDITIONS), "species": list(dataset.SPECIES)},
        hover_data=["clip_key", "latency_s", "monitor_frame", "distance_cm", "dtheta_dt_deg_per_s"],
        labels={xcol: f"{xvar} ({xunit})", ycol: f"dθ/dt sensitivity ({yunit})"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")),
                      selector=dict(type="scatter", mode="markers"))
    fig.update_layout(title=f"dθ/dt timing sensitivity vs {xvar}", height=540,
                      legend_title_text="condition / species")
    return fig


def _bubble_area(mag, mmax):
    """Marker area (pt²) for scatter: proportional to |sensitivity| with a visible floor."""
    return 30.0 + 470.0 * (mag / mmax if mmax else 0.0)


def sensitivity_bubble(df, xvar: str = "loom phase", yvar: str = "distance", rel: bool = True,
                       threshold: float = 0.0, dframes_cam: float = 1.0):
    """2D bubble chart: two chosen sensitivity-axes, marker **area ∝ |dθ/dt sensitivity|**, open
    markers with a condition-coloured border (shape = species). Only clips with |sensitivity| ≥
    ``threshold`` are drawn (0 = all). Matplotlib/static; see the ``_interactive`` twin for hover."""
    from matplotlib.lines import Line2D

    df = sens_mod.ensure_sensitivity(df, dframes_cam)
    xcol, xunit = sens_mod.XVARS[xvar]
    ycol, yunit = sens_mod.XVARS[yvar]
    scol = sens_mod.SENS_REL if rel else sens_mod.SENS_ABS
    sunit = "%/frame" if rel else "deg/s per frame"

    d = df[df[scol].abs() >= threshold].reset_index(drop=True)
    magf = d[scol].abs().to_numpy(dtype=float)
    mmax = magf.max() if magf.size else 1.0
    sizes = _bubble_area(magf, mmax)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for sp in dataset.SPECIES:
        for cond in dataset.CONDITIONS:
            m = ((d["species"] == sp) & (d["condition"] == cond)).to_numpy()
            if not m.any():
                continue
            ax.scatter(d[xcol].to_numpy()[m], d[ycol].to_numpy()[m], s=sizes[m],
                       facecolors="none", edgecolors=_COND_COLORS[cond],
                       marker=_SPECIES_MARKERS[sp], linewidths=1.3)
    ax.set_xlabel(f"{xvar} ({xunit})")
    ax.set_ylabel(f"{yvar} ({yunit})")
    ax.set_title(f"dθ/dt sensitivity — bubble ∝ |sensitivity| ({sunit})")

    handles = [Line2D([0], [0], marker="o", color=_COND_COLORS[c], linestyle="", markerfacecolor="none",
                      markeredgewidth=1.5, markersize=9, label=c) for c in dataset.CONDITIONS]
    handles += [Line2D([0], [0], marker=_SPECIES_MARKERS[s], color="0.4", linestyle="",
                       markerfacecolor="none", markeredgewidth=1.5, markersize=9, label=s)
                for s in dataset.SPECIES]
    leg1 = ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left", title="condition / species")
    ax.add_artist(leg1)

    if magf.size:                                       # size-reference legend
        refs = sorted({round(mmax * f, 1) for f in (0.25, 0.5, 1.0)} - {0.0})
        rh = [Line2D([0], [0], marker="o", color="0.5", linestyle="", markerfacecolor="none",
                     markeredgewidth=1.2, markersize=math.sqrt(_bubble_area(r, mmax)),
                     label=f"{r} {sunit}") for r in refs]
        ax.add_artist(ax.legend(handles=rh, title="|sensitivity|", loc="lower right", fontsize=8,
                                labelspacing=1.6, borderpad=1.0, handletextpad=1.4, frameon=True))
    fig.tight_layout()
    return fig


def sensitivity_bubble_interactive(df, xvar: str = "loom phase", yvar: str = "distance",
                                   rel: bool = True, threshold: float = 0.0, dframes_cam: float = 1.0):
    """Plotly twin of :func:`sensitivity_bubble` — open markers (condition-coloured border, species
    symbol), area ∝ |sensitivity|, |sensitivity| ≥ ``threshold``, with clip_key hover tooltips."""
    import plotly.graph_objects as go

    df = sens_mod.ensure_sensitivity(df, dframes_cam)
    xcol, xunit = sens_mod.XVARS[xvar]
    ycol, yunit = sens_mod.XVARS[yvar]
    scol = sens_mod.SENS_REL if rel else sens_mod.SENS_ABS
    sunit = "%/frame" if rel else "deg/s per frame"
    symbols = {"sculpin": "circle", "shiner": "triangle-up"}

    d = df.copy()
    d["_mag"] = d[scol].abs()
    d = d[d["_mag"] >= threshold]
    mmax = d["_mag"].max() if len(d) else 1.0
    sizeref = 2.0 * mmax / (40.0 ** 2)                  # plotly area-mode reference (max ≈ 40 px)

    fig = go.Figure()
    for sp in dataset.SPECIES:
        for cond in dataset.CONDITIONS:
            g = d[(d["species"].astype(str) == sp) & (d["condition"].astype(str) == cond)]
            if not len(g):
                continue
            cd = g[["clip_key", scol, "latency_s", "monitor_frame",
                    "distance_cm", "silhouette_cm", "dtheta_dt_deg_per_s"]].to_numpy()
            fig.add_trace(go.Scatter(
                x=g[xcol], y=g[ycol], mode="markers", name=f"{cond} · {sp}",
                marker=dict(size=g["_mag"], sizemode="area", sizeref=sizeref, sizemin=4,
                            color="rgba(0,0,0,0)", symbol=symbols[sp],
                            line=dict(color=_COND_COLORS[cond], width=2)),
                customdata=cd,
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    f"sensitivity=%{{customdata[1]:.2f}} {sunit}<br>"
                    "latency=%{customdata[2]:.3f}s · phase=%{customdata[3]:.1f}<br>"
                    "distance=%{customdata[4]:.1f}cm · loomØ=%{customdata[5]:.1f}cm<br>"
                    "dθ/dt=%{customdata[6]:.1f} deg/s<extra></extra>"
                ),
            ))
    fig.update_layout(
        title=f"dθ/dt sensitivity — {yvar} vs {xvar} (bubble ∝ |sensitivity|, {sunit})",
        xaxis_title=f"{xvar} ({xunit})", yaxis_title=f"{yvar} ({yunit})",
        height=560, legend_title_text="condition · species",
    )
    return fig


def _reflect_interior(xy, box):
    """Augment points with their reflections across each wall of ``box`` (interior points only, plus
    the four corner double-reflections). The mirror trick: kernel mass that would leak through a wall
    is reflected back inside, so the density isn't biased low against the tank edges. Points already
    outside a wall are *not* reflected across it (they'd land spuriously deep inside)."""
    xmin, xmax, ymin, ymax = box
    parts = [xy]
    for axis, w, interior_ge in ((0, xmin, True), (0, xmax, False), (1, ymin, True), (1, ymax, False)):
        sel = (xy[axis] >= w) if interior_ge else (xy[axis] <= w)
        p = xy[:, sel].copy()
        p[axis] = 2.0 * w - p[axis]
        parts.append(p)
    for x_w, x_ge in ((xmin, True), (xmax, False)):
        for y_w, y_ge in ((ymin, True), (ymax, False)):
            sx = (xy[0] >= x_w) if x_ge else (xy[0] <= x_w)
            sy = (xy[1] >= y_w) if y_ge else (xy[1] <= y_w)
            p = xy[:, sx & sy].copy()
            p[0] = 2.0 * x_w - p[0]
            p[1] = 2.0 * y_w - p[1]
            parts.append(p)
    return np.hstack(parts)


def _kde_reflected(xy, box, XX, YY):
    """Boundary-corrected KDE density on grid (XX, YY): a manual Gaussian KDE (Scott's bandwidth from
    the *original* points) summed over the wall-reflected point set, so mass isn't lost at the tank
    edges. Returns the density field, or None if degenerate (too few / collinear points)."""
    n = xy.shape[1]
    if n < 4:
        return None
    cov = np.cov(xy)
    if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 1e-9:
        return None
    H = cov * n ** (-1.0 / 3.0)                          # Scott bandwidth² for d=2 (factor² = n^(-1/3))
    try:
        whiten = np.linalg.cholesky(np.linalg.inv(H)).T  # q = |whiten·(x−a)|²
    except np.linalg.LinAlgError:
        return None
    aug = whiten @ _reflect_interior(xy, box)
    grid = whiten @ np.vstack([XX.ravel(), YY.ravel()])
    z = np.zeros(grid.shape[1])
    for k in range(aug.shape[1]):                        # sum kernels (few hundred–few thousand cols)
        d = grid - aug[:, k:k + 1]
        z += np.exp(-0.5 * (d[0] * d[0] + d[1] * d[1]))
    return z.reshape(XX.shape)


def _kde_fill(ax, xy, box, XX, YY, cmap, levels=6, alpha=0.45):
    """Draw one group's boundary-corrected KDE as filled contours over the tank grid."""
    z = _kde_reflected(xy, box, XX, YY)
    if z is not None:
        ax.contourf(XX, YY, z, levels=levels, cmap=cmap, alpha=alpha, zorder=2)


def _legend_below(ax, title, y=-0.14):
    """Place the axis legend below the plot (outside the data area) so it occludes nothing."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=min(len(handles), 4), fontsize=8, framealpha=0.9, title=title)


def fish_overlay(fish_df, responders: bool = True, others: bool = True,
                 heat_responders: bool = False, heat_others: bool = False, heat_all: bool = False,
                 by_condition: bool = True, others_by_condition: bool = False,
                 tank_length: float = 44.0, kde_levels: int = 6):
    """Loom-centred spatial overlay of fish head positions (cm). The screen/loom is the thick line at
    depth 0; fish sit above it, inside the tank (59 cm wide × ``tank_length`` deep). KDE density
    heatmaps toggle independently for responders (reds), other fish (grays), and **all fish combined**
    (purples), boundary-corrected against the tank walls. ``by_condition`` marks first responders by
    stimulus — colour + shape per condition (circle ○ / fixed ■ / flapping ◆); ``others_by_condition``
    does the same for other fish (else uniform gray)."""
    resp = fish_df[fish_df["is_responder"]]
    other = fish_df[~fish_df["is_responder"]]
    half = pos_mod.HALF_SCREEN_CM
    fig, ax = plt.subplots(figsize=(7.0, 6.8), layout="constrained")

    if (heat_responders or heat_others or heat_all) and len(fish_df):
        box = (-half, half, 0.0, tank_length)                        # physical tank walls
        XX, YY = np.meshgrid(np.linspace(-half, half, 140), np.linspace(0.0, tank_length, 140))

        def xy(g):
            return np.vstack([g["head_along"].to_numpy(float), g["head_depth"].to_numpy(float)])
        if heat_all:
            _kde_fill(ax, xy(fish_df), box, XX, YY, "Purples", kde_levels)
        if heat_others:
            _kde_fill(ax, xy(other), box, XX, YY, "Greys", kde_levels)
        if heat_responders:
            _kde_fill(ax, xy(resp), box, XX, YY, "Reds", kde_levels)

    # tank walls: screen (thick) + side + far walls (faint context for the density)
    for xs, ys in (([-half, -half], [0, tank_length]), ([half, half], [0, tank_length]),
                   ([-half, half], [tank_length, tank_length])):
        ax.plot(xs, ys, color="0.6", lw=1.0, zorder=4)
    ax.plot([-half, half], [0, 0], color="k", lw=3, zorder=4)          # screen (tank-corner line)
    ax.plot(0, 0, marker="o", color="k", ms=6, zorder=5)              # loom origin
    ax.annotate("screen / loom origin", (0, 0), xytext=(0, -6), textcoords="offset points",
                ha="center", va="top", fontsize=8)

    def _by_cond(group, size, edge, edge_w, base_z, faded, prefix):
        for cond in dataset.CONDITIONS:
            g = group[group["condition"] == cond]
            if not len(g):
                continue
            color = _COND_COLORS[cond]
            ax.scatter(g["head_along"], g["head_depth"], s=size, marker=_COND_MARKERS[cond],
                       facecolor=("none" if faded else color), edgecolor=edge, linewidth=edge_w,
                       alpha=(0.55 if faded else 1.0), label=f"{prefix}{cond} (n={len(g)})",
                       zorder=base_z)

    if others:
        if others_by_condition:
            _by_cond(other, 22, "0.45", 0.6, 3, faded=True, prefix="other · ")
        else:
            ax.scatter(other["head_along"], other["head_depth"], s=18, facecolor="0.75",
                       edgecolor="0.5", linewidth=0.3, label=f"other fish (n={len(other)})", zorder=3)
    if responders:
        if by_condition:
            _by_cond(resp, 46, "k", 0.5, 6, faded=False, prefix="")
        else:
            ax.scatter(resp["head_along"], resp["head_depth"], s=34, facecolor="#D6283A",
                       edgecolor="k", linewidth=0.4, label=f"first responder (n={len(resp)})", zorder=6)

    ax.set_xlabel("along screen from loom centre (cm)")
    ax.set_ylabel("depth from screen (cm)")
    ax.set_aspect("equal")
    title = "first responders by condition" if by_condition else "first responders vs others"
    _legend_below(ax, title)
    fig.suptitle("Fish positions (loom-centred)")
    return fig


def _plot_fish_groups(ax, fish_df, xy, responders, others, by_condition, others_by_condition):
    """Scatter responders + other fish onto ``ax`` using ``xy(group) -> (x, y)`` for coordinates —
    shared by the cartesian overlay and the folded tank map so the marking stays consistent."""
    resp = fish_df[fish_df["is_responder"]]
    other = fish_df[~fish_df["is_responder"]]

    def scatter(g, **kw):
        x, y = xy(g)
        ax.scatter(x, y, **kw)

    def by_cond(group, size, edge, edge_w, faded, prefix, base_z):
        for cond in dataset.CONDITIONS:
            g = group[group["condition"] == cond]
            if not len(g):
                continue
            scatter(g, s=size, marker=_COND_MARKERS[cond],
                    facecolor=("none" if faded else _COND_COLORS[cond]), edgecolor=edge,
                    linewidth=edge_w, alpha=(0.55 if faded else 1.0),
                    label=f"{prefix}{cond} (n={len(g)})", zorder=base_z)

    if others:
        if others_by_condition:
            by_cond(other, 22, "0.45", 0.6, True, "other · ", 3)
        else:
            scatter(other, s=18, facecolor="0.75", edgecolor="0.5", linewidth=0.3,
                    label=f"other fish (n={len(other)})", zorder=3)
    if responders:
        if by_condition:
            by_cond(resp, 46, "k", 0.5, False, "", 6)
        else:
            scatter(resp, s=34, facecolor="#D6283A", edgecolor="k", linewidth=0.4,
                    label=f"first responder (n={len(resp)})", zorder=6)


def fish_tankmap(fish_df, responders: bool = True, others: bool = True, by_condition: bool = True,
                 others_by_condition: bool = False, heat_responders: bool = False,
                 heat_others: bool = False, heat_all: bool = False,
                 tank_length: float = 44.0, angles=(30, 45, 60), kde_levels: int = 6):
    """Custom folded **tank map**: the tank as its physical rectangle (half-width 29.5 cm × depth
    ``tank_length``), left/right mirrored into one half so the loom origin sits at the bottom-left
    corner (middle of the 59 cm screen). Radiating guide lines at ``angles`` (from the screen: 0° =
    along the screen, 90° = perpendicular) extend to the tank edge, over faint radial distance arcs;
    fish are drawn at their registered (|along|, depth) positions. KDE heatmaps (responders/others/all,
    reds/grays/purples) are estimated on the folded half-tank and boundary-corrected against its walls.

    Tank depth: shiner = 44 cm, sculpin = 44 or 30 cm per clip → set via ``tank_length``. Registered
    depths run slightly past the wall (~perspective), so a few points may sit above the far wall."""
    half = pos_mod.HALF_SCREEN_CM
    length = float(tank_length)
    resp = fish_df[fish_df["is_responder"]]
    other = fish_df[~fish_df["is_responder"]]
    fig, ax = plt.subplots(figsize=(7.2, 6.8), layout="constrained")

    if (heat_responders or heat_others or heat_all) and len(fish_df):
        box = (0.0, half, 0.0, length)                               # folded half-tank walls
        XX, YY = np.meshgrid(np.linspace(0.0, half, 120), np.linspace(0.0, length, 120))

        def xyf(g):
            return np.vstack([np.abs(g["head_along"].to_numpy(float)), g["head_depth"].to_numpy(float)])
        if heat_all:
            _kde_fill(ax, xyf(fish_df), box, XX, YY, "Purples", kde_levels)
        if heat_others:
            _kde_fill(ax, xyf(other), box, XX, YY, "Greys", kde_levels)
        if heat_responders:
            _kde_fill(ax, xyf(resp), box, XX, YY, "Reds", kde_levels)

    rmax = max(length, float(fish_df["head_depth"].max()) if len(fish_df) else length)
    for r in range(10, int(rmax) + 10, 10):                       # faint radial distance arcs
        th = np.linspace(0, np.pi / 2, 120)
        x, y = r * np.cos(th), r * np.sin(th)
        m = (x <= half) & (y <= length)
        if m.any():
            ax.plot(x[m], y[m], color="0.86", lw=0.7, zorder=1)
    for deg in angles:                                            # radiating angle rays to tank edge
        t = math.radians(deg)
        ex, ey = ((half, half * math.tan(t)) if half * math.tan(t) <= length
                  else (length / math.tan(t), length))
        ax.plot([0, ex], [0, ey], color="0.72", lw=0.8, zorder=1)
        ax.annotate(f"{deg}°", (ex, ey), xytext=(3, 3), textcoords="offset points",
                    fontsize=8, color="0.45")

    ax.plot([0, half, half], [length, length, 0], color="k", lw=1.5, zorder=4)   # far + side walls
    ax.plot([0, 0], [0, length], color="0.5", lw=1.0, ls=(0, (4, 3)), zorder=4)  # fold centreline
    ax.plot([0, half], [0, 0], color="k", lw=3, zorder=5)                        # screen (loom edge)
    ax.plot(0, 0, marker="o", color="k", ms=6, zorder=6)                         # loom origin
    ax.annotate("0° (along screen)", (half, 0), xytext=(-2, 5), textcoords="offset points",
                ha="right", fontsize=8, color="0.45")
    ax.annotate("90°", (0, length), xytext=(5, -1), textcoords="offset points", fontsize=8, color="0.45")

    _plot_fish_groups(ax, fish_df, lambda g: (np.abs(g["head_along"]), g["head_depth"]),
                      responders, others, by_condition, others_by_condition)

    ax.set_xlabel("|along| from loom centre (cm)")
    ax.set_ylabel("depth from screen (cm)")
    ax.set_aspect("equal")
    ax.set_xlim(-2, half + 3)
    ax.set_ylim(-3, rmax + 3)
    _legend_below(ax, "first responders by condition" if by_condition else "responders vs others")
    fig.suptitle(f"Fish positions — folded tank map ({2 * half:.0f}×{length:.0f} cm)")
    return fig


def fish_interactive(fish_df, tank_length: float = 44.0):
    """Interactive **Altair** fish-position map (loom-centred cm, un-folded so trial-mates sit at their
    true spots). Hovering a fish shows a rich **tooltip** (that fish + its trial), and — linked by
    ``clip_key`` — every fish from the *same clip* stays highlighted while the rest fade. Head→tail
    sticks show heading. Returns an ``alt.Chart`` (wrap in ``mo.ui.altair_chart`` to pin a details row).
    """
    import altair as alt
    import pandas as pd

    half = pos_mod.HALF_SCREEN_CM
    d = fish_df.copy()
    d["role"] = d["is_responder"].map({True: "first responder", False: "other fish"})
    ymax = (max(float(tank_length), float(d["head_depth"].max())) if len(d) else tank_length) + 3

    hover = alt.selection_point(fields=["clip_key"], on="pointerover", nearest=True)

    tip_spec = [("clip_key", "clip", None), ("role", "role", None), ("fish_index", "fish #", None),
                ("species", "species", None), ("condition", "stimulus", None),
                ("head_along", "along (cm)", ".1f"), ("head_depth", "depth (cm)", ".1f"),
                ("latency_s", "trial latency (s)", ".3f"), ("distance_cm", "responder dist (cm)", ".1f"),
                ("angle_deg", "retinal angle (deg)", ".1f"), ("dtheta_dt_deg_per_s", "dθ/dt (deg/s)", ".0f"),
                ("monitor_side", "monitor side", None)]
    tips = [alt.Tooltip(f"{f}:{'Q' if fmt else 'N'}", title=t, **({"format": fmt} if fmt else {}))
            for f, t, fmt in tip_spec if f in d.columns]

    shared = dict(
        x=alt.X("head_along:Q", title="along screen from loom centre (cm)",
                scale=alt.Scale(domain=[-half - 2, half + 2])),
        y=alt.Y("head_depth:Q", title="depth from screen (cm)", scale=alt.Scale(domain=[-3, ymax])),
    )
    sticks = alt.Chart(d).mark_rule(color="#888").encode(
        x2="tail_along:Q", y2="tail_depth:Q",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0.05)), **shared)
    pts = alt.Chart(d).mark_point(filled=True, size=90, stroke="black", strokeWidth=0.5).encode(
        color=alt.Color("condition:N", title="stimulus",
                        scale=alt.Scale(domain=list(dataset.CONDITIONS),
                                        range=[_COND_COLORS[c] for c in dataset.CONDITIONS])),
        shape=alt.Shape("role:N", title="role",
                        scale=alt.Scale(domain=["first responder", "other fish"],
                                        range=["triangle-up", "circle"])),
        opacity=alt.condition(hover, alt.value(0.95), alt.value(0.12)),
        tooltip=tips, **shared).add_params(hover)

    screen = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="black", size=2).encode(y="y:Q")
    sides = alt.Chart(pd.DataFrame({"x": [-half, half]})).mark_rule(color="#ccc").encode(x="x:Q")
    walls = sorted({float(v) for v in d.get("tank_depth_cm", pd.Series(dtype=float)).dropna()} or {tank_length})
    farwall = alt.Chart(pd.DataFrame({"y": walls})).mark_rule(color="#ccc", strokeDash=[4, 3]).encode(y="y:Q")

    return (screen + sides + farwall + sticks + pts).properties(
        width=580, height=470,
        title="Fish positions — hover a fish for its details + trial-mates")


def save_all(out_dir: Path = Path("out/analysis")) -> None:
    """Save, per response variable: a plain histogram, a best-fit-overlay histogram, and (across all
    variables) one AICc comparison table CSV (script/headless mode)."""
    df = dataset.load_dataframe()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for label, (col, _) in dataset.RESPONSE_VARS.items():
        name = col.replace("_deg_per_s", "").replace("_", "-")
        for suffix, kwargs in (("", {}), ("_fit", {"fit": "auto"})):
            fig = response_histograms(df, label, **kwargs)
            path = out_dir / f"hist_{name}{suffix}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {path}")
        t = fits.fit_table(df, label)
        t.insert(0, "variable", label)
        tables.append(t)
    import pandas as pd

    csv_path = out_dir / "distribution_fits.csv"
    pd.concat(tables, ignore_index=True).to_csv(csv_path, index=False)
    print(f"saved {csv_path}")

    sdf = sens_mod.sensitivity_frame(df)
    for xv in ("latency", "distance"):                  # phase-driven vs the (weak) distance view
        fig = sensitivity_scatter(sdf, xvar=xv)
        path = out_dir / f"dtheta_sensitivity_vs_{xv}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")

    fig = sensitivity_bubble(sdf, xvar="loom phase", yvar="distance")
    path = out_dir / "dtheta_sensitivity_bubble.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")

    fdf = pos_mod.fish_frame(df)
    for suffix, kwargs in (("", {}), ("_heat", {"heat_responders": True, "heat_others": True})):
        fig = fish_overlay(fdf, **kwargs)
        path = out_dir / f"fish_positions{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")
    fig = fish_tankmap(fdf)
    path = out_dir / "fish_positions_tankmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    save_all()
