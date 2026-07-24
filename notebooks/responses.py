import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell
def _():
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import marimo as mo

    import dataset
    import fits
    import plots
    import sensitivity

    return dataset, fits, mo, plots, sensitivity


@app.cell
def _(mo):
    mo.md("""
    # Response distributions

    Histograms of the escape-response variables, split by **species** (sculpin, shiner) and
    overlaid by **stimulus type** (circle / fixed / flapping). Pick the variable and toggle
    conditions below.
    """)
    return


@app.cell
def _(dataset, mo):
    df = dataset.load_dataframe()
    var = mo.ui.dropdown(list(dataset.RESPONSE_VARS), value="latency", label="Variable")
    conds = mo.ui.multiselect(
        dataset.CONDITIONS, value=list(dataset.CONDITIONS), label="Conditions"
    )
    nbins = mo.ui.slider(5, 25, value=12, label="Bins")
    logx = mo.ui.switch(label="Log x")
    fitsel = mo.ui.dropdown(
        ["none", "auto", "normal", "lognormal", "gamma", "weibull"],
        value="none", label="Fit",
    )
    pool = mo.ui.switch(label="Pool conditions (table)")
    # Reversible outlier exclusion: options sorted by robust |z| so candidates surface at the top
    # (⚠ = |z|≥3.5). Excluding filters the plots + fits but never touches the dataset on disk.
    _cands = dataset.outlier_scores(df)
    excl_opts = {
        f"{r.clip_key}  ({r.variable} z={r.z:+.1f})" + ("  ⚠" if r.outlier else ""): r.clip_key
        for r in _cands.itertuples()
    }
    excluded = mo.ui.multiselect(excl_opts, value=[], label="Exclude clips")
    mo.vstack([
        mo.hstack([var, conds, nbins, logx, fitsel, pool], justify="start"),
        excluded,
    ])
    return conds, df, excluded, fitsel, logx, nbins, pool, var


@app.cell
def _(dataset, df, excluded, mo):
    df_used = dataset.exclude(df, excluded.value)
    mo.md(
        f"Using **{len(df_used)} / {len(df)}** clips"
        + (f" — excluded: `{', '.join(excluded.value)}`" if excluded.value else " (none excluded).")
    )
    return (df_used,)


@app.cell
def _(conds, df_used, fitsel, logx, nbins, plots, var):
    fig = plots.response_histograms(
        df_used, var.value, conditions=conds.value, bins=nbins.value,
        log=logx.value,
        fit=None if fitsel.value == "none" else fitsel.value,
    )
    fig
    return


@app.cell
def _(df_used, fits, mo, pool, var):
    mo.vstack([
        mo.md(
            f"**Distribution fit — {var.value}** (ranked by AICc; `normal_minus_best` > 0 ⇒ normal "
            "is worse than the winner by that many AICc units — >2 meaningful, >10 decisive):"
        ),
        fits.fit_table(df_used, var.value, pool=pool.value),
    ])
    return


@app.cell
def _(dataset, mo):
    outlier_var = mo.ui.dropdown(
        ["all"] + list(dataset.RESPONSE_VARS), value="all", label="Outlier variable"
    )
    return (outlier_var,)


@app.cell
def _(dataset, df, mo, outlier_var):
    _v = None if outlier_var.value == "all" else outlier_var.value
    mo.vstack([
        mo.md(
            "**Outlier candidates** — robust modified z-score within each species×condition group. "
            "`⚠`/`outlier=True` marks |z|≥3.5. Pick a variable to score every clip on it; `all` keeps "
            "each clip's single most-extreme variable. Candidates need judgement (some extremes are "
            "real, e.g. near-contact dθ/dt)."
        ),
        outlier_var,
        dataset.outlier_scores(df, var_label=_v),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## dθ/dt timing sensitivity

        How much the estimated dθ/dt shifts if the movement-onset frame is off by one camera frame —
        large near contact, where θ(t) is steep. The marginal histogram shows where the
        first-responders actually sit. Sensitivity tracks **loom phase / latency**, not distance (try
        the x-axis). **Hover a point** to see its `clip_key` (+ latency, phase, distance, dθ/dt).
        """
    )
    return


@app.cell
def _(mo, sensitivity):
    sxvar = mo.ui.dropdown(list(sensitivity.XVARS), value="latency", label="x-axis")
    srel = mo.ui.switch(value=True, label="Relative (%/frame)")
    mo.hstack([sxvar, srel], justify="start")
    return srel, sxvar


@app.cell
def _(df_used, plots, srel, sxvar):
    plots.sensitivity_scatter_interactive(df_used, xvar=sxvar.value, rel=srel.value)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### 2D bubble view

        Both axes are sensitivity variables; **bubble area ∝ |dθ/dt sensitivity|**. Open markers
        (border = condition, shape = species). Raise the threshold to keep only the most
        timing-sensitive clips (0 = all). Hover for `clip_key`.
        """
    )
    return


@app.cell
def _(mo, sensitivity):
    bx = mo.ui.dropdown(list(sensitivity.XVARS), value="loom phase", label="x-axis")
    by = mo.ui.dropdown(list(sensitivity.XVARS), value="distance", label="y-axis")
    bthr = mo.ui.slider(0.0, 35.0, value=0.0, step=0.5, label="|sensitivity| ≥ (%/frame)")
    mo.hstack([bx, by, bthr], justify="start")
    return bthr, bx, by


@app.cell
def _(bthr, bx, by, df_used, plots):
    plots.sensitivity_bubble_interactive(df_used, xvar=bx.value, yvar=by.value, threshold=bthr.value)
    return


if __name__ == "__main__":
    app.run()
