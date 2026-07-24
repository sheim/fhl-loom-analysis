import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import marimo as mo

    import dataset
    import plots
    import positions

    return dataset, mo, plots, positions


@app.cell
def _(mo):
    mo.md(
        """
        # Fish positions (loom-centred)

        Every fish's head mapped into a **loom-centred, screen-aligned** frame (cm): *x* = along the
        screen from the loom origin, *y* = depth from the screen (fish sit above the black screen line).
        First responders (red) vs other fish (gray), with optional density heatmaps toggled per group.

        The along-axis **sign is arbitrary per clip** (the two tank corners have no canonical order),
        so read it as left–right symmetric — only depth and distance-from-centre are meaningful.
        """
    )
    return


@app.cell
def _(dataset, positions):
    fish = positions.fish_frame(dataset.load_dataframe())
    return (fish,)


@app.cell
def _(dataset, mo):
    species = mo.ui.multiselect(dataset.SPECIES, value=list(dataset.SPECIES), label="Species")
    conds = mo.ui.multiselect(dataset.CONDITIONS, value=list(dataset.CONDITIONS), label="Conditions")
    show_resp = mo.ui.switch(value=True, label="Responders (points)")
    show_other = mo.ui.switch(value=True, label="Other fish (points)")
    by_cond = mo.ui.switch(value=True, label="Responders by condition")
    other_by_cond = mo.ui.switch(value=False, label="Others by condition")
    heat_resp = mo.ui.switch(value=False, label="Responder heatmap")
    heat_other = mo.ui.switch(value=False, label="Other-fish heatmap")
    heat_all = mo.ui.switch(value=False, label="All-fish heatmap")
    tank_len = mo.ui.slider(28, 52, value=44, step=1, label="Tank depth (cm)")
    mo.vstack([
        mo.hstack([species, conds, tank_len], justify="start"),
        mo.hstack([show_resp, show_other, by_cond, other_by_cond], justify="start"),
        mo.hstack([heat_resp, heat_other, heat_all], justify="start"),
    ])
    return (by_cond, conds, heat_all, heat_other, heat_resp, other_by_cond,
            show_other, show_resp, species, tank_len)


@app.cell
def _(conds, fish, species):
    fsel = fish[fish["species"].isin(species.value) & fish["condition"].isin(conds.value)]
    return (fsel,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Interactive map

        **Hover a fish** for its full details in the tooltip (that fish + its trial) — and every fish
        from the *same clip* stays highlighted while the rest fade. Head→tail sticks show heading.
        (Species/condition filters above apply.)
        """
    )
    return


@app.cell
def _(fsel, plots):
    plots.fish_interactive(fsel)
    return


@app.cell
def _(by_cond, fsel, heat_all, heat_other, heat_resp, other_by_cond, plots, show_other,
      show_resp, tank_len):
    plots.fish_overlay(
        fsel,
        responders=show_resp.value, others=show_other.value,
        by_condition=by_cond.value, others_by_condition=other_by_cond.value,
        heat_responders=heat_resp.value, heat_others=heat_other.value, heat_all=heat_all.value,
        tank_length=tank_len.value,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Folded tank map

        The tank drawn as its physical rectangle (59 cm wide × tank depth), left/right mirrored into one
        half with the loom origin at the bottom-left corner. Radiating lines (0° = along the screen, 90°
        = perpendicular) reach the tank edge; faint arcs mark distance from the loom. Set the tank depth:
        **shiner = 44 cm, sculpin = 44 or 30 cm**. A few points sit past the far wall — registered depths
        run ~10–15 % long (perspective), for the later correction.
        """
    )
    return


@app.cell
def _(by_cond, fsel, heat_all, heat_other, heat_resp, other_by_cond, plots, show_other,
      show_resp, tank_len):
    plots.fish_tankmap(
        fsel,
        responders=show_resp.value, others=show_other.value,
        by_condition=by_cond.value, others_by_condition=other_by_cond.value,
        heat_responders=heat_resp.value, heat_others=heat_other.value, heat_all=heat_all.value,
        tank_length=tank_len.value,
    )
    return


@app.cell
def _(fsel, mo):
    summary = (
        fsel.assign(group=fsel["is_responder"].map({True: "first responder", False: "other fish"}))
        .groupby("group")["head_depth"].agg(["count", "mean", "median", "std"]).round(1)
    )
    mo.vstack([mo.md("**Depth from screen (cm), by group:**"), summary])
    return


if __name__ == "__main__":
    app.run()
