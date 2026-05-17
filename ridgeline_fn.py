"""Ridgeline plot as a reusable function over a long-format DataFrame."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


def ridgeline(
    df,
    row_id,
    group,
    value,
    *,
    ax=None,
    palette=None,
    row_order=None,
    group_order=None,
    bw=0.3,
    overlap=1.8,
    row_height=1.0,
    n_grid=600,
    xlim=None,
    xlabel=None,
    figsize=(6, 11),
    legend=True,
    legend_kwargs=None,
):
    """Ridgeline plot with one or more overlapping KDEs per row.

    Parameters
    ----------
    df : DataFrame
        Long-format data — one row per observation.
    row_id : str
        Column whose unique values become the rows of the plot.
    group : str
        Column whose unique values become the overlapping distributions
        within each row.
    value : str
        Column holding the numeric values whose density is plotted.
    ax : matplotlib Axes, optional
    palette : dict | list, optional
        Mapping of group → color, or a list aligned with `group_order`.
        Defaults to blue/orange for 2 groups, else `tab10`.
    row_order, group_order : list, optional
        Explicit ordering. Rows are drawn top-to-bottom in this order.
    bw : float
        Bandwidth for `gaussian_kde`.
    overlap : float
        Ridge height as a multiple of `row_height`. >1 makes ridges overlap.
    row_height : float
        Vertical spacing between row baselines.
    n_grid : int
        Number of x points for KDE evaluation.
    xlim : (low, high), optional
        Defaults to data range plus 10% padding.
    xlabel : str, optional
        Defaults to `value`.
    legend : bool
        Whether to draw a legend mapping colors to `group` values.
    legend_kwargs : dict, optional
        Extra keyword arguments forwarded to `ax.legend()`.

    Returns
    -------
    ax : matplotlib Axes
    """
    # ---- order ----
    if row_order is None:
        row_order = df[row_id].drop_duplicates().tolist()
    if group_order is None:
        group_order = df[group].drop_duplicates().tolist()

    # ---- palette ----
    default_two = ["#7896a8", "#e6b97a"]
    if palette is None:
        if len(group_order) == 2:
            palette = dict(zip(group_order, default_two))
        else:
            cmap = plt.get_cmap("tab10")
            palette = {g: cmap(i % 10) for i, g in enumerate(group_order)}
    elif isinstance(palette, list):
        palette = dict(zip(group_order, palette))

    # ---- x grid ----
    if xlim is None:
        lo, hi = df[value].min(), df[value].max()
        pad = 0.1 * (hi - lo)
        xlim = (lo - pad, hi + pad)
    x_grid = np.linspace(xlim[0], xlim[1], n_grid)

    # ---- pre-compute KDEs (so we can normalize heights globally) ----
    kdes, global_max = {}, 0.0
    for r in row_order:
        for g in group_order:
            vals = df.loc[(df[row_id] == r) & (df[group] == g), value].to_numpy()
            if len(vals) < 2:
                kdes[(r, g)] = None
                continue
            y = gaussian_kde(vals, bw_method=bw)(x_grid)
            kdes[(r, g)] = y
            global_max = max(global_max, y.max())
    scale = overlap / global_max if global_max > 0 else 1.0

    # ---- draw ----
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n = len(row_order)
    for i, r in enumerate(row_order):
        y0 = (n - 1 - i) * row_height                    # top row first
        ax.hlines(y0, xlim[0], xlim[1], color="black", lw=0.6)
        for g in group_order:
            y = kdes[(r, g)]
            if y is None:
                continue
            ax.fill_between(x_grid, y0, y0 + y * scale,
                            color=palette[g], alpha=0.85, lw=0.6, ec="black")
        ax.text(xlim[0] - 0.03 * (xlim[1] - xlim[0]), y0 + 0.1,
                str(r), ha="right", va="bottom", fontsize=9)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.5, n * row_height + overlap)
    ax.set_yticks([])
    ax.set_xlabel(xlabel if xlabel is not None else value)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    # ---- legend ----
    if legend:
        handles = [Patch(facecolor=palette[g], edgecolor="black",
                         linewidth=0.6, alpha=0.85, label=str(g))
                   for g in group_order]
        kw = dict(title=group, loc="upper right", frameon=True,
                  facecolor="white", edgecolor="none", framealpha=1.0,
                  bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(handles=handles, **kw)

    return ax


# ---------- demo ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    targets = ["GHSR", "HRH3", "PPARδ", "FXR", "D4R", "Thrombin", "D3R",
               "ESR1", "MOR", "TP53", "DAT", "5-HT1A", "SERT", "GSK3",
               "AR", "GR", "JAK2", "PIK3CA", "JAK1"]

    rows = []
    for i, t in enumerate(targets):
        shift = i * 0.02
        for v in rng.normal(0.55 - shift, 0.25, 400):
            rows.append({"target": t, "method": "A", "score": v})
        for v in rng.normal(0.95, 0.20, 400):
            rows.append({"target": t, "method": "B", "score": v})
    df = pd.DataFrame(rows)

    ax = ridgeline(df, row_id="target", group="method", value="score",
                   xlabel=r"$\mathbb{U}(x)$")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/ridgeline_fn.png", dpi=160, bbox_inches="tight")
    print("saved")
