"""Plot the accuracy decline along visual-semantic axes built from self-supervised models and image retrieval."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Colors and line styles of the axes; the original consensus is always the dashed grey reference.
AXIS_STYLES = {
    "DINOv1 probe": {"color": "#264653"},
    "DINOv2 probe": {"color": "#E76F51"},
    "MAE probe": {"color": "#2A9D8F"},
    "Probe consensus": {"color": "#E9C46A", "linewidth": 2.4},
    "Retrieval, best match": {"color": "#6D597A", "linestyle": "-."},
    "Retrieval, top 5": {"color": "#B56576", "linestyle": ":"},
}
REFERENCE_STYLE = {"color": "#9AA5AE", "linestyle": "--", "linewidth": 1.4}

# Width and height of one panel in the overview figures, so that both figures have panels of the same size.
PANEL_SIZE = (3.7, 3.3)

# Colors of the single classifier heads, the CLIP text and WordNet curves, and CLIP on its own.
HEAD_COLOR = "#5B8DBE"
OTHER_COLOR = "#8C6D9E"
CLIP_COLOR = "#D4642C"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def draw_axis_curves(ax, curves, reference, title, legend_loc="lower left"):
    """Draw the mean accuracy curve of every axis, with the original consensus as a dashed reference."""
    # Draw the original axis first so the alternative axes remain visible on top.
    ax.plot(reference.index, reference.values, label="Original consensus", zorder=2, **REFERENCE_STYLE)
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, label=name, zorder=3, **{"linewidth": 1.6, **AXIS_STYLES[name]})
    # Use the same rank range for every axis and display accuracy as a percentage.
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_xlim(-10, 1010)
    ax.set_xlabel("Dissimilarity rank under the axis")
    ax.set_ylabel("Top-1 accuracy (mean over models)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend(loc=legend_loc, frameon=False)


def plot_probe_axes(probe_curves, reference):
    """
    Plot the mean accuracy along the probe axes of the self-supervised models.

    Parameters
    ----------
    probe_curves : dict
        Axis name -> mean top-1 accuracy per rank bin (pandas.Series indexed by bin center).
    reference : pandas.Series
        Mean top-1 accuracy per rank bin under the original consensus.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One curve per axis, with the original consensus as a dashed reference.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.6), layout="constrained")
        draw_axis_curves(ax, probe_curves, reference, "Self-supervised probe axes")
    return fig


def set_rank_axis(ax, title):
    """Rank ticks, axis labels and title of a panel with dissimilarity ranks on the x-axis."""
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_xlim(-10, 1010)
    ax.set_xlabel("Dissimilarity rank under the axis")
    ax.set_title(title, loc="left")


def format_overview_panels(axes, ylim):
    """Apply shared y-limits and percentage ticks, with a y-label on the first panel."""
    for i, ax in enumerate(axes):
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("Top-1 accuracy" if i == 0 else "")


def plot_visual_axes(head_curves, probe_curves, retrieval_curves, reference, ylim=(0.44, 0.70)):
    """
    Plot accuracy along the axes built from visual features: single classifier heads, probes and image retrieval.

    Parameters
    ----------
    head_curves : dict
        Head name -> mean top-1 accuracy per rank bin (pandas.Series indexed by bin center), one per single classifier head.
    probe_curves : dict
        Axis name -> mean top-1 accuracy per rank bin for the probe axes and the probe consensus.
    retrieval_curves : dict
        Same for the DINOv2 probe and the retrieval axes built from the same features.
    reference : pandas.Series
        Mean top-1 accuracy per rank bin under the original consensus, drawn as a dashed line.
    ylim : tuple
        Shared y-axis limits.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) single classifier heads, (b) self-supervised probes, (c) DINOv2 probe vs retrieval.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(3 * PANEL_SIZE[0], PANEL_SIZE[1]), layout="constrained")

        # Show the individual heads together, using one legend entry for the whole group.
        ax = axes[0]
        ax.plot(reference.index, reference.values, label="Original consensus", zorder=3, **REFERENCE_STYLE)
        for i, curve in enumerate(head_curves.values()):
            ax.plot(curve.index, curve.values, color=HEAD_COLOR, linewidth=0.9, alpha=0.6, label=f"Single heads ({len(head_curves)})" if i == 0 else None, zorder=2)
        set_rank_axis(ax, "(a) Classifier heads")
        ax.legend(loc="lower left", frameon=False)
        # Compare probes and retrieval against the same original consensus.
        draw_axis_curves(axes[1], probe_curves, reference, "(b) Self-supervised probes")
        draw_axis_curves(axes[2], retrieval_curves, reference, "(c) DINOv2 probe vs retrieval")

        format_overview_panels(axes, ylim)
    return fig


def draw_model_bundle(ax, curves, clip_model, marker=None):
    """Draw every model as a thin line, the mean over models in bold, and the CLIP model in orange."""
    # Keep individual models faint so the overall pattern is easier to see.
    for i, model in enumerate(curves.columns.drop(clip_model)):
        ax.plot(curves.index, curves[model], color=OTHER_COLOR, linewidth=0.7, alpha=0.25, label="Single models" if i == 0 else None, zorder=2)
    # The mean includes CLIP as well; draw CLIP separately to show its own behaviour.
    mean = curves.mean(axis=1)
    ax.plot(mean.index, mean.values, color=OTHER_COLOR, linewidth=2.2, marker=marker, markersize=3.5, label="Mean over models", zorder=4)
    ax.plot(curves.index, curves[clip_model], color=CLIP_COLOR, linewidth=1.6, marker=marker, markersize=3, label="CLIP ViT-B/16", zorder=3)


def plot_other_axes(clip_text_curves, wordnet_curves, reference, clip_model="CLIP-ViT-B-16", ylim=(0.20, 0.85)):
    """
    Plot accuracy along class similarities that do not come from visual features: CLIP text and WordNet.

    Parameters
    ----------
    clip_text_curves : pandas.DataFrame
        Top-1 accuracy per rank bin (indexed by bin center) under the CLIP text axis, one column per model.
    wordnet_curves : pandas.DataFrame
        Top-1 accuracy per WordNet hop count, one column per model.
    reference : pandas.Series
        Mean top-1 accuracy per rank bin under the original consensus, drawn as a dashed line in the CLIP text panel.
    clip_model : str
        Name of the CLIP model, drawn in orange.
    ylim : tuple
        Shared y-axis limits; wide enough for the single models and CLIP's own, much lower accuracy.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) CLIP text, (b) WordNet; each with every model as a thin line, the mean in bold and CLIP in orange.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_SIZE[0], PANEL_SIZE[1]), layout="constrained")

        # CLIP text uses ranks, so we can also show the original consensus here.
        ax = axes[0]
        ax.plot(reference.index, reference.values, label="Original consensus", zorder=5, **REFERENCE_STYLE)
        draw_model_bundle(ax, clip_text_curves, clip_model)
        set_rank_axis(ax, "(a) CLIP text")

        # WordNet uses discrete hop counts rather than ranks; mark each available count.
        ax = axes[1]
        draw_model_bundle(ax, wordnet_curves, clip_model, marker="o")
        ax.set_xlabel("WordNet hop count")
        ax.set_title("(b) WordNet", loc="left")

        format_overview_panels(axes, ylim)
        axes[0].legend(loc="lower left", frameon=False, ncol=2)
    return fig
