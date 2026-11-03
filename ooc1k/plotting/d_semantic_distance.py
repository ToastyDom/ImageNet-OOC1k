"""Plot recognition accuracy as a function of semantic distance."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ooc1k.plotting.c_controls import FAMILY_COLORS, MODEL_FAMILIES, get_family

# Model names and colors used in the accuracy plots.
MODEL_COLORS = {
    "DeiT-B-16": "#1f77b4",
    "ViT-L-16": "#ff7f0e",
    "ViT-B-16": "#2ca02c",
    "EfficientNet-B3": "#d62728",
    "ResNet-50": "#9467bd",
    "CLIP-ViT-B-16": "#8c564b",
}
MODEL_LABELS = {
    "DeiT-B-16": "DeiT-B/16",
    "ViT-L-16": "ViT-L/16",
    "ViT-B-16": "ViT-B/16",
    "EfficientNet-B3": "EfficientNet-B3",
    "ResNet-50": "ResNet-50",
    "CLIP-ViT-B-16": "CLIP ViT-B/16",
}

# Colors of the trend groups, background categories and alternative semantic axes.
TREND_COLORS = {
    "Strong −": "#C0392B",
    "Mild −": "#E67E22",
    "Mild +": "#5DADE2",
    "Strong +": "#2471A3",
}
CATEGORY_COLORS = {"nature": "#2D6A4F", "human_related": "#457B9D", "misc": "#9E9E9E"}
CATEGORY_LABELS = {"nature": "Nature", "human_related": "Human-related", "misc": "Misc"}
SUPERVISED_COLOR = "#2D5F8A"
SUPERVISED_MEMBER_COLOR = "#A8C8E8"
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

ACCURACY_COLUMNS = {"correct": "Top-1 accuracy", "correct_top5": "Top-5 accuracy"}


def draw_grid(ax, ylim, step=0.05):
    """Draw light horizontal grid lines."""
    for y in np.arange(np.ceil(ylim[0] / step) * step, ylim[1] + 1e-9, step):
        ax.axhline(y, color="#E0E0E0", linewidth=0.4, zorder=0)


def set_rank_axis(ax, x_min=-10):
    """Show rank ticks at 0, 200, ..., 1000."""
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_xlim(x_min, 1010)


def draw_model_accuracy(ax, curves, original_accuracy, models, column, dash_center):
    """Draw each model's accuracy curve and its original-image reference."""
    for model in models:
        curve = curves[curves["model"] == model].sort_values("bin_center")
        color = MODEL_COLORS.get(model, "#666666")
        ax.plot(
            curve["bin_center"],
            curve[column],
            color=color,
            linewidth=1.6,
            label=MODEL_LABELS.get(model, model),
            zorder=3,
        )
        ax.plot(
            [dash_center - 15, dash_center + 15],
            [original_accuracy.loc[model, column]] * 2,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=5,
        )


def plot_accuracy_vs_distance(curves, original_accuracy, ylims=((0.25, 0.90), (0.50, 0.98))):
    """
    Plot top-1 and top-5 accuracy against semantic distance rank for a few models.

    Parameters
    ----------
    curves : pandas.DataFrame
        Output of `get_accuracy_curves`.
    original_accuracy : pandas.DataFrame
        Output of `get_original_accuracy`, drawn as short dashes left of the curves.
    ylims : tuple of tuple
        Y-axis limits of the top-1 and top-5 panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Two panels (top-1 above, top-5 below) with one curve per model.
    """
    models = list(curves["model"].unique())
    bin_width = np.diff(np.sort(curves["bin_center"].unique()))[0]
    x_first = curves["bin_center"].min()

    # The original accuracies are drawn as short dashes left of the first bin.
    separator = x_first - 0.6 * bin_width
    dash_center = separator - 20

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(3.6, 6.5), gridspec_kw={"hspace": 0.12})
        for ax, (column, ylabel), ylim in zip(axes, ACCURACY_COLUMNS.items(), ylims):
            draw_grid(ax, ylim)
            draw_model_accuracy(ax, curves, original_accuracy, models, column, dash_center)

            # Separate the original accuracies from the curves.
            ax.axvline(separator, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=1)
            ax.text(
                separator,
                ylim[1] + 0.02 * (ylim[1] - ylim[0]),
                "Orig.",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#666666",
            )
            ax.set_ylim(*ylim)
            set_rank_axis(ax, x_min=dash_center - 30)
            ax.set_ylabel(ylabel)

        axes[0].set_xticklabels([])
        axes[1].set_xlabel("Semantic dissimilarity rank")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.06), frameon=False
        )
        fig.subplots_adjust(bottom=0.14)
    return fig


def draw_family_curves(ax, curves, families, column):
    """Draw individual models faintly and each family mean in bold."""
    for family in MODEL_FAMILIES:
        if family not in families:
            continue
        family_curves = curves[curves["model"].isin(families[family])]
        for _, curve in family_curves.groupby("model", observed=True):
            ax.plot(
                curve["bin_center"],
                curve[column],
                color=FAMILY_COLORS[family],
                linewidth=0.7,
                alpha=0.15,
                zorder=2,
            )
        mean_curve = family_curves.groupby("bin_center")[column].mean()
        ax.plot(
            mean_curve.index,
            mean_curve.values,
            color=FAMILY_COLORS[family],
            linewidth=2.2,
            label=f"{family} (n={len(families[family])})",
            zorder=4,
        )


def plot_accuracy_vs_distance_all_models(curves, ylims=((0.25, 0.80), (0.50, 1.00))):
    """
    Plot top-1 and top-5 accuracy against semantic distance rank for many models, grouped by family.

    Parameters
    ----------
    curves : pandas.DataFrame
        Output of `get_accuracy_curves`.
    ylims : tuple of tuple
        Y-axis limits of the top-1 and top-5 panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Two panels with a thin line per model and a bold line per family mean.
    """
    families = {}
    for model in curves["model"].unique():
        families.setdefault(get_family(model), []).append(model)

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), gridspec_kw={"wspace": 0.25})
        for ax, (column, title), ylim in zip(axes, ACCURACY_COLUMNS.items(), ylims):
            draw_grid(ax, ylim)

            # Draw every model faintly and the mean of every family in bold.
            draw_family_curves(ax, curves, families, column)

            ax.set_ylim(*ylim)
            set_rank_axis(ax)
            ax.set_xlabel("Semantic dissimilarity rank")
            ax.set_title(title, loc="left")
        axes[0].set_ylabel("Accuracy")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.subplots_adjust(bottom=0.30)
    return fig


def draw_trend_curves(ax, group_curves, trends, groups):
    """Draw the trend groups with SEM bands and the all-background reference."""
    reference = group_curves[group_curves["group"] == "All backgrounds"]
    ax.plot(
        reference["bin_center"],
        reference["accuracy"],
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.45,
        label="All backgrounds",
        zorder=2,
    )
    for group in groups:
        curve = group_curves[group_curves["group"] == group]
        color = TREND_COLORS[group]
        ax.plot(
            curve["bin_center"],
            curve["accuracy"],
            color=color,
            linewidth=1.8,
            label=f"{group} (n={(trends['group'] == group).sum()})",
            zorder=3,
        )
        ax.fill_between(
            curve["bin_center"],
            curve["accuracy"] - curve["sem"],
            curve["accuracy"] + curve["sem"],
            color=color,
            alpha=0.12,
            zorder=2,
        )


def draw_background_categories(ax, trends, groups):
    """Show the scene-category counts within each trend group."""
    counts = (
        trends.groupby(["group", "background_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(groups)
    )
    bottom = np.zeros(len(groups))
    for category, label in CATEGORY_LABELS.items():
        values = counts[category].values if category in counts else np.zeros(len(groups))
        ax.bar(
            range(len(groups)),
            values,
            bottom=bottom,
            color=CATEGORY_COLORS[category],
            alpha=0.8,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
        for i, (value, base) in enumerate(zip(values, bottom)):
            if value >= 8:
                ax.text(
                    i,
                    base + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="white",
                )
            elif value > 0:
                # Small counts do not fit inside their segment: above the bar if it is the top segment, else next to it.
                on_top = base + value == counts.loc[groups[i]].sum()
                ax.text(
                    i if on_top else i + 0.45,
                    base + value + 1 if on_top else base + value / 2,
                    str(int(value)),
                    ha="center" if on_top else "left",
                    va="bottom" if on_top else "center",
                    fontsize=7,
                    fontweight="bold",
                    color=CATEGORY_COLORS[category],
                )
        bottom += values
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Number of backgrounds")
    ax.set_title("Background categories", loc="left")
    ax.legend(loc="upper right", frameon=False)


def plot_background_trends(group_curves, trends, ylim=(0.35, 0.80)):
    """
    Plot accuracy against semantic distance for groups of backgrounds with similar trends.

    Parameters
    ----------
    group_curves : pandas.DataFrame
        Output of `get_trend_group_curves`.
    trends : pandas.DataFrame
        Output of `get_background_trends`.
    ylim : tuple
        Y-axis limits of the accuracy panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Accuracy curves per trend group (left) and the background categories in every group (right).
    """
    groups = [group for group in TREND_COLORS if (trends["group"] == group).any()]

    with mpl.rc_context(STYLE):
        fig, (ax_curves, ax_bars) = plt.subplots(
            1, 2, figsize=(7, 3.5), gridspec_kw={"width_ratios": [1.25, 1]}
        )

        # Accuracy per group with its standard error across models, and all backgrounds as reference.
        draw_grid(ax_curves, ylim)
        draw_trend_curves(ax_curves, group_curves, trends, groups)

        ax_curves.set_ylim(*ylim)
        set_rank_axis(ax_curves)
        ax_curves.set_xlabel("Semantic dissimilarity rank")
        ax_curves.set_ylabel("Top-1 accuracy")
        ax_curves.set_title("Accuracy by trend group", loc="left")
        ax_curves.legend(loc="upper right", frameon=False)

        # Stacked bars of the background categories in every group.
        draw_background_categories(ax_bars, trends, groups)

        fig.tight_layout()
    return fig


def draw_model_bundle(ax, curves, supervised_models, clip_model, markers=False):
    """Draw thin supervised-model curves, their mean with one standard deviation, and CLIP."""
    marker = {"marker": "o", "markersize": 3} if markers else {}
    supervised = curves[[model for model in supervised_models if model in curves]]
    for model in supervised:
        ax.plot(
            supervised.index,
            supervised[model],
            color=SUPERVISED_MEMBER_COLOR,
            linewidth=0.7,
            alpha=0.45,
            zorder=1,
        )

    mean, std = supervised.mean(axis=1), supervised.std(axis=1)
    ax.fill_between(
        mean.index, mean - std, mean + std, color=SUPERVISED_COLOR, alpha=0.12, zorder=2
    )
    ax.plot(
        mean.index,
        mean,
        color=SUPERVISED_COLOR,
        linewidth=2.2,
        label=f"Supervised (n={supervised.shape[1]})",
        zorder=3,
        **marker,
    )
    if clip_model in curves:
        ax.plot(
            curves.index,
            curves[clip_model],
            color=CLIP_COLOR,
            linewidth=2.2,
            label=MODEL_LABELS.get(clip_model, clip_model),
            zorder=3,
            **marker,
        )
    ax.legend(loc="lower left", frameon=False)


def draw_head_curves(ax, head_curves):
    """Compare individual classifier-head axes with the consensus."""
    for name, curve in head_curves.items():
        if name == "Consensus":
            continue
        ax.plot(
            curve.index,
            curve.values,
            color=MODEL_COLORS.get(name, "#888888"),
            linewidth=1.6,
            alpha=0.85,
            label=MODEL_LABELS.get(name, name),
            zorder=2,
        )
    consensus = head_curves["Consensus"]
    ax.plot(
        consensus.index,
        consensus.values,
        color="#1A1A1A",
        linewidth=2.8,
        label="Consensus",
        zorder=3,
    )
    ax.legend(loc="lower left", frameon=False, title="Dissimilarity matrix", title_fontsize=8)


def plot_alternative_axes(
    head_curves,
    clip_curves,
    wordnet_curves,
    supervised_models,
    clip_model="CLIP-ViT-B-16",
    ylim=(0.20, 0.80),
):
    """
    Plot accuracy when the composites are ranked by other semantic axes than the consensus.

    Parameters
    ----------
    head_curves : dict
        Axis name -> mean top-1 accuracy of the supervised models per rank bin (pandas.Series).
        The entry "Consensus" is drawn in bold.
    clip_curves : pandas.DataFrame
        Top-1 accuracy per rank bin under the CLIP text axis, one column per model.
    wordnet_curves : pandas.DataFrame
        Top-1 accuracy per WordNet hop count, one column per model.
    supervised_models : list of str
        Models averaged in the supervised curves.
    clip_model : str
        Name of the CLIP model.
    ylim : tuple
        Shared y-axis limits.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Three panels: individual classifier heads, CLIP text and WordNet.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

        # Individual classifier heads underneath, the consensus on top.
        draw_head_curves(axes[0], head_curves)

        draw_model_bundle(axes[1], clip_curves, supervised_models, clip_model)
        draw_model_bundle(axes[2], wordnet_curves, supervised_models, clip_model, markers=True)

        titles = ["Classifier heads", "CLIP text", "WordNet"]
        xlabels = ["Dissimilarity rank", "Dissimilarity rank", "WordNet hop count"]
        for ax, title, xlabel in zip(axes, titles, xlabels):
            ax.set_ylim(*ylim)
            draw_grid(ax, ylim, step=0.1)
            ax.set_title(title, loc="left")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("Top-1 accuracy")
        fig.tight_layout()
    return fig
