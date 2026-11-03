"""Plot the accuracy on object-only and background-only images and the structure of background-only errors."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


# All evaluated models by architecture family: CSV model name -> display name.
MODEL_FAMILIES = {
    "ViT": {
        "vit_small_patch16_224": "ViT-S-16",
        "ViT-B-16": "ViT-B-16",
        "vit_base_patch32_224": "ViT-B-32",
        "vit_base_patch16_384": "ViT-B-16-384",
        "ViT-L-16": "ViT-L-16",
    },
    "DeiT": {
        "deit_small_patch16_224": "DeiT-S",
        "DeiT-B-16": "DeiT-B",
        "deit3_small_patch16_224": "DeiT3-S",
        "deit3_medium_patch16_224": "DeiT3-M",
        "deit3_base_patch16_224": "DeiT3-B",
        "deit3_large_patch16_224": "DeiT3-L",
    },
    "BEiT": {
        "beit_base_patch16_224": "BEiT-B",
        "beit_large_patch16_224": "BEiT-L",
        "beitv2_base_patch16_224": "BEiTv2-B",
        "beitv2_large_patch16_224": "BEiTv2-L",
        "beit3_base_patch16_224": "BEiT3-B",
        "beit3_large_patch16_224": "BEiT3-L",
    },
    "CNN": {
        "resnet34": "ResNet-34",
        "ResNet-50": "ResNet-50",
        "resnet101": "ResNet-101",
        "ssl_resnet50": "SSL-ResNet-50",
        "inception_v3": "Inception-v3",
        "inception_v4": "Inception-v4",
    },
    "EfficientNet": {
        "efficientnet_b2": "EfficientNet-B2",
        "EfficientNet-B3": "EfficientNet-B3",
        "efficientnet_b4": "EfficientNet-B4",
        "efficientnetv2_rw_s": "EfficientNetV2-S",
        "efficientnetv2_rw_m": "EfficientNetV2-M",
    },
    "ConvNeXt": {
        "convnext_small": "ConvNeXt-S",
        "convnext_base": "ConvNeXt-B",
        "convnext_large": "ConvNeXt-L",
        "convnext_xlarge": "ConvNeXt-XL",
        "convnextv2_base": "ConvNeXtV2-B",
        "convnextv2_large": "ConvNeXtV2-L",
    },
    "CLIP": {
        "CLIP-ViT-B-16": "CLIP-ViT-B-16",
    },
}

# Colors of the main models, the model families, and the object-only backgrounds.
MODEL_COLORS = {
    "ViT-B-16": "#5B8DBE",
    "ViT-L-16": "#6AB187",
    "DeiT-B-16": "#D4726A",
    "ResNet-50": "#9B8EC4",
    "EfficientNet-B3": "#E6A756",
    "CLIP-ViT-B-16": "#5CC0C7",
}
FAMILY_COLORS = {
    "ViT": "#5B8DBE",
    "DeiT": "#D4726A",
    "BEiT": "#6AB187",
    "CNN": "#9B8EC4",
    "EfficientNet": "#E6A756",
    "ConvNeXt": "#5CC0C7",
    "CLIP": "#8C564B",
}
BACKGROUND_LABELS = {"black": "Black bg", "white": "White bg", "pink_noise": "Pink noise bg"}
BACKGROUND_COLORS = {"black": "#4A4A4A", "white": "#BABABA", "pink_noise": "#C4897E"}
BACKGROUND_EDGES = {"black": "#2A2A2A", "white": "#8A8A8A", "pink_noise": "#A06E64"}

# Colors of the five top-5 slots, from the best guess to the fifth, and of images whose true class is not among them.
SLOT_COLORS = ["#1F3B5A", "#2F5D8A", "#4A7DB0", "#7AA3CC", "#A9C4E0"]
MISSING_COLOR = "#C1292E"

ORIGINAL_LINE = {"color": "#555555", "linewidth": 1.4, "linestyle": (0, (3, 2))}
N_CLASSES = 1000

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def get_family(model):
    """Return the architecture family of a model."""
    for family, models in MODEL_FAMILIES.items():
        if model in models:
            return family
    raise ValueError(f"'{model}' is not in MODEL_FAMILIES.")


def get_display_name(model):
    """Return the display name of a model."""
    return MODEL_FAMILIES[get_family(model)][model]


def get_model_colors(models):
    """Use the paper colors if only main models are shown, otherwise one color per family."""
    if all(model in MODEL_COLORS for model in models):
        return {model: MODEL_COLORS[model] for model in models}
    return {model: FAMILY_COLORS[get_family(model)] for model in models}


def order_by_family(accuracy, family_column, model_column):
    """Order models by the family mean of one column and within families by another column."""

    # Sort the families by their mean, then the models of each family by their own value.
    families = {}
    for model in accuracy.index:
        families.setdefault(get_family(model), []).append(model)
    family_order = sorted(families, key=lambda family: accuracy.loc[families[family], family_column].mean(), reverse=True)
    return [(family, sorted(families[family], key=lambda model: accuracy.loc[model, model_column], reverse=True)) for family in family_order]


def get_bar_positions(groups, family_gap):
    """Return x positions of the models, with a gap between families."""
    positions = []
    x = 0.0
    for _, models in groups:
        for _ in models:
            positions.append(x)
            x += 1
        x += family_gap
    return np.array(positions)


def draw_original_line(ax, x, half_width, value):
    """Draw the dashed line marking the accuracy on the original images."""
    ax.plot([x - half_width, x + half_width], [value, value], **ORIGINAL_LINE, dash_capstyle="round", zorder=4)


def original_line_handle():
    """Legend entry for the dashed original-accuracy line."""
    return Line2D([0], [0], **ORIGINAL_LINE, dash_capstyle="round")


def format_model_axes(ax, x, models, top_k, values, title):
    """Label the models, the accuracy axis in percent and the title."""
    ax.set_xticks(x)
    ax.set_xticklabels([get_display_name(model) for model in models], rotation=45, ha="right")
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylabel(f"Top-{top_k} accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, np.ceil(min(1.0, np.max(values) * 1.12) * 10) / 10)
    ax.set_title(title, loc="left")


def draw_object_only(ax, accuracy, top_k, family_gap, bar_width, title="Object-only", order=None):
    """Draw grouped bars of the object-only accuracy per background, with the original accuracy as dashed lines."""

    # Order families by background-only accuracy (as in the background-only plot), models by object-only accuracy.
    groups = order_by_family(accuracy if order is None else order, "background_only", "object_only")
    models = [model for _, family_models in groups for model in family_models]
    x = get_bar_positions(groups, family_gap)
    offsets = (np.arange(len(BACKGROUND_LABELS)) - (len(BACKGROUND_LABELS) - 1) / 2) * bar_width

    for offset, background in zip(offsets, BACKGROUND_LABELS):
        ax.bar(x + offset, accuracy.loc[models, background], bar_width, label=BACKGROUND_LABELS[background],
               color=BACKGROUND_COLORS[background], edgecolor=BACKGROUND_EDGES[background], linewidth=0.5, zorder=3)

    # Mark the original accuracy across each group of bars.
    for i, model in enumerate(models):
        draw_original_line(ax, x[i], len(BACKGROUND_LABELS) * bar_width / 2, accuracy.loc[model, "original"])

    values = accuracy.loc[models, list(BACKGROUND_LABELS) + ["original"]].values
    format_model_axes(ax, x, models, top_k, values, title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [original_line_handle()], labels + ["Original"], loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=4 if len(models) > 12 else 2, frameon=False, fontsize=9)


def draw_background_only(ax, accuracy, top_k, family_gap, bar_width, title="Background-only", order=None):
    """Draw one bar of background-only accuracy per model, with its value and the original accuracy as a dashed line."""

    # Order families and models by their background-only accuracy.
    groups = order_by_family(accuracy if order is None else order, "background_only", "background_only")
    models = [model for _, family_models in groups for model in family_models]
    x = get_bar_positions(groups, family_gap)
    colors = get_model_colors(models)
    values = accuracy.loc[models, "background_only"]

    ax.bar(x, values, width=bar_width, color=[colors[model] for model in models], edgecolor="white", zorder=3)

    # Mark the original accuracy and write each value above its bar.
    for i, model in enumerate(models):
        draw_original_line(ax, x[i], bar_width / 2, accuracy.loc[model, "original"])
        ax.text(x[i], values[model] + 0.006, f"{values[model]:.1%}", ha="center", va="bottom", fontsize=8, color="#333333", rotation=90)

    format_model_axes(ax, x, models, top_k, list(values) + list(accuracy.loc[models, "original"]), title)
    ax.legend([original_line_handle()], ["Original"], loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)


def draw_error_ranks(ax, error_ranks, max_rank, marked_rank, title="Semantic structure of errors"):
    """Draw cumulative neighbor-rank curves of the wrong background-only predictions, their mean and chance."""
    curves = {model: get_cumulative_curve(ranks) for model, ranks in error_ranks.items()}
    colors = get_model_colors(list(curves))
    mean_curve = np.mean(list(curves.values()), axis=0)
    ranks = np.arange(1, max_rank + 1)

    # Draw each model faintly.
    for model, curve in curves.items():
        ax.plot(ranks, curve[:max_rank], color=colors[model], linewidth=1.0, alpha=0.35, zorder=2)

    # With more than the main models, also draw the mean of every family.
    if not all(model in MODEL_COLORS for model in curves):
        for family in MODEL_FAMILIES:
            family_curves = [curve for model, curve in curves.items() if get_family(model) == family]
            if family_curves:
                ax.plot(ranks, np.mean(family_curves, axis=0)[:max_rank], color=FAMILY_COLORS[family], linewidth=2.0, label=f"{family} mean", zorder=3)
    ax.plot(ranks, mean_curve[:max_rank], color="#1A1A1A", linewidth=2.5, label="Mean across models", zorder=4)
    ax.plot(ranks, ranks / (N_CLASSES - 1), color="#AAAAAA", linewidth=1.3, linestyle="--", label="Chance", zorder=1)

    # Annotate the mean at the marked rank.
    value = mean_curve[marked_rank - 1]
    ax.axvline(marked_rank, color="#CCCCCC", linewidth=0.8, linestyle=":", zorder=0)
    ax.annotate(f"~{value:.0%}", xy=(marked_rank, value), xytext=(marked_rank + 30, value + 0.06), fontweight="bold",
                color="#1A1A1A", arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2}, zorder=5)

    ax.set_xlabel("Semantic neighbor rank")
    ax.set_ylabel("Cumulative fraction of\nincorrect predictions")
    ax.set_xlim(1, max_rank)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)


def plot_object_only(accuracy, top_k=1, family_gap=0.5, bar_width=0.22, order=None):
    """
    Plot the object-only accuracy of each model on the black, white and pink-noise backgrounds.

    Parameters
    ----------
    accuracy : pandas.DataFrame
        One row per model with "black", "white", "pink_noise", "object_only", "background_only" and "original" columns.
    top_k : int
        Accuracy measure, used for the axis label.
    family_gap : float
        Extra space between architecture families.
    bar_width : float
        Width of one background bar.
    order : pandas.DataFrame or None
        Accuracy table used only to order the models (we pass the top-1 table, so that top-1 and top-5 plots share
        one order). Defaults to `accuracy`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Grouped bar plot with one bar per background and a dashed line for the original accuracy.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(5.5, 0.5 * len(accuracy)), 4.2))
        draw_object_only(ax, accuracy, top_k, family_gap, bar_width, order=order)
        fig.tight_layout()
    return fig


def plot_background_only(accuracy, top_k=1, family_gap=0.35, bar_width=0.65, order=None):
    """
    Plot the background-only accuracy of each model next to its accuracy on the original images.

    Parameters
    ----------
    accuracy : pandas.DataFrame
        One row per model with "background_only" and "original" columns.
    top_k : int
        Accuracy measure, used for the axis label.
    family_gap : float
        Extra space between architecture families.
    bar_width : float
        Width of one bar.
    order : pandas.DataFrame or None
        Accuracy table used only to order the models (we pass the top-1 table, so that top-1 and top-5 plots share
        one order). Defaults to `accuracy`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Bar plot with one bar per model and a dashed line for the original accuracy.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(5.5, 0.45 * len(accuracy)), 4.2))
        draw_background_only(ax, accuracy, top_k, family_gap, bar_width, order=order)
        fig.tight_layout()
    return fig


def get_cumulative_curve(ranks, max_rank=N_CLASSES - 1):
    """Fraction of errors whose predicted class is within the k nearest neighbors, for k = 1..max_rank."""
    return np.array([np.mean(ranks <= k) for k in range(1, max_rank + 1)])


def plot_error_ranks(error_ranks, max_rank=200, marked_rank=25):
    """
    Plot how close wrong background-only predictions are to the true class.

    Parameters
    ----------
    error_ranks : dict
        Model name -> semantic neighbor rank of the predicted class for every wrong prediction.
    max_rank : int
        Largest neighbor rank shown.
    marked_rank : int
        Rank at which the mean across models is annotated.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Cumulative curves per model, per family (when not only main models are shown), their mean and the chance level.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        draw_error_ranks(ax, error_ranks, max_rank, marked_rank)
        fig.tight_layout()
    return fig


def draw_accuracy_bars(ax, values, original, top_k, title, decimals):
    """Draw one bar per model with its value on top and its accuracy on the original images as a dashed line."""
    models = list(values.index)
    colors = get_model_colors(models)
    x = np.arange(len(models))
    ax.bar(x, values, width=0.65, color=[colors[model] for model in models], edgecolor="white", zorder=3)
    for i, model in enumerate(models):
        draw_original_line(ax, x[i], 0.35, original[model])
        ax.text(x[i], values[model] + 0.015, f"{values[model]:.{decimals}%}", ha="center", va="bottom", fontsize=10, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([get_display_name(model) for model in models], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Top-{top_k} accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend([original_line_handle()], ["Original"], loc="upper right", fontsize=9)


def plot_controls_overview(accuracy, error_ranks, top_k=1, max_rank=200, marked_rank=25, order=None):
    """
    Plot object-only accuracy, background-only accuracy and the semantic structure of background-only errors side by side.

    Parameters
    ----------
    accuracy : pandas.DataFrame
        One row per model with "object_only", "background_only" and "original" columns.
    error_ranks : dict
        Model name -> semantic neighbor rank of the predicted class for every wrong background-only prediction.
    top_k : int
        Accuracy measure, used for the axis labels.
    max_rank : int
        Largest neighbor rank shown in the error panel.
    marked_rank : int
        Rank at which the mean across models is annotated.
    order : pandas.DataFrame or None
        Accuracy table used only to order the models (we pass the top-1 table, so that top-1 and top-5 plots share
        one order). Defaults to `accuracy`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) object-only accuracy (mean over the three backgrounds), (b) background-only accuracy, both with one bar per
        model in the same order, and (c) the semantic structure of background-only errors.
    """

    # The same model order in both bar panels, by background-only accuracy.
    accuracy = accuracy.loc[(accuracy if order is None else order)["background_only"].sort_values(ascending=False).index]

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), width_ratios=[1, 1, 1.15], layout="constrained")
        draw_accuracy_bars(axes[0], accuracy["object_only"], accuracy["original"], top_k, "(a) Object-only", decimals=0)
        draw_accuracy_bars(axes[1], accuracy["background_only"], accuracy["original"], top_k, "(b) Background-only", decimals=1)
        draw_error_ranks(axes[2], error_ranks, max_rank, marked_rank, title="(c) Semantic structure of errors")
    return fig


def draw_top5_ranks(ax, top5_ranks, max_rank, marked_rank, title="Wrong guesses by position in the top 5"):
    """Draw cumulative neighbor-rank curves of the wrong top-5 guesses per position, for images without the true class in the top 5, and chance."""
    ranks = np.arange(1, max_rank + 1)

    # One curve per slot, from the best guess (dark) to the fifth (light).
    for slot, slot_ranks in top5_ranks.groupby("slot")["rank"]:
        curve = get_cumulative_curve(slot_ranks.values, max_rank)
        ax.plot(ranks, curve, color=SLOT_COLORS[slot - 1], linewidth=1.8, label=f"Guess {slot}", zorder=3)

    # Images where the true class is not among the five guesses at all.
    missing = top5_ranks.loc[~top5_ranks["true_class_in_top5"], "rank"].values
    ax.plot(ranks, get_cumulative_curve(missing, max_rank), color=MISSING_COLOR, linewidth=2.2, label="True class not in top 5", zorder=4)
    ax.plot(ranks, ranks / (N_CLASSES - 1), color="#AAAAAA", linewidth=1.3, linestyle="--", label="Chance", zorder=1)
    ax.axvline(marked_rank, color="#CCCCCC", linewidth=0.8, linestyle=":", zorder=0)

    ax.set_xlabel("Semantic neighbor rank")
    ax.set_ylabel("Cumulative fraction of\nwrong guesses")
    ax.set_xlim(1, max_rank)
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)


def plot_top5_ranks(top5_ranks, max_rank=200, marked_rank=25):
    """
    Plot how close the wrong classes among the five best background-only guesses are to the true class.

    Parameters
    ----------
    top5_ranks : pandas.DataFrame
        One row per wrong class among the five best guesses, with "slot" (1-5), "rank" (semantic neighbor rank of the
        guess) and "true_class_in_top5" columns.
    max_rank : int
        Largest neighbor rank shown.
    marked_rank : int
        Rank marked with a dotted line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Cumulative curves per slot, for images whose true class is not among the five guesses, and the chance level.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        draw_top5_ranks(ax, top5_ranks, max_rank, marked_rank)
        fig.tight_layout()
    return fig


def plot_error_structure(error_ranks, top5_ranks, max_rank=200, marked_rank=25):
    """
    Plot how close the wrong background-only predictions and the wrong top-5 guesses are to the true class side by side.

    Parameters
    ----------
    error_ranks : dict
        Model name -> semantic neighbor rank of the predicted class for every wrong background-only prediction.
    top5_ranks : pandas.DataFrame
        One row per wrong class among the five best guesses, as for `plot_top5_ranks`.
    max_rank : int
        Largest neighbor rank shown.
    marked_rank : int
        Rank marked with a dotted line; in (a) the mean across models is annotated there.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) the semantic structure of the wrong top-1 predictions, (b) the wrong guesses by their position in the top 5,
        with a shared y-axis.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
        draw_error_ranks(axes[0], error_ranks, max_rank, marked_rank, title="(a)  Wrong top-1 predictions")
        draw_top5_ranks(axes[1], top5_ranks, max_rank, marked_rank, title="(b)  Wrong guesses by position in the top 5")

        # The shared y-axis is labelled once and needs no headroom, since the legends go below the panels.
        axes[0].set_ylabel("Cumulative fraction of\nwrong guesses")
        axes[1].set_ylabel("")
        axes[0].autoscale(axis="y")
        axes[0].set_ylim(bottom=0)
        for ax in axes:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9)
    return fig
