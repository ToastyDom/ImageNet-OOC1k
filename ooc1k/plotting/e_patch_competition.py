"""Plot what the patches of a Vision Transformer encode and how object and background evidence compete."""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# Object, background and undecided colors; semantic pull runs from red to green.
OBJECT_COLOR = "#2D6A4F"
BACKGROUND_COLOR = "#C1292E"
UNDECIDED_COLOR = "#ADB5BD"
PULL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pull", [BACKGROUND_COLOR, "#F5F5F5", OBJECT_COLOR]
)

# Width of one column of the patch heatmap figure, in inches.
PANEL_WIDTH = 1.408

# Colors and names of the five ViTs of the patch analysis.
MODEL_COLORS = {
    "ViT-B-16": "#2ca02c",
    "ViT-S-16": "#17becf",
    "ViT-L-16": "#ff7f0e",
    "DeiT-B-16": "#1f77b4",
    "ViT-RelPos-B-16": "#e377c2",
}
MODEL_LABELS = {
    "ViT-B-16": "ViT-B/16",
    "ViT-S-16": "ViT-S/16",
    "ViT-L-16": "ViT-L/16",
    "DeiT-B-16": "DeiT-B/16",
    "ViT-RelPos-B-16": "ViT-RelPos-B/16",
}

# Colors and markers of the distance groups.
DISTANCE_COLORS = {"Low": "#2D6A4F", "Mid": "#D4A843", "High": "#C1292E"}
DISTANCE_MARKERS = {"Low": "o", "Mid": "D", "High": "s"}
ORIGINALS_COLOR = "#555555"

# Colors of the four outcome types and of the rank bands of the true class.
OUTCOME_COLORS = {
    "Clean win": "#2D6A4F",
    "Contested win": "#84A98C",
    "Found-but-lost": "#E76F51",
    "No detection": "#ADB5BD",
}
RANK_COLORS = {
    "Rank 1": "#2D6A4F",
    "Rank 2-5": "#52B788",
    "Rank 6-20": "#D4A843",
    "Rank > 20": "#C1292E",
}

# Short names of the outcome types and legend names of the rank bands, with the text color on each band.
OUTCOME_SHORT = {"Clean win": "Clean", "Contested win": "Contested", "Found-but-lost": "FBL", "No detection": "No-det"}
RANK_LABELS = {"Rank 1": "Rank 1 (correct)", "Rank 2-5": "Rank 2–5", "Rank 6-20": "Rank 6–20", "Rank > 20": "Rank >20"}
RANK_TEXT_COLORS = {"Rank 1": "white", "Rank 2-5": "black", "Rank 6-20": "black", "Rank > 20": "white"}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def center_crop(image):
    """Cut the largest centred square out of a PIL image."""
    width, height = image.size
    size = min(width, height)
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def get_cls_tag(cls_prediction, example, class_labels):
    """Short text for the CLS prediction: the object, the background or the predicted class."""
    if cls_prediction == example["object_class"]:
        return "CLS → obj"
    if cls_prediction == example["background_class"]:
        return "CLS → bg"
    label = class_labels[cls_prediction]
    return f"CLS → {label[:13]}…" if len(label) > 14 else f"CLS → {label}"


def draw_coverage_bars(ax, pull, layers, tau):
    """Stacked shares of object-realm, undecided and background-realm patches per layer."""
    x = np.arange(len(layers))
    share_object = np.array([(pull[layer] > tau).mean() for layer in layers])
    share_background = np.array([(pull[layer] < -tau).mean() for layer in layers])
    share_undecided = 1 - share_object - share_background

    ax.bar(x, share_object, width=0.72, color=OBJECT_COLOR, alpha=0.88, edgecolor="white", linewidth=0.6)
    ax.bar(
        x,
        share_undecided,
        width=0.72,
        bottom=share_object,
        color=UNDECIDED_COLOR,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x,
        share_background,
        width=0.72,
        bottom=share_object + share_undecided,
        color=BACKGROUND_COLOR,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
    )

    # Percentages only fit when there are few bars.
    if len(layers) <= 5:
        for xi, (obj, undecided, bg) in enumerate(zip(share_object, share_undecided, share_background)):
            if obj > 0.08:
                ax.text(xi, obj / 2, f"{obj:.0%}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if undecided > 0.15:
                ax.text(xi, obj + undecided / 2, f"{undecided:.0%}", ha="center", va="center", fontsize=7.5, color="#333333")
            if bg > 0.08:
                ax.text(xi, 1 - bg / 2, f"{bg:.0%}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=9 if len(layers) <= 5 else 6)
    ax.set_xlabel("Block", fontsize=9, labelpad=2)
    ax.set_xlim(-0.55, len(layers) - 0.45)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.yaxis.tick_right()
    ax.set_box_aspect(1)


def draw_composite(ax, example, class_labels, final_prediction, correct):
    """Show the composite with its final prediction, object, background and rank."""
    ax.imshow(center_crop(example["image"]))
    ax.set_xticks([])
    ax.set_yticks([])
    caption = (
        f"{'✓' if correct else '✗'} pred: {class_labels[final_prediction]}\n"
        f"obj: {class_labels[example['object_class']]}\n"
        f"bg: {class_labels[example['background_class']]}  ·  rank {example['rank']}"
    )
    ax.set_xlabel(
        caption, fontsize=7.5, linespacing=1.4, color=OBJECT_COLOR if correct else BACKGROUND_COLOR, labelpad=3
    )


def draw_patch_pull(ax, example, layer, grid_size, norm, class_labels):
    """Draw semantic pull and outline exact object or background predictions."""
    pull = example["pull"][layer].reshape(grid_size, grid_size)
    predictions = example["patch_predictions"][layer].reshape(grid_size, grid_size)
    ax.imshow(pull, cmap=PULL_CMAP, norm=norm, interpolation="nearest")
    exact = (predictions == example["object_class"]) | (predictions == example["background_class"])
    for r, c in zip(*np.nonzero(exact)):
        ax.add_patch(
            Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1.3, edgecolor="black", facecolor="none", zorder=2)
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        get_cls_tag(example["cls_predictions"][layer], example, class_labels),
        fontsize=9,
        labelpad=3,
    )


def plot_patch_evidence(
    examples, class_labels, layers, tau, row_labels=None, show_coverage=True, pull_range=0.7, row_height=1.8, row_spacing=0.15
):
    """
    Show, for a few composites, which class every patch points to in selected blocks.

    Every patch is colored by its semantic pull: green if its predicted class is closer to the object,
    red if it is closer to the background, and white near zero. Patches that predict exactly the object
    or background class are outlined in black. Below every heatmap we note what the CLS token predicts.
    Every column has a fixed width, so the figure grows with the number of blocks.

    Parameters
    ----------
    examples : list of dict
        Output of `get_example_predictions`.
    class_labels : list of str
        ImageNet class labels by index.
    layers : list of int
        Blocks to show (0-based).
    tau : float
        Threshold of the undecided zone, used for the coverage bars.
    row_labels : list of str, optional
        A label left of every row.
    show_coverage : bool
        Add a column with the share of object, undecided and background patches per block.
    pull_range : float
        Pull values mapped to the strongest colors.
    row_height : float
        Height of every row in inches.
    row_spacing : float
        Vertical space between the rows, as a fraction of the row height; it has to fit the caption below every
        composite.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One row per composite: the composite, a color scale, one heatmap per block and optionally the coverage bars.
    """
    n_rows = len(examples)
    norm = mcolors.TwoSlopeNorm(vmin=-pull_range, vcenter=0, vmax=pull_range)

    # Columns: the composite, a narrow color scale, one heatmap per block and optionally the coverage bars.
    width_ratios = [1.0, 0.09] + [1.0] * len(layers) + [1.0] * int(show_coverage)
    n_cols = len(width_ratios)

    # Every column gets a fixed width in inches (the three rows of the paper figure are 11 inches wide),
    # plus fixed margins for the row labels, the column titles and the legend.
    wspace = 0.15
    left, right, top, bottom = (0.88 if row_labels is not None else 0.66), 0.44, 0.5, 0.65
    width = left + right + PANEL_WIDTH * sum(width_ratios) * (1 + wspace * (n_cols - 1) / n_cols)
    height = row_height * n_rows + top + bottom
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(width, height),
            squeeze=False,
            gridspec_kw={
                "width_ratios": width_ratios,
                "wspace": wspace,
                "hspace": row_spacing if n_rows > 1 else 0,
                "left": left / width,
                "right": 1 - right / width,
                "top": 1 - top / height,
                "bottom": bottom / height,
            },
        )

        for row, example in enumerate(examples):
            grid_size = int(np.sqrt(example["pull"].shape[-1]))
            final_prediction = example["cls_predictions"][-1]
            correct = final_prediction == example["object_class"]

            # The composite, with its final prediction, object, background and rank.
            ax = axes[row, 0]
            draw_composite(ax, example, class_labels, final_prediction, correct)
            if row_labels is not None:
                ax.set_ylabel(row_labels[row], fontsize=10, fontweight="bold")

            # Color scale of the pull; the exact range belongs in the caption.
            colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=PULL_CMAP), cax=axes[row, 1])
            colorbar.set_ticks([-pull_range, 0, pull_range], labels=["−", "0", "+"])
            colorbar.ax.tick_params(labelsize=9)

            # One heatmap of the semantic pull per block; exact object or background patches are outlined.
            for ax, layer in zip(axes[row, 2:], layers):
                draw_patch_pull(ax, example, layer, grid_size, norm, class_labels)

            if show_coverage:
                draw_coverage_bars(axes[row, -1], example["pull"], layers, tau)

        # Column titles above the first row, none above the color scale.
        titles = ["Composite", None] + [f"Block {layer}" for layer in layers] + ["Realm coverage"] * int(show_coverage)
        for ax, title in zip(axes[0], titles):
            if title is not None:
                box = ax.get_position()
                fig.text(box.x0 + box.width / 2, box.y1 + 0.012, title, ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Legend of the realms, centred below the heatmaps.
        first, last = axes[0, 2].get_position(), axes[0, 1 + len(layers)].get_position()
        handles = [
            Patch(facecolor=OBJECT_COLOR, alpha=0.88, edgecolor="white", label="Object (semantic pull > τ)"),
            Patch(facecolor=UNDECIDED_COLOR, alpha=0.55, edgecolor="white", label="Undecided (|pull| < τ)"),
            Patch(facecolor=BACKGROUND_COLOR, alpha=0.88, edgecolor="white", label="Background (semantic pull < −τ)"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9.5,
            handlelength=1.4,
            handleheight=1.0,
            columnspacing=2.5,
            bbox_to_anchor=((first.x0 + last.x1) / 2, 0.06 / height),
        )
    return fig


def plot_subset_accuracy(curves):
    """
    Plot the top-1 accuracy of the ViTs on the mechanism subset along the semantic axis.

    Parameters
    ----------
    curves : pandas.DataFrame
        Output of `get_subset_accuracy`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One line per model.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for model in curves.columns:
            ax.plot(
                curves.index,
                100 * curves[model],
                marker="o",
                markersize=4,
                color=MODEL_COLORS.get(model),
                label=MODEL_LABELS.get(model, model),
            )
        ax.set_xticks(np.arange(0, 1001, 200))
        ax.set_xlabel("Dissimilarity rank")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        fig.tight_layout()
    return fig


def draw_realm_balance(ax, balance, title=None):
    """Draw the object share of the decided patches across blocks, by outcome and distance group, on an existing axis."""
    x = np.arange(len(balance))

    # Background-leaning and object-leaning halves of the plot.
    ax.axhspan(0, 50, color=BACKGROUND_COLOR, alpha=0.04, zorder=0)
    ax.axhspan(50, 95, color=OBJECT_COLOR, alpha=0.04, zorder=0)
    ax.axhline(50, color="#AAAAAA", linewidth=1.0, zorder=1)

    for outcome, linestyle in [("Correct", "-"), ("Incorrect", "--")]:
        for group, color in DISTANCE_COLORS.items():
            ax.plot(
                x,
                100 * balance[(outcome, group)],
                linestyle=linestyle,
                marker=DISTANCE_MARKERS[group],
                color=color,
                linewidth=1.8,
                markersize=4.5,
                zorder=5,
            )
    ax.plot(
        x,
        100 * balance[("Originals", "")],
        marker="^",
        color=ORIGINALS_COLOR,
        linewidth=1.6,
        markersize=4,
        alpha=0.85,
        zorder=6,
    )

    # Name the solid and dashed groups at the right end.
    correct = 100 * np.mean([balance[("Correct", group)].iloc[-1] for group in DISTANCE_COLORS])
    incorrect = 100 * np.mean([balance[("Incorrect", group)].iloc[-1] for group in DISTANCE_COLORS])
    ax.annotate("Correct", xy=(x[-1] + 0.08, correct), fontsize=8, fontweight="bold", color="#333333", va="center")
    ax.annotate("Incorrect", xy=(x[-1] - 0.05, incorrect + 7), fontsize=8, fontweight="bold", color="#333333", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {block}" for block in balance.index])
    ax.set_xlim(-0.3, len(x) - 0.7)
    low = min(33, 100 * balance.values.min() - 5)
    ax.set_ylim(low, 100)
    ax.set_yticks(np.arange(10 * np.ceil(low / 10), 101, 10))
    ax.set_ylabel("Object share of\ndecided patches (%)")

    handles = [
        Line2D([0], [0], color=color, marker=DISTANCE_MARKERS[group], linewidth=2, markersize=4, label=f"{group} diss.")
        for group, color in DISTANCE_COLORS.items()
    ]
    handles += [
        Line2D([0], [0], color="#666666", linewidth=2, label="Correct"),
        Line2D([0], [0], color="#666666", linewidth=2, linestyle="--", label="Incorrect"),
        Line2D([0], [0], color=ORIGINALS_COLOR, marker="^", linewidth=2, markersize=4, label="Originals"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    if title:
        ax.set_title(title, loc="left", pad=8)


def plot_realm_balance(balance):
    """
    Plot the object share of the decided patches across blocks, by outcome and distance group.

    Parameters
    ----------
    balance : pandas.DataFrame
        Output of `get_realm_balance`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Correct composites as solid lines, incorrect ones dashed, the originals in grey.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.4, 3.5))
        draw_realm_balance(ax, balance)
        fig.tight_layout()
    return fig
def draw_stacked_shares(ax, shares, colors, scale=1, width=0.55, labelled=None, text_colors=None):
    """Stacked bars of shares (rows: bars, columns: segments); segments above 6% get a label, black up to 12% unless `text_colors` is given."""
    x = np.arange(len(shares))
    bottom = np.zeros(len(shares))
    for column in shares.columns:
        values = shares[column].values
        ax.bar(
            x,
            scale * values,
            width=width,
            bottom=scale * bottom,
            color=colors[column],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
            label=column,
        )
        if labelled is None or column in labelled:
            for xi, (value, low) in enumerate(zip(values, bottom)):
                if value > 0.06:
                    color = text_colors[column] if text_colors else ("white" if value > 0.12 else "black")
                    ax.text(
                        xi,
                        scale * (low + value / 2),
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=color,
                        fontweight="bold",
                    )
        bottom += values
    ax.set_xticks(x)


def draw_outcome_shares(ax, shares, labels=None, title=None, detailed=True):
    """Draw the outcome shares as stacked bars with the accuracy on top; `detailed` labels every segment, otherwise only clean wins and no detection."""
    labels = labels or {}
    labelled = None if detailed else ["Clean win", "No detection"]
    draw_stacked_shares(ax, shares, OUTCOME_COLORS, scale=100, width=0.55 if detailed else 0.62, labelled=labelled)

    # Accuracy is the share of clean and contested wins.
    accuracy = 100 * (shares["Clean win"] + shares["Contested win"])
    for xi, value in enumerate(accuracy):
        if detailed:
            ax.text(xi, 102, f"Acc: {value:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#444444")
        else:
            ax.text(xi, 101.5, f"Acc: {value:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333")

    ax.set_xticklabels([labels.get(name, name) for name in shares.index])
    if detailed:
        ax.set_ylim(0, 112)
        ax.set_ylabel("Fraction of images (%)")
    else:
        ax.set_xlim(-0.6, len(shares) - 0.4)
        ax.set_ylim(0, 113)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylabel("Fraction of images")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4 if detailed else 2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    if title:
        ax.set_title(title, loc="left", pad=8)


def plot_outcome_shares(shares, labels=None, detailed=True, title=None):
    """
    Plot the share of the four outcome types as one stacked bar per group, with the accuracy on top.

    Parameters
    ----------
    shares : pandas.DataFrame
        Output of `get_outcome_shares_by_model` or `get_outcome_shares_by_distance`.
    labels : dict, optional
        Display name of every row of `shares`.
    detailed : bool
        Label every segment and give the accuracy with one decimal, as in the comparison across models. Otherwise
        only clean wins and no detection are labelled, as in the overview by distance.
    title : str, optional
        Title above the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One bar per row of `shares`.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(shares), 3.5))
        draw_outcome_shares(ax, shares, labels, title=title, detailed=detailed)
        fig.tight_layout()
    return fig
def draw_object_expansion(ax, expansion, n_patches=196, title=None):
    """Draw the mean number of exact object patches across blocks, by outcome type, on an existing axis."""
    x = np.arange(len(expansion))
    for outcome, color in OUTCOME_COLORS.items():
        ax.plot(
            x,
            expansion[outcome],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=outcome,
        )
        ax.annotate(
            f"{expansion[outcome].iloc[-1]:.0f}",
            xy=(x[-1], expansion[outcome].iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            va="center",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{block}" for block in expansion.index])
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Mean exact-object patches (of {n_patches})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    if title:
        ax.set_title(title, loc="left", pad=8)


def plot_object_expansion(expansion, n_patches=196):
    """
    Plot the mean number of exact object patches across blocks, by outcome type.

    Parameters
    ----------
    expansion : pandas.DataFrame
        Output of `get_object_expansion`.
    n_patches : int
        Patches per image, for the axis label.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One line per outcome type, with its value in the last block.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        draw_object_expansion(ax, expansion, n_patches)
        fig.tight_layout()
    return fig
def draw_found_but_lost_aggregation(ax, aggregation, title=None):
    """Draw the realm balance of the patches and the classification proximity of the CLS token for found-but-lost composites, on an existing axis."""
    x = np.arange(len(aggregation))
    balance, proximity = 100 * aggregation["Realm balance"], 100 * aggregation["CLS proximity"]
    ax.plot(
        x,
        balance,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        color=OBJECT_COLOR,
        label="Patch realm balance",
        zorder=5,
    )
    ax.plot(
        x,
        proximity,
        marker="s",
        markersize=4.5,
        linewidth=1.8,
        color=BACKGROUND_COLOR,
        label="CLS proximity to object (CP)",
        zorder=5,
    )

    # Shade the gap where the patches favour the object more than the CLS token does.
    ax.fill_between(
        x,
        balance,
        proximity,
        where=balance > proximity,
        interpolate=True,
        color=OBJECT_COLOR,
        alpha=0.1,
        zorder=1,
    )
    ax.plot(
        x,
        100 * aggregation["CLS proximity (correct)"],
        marker="^",
        markersize=3,
        linewidth=1.0,
        linestyle="--",
        color=OBJECT_COLOR,
        alpha=0.4,
        label="CP — correct (ref.)",
        zorder=3,
    )

    # Mark the drop of the CLS proximity in the last block and the gap it leaves.
    if len(x) >= 2 and proximity.iloc[-2] > proximity.iloc[-1] + 2:
        ax.annotate(
            "",
            xy=(x[-1], proximity.iloc[-1] + 1),
            xytext=(x[-2], proximity.iloc[-2] - 1),
            arrowprops=dict(arrowstyle="->", color=OUTCOME_COLORS["Found-but-lost"], lw=1.4, connectionstyle="arc3,rad=-0.3"),
            zorder=10,
        )
    gap_y = (balance.iloc[-1] + proximity.iloc[-1]) / 2
    ax.text(x[-1] + 0.12, gap_y, "gap", fontsize=7.5, color="#555555", fontstyle="italic", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {block}" for block in aggregation.index])
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Proximity / balance (%)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    if title:
        ax.set_title(title, loc="left", pad=8)


def plot_found_but_lost_aggregation(aggregation):
    """
    Plot the realm balance of the patches and the classification proximity of the CLS token for found-but-lost composites.

    Parameters
    ----------
    aggregation : pandas.DataFrame
        Output of `get_found_but_lost_aggregation`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Patch evidence in green, the CLS token in red and correctly classified composites as a dashed reference.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.4, 3.5))
        draw_found_but_lost_aggregation(ax, aggregation)
        fig.tight_layout()
    return fig
def draw_true_class_rank(ax, shares, title=None):
    """Draw where the true class ranks in the final CLS logits, by outcome type, on an existing axis."""
    draw_stacked_shares(ax, shares, RANK_COLORS, text_colors=RANK_TEXT_COLORS)
    ax.set_xticklabels([OUTCOME_SHORT.get(name, name) for name in shares.index])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction of images")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [RANK_LABELS[column] for column in shares.columns],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    if title:
        ax.set_title(title, loc="left", pad=8)


def plot_true_class_rank(shares):
    """
    Plot where the true class ranks in the final CLS logits, by outcome type.

    Parameters
    ----------
    shares : pandas.DataFrame
        Output of `get_rank_shares`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One stacked bar of rank bands per outcome type.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        draw_true_class_rank(ax, shares)
        fig.tight_layout()
    return fig
def draw_compactness(ax, compactness, title=None):
    """Draw how often the runner-up class is spatially more compact than the predicted class, by outcome type, on an existing axis."""
    x = np.arange(len(compactness))
    for outcome, color in OUTCOME_COLORS.items():
        values = 100 * compactness[outcome]
        ax.plot(x, values, marker="o", markersize=4, linewidth=1.8, color=color, label=outcome, zorder=3)
        ax.annotate(
            f"{values.iloc[-1]:.0f}%",
            xy=(x[-1], values.iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            va="center",
        )

    # Above 50% the runner-up (top-2) is more often the more compact class, below it the prediction (top-1).
    low = min(10, 10 * np.floor(10 * compactness.values.min()))
    high = max(92, 100 * compactness.values.max() + 5)
    ax.axhline(50, color="#888888", linewidth=0.8, zorder=1)
    ax.fill_between(x, 50, high, color=OUTCOME_COLORS["Found-but-lost"], alpha=0.04, zorder=0)
    ax.fill_between(x, low, 50, color=OUTCOME_COLORS["Clean win"], alpha=0.04, zorder=0)
    ax.text(x[0] + 0.1, high - 3, "Top-2 more compact", fontsize=7.5, color=OUTCOME_COLORS["Found-but-lost"], va="top", fontstyle="italic")
    ax.text(x[0] + 0.1, 48, "Top-1 more compact", fontsize=7.5, color=OUTCOME_COLORS["Clean win"], va="top", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"B{block}" for block in compactness.index])
    ax.set_xlabel("Layer")
    ax.set_ylim(low, high)
    ax.set_yticks(np.arange(low, high, 10))
    ax.set_ylabel("Fraction of images (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    if title:
        ax.set_title(title, loc="left", pad=8)


def plot_compactness(compactness):
    """
    Plot how often the runner-up class is spatially more compact than the predicted class, by outcome type.

    Parameters
    ----------
    compactness : pandas.DataFrame
        Output of `get_compactness`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One line per outcome type; above 50% the runner-up is more often the more compact class.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        draw_compactness(ax, compactness)
        fig.tight_layout()
    return fig


def plot_mechanism_overview(distance_shares, balance, aggregation):
    """
    Plot the outcome types by distance, the object-background competition and the aggregation failure side by side.

    Parameters
    ----------
    distance_shares : pandas.DataFrame
        Output of `get_outcome_shares_by_distance`.
    balance : pandas.DataFrame
        Output of `get_realm_balance`.
    aggregation : pandas.DataFrame
        Output of `get_found_but_lost_aggregation`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) outcome shares per distance group, (b) realm balance across blocks, (c) found-but-lost aggregation.
    """
    labels = {"Low": "Low diss.", "Mid": "Mid diss.", "High": "High diss."}
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(10.8, 3.8),
            gridspec_kw={"width_ratios": [1.1, 1.2, 1.0], "left": 0.06, "right": 0.95, "bottom": 0.2, "top": 0.86, "wspace": 0.36},
        )
        draw_outcome_shares(axes[0], distance_shares, labels=labels, detailed=False)
        draw_realm_balance(axes[1], balance)
        draw_found_but_lost_aggregation(axes[2], aggregation)

        titles = ["(a)  Recognition outcomes shift\n      with semantic distance", "(b)  Object–background competition\n      across layers", "(c)  Aggregation failure despite\n      strong object signal"]
        for ax, title, x in zip(axes, titles, [-0.02, -0.02, -0.04]):
            ax.set_title(title, x=x, ha="left", pad=8)
    return fig


def plot_mechanism_appendix(rank_shares, expansion, compactness):
    """
    Plot the true-class rank, the object patch expansion and the spatial compactness side by side.

    Parameters
    ----------
    rank_shares : pandas.DataFrame
        Output of `get_rank_shares`.
    expansion : pandas.DataFrame
        Output of `get_object_expansion`.
    compactness : pandas.DataFrame
        Output of `get_compactness`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) true-class rank in the CLS logits, (b) object patch expansion, (c) spatial compactness across blocks.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"left": 0.07, "right": 0.95, "bottom": 0.26, "top": 0.91, "wspace": 0.3})
        draw_true_class_rank(axes[0], rank_shares, title="(a)  True class rank in CLS logits")
        draw_object_expansion(axes[1], expansion, title="(b)  Object patch expansion across layers")
        draw_compactness(axes[2], compactness, title="(c)  Spatial compactness across layers")

        # All three legends at the same height near the bottom of the figure.
        for ax in axes:
            legend = ax.get_legend()
            legend.set_loc("lower center")
            legend.set_bbox_to_anchor((0.5, 0.01), transform=mpl.transforms.blended_transform_factory(ax.transAxes, fig.transFigure))
    return fig


def plot_expansion_by_model(expansions):
    """
    Plot the object patch expansion of several models next to each other, as in `plot_object_expansion`.

    Parameters
    ----------
    expansions : dict
        Model name -> output of `get_object_expansion` for that model's last blocks.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One panel per model with a shared y-axis and one legend below all panels.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            len(expansions),
            figsize=(2.8 * len(expansions), 3.8),
            sharey=True,
            squeeze=False,
            gridspec_kw={"left": 0.07, "right": 0.96, "bottom": 0.24, "top": 0.9, "wspace": 0.08},
        )
        for i, (ax, (model, expansion)) in enumerate(zip(axes[0], expansions.items())):
            draw_object_expansion(ax, expansion)

            # One shared legend, and the y-label only on the first panel.
            ax.get_legend().remove()
            ax.set_ylabel("Exact-object patches" if i == 0 else "")
            ax.set_title(f"({chr(ord('a') + i)})  {MODEL_LABELS.get(model, model)}", loc="left", pad=8, fontsize=11)

        handles = [
            Line2D([0], [0], color=color, marker="o", markersize=6, linewidth=1.8, label=outcome)
            for outcome, color in OUTCOME_COLORS.items()
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(handles), frameon=False, fontsize=10)
    return fig
