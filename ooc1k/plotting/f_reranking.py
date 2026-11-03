"""Plot spatial re-ranking signals, recoveries and accuracy."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ooc1k.plotting.e_patch_competition import (
    BACKGROUND_COLOR,
    MODEL_LABELS,
    OBJECT_COLOR,
    OUTCOME_COLORS,
    STYLE,
)

# Signal column, panel title, rule threshold and histogram range.
SIGNALS = [
    ("advantage", "(a)  Compactness advantage $a$", "a_min", (-8, 8)),
    ("gap", "(b)  Logit gap $g$", "g_max", (0, 8)),
    ("distance", "(c)  Semantic distance $d$", "d_min", (0, 1)),
]

# Compare the target errors with correct predictions that a swap could break.
SIGNAL_OUTCOMES = ["Clean win", "Contested win", "Found-but-lost"]

# Accuracy before re-ranking; after re-ranking it is drawn in the object color.
ORIGINAL_COLOR = "#ADB5BD"


def draw_signal_histograms(ax, signals, outcomes, column, low, high):
    """Compare the signal distributions, clipping outliers into the outer bins."""
    bins = np.linspace(low, high, 41)

    # Values outside the range are added to the outermost bins.
    for outcome in SIGNAL_OUTCOMES:
        values = np.clip(signals.loc[outcomes == outcome, column], low, high)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=OUTCOME_COLORS[outcome],
            label=outcome,
        )


def plot_rerank_signals(signals, outcomes, rule):
    """
    Plot the distribution of the three re-ranking signals for correct and found-but-lost composites.

    Parameters
    ----------
    signals : pandas.DataFrame
        Output of `get_rerank_signals`.
    outcomes : pandas.Series
        Output of `get_outcomes`.
    rule : dict
        Thresholds `a_min`, `g_max` and optionally `d_min`; the region where each condition holds is shaded.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One panel per signal, one histogram per outcome type.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
        for ax, (column, title, key, (low, high)) in zip(axes, SIGNALS):
            draw_signal_histograms(ax, signals, outcomes, column, low, high)

            # Shade where the condition of the rule holds.
            threshold = rule.get(key)
            if threshold is not None:
                swap_low, swap_high = (low, threshold) if key == "g_max" else (threshold, high)
                ax.axvspan(swap_low, swap_high, color=OBJECT_COLOR, alpha=0.06, zorder=0)
                ax.axvline(threshold, color="#888888", linewidth=0.8, linestyle="--")

            ax.set_xlim(low, high)
            ax.set_title(title, loc="left", pad=8)
            ax.set_yticks([])
        axes[0].set_ylabel("Density")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        fig.tight_layout()
    return fig


def get_rule_label(name):
    """Rule name as math text with one condition per line, e.g. "a > 1.0, g < 1.0" -> "$a{>}1.0$\\n$g{<}1.0$"."""
    conditions = [condition.split() for condition in name.split(", ")]
    return "\n".join(f"${signal}{{{operator}}}{value}$" for signal, operator, value in conditions)


def draw_recovery_breakage(ax, table, selected, gate_block):
    """Draw recovered and broken counts per re-ranking rule side by side, with the net change and the ratio on top."""
    x = np.arange(len(table))
    recovered, broken = table["FBL recovered"].values, table["Broken"].values

    # Rules without and with the semantic distance condition, each group shaded and named below the bars.
    label_y = -0.12 * recovered.max()
    for protected, color, text_color, text in [
        (False, "#888888", "#666666", f"B{gate_block} unprotected"),
        (True, OBJECT_COLOR, OBJECT_COLOR, f"B{gate_block} + semantic protection"),
    ]:
        group = [xi for xi, name in enumerate(table.index) if ("d >" in name) == protected]
        if group:
            ax.axvspan(min(group) - 0.5, max(group) + 0.5, color=color, alpha=0.04, zorder=0)
            ax.text((min(group) + max(group)) / 2, label_y, text, ha="center", fontsize=8, color=text_color, fontstyle="italic")

    # The selected rule gets a black outline and a star.
    for xi, name in enumerate(table.index):
        edge, width = ("black", 2.0) if name == selected else ("white", 0.5)
        ax.bar(xi - 0.15, recovered[xi], 0.30, color=OBJECT_COLOR, alpha=0.85, edgecolor=edge, linewidth=width)
        ax.bar(xi + 0.15, broken[xi], 0.30, color=BACKGROUND_COLOR, alpha=0.85, edgecolor=edge, linewidth=width)
        text = f"{table['Net change (pp)'].iloc[xi]:+.1f}pp\n{table['Ratio'].iloc[xi]:.1f}:1"
        ax.text(
            xi,
            max(recovered[xi], broken[xi]) + 0.027 * recovered.max(),
            f"★ {text}" if name == selected else text,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold" if name == selected else "normal",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([get_rule_label(name) for name in table.index], fontsize=8, linespacing=0.9)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(-0.15 * recovered.max(), 1.5 * recovered.max())
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Number of images")
    handles = [
        Patch(facecolor=OBJECT_COLOR, alpha=0.85, label="FBL recovered"),
        Patch(facecolor=BACKGROUND_COLOR, alpha=0.85, label="Correct broken"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)


def plot_recovery_breakage(table, selected, gate_block):
    """
    Plot how many found-but-lost errors each re-ranking rule recovers and how many correct predictions it breaks.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per rule (indexed by its name, conditions separated by ", ") with the columns
        "FBL recovered", "Broken", "Ratio" and "Net change (pp)". Rules without and with a distance condition
        ("d > …") are shaded as two groups; the rules of each group should be next to each other.
    selected : str
        Name of the rule marked with a star.
    gate_block : int
        Block of the compactness signal, for the group names.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Recovered and broken counts side by side, with the net change and the ratio on top.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(table), 3.2))
        draw_recovery_breakage(ax, table, selected, gate_block)
        fig.tight_layout()
    return fig


def get_accuracy_limits(curves):
    """Y-limits in percent that leave at least 5 points around all accuracy curves, rounded to multiples of 5."""
    low = min(curve[["Before", "After"]].values.min() for curve in curves)
    high = max(curve[["Before", "After"]].values.max() for curve in curves)
    return 5 * np.floor((100 * low - 5) / 5), 5 * np.ceil((100 * high + 5) / 5)


def draw_rerank_curve(ax, curve):
    """Draw accuracy before and after re-ranking along the dissimilarity rank, in percent."""
    before, after = 100 * curve["Before"], 100 * curve["After"]
    ax.plot(curve.index, before, "-o", color=ORIGINAL_COLOR, markersize=4, label="Original", zorder=2)
    ax.plot(curve.index, after, "-o", color=OBJECT_COLOR, markersize=4, label="After re-ranking", zorder=3)
    ax.fill_between(curve.index, before, after, color=OBJECT_COLOR, alpha=0.10)
    ax.set_xlabel("Dissimilarity rank")


def plot_rerank_accuracy(curves):
    """
    Plot the top-1 accuracy along the semantic axis before and after re-ranking, one panel per model.

    Parameters
    ----------
    curves : dict
        Model name -> output of `get_rerank_accuracy`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One row of panels with a shared y-axis, the mean bin-wise gain in the lower right corner of each.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(curves), figsize=(max(5.5, 2.8 * len(curves)), 3.0), sharey=True, squeeze=False)
        for ax, (model, curve) in zip(axes[0], curves.items()):
            draw_rerank_curve(ax, curve)
            gain = 100 * (curve["After"] - curve["Before"]).mean()
            ax.text(0.95, 0.05, f"Δ={gain:+.1f}pp", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color=OBJECT_COLOR)
            ax.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
        axes[0, 0].set_ylim(get_accuracy_limits(curves.values()))
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].legend(loc="lower left", frameon=False)
        fig.tight_layout()
    return fig


def plot_rerank_overview(table, selected, gate_block, curve):
    """
    Plot the recovery and breakage of the re-ranking rules next to the accuracy before and after re-ranking.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per rule, as for `plot_recovery_breakage`.
    selected : str
        Name of the rule marked with a star; it is the rule used for the accuracy panel.
    gate_block : int
        Block of the compactness signal, for the group names.
    curve : pandas.DataFrame
        Output of `get_rerank_accuracy` for the selected rule.

    Returns
    -------
    fig : matplotlib.figure.Figure
        (a) error recovery via spatial re-ranking, (b) accuracy before and after re-ranking along the semantic axis.
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), gridspec_kw={"left": 0.08, "right": 0.97, "bottom": 0.31, "top": 0.89, "wspace": 0.3})
        draw_recovery_breakage(axes[0], table, selected, gate_block)
        draw_rerank_curve(axes[1], curve)
        axes[1].set_ylim(get_accuracy_limits([curve]))
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend(loc="lower left", frameon=False, fontsize=8)
        axes[0].set_title("(a)  Error recovery via spatial re-ranking", loc="left", pad=8)
        axes[1].set_title("(b)  Accuracy gain is uniform across dissimilarity", loc="left", pad=8)
    return fig
