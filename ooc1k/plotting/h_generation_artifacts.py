"""Plot the accuracy decline along the semantic axis before and after controlling for generation artifacts."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Colors of the curve without controls and the curve with all controls.
MARGINAL_COLOR = "#9AA5AE"
ADJUSTED_COLOR = "#1F4E79"

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


def plot_adjusted_curve(marginal_curve, adjusted_curve):
    """
    Plot the accuracy change along the semantic axis without and with all generation controls.

    Parameters
    ----------
    marginal_curve, adjusted_curve : pandas.Series
        Mean accuracy change relative to the first rank bin, in percentage points, indexed by bin center: without
        controls, and with placement, scale, background and object-instance controls.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Both curves, starting at zero in the first bin.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.6), layout="constrained")
        ax.axhline(0, color="#CCCCCC", linewidth=0.8, zorder=1)
        ax.plot(marginal_curve.index, marginal_curve.values, color=MARGINAL_COLOR, linewidth=1.8, label="No controls", zorder=2)
        ax.plot(adjusted_curve.index, adjusted_curve.values, color=ADJUSTED_COLOR, linewidth=2.2, label="Same background and object", zorder=3)
        ax.set_xticks(np.arange(0, 1001, 200))
        ax.set_xlim(-10, 1010)
        ax.set_xlabel("Semantic dissimilarity rank")
        ax.set_ylabel("Accuracy change vs first bin (pp)")
        ax.set_title("Decline with and without controls", loc="left")
        ax.legend(loc="lower left", frameon=False)
    return fig
