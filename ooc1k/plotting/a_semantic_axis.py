"""Plot semantic dissimilarity matrices and local ImageNet class neighborhoods."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


# Figure layout and colors.
N_NEIGHBORS = 15
TREE_SHRINK = 0.45
TREE_LINE_WIDTH = 1.0
LABEL_PAD = 0.055

RDM_CMAP = "viridis"
FOCAL_COLOR = "#E05A2B"
LINK_COLOR = "#444444"
LEAF_COLOR = "#222222"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}


def find_class_index(name, class_labels):
    """Find an ImageNet class by exact match, then by first substring match."""

    # Match labels regardless of capitalization or spaces.
    query = name.lower().replace(" ", "_")
    labels = [label.lower() for label in class_labels]
    if query in labels:
        return labels.index(query)

    # Use the first partial match if no exact match exists.
    for index, label in enumerate(labels):
        if query in label:
            return index

    raise ValueError(f"'{name}' not found in the ImageNet class labels.")


def display_label(label):
    """Replace underscores and capitalize class names for display."""
    return label.replace("_", " ").title()


def get_neighborhood_submatrix(matrix, focal_idx):
    """Return the focal class and its nearest neighbors as a submatrix and indices."""

    # Select the nearest classes, excluding the focal class itself.
    distances = matrix[focal_idx].copy()
    distances[focal_idx] = np.inf
    neighbours = np.argsort(distances)[:N_NEIGHBORS]
    indices = [focal_idx, *neighbours.tolist()]

    # Remove small negative rounding errors and set self-distances to zero.
    submatrix = np.clip(matrix[np.ix_(indices, indices)], 0, None)
    np.fill_diagonal(submatrix, 0.0)
    return submatrix, indices


def draw_dendrogram(ax, matrix, focal_idx, class_labels):
    """Plot a neighborhood dendrogram on an existing axis."""

    # Select neighboring classes and prepare their labels.
    submatrix, indices = get_neighborhood_submatrix(matrix, focal_idx)
    labels = [display_label(class_labels[i]) for i in indices]
    focal_label = labels[0]

    # Cluster neighboring classes using Ward linkage.
    distances = squareform(submatrix)
    linkage_matrix = linkage(distances, method="ward")
    dendrogram(linkage_matrix, labels=labels, orientation="left", ax=ax, link_color_func=lambda _: LINK_COLOR, leaf_font_size=10)

    # Set branch widths and highlight the focal class.
    for collection in ax.collections:
        collection.set_linewidth(TREE_LINE_WIDTH)

    for label in ax.get_yticklabels():
        if label.get_text() == focal_label:
            label.set_color(FOCAL_COLOR)
            label.set_fontweight("bold")
            label.set_fontsize(10.5)
        else:
            label.set_color(LEAF_COLOR)
            label.set_fontsize(10)

    # Hide distance ticks and the frame, then add the class title.
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'"{focal_label}"', fontsize=12, fontweight="bold", color=FOCAL_COLOR, pad=6)


def _plot_matrix(ax, matrix, title, vmin=None, vmax=None):
    """Plot a dissimilarity matrix and colorbar on an existing axis."""

    # Display the matrix with its own colorbar.
    image = ax.imshow(matrix, aspect="equal", cmap=RDM_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
    colorbar.set_label("Dissimilarity", fontsize=10, color="#444444")
    colorbar.ax.tick_params(labelsize=9, colors="#444444")
    colorbar.outline.set_visible(False)

    # Label the panel and hide individual class ticks and the frame.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("ImageNet classes (sorted by index)", fontsize=10, color="#555555")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color="#222222")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_dissimilarity_matrices(matrices):
    """
    Plot dissimilarity matrices side by side with individual color scales.

    Parameters
    ----------
    matrices : dict
        Mapping from panel titles to dissimilarity matrices, in display order.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the matrix panels.
    """

    # Apply the figure style locally and plot one panel per matrix.
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.4), squeeze=False)
        for ax, (title, matrix) in zip(axes[0], matrices.items()):
            _plot_matrix(ax, matrix, title)

        fig.tight_layout()
    return fig


def plot_rdm_with_dendrograms(matrix, title, class_labels, focal_classes=("hen", "holster", "coral_reef"), show_matrix=True):
    """
    Plot a dissimilarity matrix alongside focal-class dendrograms.

    Parameters
    ----------
    matrix : numpy.ndarray
        Dissimilarity matrix in ImageNet class-index order, displayed from 0 to 1.
    title : str
        Full name of the semantic axis (e.g. "Consensus Visual-Semantic Axis"), used in the figure titles.
    class_labels : list of str
        ImageNet class labels in class-index order.
    focal_classes : tuple of str
        Class names whose local neighborhoods are shown.
    show_matrix : bool
        Draw the dissimilarity matrix left of the dendrograms; set to False for the dendrograms only.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the matrix (optional) and neighborhood dendrograms.
    """

    focal_indices = [find_class_index(name, class_labels) for name in focal_classes]
    with mpl.rc_context(STYLE):

        # Dendrograms only: one plain panel per focal class, with room for the class labels.
        if not show_matrix:
            fig, tree_axes = plt.subplots(1, len(focal_indices), figsize=(4.2 * len(focal_indices), 4.4), layout="constrained", squeeze=False)
            for ax, focal_idx in zip(tree_axes[0], focal_indices):
                draw_dendrogram(ax, matrix, focal_idx, class_labels)
            fig.suptitle(f"Semantic Neighborhoods in the {title}", fontsize=13, fontweight="bold", color="#222222")
            return fig

        # Create the matrix panel and one dendrogram panel per focal class.
        n_panels = 1 + len(focal_indices)
        fig = plt.figure(figsize=(6.5 * n_panels, 4.4))
        grid = fig.add_gridspec(1, n_panels, wspace=0.25, left=0.03, right=0.97, top=0.84, bottom=0.06)
        ax_matrix = fig.add_subplot(grid[0])
        tree_axes = [fig.add_subplot(grid[i]) for i in range(1, n_panels)]
        _plot_matrix(ax_matrix, matrix, title, vmin=0, vmax=1)

        for ax, focal_idx in zip(tree_axes, focal_indices):
            draw_dendrogram(ax, matrix, focal_idx, class_labels)

        # Narrow the tree panels and leave room for the class labels.
        cursor = ax_matrix.get_position().x1
        for ax in tree_axes:
            position = ax.get_position()
            width = position.width * TREE_SHRINK
            x0 = cursor + LABEL_PAD
            ax.set_position([x0, position.y0, width, position.height])
            cursor = x0 + width

        fig.suptitle(f"Semantic Neighborhoods in the {title}", fontsize=13, fontweight="bold", y=0.97, color="#222222")
    return fig
