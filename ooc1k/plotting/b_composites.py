"""Plot generated composites of one background across the semantic axis."""

import io

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


# Panel size and figure style.
PANEL_SIZE = 3.0

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def plot_composite_examples(h5_path, metadata, background_id, ranks=(1, 50, 250, 500, 750, 950)):
    """
    Plot composites of one background at selected dissimilarity ranks.

    Parameters
    ----------
    h5_path : str or pathlib.Path
        HDF5 file containing the composite images.
    metadata : pandas.DataFrame
        Composite metadata in file order, with indices matching HDF5 image rows.
    background_id : str
        Background identifier to display.
    ranks : tuple of int
        Requested ranks, each matched to the closest available rank.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing one composite per requested rank.
    """

    # Select all composites generated for the requested background.
    rows = metadata[metadata["background_id"] == background_id]
    if rows.empty:
        raise ValueError(f"No composites found for background '{background_id}'.")

    # Find the closest available rank and select its first object instance.
    examples = []
    for rank in ranks:
        closest = (rows["dissimilarity_rank"] - rank).abs().idxmin()
        closest_rank = rows.loc[closest, "dissimilarity_rank"]
        examples.append(rows[rows["dissimilarity_rank"] == closest_rank].iloc[0])

    # Open the image file and apply the figure style locally.
    with mpl.rc_context(STYLE), h5py.File(h5_path, "r") as f:
        fig, axes = plt.subplots(1, len(examples), figsize=(PANEL_SIZE * len(examples), PANEL_SIZE + 0.4), squeeze=False)

        # Decode each composite and label it with its actual rank and class.
        for ax, row in zip(axes[0], examples):
            img_bytes = f["images"][row.name]
            ax.imshow(Image.open(io.BytesIO(img_bytes.tobytes())).convert("RGB"))
            label = row["object_class_label"].replace("_", " ")
            ax.set_title(f"Rank {row['dissimilarity_rank']}\n{label}", fontsize=10)
            ax.axis("off")

        # Add the background class as the figure title and adjust spacing.
        background_label = rows["background_class_label"].iloc[0].replace("_", " ")
        fig.suptitle(f"Background: {background_label}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
