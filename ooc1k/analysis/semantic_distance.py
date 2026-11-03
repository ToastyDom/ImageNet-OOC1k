"""Measure recognition accuracy along the semantic axis from the composite predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# Trend groups use (lower, upper] intervals of the rank–accuracy correlation.
TREND_GROUPS = {
    "Strong −": (-np.inf, -0.50),
    "Mild −": (-0.50, 0.00),
    "Mild +": (0.00, 0.50),
    "Strong +": (0.50, np.inf),
}

METADATA_COLUMNS = [
    "filename",
    "background_id",
    "background_category",
    "background_class_hash",
    "object_class_hash",
    "dissimilarity_rank",
]


# Load predictions and match them to the composite metadata.


def load_class_hash_index(class_index_path):
    """Map WordNet IDs (e.g. n01440764) to ImageNet class indices."""
    with open(class_index_path) as f:
        return {wnid: int(idx) for idx, (wnid, _) in json.load(f).items()}


def find_prediction_csv(predictions_dir, model, condition):
    """Return the newest prediction CSV of a model for one image condition."""
    return sorted(Path(predictions_dir).glob(f"{model}_{condition}_*.csv"))[-1]


def parse_guesses(top5_indices):
    """Turn the stored top-5 class lists into an array with one row per image."""
    return np.array([json.loads(top5) for top5 in top5_indices])


def is_correct(guesses, true_classes, top_k):
    """Whether the true class is among the first top_k guesses of every image."""
    return (guesses[:, :top_k] == np.asarray(true_classes)[:, None]).any(axis=1)


def load_composite_metadata(metadata_path, class_index_path):
    """Read the composite metadata, image IDs and object class indices."""
    # Read the metadata once; repeated strings are stored as categories to save memory.
    metadata = pd.read_csv(metadata_path, usecols=METADATA_COLUMNS)
    image_ids = metadata.pop("filename").str.replace(".jpg", "", regex=False).values
    for column in [
        "background_id",
        "background_category",
        "background_class_hash",
        "object_class_hash",
    ]:
        metadata[column] = metadata[column].astype("category")

    # The true class is the class of the placed object.
    hash_to_idx = load_class_hash_index(class_index_path)
    true_classes = metadata["object_class_hash"].map(hash_to_idx).astype(int).values

    return metadata, image_ids, true_classes


def load_composite_predictions(predictions_dir, metadata_path, class_index_path, models):
    """
    Combine the composite predictions of several models with the composite metadata.

    Parameters
    ----------
    predictions_dir : str or pathlib.Path
        Folder with the prediction CSVs of notebook 03.
    metadata_path : str or pathlib.Path
        Composite metadata.csv, in the same order as composites.h5.
    class_index_path : str or pathlib.Path
        JSON mapping ImageNet indices to class hashes and labels.
    models : list of str
        Model names as used in the CSV file names.

    Returns
    -------
    composites : pandas.DataFrame
        One row per model and composite, with the background, the dissimilarity rank and whether
        the prediction was correct (top-1 and top-5).
    """

    metadata, image_ids, true_classes = load_composite_metadata(metadata_path, class_index_path)

    frames = []
    for model in models:
        predictions = pd.read_csv(
            find_prediction_csv(predictions_dir, model, "composite_ooc"),
            usecols=["image_id", "top5_indices"],
        )

        # Predictions and metadata both follow the order of composites.h5, so rows match one to one.
        if (
            len(predictions) != len(metadata)
            or not (predictions["image_id"].values == image_ids).all()
        ):
            raise ValueError(
                f"The composite predictions of {model} are not in the order of {metadata_path}."
            )

        frame = metadata.copy()
        frame["model"] = model
        guesses = parse_guesses(predictions["top5_indices"])
        frame["correct"] = is_correct(guesses, true_classes, top_k=1)
        frame["correct_top5"] = is_correct(guesses, true_classes, top_k=5)
        frames.append(frame)

    composites = pd.concat(frames, ignore_index=True)
    composites["model"] = composites["model"].astype("category")
    return composites


def load_composite_correctness(predictions_dir, metadata_path, class_index_path, models):
    """
    Top-1 correctness of many models on every composite, with one column per model.

    Much smaller than `load_composite_predictions`, which is useful when we look at all models at once.

    Parameters
    ----------
    predictions_dir : str or pathlib.Path
        Folder with the prediction CSVs of notebook 03.
    metadata_path : str or pathlib.Path
        Composite metadata.csv, in the same order as composites.h5.
    class_index_path : str or pathlib.Path
        JSON mapping ImageNet indices to class hashes and labels.
    models : list of str
        Model names as used in the CSV file names.

    Returns
    -------
    correct : pandas.DataFrame
        One row per composite, in the order of metadata.csv, and one boolean column per model.
    """
    _, image_ids, true_classes = load_composite_metadata(metadata_path, class_index_path)

    correct = {}
    for model in models:
        predictions = pd.read_csv(find_prediction_csv(predictions_dir, model, "composite_ooc"), usecols=["image_id", "pred_class_idx"])

        # Predictions and metadata both follow the order of composites.h5, so rows match one to one.
        if len(predictions) != len(image_ids) or not (predictions["image_id"].values == image_ids).all():
            raise ValueError(f"The composite predictions of {model} are not in the order of {metadata_path}.")
        correct[model] = predictions["pred_class_idx"].values == true_classes
    return pd.DataFrame(correct)


# Accuracy along the semantic axis.


def get_original_accuracy(predictions_dir, models):
    """Top-1 and top-5 accuracy of every model on the original object-library images."""
    rows = {}
    for model in models:
        predictions = pd.read_csv(find_prediction_csv(predictions_dir, model, "original_library"))
        guesses = parse_guesses(predictions["top5_indices"])
        rows[model] = {
            "correct": is_correct(guesses, predictions["true_class_idx"], top_k=1).mean(),
            "correct_top5": is_correct(guesses, predictions["true_class_idx"], top_k=5).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def get_bin_centers(ranks, bin_width):
    """Map ranks to bin centers, starting with ranks 1 through bin_width."""
    bins = (np.asarray(ranks) - 1) // bin_width
    return 1 + bins * bin_width + bin_width / 2


def get_accuracy_curves(composites, bin_width=25):
    """Top-1 and top-5 accuracy of every model in bins of dissimilarity rank."""
    binned = composites.assign(
        bin_center=get_bin_centers(composites["dissimilarity_rank"], bin_width)
    )
    return (
        binned.groupby(["model", "bin_center"], observed=True)[["correct", "correct_top5"]]
        .mean()
        .reset_index()
    )


def get_accuracy_drop(curves):
    """Accuracy in the first and last rank bin of every model and the drop between them."""
    curves = curves.sort_values(["model", "bin_center"])
    first = curves.groupby("model", observed=True)[["correct", "correct_top5"]].first()
    last = curves.groupby("model", observed=True)[["correct", "correct_top5"]].last()
    return pd.concat({"first_bin": first, "last_bin": last, "drop": first - last}, axis=1)


def fit_logistic_regressions(composites):
    """
    Fit P(correct) ~ dissimilarity rank for every model on the individual composites.

    Parameters
    ----------
    composites : pandas.DataFrame
        Output of `load_composite_predictions`.

    Returns
    -------
    regressions : pandas.DataFrame
        One row per model with the slope (change in log-odds per rank), its 95% confidence
        interval and p-value.
    """
    rows = {}
    for model, group in composites.groupby("model", observed=True):
        # Logistic regression of top-1 correctness on the rank, with an intercept.
        X = sm.add_constant(group["dissimilarity_rank"].values.astype(float))
        fit = sm.GLM(group["correct"].values.astype(float), X, family=sm.families.Binomial()).fit()
        ci = fit.conf_int(alpha=0.05)
        rows[model] = {
            "slope": fit.params[1],
            "ci_low": ci[1, 0],
            "ci_high": ci[1, 1],
            "p_value": fit.pvalues[1],
            "n_images": len(group),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def get_background_correlations(accuracy, min_bins):
    """Correlate rank with binned accuracy for backgrounds with enough bins."""
    rows = {}
    for background_id, group in accuracy.groupby("background_id", observed=True):
        if len(group) >= min_bins:
            rows[background_id] = {"rho": spearmanr(group["bin_center"], group["correct"])[0]}
    trends = pd.DataFrame.from_dict(rows, orient="index")

    return trends


def get_background_trends(composites, models, bin_width=50, min_bins=5):
    """
    Measure for every background whether accuracy falls or rises with semantic distance.

    Parameters
    ----------
    composites : pandas.DataFrame
        Output of `load_composite_predictions`.
    models : list of str
        Models whose composites are pooled.
    bin_width : int
        Width of the rank bins.
    min_bins : int
        Backgrounds with fewer filled bins are left out.

    Returns
    -------
    trends : pandas.DataFrame
        One row per background with its category, the Spearman correlation between rank and
        accuracy (rho) and its trend group.
    """

    # Accuracy per background and rank bin, pooled over the selected models.
    data = composites[composites["model"].isin(models)]
    data = data.assign(bin_center=get_bin_centers(data["dissimilarity_rank"], bin_width))
    accuracy = (
        data.groupby(["background_id", "bin_center"], observed=True)["correct"].mean().reset_index()
    )

    # One Spearman correlation between bin and accuracy per background.
    trends = get_background_correlations(accuracy, min_bins)

    # Add the background category and sort every background into a trend group.
    categories = (
        data.drop_duplicates("background_id")
        .set_index("background_id")["background_category"]
        .astype(str)
    )
    trends["background_category"] = categories.reindex(trends.index).values
    trends["group"] = None
    for group, (lower, upper) in TREND_GROUPS.items():
        trends.loc[(trends["rho"] > lower) & (trends["rho"] <= upper), "group"] = group
    return trends


def get_trend_group_curves(composites, trends, models, bin_width=50):
    """Accuracy curves of every trend group and of all backgrounds, as mean and SEM across models."""
    data = composites[composites["model"].isin(models)]
    data = data.assign(bin_center=get_bin_centers(data["dissimilarity_rank"], bin_width))
    groups = trends["group"].to_dict()

    curves = []
    for group in ["All backgrounds", *TREND_GROUPS]:
        rows = (
            data
            if group == "All backgrounds"
            else data[data["background_id"].astype(str).map(groups) == group]
        )

        # Accuracy per model and bin, then mean and standard error across models.
        per_model = rows.groupby(["bin_center", "model"], observed=True)["correct"].mean()
        summary = per_model.groupby("bin_center").agg(accuracy="mean", sem="sem").reset_index()
        curves.append(summary.assign(group=group))
    return pd.concat(curves, ignore_index=True)


# Reorder the same predictions using alternative semantic axes.


def get_rank_table(dissimilarity):
    """Rank of every class in every row of a dissimilarity matrix; the class itself has rank 1."""
    return np.argsort(np.argsort(dissimilarity, axis=1), axis=1) + 1


def rerank_composites(composites, dissimilarity, hash_to_idx):
    """Replace the consensus ranks by the ranks of the object classes under another dissimilarity matrix."""
    ranks = get_rank_table(dissimilarity)
    background_idx = composites["background_class_hash"].map(hash_to_idx).astype(int).values
    object_idx = composites["object_class_hash"].map(hash_to_idx).astype(int).values
    return composites.assign(dissimilarity_rank=ranks[background_idx, object_idx])


def get_model_curves(composites, bin_width=25):
    """Top-1 accuracy per rank bin, with one column per model."""
    curves = get_accuracy_curves(composites, bin_width)
    return curves.pivot(index="bin_center", columns="model", values="correct")


def get_wordnet_hops(wordnet):
    """Recover hop counts from rescaled WordNet distances; leave self-pairs as NaN."""
    values = np.unique(np.round(wordnet[~np.eye(len(wordnet), dtype=bool)], 6))
    step = np.min(np.diff(values)[np.diff(values) > 1e-6])
    hops = np.round(np.round(wordnet, 6) / step) - np.round(values.min() / step) + 1
    np.fill_diagonal(hops, np.nan)
    return hops


def get_wordnet_curves(composites, wordnet, hash_to_idx, min_images=500):
    """Top-1 accuracy per model and WordNet hop count, excluding rare hop counts."""
    hops = get_wordnet_hops(wordnet)
    background_idx = composites["background_class_hash"].map(hash_to_idx).astype(int).values
    object_idx = composites["object_class_hash"].map(hash_to_idx).astype(int).values
    data = composites.assign(hop=hops[background_idx, object_idx]).dropna(subset=["hop"])
    data["hop"] = data["hop"].astype(int)

    # Keep hop counts with enough composites, summed over all models.
    counts = data.groupby("hop").size()
    data = data[data["hop"].isin(counts[counts >= min_images].index)]
    return data.groupby(["hop", "model"], observed=True)["correct"].mean().unstack("model")
