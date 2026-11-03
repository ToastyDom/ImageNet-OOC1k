"""Measure how object and background evidence compete across the patches of a Vision Transformer."""

import io

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ooc1k.analysis.semantic_distance import load_class_hash_index

# Inclusive rank ranges: 7, 7 and 6 of the 20 subset bins.
DISTANCE_GROUPS = {"Low": (1, 350), "Mid": (351, 700), "High": (701, 1000)}

# The four outcome types of a composite, in plotting order.
OUTCOMES = ["Clean win", "Contested win", "Found-but-lost", "No detection"]

# Inclusive bands for the true class rank in the final CLS logits.
RANK_BANDS = {"Rank 1": (1, 1), "Rank 2-5": (2, 5), "Rank 6-20": (6, 20), "Rank > 20": (21, 1000)}


def load_subset(subset_path, class_index_path):
    """Load the mechanism subset with the ImageNet indices of the object and background classes."""
    subset = pd.read_csv(subset_path)
    hash_to_idx = load_class_hash_index(class_index_path)
    subset["object_class_idx"] = subset["object_class_hash"].map(hash_to_idx)
    subset["background_class_idx"] = subset["background_class_hash"].map(hash_to_idx)
    return subset


def load_logit_lens_results(path, subset):
    """
    Load the logit-lens results of one model, written by `run_logit_lens`.

    Parameters
    ----------
    path : str or pathlib.Path
        `{model}_logit_lens.h5` file.
    subset : pandas.DataFrame
        The mechanism subset from `load_subset`, in the order used for the extraction.

    Returns
    -------
    results : dict
        `composites` and `originals`, each a dict with `image_id`, `patch_predictions` (images, blocks, patches),
        `cls_predictions` (images, blocks) and `final_logits` (images, classes). The composites also get
        `correct`, whether the final prediction is the object class.
    """
    results = {}
    with h5py.File(path, "r") as f:
        for group in ["composites", "originals"]:
            results[group] = {name: f[group][name][:] for name in f[group]}
            results[group]["image_id"] = results[group]["image_id"].astype(str)

    # The rows follow the subset, so we can use the subset columns directly.
    composites = results["composites"]
    assert (
        composites["image_id"] == subset["filename"].str.replace(".jpg", "", regex=False)
    ).all(), "results and subset differ"
    composites["correct"] = (
        composites["cls_predictions"][:, -1] == subset["object_class_idx"].values
    )
    return results


def get_semantic_pull(patch_predictions, object_classes, background_classes, consensus):
    """
    Semantic pull of every patch: how much closer its predicted class is to the object than to the background.

    Parameters
    ----------
    patch_predictions : numpy.ndarray
        Predicted class of every patch, with the images along the first axis.
    object_classes, background_classes : numpy.ndarray
        Object and background class of every image.
    consensus : numpy.ndarray
        Consensus dissimilarity matrix (1000, 1000).

    Returns
    -------
    pull : numpy.ndarray
        Same shape as `patch_predictions`; positive values lean towards the object, negative ones towards the background.
    """
    shape = (-1,) + (1,) * (patch_predictions.ndim - 1)
    object_classes = np.asarray(object_classes).reshape(shape)
    background_classes = np.asarray(background_classes).reshape(shape)
    return (
        consensus[patch_predictions, background_classes]
        - consensus[patch_predictions, object_classes]
    )


def load_example_images(composites_h5, examples):
    """Read the selected composites as RGB images in example order."""
    with h5py.File(composites_h5, "r") as f:
        images = [
            Image.open(io.BytesIO(f["images"][i].tobytes())).convert("RGB")
            for i in examples["h5_index"]
        ]

    return images


def get_example_predictions(lens, transform, composites_h5, examples, consensus):
    """
    Run the logit lens on a few composites and compute the semantic pull of their patches.

    Parameters
    ----------
    lens : ooc1k.inference.logit_lens.LogitLens
        The model, from `load_logit_lens`.
    transform : callable
        Its evaluation transform.
    composites_h5 : str or pathlib.Path
        composites.h5 from notebook 02.
    examples : pandas.DataFrame
        Rows of the subset from `load_subset`.
    consensus : numpy.ndarray
        Consensus dissimilarity matrix.

    Returns
    -------
    examples : list of dict
        One dict per composite with its `image`, `patch_predictions` and `pull` (blocks, patches),
        `cls_predictions` (blocks), `object_class`, `background_class` and `rank`.
    """
    images = load_example_images(composites_h5, examples)

    device = next(lens.parameters()).device
    with torch.no_grad():
        outputs = lens(torch.stack([transform(image) for image in images]).to(device))
    patch_predictions = outputs["patch_predictions"].cpu().numpy()
    cls_predictions = outputs["cls_predictions"].cpu().numpy()
    pull = get_semantic_pull(
        patch_predictions, examples["object_class_idx"], examples["background_class_idx"], consensus
    )

    return [
        {
            "image": image,
            "patch_predictions": patch_predictions[i],
            "cls_predictions": cls_predictions[i],
            "pull": pull[i],
            "object_class": row.object_class_idx,
            "background_class": row.background_class_idx,
            "rank": row.dissimilarity_rank,
        }
        for i, (image, row) in enumerate(zip(images, examples.itertuples()))
    ]


def get_subset_accuracy(results_by_model, subset, bin_width=50):
    """
    Top-1 accuracy of every model in bins of the dissimilarity rank.

    Parameters
    ----------
    results_by_model : dict
        Model name -> output of `load_logit_lens_results`.
    subset : pandas.DataFrame
        The mechanism subset.
    bin_width : int
        Ranks per bin.

    Returns
    -------
    curves : pandas.DataFrame
        Accuracy with one row per bin (indexed by the bin centre) and one column per model.
    """
    bins = (subset["dissimilarity_rank"].values - 1) // bin_width
    centers = bins * bin_width + (bin_width + 1) / 2
    curves = {
        model: pd.Series(results["composites"]["correct"]).groupby(centers).mean()
        for model, results in results_by_model.items()
    }
    return pd.DataFrame(curves)


def get_realm_shares(pull, tau):
    """Mean share of object-realm (pull > tau) and background-realm (pull < -tau) patches."""
    return (pull > tau).mean(), (pull < -tau).mean()


def get_composite_realm_balance(
    composites, object_classes, background_classes, ranks, consensus, tau, blocks
):
    """Measure realm balance for correct and incorrect composites in each distance group."""
    balance = {}
    for outcome, correct in [("Correct", True), ("Incorrect", False)]:
        for group, (low, high) in DISTANCE_GROUPS.items():
            mask = (composites["correct"] == correct) & (ranks >= low) & (ranks <= high)
            values = []
            for block in blocks:
                pull = get_semantic_pull(
                    composites["patch_predictions"][mask, block],
                    object_classes[mask],
                    background_classes[mask],
                    consensus,
                )
                share_object, share_background = get_realm_shares(pull, tau)
                values.append(share_object / (share_object + share_background))
            balance[(outcome, group)] = values

    return balance


def get_original_realm_balance(originals, subset, consensus, tau, blocks):
    """Measure realm balance for correct originals against their nearest other class."""
    id_to_class = dict(zip(subset["background_id"], subset["background_class_idx"]))
    original_classes = np.array([id_to_class[image_id] for image_id in originals["image_id"]])
    correct = originals["cls_predictions"][:, -1] == original_classes
    distances = consensus[original_classes[correct]].copy()
    distances[np.arange(correct.sum()), original_classes[correct]] = np.inf
    competitors = distances.argmin(axis=1)
    values = []
    for block in blocks:
        pull = get_semantic_pull(
            originals["patch_predictions"][correct, block],
            original_classes[correct],
            competitors,
            consensus,
        )
        share_object, share_background = get_realm_shares(pull, tau)
        values.append(share_object / (share_object + share_background))
    return values


def get_realm_balance(results, subset, consensus, tau, blocks):
    """
    Share of the decided patches that favour the object, per block, outcome and distance group.

    A patch is decided if its semantic pull is larger than `tau` in either direction. For every group of
    images, we take the mean share of object-realm and background-realm patches and report
    object / (object + background). As a reference, we compute the same for the correctly classified
    original images, with the nearest other class as the competitor (originals have no background class).

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    subset : pandas.DataFrame
        The mechanism subset.
    consensus : numpy.ndarray
        Consensus dissimilarity matrix.
    tau : float
        Threshold of the undecided zone.
    blocks : list of int
        Blocks to evaluate.

    Returns
    -------
    balance : pandas.DataFrame
        One row per block and one column per (outcome, distance group), plus ("Originals", "").
    """
    composites, originals = results["composites"], results["originals"]
    object_classes, background_classes = (
        subset["object_class_idx"].values,
        subset["background_class_idx"].values,
    )
    ranks = subset["dissimilarity_rank"].values

    # Composites, split by correctness and distance group.
    balance = get_composite_realm_balance(
        composites, object_classes, background_classes, ranks, consensus, tau, blocks
    )

    # Correctly classified originals, against the nearest other class.
    values = get_original_realm_balance(originals, subset, consensus, tau, blocks)

    balance[("Originals", "")] = values

    return pd.DataFrame(balance, index=pd.Index(blocks, name="block"))


def get_distance_group(ranks):
    """Low, mid or high distance group of every dissimilarity rank."""
    edges = [0] + [high for _, high in DISTANCE_GROUPS.values()]
    return pd.cut(ranks, bins=edges, labels=list(DISTANCE_GROUPS))


def get_outcomes(results, subset, consensus, tau, min_object_patches=20, clean_balance=0.75):
    """
    Sort every composite into one of four outcome types, from its patches and its prediction after the last block.

    A composite is detected if at least `min_object_patches` patches predict exactly the object class. Correct
    composites are clean wins if they are detected and their realm balance is above `clean_balance`, and contested
    wins otherwise. Incorrect composites are found-but-lost if they are detected, and no detection otherwise.

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    subset : pandas.DataFrame
        The mechanism subset.
    consensus : numpy.ndarray
        Consensus dissimilarity matrix.
    tau : float
        Threshold of the undecided zone.
    min_object_patches : int
        Exact object patches needed to count as detected.
    clean_balance : float
        Realm balance above which a correct composite is a clean win.

    Returns
    -------
    outcomes : pandas.Series
        Outcome type of every composite, in the order of the subset.
    """
    composites = results["composites"]
    object_classes = subset["object_class_idx"].values
    last_block = composites["patch_predictions"][:, -1]

    # Exact object patches and realm balance of every composite; composites without decided patches count as a tie.
    object_patches = (last_block == object_classes[:, None]).sum(axis=1)
    pull = get_semantic_pull(
        last_block, object_classes, subset["background_class_idx"].values, consensus
    )
    n_object, n_background = (pull > tau).sum(axis=1), (pull < -tau).sum(axis=1)
    balance = np.where(
        n_object + n_background > 0, n_object / np.maximum(n_object + n_background, 1), 0.5
    )

    # The first matching condition decides.
    detected = object_patches >= min_object_patches
    correct = composites["correct"]
    outcomes = np.select(
        [correct & detected & (balance > clean_balance), correct, detected],
        OUTCOMES[:3],
        default=OUTCOMES[3],
    )
    return pd.Series(
        pd.Categorical(outcomes, categories=OUTCOMES), index=subset.index, name="outcome"
    )


def get_outcome_shares_by_model(outcomes_by_model):
    """
    Share of every outcome type for every model.

    Parameters
    ----------
    outcomes_by_model : dict
        Model name -> output of `get_outcomes`.

    Returns
    -------
    shares : pandas.DataFrame
        One row per model and one column per outcome type.
    """
    shares = {
        model: outcomes.value_counts(normalize=True)
        for model, outcomes in outcomes_by_model.items()
    }
    return pd.DataFrame(shares).T[OUTCOMES]


def get_outcome_shares_by_distance(outcomes, subset):
    """
    Share of every outcome type in the low, mid and high distance groups.

    Parameters
    ----------
    outcomes : pandas.Series
        Output of `get_outcomes`.
    subset : pandas.DataFrame
        The mechanism subset.

    Returns
    -------
    shares : pandas.DataFrame
        One row per distance group and one column per outcome type.
    """
    groups = get_distance_group(subset["dissimilarity_rank"]).rename("distance")
    shares = pd.crosstab(groups, outcomes, normalize="index")
    return shares.reindex(index=list(DISTANCE_GROUPS), columns=OUTCOMES, fill_value=0)


def get_object_expansion(results, subset, outcomes, blocks):
    """
    Mean number of patches that predict exactly the object class, per block and outcome type.

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    subset : pandas.DataFrame
        The mechanism subset.
    outcomes : pandas.Series
        Output of `get_outcomes`.
    blocks : list of int
        Blocks to evaluate.

    Returns
    -------
    expansion : pandas.DataFrame
        One row per block and one column per outcome type.
    """
    patch_predictions = results["composites"]["patch_predictions"][:, blocks]
    object_patches = (patch_predictions == subset["object_class_idx"].values[:, None, None]).sum(
        axis=2
    )
    expansion = (
        pd.DataFrame(object_patches, columns=blocks, index=subset.index)
        .groupby(outcomes, observed=False)
        .mean()
    )
    return expansion.T.rename_axis("block")[OUTCOMES]


def get_found_but_lost_aggregation(results, subset, outcomes, consensus, tau, blocks):
    """
    Patch evidence and CLS prediction of the found-but-lost composites across blocks.

    The realm balance measures how much the decided patches favour the object. The classification proximity,
    1 - D(CLS prediction, object class), measures how close the CLS prediction is to the object class.
    As a reference, we also compute the classification proximity of the correctly classified composites.

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    subset : pandas.DataFrame
        The mechanism subset.
    outcomes : pandas.Series
        Output of `get_outcomes`.
    consensus : numpy.ndarray
        Consensus dissimilarity matrix.
    tau : float
        Threshold of the undecided zone.
    blocks : list of int
        Blocks to evaluate.

    Returns
    -------
    aggregation : pandas.DataFrame
        One row per block, with the columns "Realm balance", "CLS proximity" and "CLS proximity (correct)".
    """
    composites = results["composites"]
    object_classes, background_classes = (
        subset["object_class_idx"].values,
        subset["background_class_idx"].values,
    )
    found_but_lost = (outcomes == "Found-but-lost").values

    rows = []
    for block in blocks:
        pull = get_semantic_pull(
            composites["patch_predictions"][found_but_lost, block],
            object_classes[found_but_lost],
            background_classes[found_but_lost],
            consensus,
        )
        share_object, share_background = get_realm_shares(pull, tau)
        proximity = 1 - consensus[composites["cls_predictions"][:, block], object_classes]
        rows.append(
            {
                "Realm balance": share_object / (share_object + share_background),
                "CLS proximity": float(proximity[found_but_lost].mean()),
                "CLS proximity (correct)": float(proximity[composites["correct"]].mean()),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(blocks, name="block"))


def get_true_class_rank(results, subset):
    """
    Rank of the object class in the final CLS logits of every composite.

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    subset : pandas.DataFrame
        The mechanism subset.

    Returns
    -------
    ranks : pandas.Series
        1 if the object class is the prediction, 2 if it is the runner-up, and so on.
    """
    composites = results["composites"]
    logits = composites["final_logits"].astype(np.float32)

    # The logits are stored as float16 and can tie, so we make sure the prediction always ranks first.
    logits[np.arange(len(logits)), composites["cls_predictions"][:, -1]] = np.inf
    object_logits = logits[np.arange(len(logits)), subset["object_class_idx"].values]
    return pd.Series(
        (logits > object_logits[:, None]).sum(axis=1) + 1, index=subset.index, name="rank"
    )


def get_rank_shares(ranks, outcomes):
    """
    Share of the rank bands of the true class for every outcome type.

    Parameters
    ----------
    ranks : pandas.Series
        Output of `get_true_class_rank`.
    outcomes : pandas.Series
        Output of `get_outcomes`.

    Returns
    -------
    shares : pandas.DataFrame
        One row per outcome type and one column per rank band.
    """
    bands = pd.cut(
        ranks, bins=[0] + [high for _, high in RANK_BANDS.values()], labels=list(RANK_BANDS)
    )
    shares = pd.crosstab(outcomes, bands, normalize="index")
    return shares.reindex(index=OUTCOMES, columns=list(RANK_BANDS), fill_value=0)


def get_spatial_spread(patch_predictions, classes):
    """Count matching patches and measure their mean distance from their grid centre."""
    grid_size = int(np.sqrt(patch_predictions.shape[1]))
    rows, cols = np.divmod(np.arange(patch_predictions.shape[1]), grid_size)
    match = patch_predictions == classes[:, None]
    count = match.sum(axis=1)
    center_row = (match * rows).sum(axis=1) / np.maximum(count, 1)
    center_col = (match * cols).sum(axis=1) / np.maximum(count, 1)
    distance = np.sqrt((rows - center_row[:, None]) ** 2 + (cols - center_col[:, None]) ** 2)
    return count, (distance * match).sum(axis=1) / np.maximum(count, 1)


def get_compactness(results, outcomes, blocks):
    """
    Share of composites in which the runner-up class of the CLS token is spatially more compact than its prediction.

    The prediction and the runner-up are the two largest final CLS logits. For both classes we take the patches
    that predict them in a block and measure their mean distance from their centre; the smaller distance is the
    more compact class. Only composites where both classes are predicted by at least two patches count.

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    outcomes : pandas.Series
        Output of `get_outcomes`.
    blocks : list of int
        Blocks to evaluate.

    Returns
    -------
    compactness : pandas.DataFrame
        One row per block and one column per outcome type.
    """
    composites = results["composites"]

    # The prediction and the class with the second-largest logit.
    prediction = composites["cls_predictions"][:, -1]
    logits = composites["final_logits"].astype(np.float32)
    logits[np.arange(len(logits)), prediction] = -np.inf
    runner_up = logits.argmax(axis=1)

    compactness = {}
    for block in blocks:
        count_prediction, spread_prediction = get_spatial_spread(
            composites["patch_predictions"][:, block], prediction
        )
        count_runner_up, spread_runner_up = get_spatial_spread(
            composites["patch_predictions"][:, block], runner_up
        )
        valid = (count_prediction >= 2) & (count_runner_up >= 2)
        runner_up_compact = pd.Series(spread_runner_up < spread_prediction, index=outcomes.index)[
            valid
        ]
        compactness[block] = runner_up_compact.groupby(outcomes[valid], observed=False).mean()
    return pd.DataFrame(compactness).T.rename_axis("block")[OUTCOMES]


def get_rerank_signals(results, consensus, block):
    """
    The three signals of the spatial re-ranking for every composite: logit gap, compactness advantage and distance.

    The prediction (top-1) and the runner-up (top-2) are the two largest final CLS logits. The logit gap is the
    difference of their logits. The compactness advantage is the spatial spread of the top-1 class minus that of
    the top-2 class in `block`, so positive values mean the runner-up forms the tighter group of patches. A class
    that no patch predicts gets the largest possible spread, the diagonal of the patch grid. The distance is the
    consensus dissimilarity between the two classes.

    Parameters
    ----------
    results : dict
        Output of `load_logit_lens_results`.
    consensus : numpy.ndarray
        Consensus dissimilarity matrix.
    block : int
        Block in which the compactness is measured, e.g. -2 for the penultimate block.

    Returns
    -------
    signals : pandas.DataFrame
        One row per composite with the columns "top1", "top2", "gap", "advantage" and "distance".
    """
    composites = results["composites"]
    rows = np.arange(len(composites["final_logits"]))

    # The prediction and the class with the second-largest logit.
    top1 = composites["cls_predictions"][:, -1]
    logits = composites["final_logits"].astype(np.float32)
    top1_logits = logits[rows, top1]
    logits[rows, top1] = -np.inf
    top2 = logits.argmax(axis=1)
    gap = top1_logits - logits[rows, top2]

    # Spread of both classes in the gate block; classes without any patch count as maximally spread out.
    patch_predictions = composites["patch_predictions"][:, block]
    max_spread = np.sqrt(2) * (np.sqrt(patch_predictions.shape[1]) - 1)
    count_top1, spread_top1 = get_spatial_spread(patch_predictions, top1)
    count_top2, spread_top2 = get_spatial_spread(patch_predictions, top2)
    spread_top1 = np.where(count_top1 > 0, spread_top1, max_spread)
    spread_top2 = np.where(count_top2 > 0, spread_top2, max_spread)

    return pd.DataFrame(
        {
            "top1": top1,
            "top2": top2,
            "gap": gap,
            "advantage": spread_top1 - spread_top2,
            "distance": consensus[top1, top2],
        }
    )


def get_rerank_accuracy(correct_before, correct_after, subset, bin_width=50):
    """
    Top-1 accuracy before and after re-ranking in bins of the dissimilarity rank.

    Parameters
    ----------
    correct_before, correct_after : array-like
        Whether every composite is classified correctly before and after re-ranking.
    subset : pandas.DataFrame
        The mechanism subset.
    bin_width : int
        Ranks per bin.

    Returns
    -------
    curves : pandas.DataFrame
        One row per bin (indexed by the bin centre), with the columns "Before" and "After".
    """
    bins = (subset["dissimilarity_rank"].values - 1) // bin_width
    centers = bins * bin_width + (bin_width + 1) / 2
    correct = pd.DataFrame(
        {"Before": np.asarray(correct_before), "After": np.asarray(correct_after)}
    )
    return correct.groupby(centers).mean()
