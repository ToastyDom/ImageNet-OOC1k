"""Build alternative visual-semantic axes and measure the accuracy decline along each of them."""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ooc1k.analysis.semantic_distance import get_rank_table


# Accuracy along any axis.


def get_axis_ranks(dissimilarity, background_idx, object_idx):
    """Rank of every composite's object class among the neighbours of its background class, under one axis."""
    return get_rank_table(dissimilarity)[background_idx, object_idx]


def get_axis_curves(ranks, correct, bin_width=25):
    """
    Top-1 accuracy of every model in bins of dissimilarity rank.

    Parameters
    ----------
    ranks : numpy.ndarray
        Rank of every composite under the axis, from 1 to 1000.
    correct : pandas.DataFrame
        Output of `load_composite_correctness`, one boolean column per model.
    bin_width : int
        Number of ranks per bin.

    Returns
    -------
    curves : pandas.DataFrame
        One row per bin (indexed by its center rank) and one column per model.
    """
    bins = (np.asarray(ranks) - 1) // bin_width
    counts = np.bincount(bins)
    curves = {model: np.bincount(bins, weights=correct[model].values) / counts for model in correct.columns}
    return pd.DataFrame(curves, index=pd.Index(1 + np.arange(len(counts)) * bin_width + bin_width / 2, name="bin_center"))


def get_hop_curves(hops, correct, min_composites=1000):
    """Top-1 accuracy of every model per WordNet hop count, leaving out self-pairs and rare hop counts."""
    known = ~np.isnan(hops)
    curves = correct[known].groupby(hops[known].astype(int)).mean()
    counts = pd.Series(hops[known].astype(int)).value_counts()
    return curves[counts.reindex(curves.index) >= min_composites].rename_axis("hop")


def get_accuracy_drops(curves):
    """Drop in accuracy from the first to the last bin for every model, in percentage points."""
    return 100 * (curves.iloc[0] - curves.iloc[-1])


def get_curve_summary(curve):
    """First bin, lowest bin, last bin, decline and monotonicity of one accuracy curve."""
    return {
        "first bin (%)": 100 * curve.iloc[0],
        "lowest bin (%)": 100 * curve.min(),
        "lowest at bin": int(np.argmin(curve.values)) + 1,
        "last bin (%)": 100 * curve.iloc[-1],
        "decline (pp)": 100 * (curve.iloc[0] - curve.iloc[-1]),
        "monotonicity": spearmanr(np.arange(len(curve)), curve.values)[0],
    }


def get_axis_agreement(D_a, D_b):
    """Spearman correlation between two dissimilarity matrices, over all class pairs."""
    upper = np.triu_indices(len(D_a), k=1)
    return spearmanr(D_a[upper], D_b[upper])[0]


def permutation_test(values_a, values_b, n_permutations=20000, seed=42):
    """Two-sided permutation test on the difference between the means of two groups."""
    values_a, values_b = np.asarray(values_a), np.asarray(values_b)
    observed = values_a.mean() - values_b.mean()
    pooled = np.concatenate([values_a, values_b])
    rng = np.random.default_rng(seed)

    differences = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        differences[i] = shuffled[:len(values_a)].mean() - shuffled[len(values_a):].mean()
    return observed, np.mean(np.abs(differences) >= abs(observed))


# Building new axes.


def get_cosine_dissimilarity(rows):
    """Cosine dissimilarity between the rows of a (classes x dimensions) matrix, symmetric with zero diagonal (Eq. 1)."""
    rows = torch.nn.functional.normalize(torch.as_tensor(rows, dtype=torch.float64), dim=1)
    D = (1 - rows @ rows.T).numpy()
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def build_consensus(matrices):
    """Z-score the off-diagonal of every matrix, average, symmetrise and min-max rescale to [0, 1] (Eq. 2, as in notebook 01)."""
    mask = ~np.eye(matrices[0].shape[0], dtype=bool)

    z_matrices = []
    for D in matrices:
        Z = D.copy()
        Z[mask] = (D[mask] - D[mask].mean()) / D[mask].std()
        z_matrices.append(Z)

    consensus = np.mean(np.stack(z_matrices), axis=0)
    consensus = (consensus + consensus.T) / 2
    consensus[mask] = (consensus[mask] - consensus[mask].min()) / (consensus[mask].max() - consensus[mask].min())
    np.fill_diagonal(consensus, 0.0)
    return consensus


def fit_linear_probe(features, class_idx, is_train, epochs=20, lr=1e-3, weight_decay=1e-4, batch_size=1024, seed=42, device="cuda"):
    """
    Train a linear classifier on frozen features.

    Parameters
    ----------
    features : numpy.ndarray
        Images x feature dimensions.
    class_idx : numpy.ndarray
        ImageNet class index of every image.
    is_train : numpy.ndarray
        Boolean mask of the training images; all others are held out to measure accuracy.
    epochs, lr, weight_decay, batch_size : float or int
        AdamW with a cosine learning-rate schedule.
    seed : int
        Seed of the weight initialisation and the batch order.
    device : str
        Torch device.

    Returns
    -------
    weights : numpy.ndarray
        The probe weights, one row per class (1000 x feature dimensions).
    accuracy : float
        Top-1 accuracy on the held-out images.
    """
    torch.manual_seed(seed)
    X_train = torch.as_tensor(features[is_train], dtype=torch.float32)
    y_train = torch.as_tensor(class_idx[is_train], dtype=torch.long)
    X_test = torch.as_tensor(features[~is_train], dtype=torch.float32, device=device)
    y_test = torch.as_tensor(class_idx[~is_train], dtype=torch.long, device=device)

    probe = torch.nn.Linear(X_train.shape[1], 1000).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Plain minibatch training on the frozen features.
    for _ in range(epochs):
        order = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(X_train[batch].to(device)), y_train[batch].to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

    with torch.no_grad():
        accuracy = (probe(X_test).argmax(dim=1) == y_test).float().mean().item()
    return probe.weight.detach().cpu().numpy(), accuracy


def get_retrieval_axis(features, class_idx, top_k=1, block=25, device="cuda"):
    """
    Class dissimilarity from image retrieval: how well the images of one class find matches in another class.

    Parameters
    ----------
    features : numpy.ndarray
        Images x feature dimensions, with the same number of images for every class.
    class_idx : numpy.ndarray
        ImageNet class index of every image.
    top_k : int
        For every image, we average the cosine similarity of its top_k best matches in the other class
        (1 = the single best match).
    block : int
        Number of classes compared at once, to limit GPU memory.
    device : str
        Torch device.

    Returns
    -------
    D : numpy.ndarray
        1000 x 1000 dissimilarity matrix (1 - mean best-match similarity), symmetric with zero diagonal.
    """

    # Sort the images by class, so that each class is one contiguous block of rows.
    order = np.argsort(class_idx, kind="stable")
    n_per_class = np.bincount(class_idx)[0]
    X = torch.nn.functional.normalize(torch.as_tensor(features[order], dtype=torch.float32, device=device), dim=1)

    # Similarities of every image in a block of classes to every image in every class.
    D = np.zeros((1000, 1000))
    for start in range(0, 1000, block):
        stop = min(start + block, 1000)
        S = (X[start * n_per_class:stop * n_per_class] @ X.T).view(stop - start, n_per_class, 1000, n_per_class)
        best = S.topk(top_k, dim=3).values.mean(dim=3)
        D[start:stop] = 1 - best.mean(dim=1).double().cpu().numpy()

    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D
