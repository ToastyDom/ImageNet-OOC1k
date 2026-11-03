"""Test whether properties of the composite generation could explain the accuracy decline along the semantic axis."""

import random
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.color import rgb2lab
from tqdm.auto import tqdm

from ooc1k.analysis.semantic_distance import find_prediction_csv
from ooc1k.dataset_generation.composites import (
    BRIGHTNESS_RANGE,
    COLOR_RANGE,
    MASK_THRESHOLD,
    ROTATION_PROBABILITY,
    create_composite,
    create_cutout,
    get_background_path,
)

# The 18 generation variables, grouped by the concern they address.
VARIABLE_GROUPS = {
    "placement": ["center_x", "center_y", "eccentricity", "border_distance", "rotated"],
    "scale": [
        "scale_factor",
        "relative_width",
        "relative_height",
        "bbox_area",
        "mask_area",
        "aspect_ratio",
        "aspect_distortion",
        "min_clamp",
        "max_clamp_width",
        "max_clamp_height",
        "both_clamps",
    ],
    "cutout": ["cutout_complexity", "mask_fill"],
}

# Covariates of the regressions: placement, realized scale and shape, and which size limits were hit.
COVARIATES = [
    "center_x",
    "center_y",
    "eccentricity",
    "border_distance",
    "scale_factor",
    "mask_area",
    "aspect_ratio",
    "max_clamp_width",
    "max_clamp_height",
    "min_clamp",
    "both_clamps",
]

# Boundary bands: pixels within BAND_INSIDE of the edge inside the object, within BAND_OUTSIDE outside of it.
BAND_INSIDE = 4
BAND_OUTSIDE = 7
BOUNDARY_MEASURES = ["luminance", "color", "gradient"]


# Generation variables of every composite.


def get_cutout_geometry(source_dir, object_names):
    """Size, mask fill ratio and outline complexity of every object cutout, as `create_cutout` makes them."""
    library = Path(source_dir) / "object_library"
    rows = {}
    for name in tqdm(object_names, desc="Cutouts"):
        with Image.open(library / "images" / name) as image:
            size = image.size
        with Image.open(library / "masks" / f"{Path(name).stem}.png") as mask:
            mask = mask.convert("L")
            if mask.size != size:
                mask = mask.resize(size, Image.Resampling.NEAREST)
            alpha = np.array(mask) > MASK_THRESHOLD

        # Crop to the bounding box of the mask, as the cutout is cropped.
        ys, xs = np.nonzero(alpha)
        crop = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        area = crop.sum()
        perimeter = area - ndimage.binary_erosion(crop).sum()
        rows[name] = {
            "cutout_width": crop.shape[1],
            "cutout_height": crop.shape[0],
            "mask_fill": area / crop.size,
            "cutout_complexity": perimeter / np.sqrt(area),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def get_background_sizes(source_dir, backgrounds):
    """Width and height of every background image."""
    sizes = {}
    for _, row in backgrounds.iterrows():
        with Image.open(get_background_path(row, source_dir)) as image:
            sizes[row["image_id"]] = image.size
    return pd.DataFrame.from_dict(sizes, orient="index", columns=["background_width", "background_height"])


def replay_random_draws(image_seed, scale_min, scale_max):
    """Replay the first draws of `create_composite`: the scale factor and whether the object was rotated."""
    rng = random.Random(image_seed)
    scale_factor = rng.uniform(scale_min, scale_max)
    rng.uniform(*BRIGHTNESS_RANGE)
    rng.uniform(*COLOR_RANGE)
    return scale_factor, rng.random() < ROTATION_PROBABILITY


def get_generation_variables(metadata, cutouts, background_sizes, config):
    """
    Reconstruct the placement, scale and cutout properties of every composite.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Composite metadata.csv of notebook 02.
    cutouts : pandas.DataFrame
        Output of `get_cutout_geometry`, indexed by object image name.
    background_sizes : pandas.DataFrame
        Output of `get_background_sizes`, indexed by background ID.
    config : dict
        The generation settings in config.json of notebook 02.

    Returns
    -------
    variables : pandas.DataFrame
        One row per composite and one column per variable in `VARIABLE_GROUPS`, plus "size_matches": whether the
        replayed object size equals the recorded one (only checked for objects that were not rotated, since rotation
        enlarges the recorded bounding box).
    """
    cutout = cutouts.loc[metadata["object_image_name"]].reset_index(drop=True)
    background = background_sizes.loc[metadata["background_id"]].reset_index(drop=True)
    bg_w, bg_h = background["background_width"].values, background["background_height"].values
    x, y = metadata["placement_x"].values, metadata["placement_y"].values
    w, h = metadata["object_width"].values, metadata["object_height"].values

    # The scale factor is stored rounded, and the rotation is not stored, so we replay both from the seed.
    draws = [replay_random_draws(seed, config["scale_min"], config["scale_max"]) for seed in metadata["image_seed"]]
    scale_factor = np.array([draw[0] for draw in draws])
    rotated = np.array([draw[1] for draw in draws])

    # Replay `scale_object`: scale the longer side, then apply the minimum size and the cap at half the background.
    aspect = cutout["cutout_width"].values / cutout["cutout_height"].values
    wide = aspect > 1
    scaled_w = np.where(wide, np.floor(bg_w * scale_factor), np.floor(np.floor(bg_h * scale_factor) * aspect))
    scaled_h = np.where(wide, np.floor(np.floor(bg_w * scale_factor) / aspect), np.floor(bg_h * scale_factor))
    min_object_dim = config["min_object_dim"]
    min_clamp = (scaled_w < min_object_dim) | (scaled_h < min_object_dim)
    max_clamp_width = np.maximum(scaled_w, min_object_dim) > bg_w // 2
    max_clamp_height = np.maximum(scaled_h, min_object_dim) > bg_h // 2
    final_w = np.minimum(np.maximum(scaled_w, min_object_dim), bg_w // 2)
    final_h = np.minimum(np.maximum(scaled_h, min_object_dim), bg_h // 2)

    # Placement uses the recorded bounding box, which is what ends up in the image.
    center_x = (x + w / 2) / bg_w
    center_y = (y + h / 2) / bg_h
    variables = pd.DataFrame({
        "center_x": center_x,
        "center_y": center_y,
        "eccentricity": np.hypot(center_x - 0.5, center_y - 0.5),
        "border_distance": np.minimum.reduce([x, y, bg_w - x - w, bg_h - y - h]) / np.minimum(bg_w, bg_h),
        "rotated": rotated.astype(float),
        "scale_factor": scale_factor,
        "relative_width": w / bg_w,
        "relative_height": h / bg_h,
        "bbox_area": w * h / (bg_w * bg_h),
        "mask_area": cutout["mask_fill"].values * final_w * final_h / (bg_w * bg_h),
        "aspect_ratio": final_w / final_h,
        "aspect_distortion": np.abs(np.log(final_w / final_h / aspect)),
        "min_clamp": min_clamp.astype(float),
        "max_clamp_width": max_clamp_width.astype(float),
        "max_clamp_height": max_clamp_height.astype(float),
        "both_clamps": (max_clamp_width & max_clamp_height).astype(float),
        "cutout_complexity": cutout["cutout_complexity"].values,
        "mask_fill": cutout["mask_fill"].values,
    })
    variables["size_matches"] = np.where(rotated, np.nan, (final_w == w) & (final_h == h))
    return variables


def get_variable_drift(variables, ranks, accuracy, bin_width=25):
    """
    How much every generation variable changes along the semantic axis, and how much accuracy that change could explain.

    Parameters
    ----------
    variables : pandas.DataFrame
        Output of `get_generation_variables`.
    ranks : numpy.ndarray
        Dissimilarity rank of every composite.
    accuracy : numpy.ndarray
        Top-1 accuracy of every composite, averaged over models.
    bin_width : int
        Number of ranks per bin.

    Returns
    -------
    drift : pandas.DataFrame
        One row per variable: its Spearman correlation with rank, its mean in the first and last bin, and its largest
        plausible contribution to the accuracy change between these bins (in percentage points and as a share of the
        observed change). The contribution is the variable's change times its univariate effect on accuracy, which
        also credits the variable with everything it correlates with, so it is generous.
    """
    bins = (np.asarray(ranks) - 1) // bin_width
    first, last = bins == bins.min(), bins == bins.max()
    observed = 100 * (accuracy[last].mean() - accuracy[first].mean())

    rows = {}
    for group, names in VARIABLE_GROUPS.items():
        for name in names:
            values = variables[name].values
            change = values[last].mean() - values[first].mean()
            contribution = 100 * np.polyfit(values, accuracy, 1)[0] * change
            rows[name] = {
                "group": group,
                "rho with rank": spearmanr(ranks, values)[0],
                "first bin": values[first].mean(),
                "last bin": values[last].mean(),
                "contribution (pp)": contribution,
                "share of change (%)": 100 * contribution / observed,
            }
    drift = pd.DataFrame.from_dict(rows, orient="index")
    drift.attrs["observed_change"] = observed
    return drift


# Rank slopes with fixed effects.


def absorb_fixed_effects(Z, groups, tol=1e-8, max_iterations=1000):
    """Remove group means from every column of Z, alternating between the groupings until nothing changes."""
    Z = np.array(Z, dtype=np.float64)
    for _ in range(max_iterations):
        largest_change = 0.0
        for codes, n_groups in groups:
            counts = np.bincount(codes, minlength=n_groups)
            for j in range(Z.shape[1]):
                means = np.bincount(codes, weights=Z[:, j], minlength=n_groups) / counts
                Z[:, j] -= means[codes]
                largest_change = max(largest_change, np.abs(means).max())
        if largest_change < tol or len(groups) == 1:
            break
    return Z


def fit_clustered_ols(X, y, clusters, n_absorbed):
    """Least squares with standard errors clustered two ways (by background and by object instance)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    scores = X * (y - X @ beta)[:, None]
    dof = (len(y) - 1) / (len(y) - X.shape[1] - n_absorbed)

    # Cameron-Gelbach-Miller: both one-way variances minus that of their intersection (one composite per pair).
    meat = -scores.T @ scores * dof
    for codes, n_groups in clusters:
        sums = np.stack([np.bincount(codes, weights=scores[:, j], minlength=n_groups) for j in range(X.shape[1])], axis=1)
        meat += sums.T @ sums * n_groups / (n_groups - 1) * dof
    variance = np.diag(XtX_inv @ meat @ XtX_inv)
    return beta, np.sqrt(np.maximum(variance, 0))


def get_group_codes(values):
    """Integer codes of a grouping variable and the number of groups."""
    codes, uniques = pd.factorize(values)
    return codes, len(uniques)


def fit_rank_slopes(ranks, correct, clusters, covariates=None, fixed_effects=(), model_covariates=None):
    """
    Fit accuracy ~ rank for every model, optionally with covariates and fixed effects.

    Parameters
    ----------
    ranks : numpy.ndarray
        Dissimilarity rank of every composite.
    correct : pandas.DataFrame
        One boolean column per model.
    clusters : list
        Two groupings of every composite (background ID and object instance) used to cluster the standard errors.
    covariates : pandas.DataFrame or None
        Covariates shared by all models, one row per composite.
    fixed_effects : list
        Groupings of every composite that each get their own intercept (e.g. background ID and object instance).
    model_covariates : pandas.DataFrame or None
        One extra covariate per model (one column per model), e.g. the model's own object-only accuracy.

    Returns
    -------
    slopes : pandas.DataFrame
        One row per model with the rank slope in percentage points per 100 ranks and its 95% confidence interval.
    """
    clusters = [get_group_codes(values) for values in clusters]
    groups = [get_group_codes(values) for values in fixed_effects]
    n_absorbed = sum(n for _, n in groups) - max(len(groups) - 1, 0)

    # Shared columns, outcomes and model-specific columns are demeaned together, once.
    shared = np.column_stack([np.asarray(ranks, dtype=float) / 100] + ([covariates.values] if covariates is not None else []))
    blocks = [shared, correct.values.astype(float)] + ([model_covariates[correct.columns].values] if model_covariates is not None else [])
    Z = np.column_stack(blocks)
    Z = absorb_fixed_effects(Z, groups) if groups else np.column_stack([Z, np.ones(len(Z))])
    n_shared, n_models = shared.shape[1], correct.shape[1]

    rows = {}
    for i, model in enumerate(correct.columns):
        columns = [Z[:, :n_shared]]
        if model_covariates is not None:
            columns.append(Z[:, n_shared + n_models + i][:, None])
        if not groups:
            columns.append(Z[:, -1:])
        beta, se = fit_clustered_ols(np.column_stack(columns), Z[:, n_shared + i], clusters, n_absorbed)
        rows[model] = {"slope": 100 * beta[0], "ci_low": 100 * (beta[0] - 1.96 * se[0]), "ci_high": 100 * (beta[0] + 1.96 * se[0])}
    return pd.DataFrame.from_dict(rows, orient="index")


def fit_binned_curve(ranks, correct, covariates, fixed_effects, bin_width=25):
    """
    Accuracy change of every rank bin relative to the first bin, after removing covariates and fixed effects.

    Parameters
    ----------
    ranks : numpy.ndarray
        Dissimilarity rank of every composite.
    correct : pandas.DataFrame
        One boolean column per model.
    covariates : pandas.DataFrame
        Covariates shared by all models, one row per composite.
    fixed_effects : list
        Groupings of every composite whose means are removed (e.g. background ID and object instance).
    bin_width : int
        Number of ranks per bin.

    Returns
    -------
    curves : pandas.DataFrame
        One row per bin (indexed by its center rank) and one column per model, in percentage points; the first bin is 0.
    """

    # One indicator per bin except the first, which is the reference.
    bins = (np.asarray(ranks) - 1) // bin_width
    n_bins = bins.max() + 1
    indicators = (bins[:, None] == np.arange(1, n_bins)[None, :]).astype(float)

    groups = [get_group_codes(values) for values in fixed_effects]
    Z = absorb_fixed_effects(np.column_stack([indicators, covariates.values, correct.values.astype(float)]), groups)
    X = Z[:, :indicators.shape[1] + covariates.shape[1]]
    beta = np.linalg.lstsq(X, Z[:, X.shape[1]:], rcond=None)[0][:n_bins - 1]

    centers = 1 + np.arange(n_bins) * bin_width + bin_width / 2
    return pd.DataFrame(100 * np.vstack([np.zeros(correct.shape[1]), beta]), index=pd.Index(centers, name="bin_center"), columns=correct.columns)


def get_object_only_accuracy(predictions_dir, models, object_image_names, backgrounds=("black", "white", "pink_noise")):
    """Every model's accuracy on the object-only image of each composite's object, averaged over the neutral backgrounds."""
    stems = pd.Series(object_image_names).str.replace(".JPEG", "", regex=False)
    accuracy = {}
    for model in models:
        per_background = []
        for background in backgrounds:
            predictions = pd.read_csv(find_prediction_csv(predictions_dir, model, f"object_library_{background}"), usecols=["image_id", "true_class_idx", "pred_class_idx"])
            per_background.append((predictions["pred_class_idx"] == predictions["true_class_idx"]).groupby(predictions["image_id"]).mean())
        accuracy[model] = pd.concat(per_background, axis=1).mean(axis=1).reindex(stems).values
    return pd.DataFrame(accuracy)


# Boundary discontinuities.


def sample_composites(metadata, n_objects, n_per_object, seed=42):
    """Draw object instances, then a few of their composites, so that object-instance fixed effects stay estimable."""
    rng = np.random.default_rng(seed)
    objects = rng.choice(metadata["object_image_name"].unique(), size=n_objects, replace=False)
    candidates = metadata[metadata["object_image_name"].isin(objects)]
    return candidates.groupby("object_image_name", group_keys=False).sample(n=n_per_object, random_state=seed).sort_index()


def get_boundary_bands(alpha):
    """Pixels just inside and just outside the object edge, and the soft edge itself, from the pasted alpha."""
    solid = alpha >= 0.96
    empty = alpha <= 0.04
    if solid.sum() < 20 or empty.sum() < 20:
        return None
    inside = solid & (ndimage.distance_transform_edt(solid) <= BAND_INSIDE)
    outside = empty & (ndimage.distance_transform_edt(empty) <= BAND_OUTSIDE)
    return inside, outside, ~(solid | empty)


def get_boundary_contrast(rgb, inside, outside, edge):
    """Luminance, color and gradient discontinuity across an object boundary."""
    lab = rgb2lab(rgb)
    lab_inside, lab_outside = lab[inside].mean(axis=0), lab[outside].mean(axis=0)
    gradient_y, gradient_x = np.gradient(lab[..., 0])
    return {
        "luminance": abs(lab_inside[0] - lab_outside[0]),
        "color": np.linalg.norm(lab_inside - lab_outside),
        "gradient": np.hypot(gradient_x, gradient_y)[inside | outside | edge].mean(),
    }


@lru_cache(maxsize=256)
def load_cutout(source_dir, object_image_name):
    """Cutout of one library object, cached because every object is used for several composites."""
    library = Path(source_dir) / "object_library"
    return create_cutout(library / "images" / object_image_name, library / "masks" / f"{Path(object_image_name).stem}.png")


def measure_background(args):
    """Rebuild the sampled composites of one background and measure their boundaries, with the object and without it."""
    rows, source_dir, config, padding = args
    first = rows.iloc[0]
    background_row = {"image_id": first["background_id"], "background_category": first["background_category"], "background_subcategory": first["background_subcategory"]}
    background = Image.open(get_background_path(background_row, source_dir)).convert("RGB")
    background_rgb = np.asarray(background, dtype=np.float32) / 255

    measures = {}
    for index, row in rows.iterrows():
        cutout = load_cutout(str(source_dir), row["object_image_name"])
        composite, _, obj, placement = create_composite(background, cutout, row["image_seed"], config["scale_min"], config["scale_max"], config["margin"], config["min_object_dim"])

        # The soft alpha of the pasted object; we work on its bounding box plus some padding.
        alpha = np.zeros(background.size[::-1], dtype=np.float32)
        x, y = placement["placement_x"], placement["placement_y"]
        alpha[y:y + obj.size[1], x:x + obj.size[0]] = np.asarray(obj.split()[-1], dtype=np.float32) / 255
        x0, y0 = max(x - padding, 0), max(y - padding, 0)
        x1, y1 = min(x + obj.size[0] + padding, background.size[0]), min(y + obj.size[1] + padding, background.size[1])
        bands = get_boundary_bands(alpha[y0:y1, x0:x1])
        if bands is None:
            continue

        # The same bands on the background alone tell us how much contrast the background has there anyway.
        real = get_boundary_contrast(np.asarray(composite, dtype=np.float32)[y0:y1, x0:x1] / 255, *bands)
        sham = get_boundary_contrast(background_rgb[y0:y1, x0:x1], *bands)
        measures[index] = {
            **{f"{name}_excess": real[name] - sham[name] for name in BOUNDARY_MEASURES},
            "replay_matches": (x, y, *obj.size) == (row["placement_x"], row["placement_y"], row["object_width"], row["object_height"]),
        }
    return measures


def get_boundary_measures(sample, source_dir, config, padding=8, num_workers=16):
    """
    Rebuild sampled composites from their seeds and measure the discontinuities across the object boundary.

    Parameters
    ----------
    sample : pandas.DataFrame
        Rows of the composite metadata, e.g. from `sample_composites`.
    source_dir : str or pathlib.Path
        Dataset source directory with the backgrounds and the object library.
    config : dict
        The generation settings in config.json of notebook 02.
    padding : int
        Pixels around the object's bounding box included in the measurement.
    num_workers : int
        Number of processes; each handles all composites of one background at a time.

    Returns
    -------
    measures : pandas.DataFrame
        One row per measured composite (same index as `sample`) with the excess luminance, color and gradient
        discontinuity (composite minus background alone, in the same bands) and whether the replayed placement
        matches the metadata.
    """
    jobs = [(rows, Path(source_dir), config, padding) for _, rows in sample.groupby("background_id")]
    measures = {}
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(measure_background, jobs), total=len(jobs), desc="Backgrounds"):
            measures.update(result)
    return pd.DataFrame.from_dict(measures, orient="index").sort_index()
