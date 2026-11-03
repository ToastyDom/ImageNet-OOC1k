"""Read out token predictions after each Vision Transformer block with the logit lens."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ooc1k.inference.model_inference import (
    CompositeH5Dataset,
    ImageFolderDataset,
    get_device,
    load_class_index,
    load_model,
)


class LogitLens(torch.nn.Module):
    """ViT that applies its final LayerNorm and classifier head to every token after every block."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.global_pool = model.global_pool
        self.n_prefix_tokens = model.num_prefix_tokens

        # CLS-token ViTs normalise the tokens with model.norm, average-pooling ViTs with model.fc_norm.
        self.final_norm = model.norm if self.global_pool == "token" else model.fc_norm

        # Every block stores its output tokens during the forward pass.
        self.block_outputs = []
        for block in model.blocks:
            block.register_forward_hook(
                lambda module, inputs, output: self.block_outputs.append(output)
            )

    def forward(self, images):
        self.block_outputs.clear()
        self.model.forward_features(images)
        tokens = torch.stack(self.block_outputs, dim=1)
        logits = self.model.head(self.final_norm(tokens))

        # The "CLS" prediction is the CLS token, or the mean over all patches for average-pooling ViTs.
        patch_logits = logits[:, :, self.n_prefix_tokens :]
        cls_logits = logits[:, :, 0] if self.global_pool == "token" else patch_logits.mean(dim=2)
        return {
            "patch_predictions": patch_logits.argmax(dim=-1),
            "cls_predictions": cls_logits.argmax(dim=-1),
            "final_logits": cls_logits[:, -1],
        }


def load_logit_lens(timm_name, device=None):
    """
    Load a pretrained ViT from timm, wrapped for the logit lens.

    Parameters
    ----------
    timm_name : str
        Pretrained timm model, e.g. "vit_base_patch16_224.augreg2_in21k_ft_in1k".
    device : torch.device, optional
        Defaults to the best available device.

    Returns
    -------
    lens : LogitLens
        The model in evaluation mode on `device`.
    transform : callable
        The evaluation transform of the model, the same as in notebook 03.
    """
    device = device or get_device()
    model, transform = load_model(timm_name, class_labels=None)
    return LogitLens(model).to(device).eval(), transform


def get_logit_lens_predictions(lens, dataset, batch_size, num_workers, desc):
    """Patch predictions, CLS predictions and final logits of every image in a dataset."""
    device = next(lens.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    outputs = {"patch_predictions": [], "cls_predictions": [], "final_logits": []}
    image_ids = []
    with torch.no_grad():
        for images, batch_ids, _ in tqdm(loader, desc=desc, leave=False):
            batch = lens(images.to(device))
            outputs["patch_predictions"].append(batch["patch_predictions"].to(torch.int16).cpu())
            outputs["cls_predictions"].append(batch["cls_predictions"].to(torch.int16).cpu())
            outputs["final_logits"].append(batch["final_logits"].to(torch.float16).cpu())
            image_ids += list(batch_ids)

    outputs = {name: torch.cat(arrays).numpy() for name, arrays in outputs.items()}
    outputs["image_id"] = np.array(image_ids)
    return outputs


def save_logit_lens_group(f, group_name, outputs):
    """Write the predictions of one image set into an HDF5 group, one chunk per image."""
    group = f.create_group(group_name)
    group.create_dataset("image_id", data=outputs["image_id"].astype(h5py.string_dtype()))
    for name in ["patch_predictions", "cls_predictions", "final_logits"]:
        data = outputs[name]
        group.create_dataset(
            name, data=data, chunks=(1, *data.shape[1:]), compression="gzip", compression_opts=4
        )


def load_logit_lens_inputs(subset_path, class_index_path):
    """Load the subset order and ground truth for composites and original backgrounds."""
    # Ground truth: the object class of every composite and the class of every original.
    hash_to_idx = {wnid: idx for idx, (wnid, _) in load_class_index(class_index_path).items()}
    subset = pd.read_csv(subset_path)
    subset_ids = subset["filename"].str.replace(".jpg", "", regex=False).values
    gt_composites = dict(zip(subset_ids, subset["object_class_hash"].map(hash_to_idx)))
    backgrounds = subset.drop_duplicates("background_id")
    gt_originals = dict(
        zip(backgrounds["background_id"], backgrounds["background_class_hash"].map(hash_to_idx))
    )

    return subset, subset_ids, hash_to_idx, gt_composites, gt_originals


def save_logit_lens_results(path, model_name, timm_name, lens, composites, originals):
    """Save both image sets with the model name, pooling mode and patch grid."""
    with h5py.File(path, "w") as f:
        f.attrs["model"] = model_name
        f.attrs["timm_name"] = timm_name
        f.attrs["global_pool"] = lens.global_pool
        f.attrs["grid_size"] = lens.model.patch_embed.grid_size
        save_logit_lens_group(f, "composites", composites)
        save_logit_lens_group(f, "originals", originals)


def run_logit_lens(
    models,
    subset_path,
    composites_h5,
    originals_dir,
    class_index_path,
    output_dir,
    batch_size=128,
    num_workers=4,
    skip_existing=True,
):
    """
    Apply the logit lens to the mechanism subset and the matching original images, and save the results.

    Parameters
    ----------
    models : dict
        Model name -> pretrained timm ViT.
    subset_path : str or pathlib.Path
        mechanism_subset.csv from notebook 02.
    composites_h5 : str or pathlib.Path
        composites.h5 from notebook 02.
    originals_dir : str or pathlib.Path
        Folder with the original images; we use the originals of the subset's backgrounds as a reference.
    class_index_path : str or pathlib.Path
        JSON mapping ImageNet indices to class hashes and labels.
    output_dir : str or pathlib.Path
        Folder for the result files, one `{model}_logit_lens.h5` per model.
    batch_size, num_workers : int
        DataLoader settings.
    skip_existing : bool
        Do not run models that already have a result file.

    Returns
    -------
    paths : list of pathlib.Path
        The result file of every model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset, subset_ids, hash_to_idx, gt_composites, gt_originals = load_logit_lens_inputs(
        subset_path, class_index_path
    )

    paths = []
    for model_name, timm_name in models.items():
        path = output_dir / f"{model_name}_logit_lens.h5"
        paths.append(path)
        if skip_existing and path.exists():
            print(f"{model_name}: {path.name} exists, skipped")
            continue

        lens, transform = load_logit_lens(timm_name)
        composites = get_logit_lens_predictions(
            lens,
            CompositeH5Dataset(composites_h5, gt_composites, transform),
            batch_size,
            num_workers,
            f"{model_name} | composites",
        )
        originals = get_logit_lens_predictions(
            lens,
            ImageFolderDataset(originals_dir, gt_originals, transform),
            batch_size,
            num_workers,
            f"{model_name} | originals",
        )

        # The composites are read in file order; store them in the order of the subset.
        order = pd.Series(np.arange(len(composites["image_id"])), index=composites["image_id"])[
            subset_ids
        ].values
        composites = {name: values[order] for name, values in composites.items()}

        save_logit_lens_results(path, model_name, timm_name, lens, composites, originals)

        accuracy = (
            composites["cls_predictions"][:, -1]
            == subset["object_class_hash"].map(hash_to_idx).values
        ).mean()
        print(
            f"{model_name}: top-1 accuracy {accuracy:.1%} on {len(subset_ids):,} composites, saved to {path.name}"
        )

        del lens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return paths
