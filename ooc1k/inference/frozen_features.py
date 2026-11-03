"""Extract frozen backbone features of ImageNet training images, used to build new semantic axes."""

import io
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ooc1k.analysis.semantic_distance import load_class_hash_index


def list_imagenet_train(imagenet_train):
    """List the ImageNet training images, either from a train/<wnid>/ folder or from our H5 file with its mapper."""
    imagenet_train = Path(imagenet_train)

    # Our server keeps ImageNet in one H5 file, with a parquet file that lists the images in it.
    if imagenet_train.suffix == ".h5":
        mapper = pd.read_parquet(imagenet_train.with_name("train_mapper.parquet"), columns=["h5_index", "filename", "wn_name"])
        return pd.DataFrame({"filename": mapper["filename"], "wnid": mapper["wn_name"], "source": mapper["h5_index"]})

    # The usual layout: one folder per class, named after its WordNet ID.
    paths = sorted(imagenet_train.glob("*/*.JPEG"))
    return pd.DataFrame({"filename": [path.name for path in paths], "wnid": [path.parent.name for path in paths], "source": [str(path) for path in paths]})


def sample_imagenet_train(imagenet_train, class_index_path, n_per_class, seed=42):
    """
    Draw the same number of ImageNet training images for every class.

    Parameters
    ----------
    imagenet_train : str or pathlib.Path
        Either the ImageNet train folder (one subfolder per class) or our train.h5 file.
    class_index_path : str or pathlib.Path
        JSON mapping ImageNet indices to class hashes and labels.
    n_per_class : int
        Number of images drawn per class.
    seed : int
        Seed of the draw. Images are sorted by file name first, so the draw does not depend on the storage format.

    Returns
    -------
    images : pandas.DataFrame
        One row per image with its file name, class index, position within its class (0 to n_per_class - 1) and
        where to read it from.
    """
    hash_to_idx = load_class_hash_index(class_index_path)
    all_images = list_imagenet_train(imagenet_train).sort_values("filename")

    rng = np.random.default_rng(seed)
    samples = []
    for wnid, class_images in sorted(all_images.groupby("wnid"), key=lambda item: hash_to_idx[item[0]]):
        picked = class_images.iloc[rng.choice(len(class_images), size=n_per_class, replace=False)]
        samples.append(picked.assign(class_idx=hash_to_idx[wnid], within_class=np.arange(n_per_class)))
    return pd.concat(samples, ignore_index=True)


class ImageNetTrainImages(Dataset):
    """Read sampled ImageNet training images from a folder or from our H5 file, and apply a transform."""

    def __init__(self, images, imagenet_train, transform):
        self.sources = images["source"].tolist()
        self.h5_path = str(imagenet_train) if Path(imagenet_train).suffix == ".h5" else None
        self.transform = transform
        self.h5_file = None

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        if self.h5_path is None:
            image = Image.open(self.sources[i])
        else:
            # Every worker opens its own handle, since an open H5 file cannot be shared between processes.
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, "r")
            image = Image.open(io.BytesIO(self.h5_file["images"][self.sources[i]].tobytes()))
        return self.transform(image.convert("RGB"))


def extract_features(timm_name, images, imagenet_train, output_path, img_size=None, batch_size=256, num_workers=8, device="cuda", skip_existing=True):
    """
    Pass ImageNet training images through a frozen backbone and save the features its head would see.

    Parameters
    ----------
    timm_name : str
        Backbone with its weight tag, e.g. "vit_base_patch16_224.dino".
    images : pandas.DataFrame
        Output of `sample_imagenet_train`.
    imagenet_train : str or pathlib.Path
        ImageNet train folder or our train.h5 file.
    output_path : str or pathlib.Path
        NPZ file the features, class indices and within-class positions are written to.
    img_size : int or None
        Input size, if it should differ from the backbone's default (DINOv2 defaults to 518 pixels).
    batch_size, num_workers : int
        Data loader settings.
    device : str
        Torch device.
    skip_existing : bool
        Load the saved features instead of extracting them again.

    Returns
    -------
    features : dict
        "features" (images x feature dimensions, float16), "class_idx" and "within_class".
    """
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        return dict(np.load(output_path))

    # The backbone without a classifier head, with its own preprocessing.
    model_kwargs = {"num_classes": 0} if img_size is None else {"num_classes": 0, "img_size": img_size}
    model = timm.create_model(timm_name, pretrained=True, **model_kwargs).eval().to(device)
    data_config = timm.data.resolve_data_config({}, model=model)
    if img_size is not None:
        data_config["input_size"] = (3, img_size, img_size)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Pre-logit features: the representation a linear classifier on top of the backbone would use.
    loader = DataLoader(ImageNetTrainImages(images, imagenet_train, transform), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=timm_name):
            output = model.forward_head(model.forward_features(batch.to(device, non_blocking=True)), pre_logits=True)
            features.append(output.float().cpu().numpy().astype(np.float16))

    result = {
        "features": np.concatenate(features),
        "class_idx": images["class_idx"].values.astype(np.int16),
        "within_class": images["within_class"].values.astype(np.int16),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)
    return result
