"""Run ImageNet classifiers on the ImageNet-OOC1k image conditions and save their predictions."""

import io
import json
from datetime import datetime
from pathlib import Path

import clip
import h5py
import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# CLIP is not a timm model; this name selects the zero-shot CLIP ViT-B/16 instead.
CLIP_MODEL = "clip"
CLIP_PROMPT_TEMPLATE = "a photo of a {}"
CLIP_LOGIT_SCALE = 100.0

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
TOP_K = 5


def get_device():
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_class_index(class_index_path):
    """Load the ImageNet class index as `{class_idx: (wordnet_id, label)}`."""
    with open(class_index_path) as f:
        class_index = json.load(f)
    return {int(idx): (wnid, label) for idx, (wnid, label) in class_index.items()}


# Ground truth labels.
def load_ground_truth_main(metadata_path, class_index_path):
    """Ground truth of the main images (originals, background-only and object-only)."""
    hash_to_idx = {wnid: idx for idx, (wnid, _) in load_class_index(class_index_path).items()}
    metadata = pd.read_csv(metadata_path)
    return {
        image_id: hash_to_idx[class_hash]
        for image_id, class_hash in zip(metadata["image_id"], metadata["class_hash"])
    }


def load_ground_truth_library(metadata_path):
    """Ground truth of the object-library images."""
    metadata = pd.read_csv(metadata_path)
    return {
        Path(image_name).stem: int(class_idx)
        for image_name, class_idx in zip(metadata["image_name"], metadata["class_index"])
    }


def load_ground_truth_composites(metadata_path, class_index_path):
    """Ground truth of the composites: the class of the placed object."""
    hash_to_idx = {wnid: idx for idx, (wnid, _) in load_class_index(class_index_path).items()}
    metadata = pd.read_csv(metadata_path)
    return {
        Path(filename).stem: hash_to_idx[class_hash]
        for filename, class_hash in zip(metadata["filename"], metadata["object_class_hash"])
    }


# Models and their evaluation transforms.
class CLIPZeroShot(torch.nn.Module):
    """CLIP image encoder that scores images against fixed class-prompt embeddings."""

    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_features", text_features)

    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return CLIP_LOGIT_SCALE * image_features @ self.text_features.T


def load_clip_zero_shot(class_labels):
    """Load CLIP ViT-B/16 as a zero-shot classifier over the ImageNet class labels."""
    # Loaded on CPU so the model stays in float32 when moved to the GPU.
    clip_model, transform = clip.load("ViT-B/16", device="cpu")

    # The raw labels are used, underscores included (e.g. "great_white_shark").
    tokens = clip.tokenize([CLIP_PROMPT_TEMPLATE.format(label) for label in class_labels])
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return CLIPZeroShot(clip_model, text_features), transform


def load_model(model_name, class_labels):
    """Load a pretrained timm model (or CLIP) with its evaluation transform."""
    if model_name == CLIP_MODEL:
        return load_clip_zero_shot(class_labels)

    model = timm.create_model(model_name, pretrained=True)
    transform = create_transform(**resolve_model_data_config(model))
    return model, transform


# Image sources.
class ImageFolderDataset(Dataset):
    """Images found recursively in a folder, matched to the ground truth by file name."""

    def __init__(self, image_dir, ground_truth, transform):
        self.transform = transform
        paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.items = [(path, path.stem, ground_truth[path.stem]) for path in paths if path.stem in ground_truth]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, image_id, true_idx = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), image_id, true_idx


class CompositeH5Dataset(Dataset):
    """Composites stored as JPEG bytes in `composites.h5`, in file order."""

    def __init__(self, h5_path, ground_truth, transform):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, "r") as f:
            keys = [k.decode() if isinstance(k, bytes) else k for k in f["keys"][:]]
        self.items = [(i, key, ground_truth[key]) for i, key in enumerate(keys) if key in ground_truth]

        # Opened lazily, so every DataLoader worker gets its own file handle.
        self.file = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        h5_idx, image_id, true_idx = self.items[idx]
        img_bytes = self.file["images"][h5_idx].tobytes()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.transform(image), image_id, true_idx


def make_dataset(image_source, ground_truth, transform):
    """Dataset for an image folder or a composites HDF5 file."""
    image_source = Path(image_source)
    if image_source.suffix == ".h5":
        return CompositeH5Dataset(image_source, ground_truth, transform)
    return ImageFolderDataset(image_source, ground_truth, transform)


# Batch prediction and CSV output.
def make_data_loader(dataset, device, batch_size, num_workers):
    """Batch images in dataset order, pinning memory when using CUDA."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def predict_batch(model, images, device):
    """Return the top-five probabilities and class indices for one image batch."""
    probabilities = torch.softmax(model(images.to(device)), dim=-1)
    return probabilities.topk(TOP_K, dim=-1)


def format_prediction_rows(image_ids, true_indices, top_indices, top_probabilities, class_labels):
    """Convert one batch's predictions into rows for the output CSV."""
    rows = []
    for image_id, true_idx, indices, confidences in zip(
        image_ids, true_indices.tolist(), top_indices.tolist(), top_probabilities.tolist()
    ):
        rows.append(
            {
                "image_id": image_id,
                "true_class_idx": true_idx,
                "true_class_label": class_labels[true_idx],
                "pred_class_idx": indices[0],
                "pred_class_label": class_labels[indices[0]],
                "pred_confidence": round(confidences[0], 6),
                "top5_indices": str(indices),
                "top5_confidences": str([round(p, 6) for p in confidences]),
            }
        )
    return rows


def predict(model, dataset, class_labels, device, batch_size, num_workers, desc):
    """Predict each image's top-five classes and return the results as a table."""
    loader = make_data_loader(dataset, device, batch_size, num_workers)

    rows = []
    with torch.no_grad():
        for images, image_ids, true_indices in tqdm(loader, desc=desc, leave=False):
            top_probabilities, top_indices = predict_batch(model, images, device)
            rows.extend(
                format_prediction_rows(
                    image_ids, true_indices, top_indices, top_probabilities, class_labels
                )
            )
    return pd.DataFrame(rows)


def get_pending_conditions(model_name, conditions, output_dir, skip_existing):
    """Select conditions to run for this model, keeping their original order."""
    return [
        condition
        for condition in conditions
        if not (skip_existing and any(output_dir.glob(f"{model_name}_{condition}_*.csv")))
    ]


def save_predictions(predictions, output_path, description):
    """Save the prediction table and print its top-one accuracy."""
    predictions.to_csv(output_path, index=False)
    top1 = (predictions["pred_class_idx"] == predictions["true_class_idx"]).mean()
    print(f"{description}: top-1 accuracy {top1:.1%} on {len(predictions):,} images")


def run_inference(
    models,
    conditions,
    class_index_path,
    output_dir,
    batch_size=128,
    num_workers=4,
    skip_existing=True,
):
    """
    Run every model on every condition and save one CSV per pair.

    The CSVs are named `{model}_{condition}_{timestamp}.csv`.

    Parameters
    ----------
    models : dict
        CSV model name -> timm model name (or `CLIP_MODEL`).
    conditions : dict
        Condition name -> (image folder or composites.h5, ground truth).
    class_index_path : str or pathlib.Path
        JSON mapping ImageNet indices to class hashes and labels.
    output_dir : str or pathlib.Path
        Folder the prediction CSVs are written to.
    batch_size, num_workers : int
        DataLoader settings.
    skip_existing : bool
        Do not run pairs that already have a CSV in `output_dir`.
    """
    # Set up the output directory, class labels, and device.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_labels = [label for _, (_, label) in sorted(load_class_index(class_index_path).items())]
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, timm_name in models.items():
        pending = get_pending_conditions(model_name, conditions, output_dir, skip_existing)
        if not pending:
            print(f"{model_name}: all conditions done, skipped")
            continue

        # Load the model once and reuse it across the pending conditions.
        model, transform = load_model(timm_name, class_labels)
        model = model.to(device).eval()

        for condition in pending:
            image_source, ground_truth = conditions[condition]
            dataset = make_dataset(image_source, ground_truth, transform)
            predictions = predict(
                model,
                dataset,
                class_labels,
                device,
                batch_size,
                num_workers,
                f"{model_name} | {condition}",
            )
            save_predictions(
                predictions,
                output_dir / f"{model_name}_{condition}_{timestamp}.csv",
                f"{model_name} | {condition}",
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
