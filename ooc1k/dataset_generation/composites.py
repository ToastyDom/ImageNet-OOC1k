"""Generate ImageNet-OOC1k composites by placing object cutouts on background images."""

import hashlib
import io
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm


# Object appearance and storage settings.
BRIGHTNESS_RANGE = (0.9, 1.1)
COLOR_RANGE = (0.95, 1.05)
ROTATION_PROBABILITY = 0.3
ROTATION_RANGE = (-10, 10)

MASK_THRESHOLD = 128
MIN_CUTOUT_SIZE = 5
H5_CHUNK_SIZE = 500


def load_data(source_dir, similarity_matrix_path, class_index_path):
    """Load background and object metadata, class mappings, and dissimilarities."""

    # Read the background and object tables.
    root = Path(source_dir)
    bg_metadata = pd.read_csv(root / "metadata_backgrounds.csv")
    obj_metadata = pd.read_csv(root / "metadata_objects.csv")

    # Group object filenames by class, preserving their metadata order.
    objects_by_hash = {}
    for class_hash, image_name in zip(obj_metadata["class_hash"], obj_metadata["image_name"]):
        objects_by_hash.setdefault(class_hash, []).append(image_name)

    # Map class hashes to matrix indices and back.
    with open(class_index_path) as f:
        class_index = json.load(f)
    hash_to_idx = {class_hash: int(idx) for idx, (class_hash, _) in class_index.items()}
    idx_to_hash = {idx: class_hash for class_hash, idx in hash_to_idx.items()}

    return {
        "bg_metadata": bg_metadata,
        "obj_metadata": obj_metadata,
        "D": np.load(similarity_matrix_path),
        "class_index": class_index,
        "hash_to_idx": hash_to_idx,
        "idx_to_hash": idx_to_hash,
        "objects_by_hash": objects_by_hash,
    }


def sample_backgrounds(bg_metadata, bg_per_subcategory, seed):
    """Sample up to the requested number of backgrounds per category/subcategory."""

    # Assign missing subcategories to the miscellaneous group.
    rng = np.random.RandomState(seed)
    meta = bg_metadata.copy()
    meta["background_subcategory"] = meta["background_subcategory"].fillna("misc")

    # Sample without replacement in sorted group order.
    sampled = []
    groups = meta.groupby(["background_category", "background_subcategory"], sort=True)
    for _, group in groups:
        n = min(bg_per_subcategory, len(group))
        indices = rng.choice(len(group), size=n, replace=False)
        sampled.append(group.iloc[indices])
    return pd.concat(sampled, ignore_index=True)


def get_background_path(row, source_dir):
    """Return the inpainted image path for a background metadata row."""
    backgrounds_dir = Path(source_dir) / "backgrounds"
    category = row["background_category"]
    subcategory = row["background_subcategory"]
    if category == "misc" or pd.isna(subcategory):
        return backgrounds_dir / "misc" / f"{row['image_id']}.JPEG"
    return backgrounds_dir / category / subcategory / f"{row['image_id']}.JPEG"


def make_image_seed(bg_id, name, global_seed):
    """Derive a seed from the background, object or class name, and global seed."""
    key = f"{global_seed}_{bg_id}_{name}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def get_ranked_classes(bg_class_hash, data):
    """Return available object classes with ranks and dissimilarity scores.

    Ranks start at 1 and include classes without objects, leaving gaps in the result.
    """
    if bg_class_hash not in data["hash_to_idx"]:
        return []

    # Sort all classes by dissimilarity, then retain classes with library objects.
    scores = data["D"][data["hash_to_idx"][bg_class_hash]]
    rankings = []
    for rank_0, class_idx in enumerate(np.argsort(scores)):
        class_hash = data["idx_to_hash"].get(class_idx)
        if class_hash in data["objects_by_hash"]:
            rankings.append((class_hash, rank_0 + 1, float(scores[class_idx])))
    return rankings


def create_cutout(obj_image_path, mask_path):
    """Apply the segmentation mask and crop the object to its bounding box."""

    # Match the mask dimensions to the source image.
    original = Image.open(obj_image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.NEAREST)

    # Convert the mask to binary transparency and crop the visible region.
    alpha = np.where(np.array(mask) > MASK_THRESHOLD, 255, 0).astype(np.uint8)
    original.putalpha(Image.fromarray(alpha, "L"))

    bbox = original.getbbox()
    if bbox:
        original = original.crop(bbox)
    return original


def scale_object(obj_img, bg_size, scale_factor, min_object_dim):
    """Scale an object relative to the background and clamp its dimensions."""
    bg_w, bg_h = bg_size
    obj_w, obj_h = obj_img.size
    aspect = obj_w / obj_h

    # Scale wide objects by background width and tall objects by background height.
    if aspect > 1:
        new_w = int(bg_w * scale_factor)
        new_h = int(new_w / aspect)
    else:
        new_h = int(bg_h * scale_factor)
        new_w = int(new_h * aspect)

    # Apply the minimum size, then cap each side at half the background dimension.
    new_w = min(max(new_w, min_object_dim), bg_w // 2)
    new_h = min(max(new_h, min_object_dim), bg_h // 2)
    return obj_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def adjust_appearance(obj_img, rng):
    """Vary object brightness and color, with an optional small rotation."""
    obj_img = ImageEnhance.Brightness(obj_img).enhance(rng.uniform(*BRIGHTNESS_RANGE))
    obj_img = ImageEnhance.Color(obj_img).enhance(rng.uniform(*COLOR_RANGE))

    if rng.random() < ROTATION_PROBABILITY:
        angle = rng.uniform(*ROTATION_RANGE)
        obj_img = obj_img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    return obj_img


def sample_position(bg_size, obj_size, margin, rng):
    """Sample a top-left position within the background margin when the object fits."""
    bg_w, bg_h = bg_size
    x_min = int(bg_w * margin)
    y_min = int(bg_h * margin)
    x_max = max(x_min, int(bg_w * (1 - margin)) - obj_size[0])
    y_max = max(y_min, int(bg_h * (1 - margin)) - obj_size[1])
    return rng.randint(x_min, x_max), rng.randint(y_min, y_max)


def place_object(bg_img, obj_img, x, y):
    """Place an object on the background and return the RGB image and binary mask."""

    # Paste the object using its transparency and convert the result to RGB.
    composite = bg_img.convert("RGBA")
    composite.paste(obj_img, (x, y), obj_img)

    composite_rgb = Image.new("RGB", composite.size, (255, 255, 255))
    composite_rgb.paste(composite, mask=composite.split()[-1])

    # Place the object's binary mask at the same image coordinates.
    mask = Image.new("L", bg_img.size, 0)
    alpha = obj_img.split()[-1].point(lambda p: 255 if p > MASK_THRESHOLD else 0, mode="L")
    mask.paste(alpha, (x, y))
    return composite_rgb, mask


def create_composite(bg_img, cutout, img_seed, scale_min, scale_max, margin, min_object_dim):
    """Scale, adjust, and place an object using its per-image random seed."""

    # Draw the scale, appearance changes, and position from one random generator.
    rng = random.Random(img_seed)
    scale_factor = rng.uniform(scale_min, scale_max)
    obj_img = scale_object(cutout, bg_img.size, scale_factor, min_object_dim)
    obj_img = adjust_appearance(obj_img, rng)
    x, y = sample_position(bg_img.size, obj_img.size, margin, rng)

    # Create the image and mask, and record the transformed object's placement.
    composite, mask = place_object(bg_img, obj_img, x, y)
    placement = {
        "placement_x": x,
        "placement_y": y,
        "object_width": obj_img.size[0],
        "object_height": obj_img.size[1],
        "scale_factor": round(scale_factor, 4),
    }
    return composite, mask, obj_img, placement


def _encode_image(img, fmt="JPEG", **kwargs):
    """Encode an image as a byte array in memory."""
    buffer = io.BytesIO()
    img.save(buffer, fmt, **kwargs)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _open_h5_writer(path, batch_size):
    """Open an HDF5 image file and initialize its batch buffer."""
    file = h5py.File(path, "w")
    for name, dtype in [("images", h5py.vlen_dtype(np.uint8)), ("keys", h5py.string_dtype())]:
        file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(H5_CHUNK_SIZE,))
    return {"file": file, "batch_size": batch_size, "keys": [], "images": []}


def _add_image(writer, key, img_bytes):
    """Buffer an encoded image and write the batch when full."""
    writer["keys"].append(key)
    writer["images"].append(img_bytes)
    if len(writer["keys"]) >= writer["batch_size"]:
        _flush_image_batch(writer)


def _flush_image_batch(writer):
    """Append buffered images and keys to the HDF5 datasets."""
    if not writer["keys"]:
        return

    # Extend the datasets to accommodate the buffered images.
    file = writer["file"]
    start = file["images"].shape[0]
    end = start + len(writer["keys"])
    file["images"].resize(end, axis=0)
    file["keys"].resize(end, axis=0)

    # Write images and keys in matching order, then clear the buffer.
    for i, (key, img_bytes) in enumerate(zip(writer["keys"], writer["images"])):
        file["images"][start + i] = img_bytes
        file["keys"][start + i] = key
    writer["keys"].clear()
    writer["images"].clear()


def _write_h5_metadata(file, metadata_df):
    """Store metadata columns as HDF5 datasets and record the image count."""
    group = file.create_group("metadata")
    for column in metadata_df.columns:
        values = metadata_df[column]
        if values.dtype == object:
            group.create_dataset(column, data=values.astype(str).tolist(), dtype=h5py.string_dtype())
        elif np.issubdtype(values.dtype, np.floating):
            group.create_dataset(column, data=values.values.astype(np.float32))
        else:
            group.create_dataset(column, data=values.values.astype(np.int64))
    file.attrs["n_images"] = len(metadata_df)


def open_output_files(output_dir, save_masks, save_silhouettes, h5_batch_size):
    """Open the composite HDF5 file and any requested mask or cutout files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {"composites": _open_h5_writer(output_dir / "composites.h5", h5_batch_size)}
    if save_masks:
        writers["masks"] = _open_h5_writer(output_dir / "masks.h5", h5_batch_size)
    if save_silhouettes:
        writers["silhouettes"] = _open_h5_writer(output_dir / "silhouettes.h5", h5_batch_size)
    return writers


def generate_background_composites(bg_row, data, writers, config):
    """Generate and write one background's composites, returning metadata and skips.

    A missing background or class ranking returns None instead of metadata rows.
    """
    obj_images_dir = Path(config["source_dir"]) / "object_library" / "images"
    obj_masks_dir = Path(config["source_dir"]) / "object_library" / "masks"
    metadata_rows = []
    n_skipped = 0

    # Load the background and rank object classes by semantic dissimilarity.
    bg_id = bg_row["image_id"]
    bg_path = get_background_path(bg_row, config["source_dir"])
    rankings = get_ranked_classes(bg_row["class_hash"], data)
    if not bg_path.exists() or not rankings:
        tqdm.write(f"Skipped background {bg_id}: image or class ranking missing")
        return None, 1
    bg_img = Image.open(bg_path).convert("RGB")

    for class_hash, rank, dissimilarity in rankings:
        # Sample object instances using a seed for this background and class.
        available = data["objects_by_hash"].get(class_hash, [])
        n_objects = min(config["objects_per_class"], len(available))
        if n_objects == 0:
            continue
        class_rng = random.Random(make_image_seed(bg_id, class_hash, config["seed"]))
        selected_objects = class_rng.sample(available, n_objects)
        obj_class_label = data["class_index"][str(data["hash_to_idx"][class_hash])][1]

        for obj_name in selected_objects:
            # Locate the object image and its segmentation mask.
            obj_path = obj_images_dir / obj_name
            mask_path = obj_masks_dir / f"{Path(obj_name).stem}.png"
            if not obj_path.exists() or not mask_path.exists():
                n_skipped += 1
                continue

            # Extract the cutout and generate its composite and placement data.
            img_seed = make_image_seed(bg_id, obj_name, config["seed"])
            try:
                cutout = create_cutout(obj_path, mask_path)
                if min(cutout.size) < MIN_CUTOUT_SIZE:
                    n_skipped += 1
                    continue
                composite, mask, silhouette, placement = create_composite(
                    bg_img,
                    cutout,
                    img_seed,
                    config["scale_min"],
                    config["scale_max"],
                    config["margin"],
                    config["min_object_dim"],
                )
            except Exception as error:
                tqdm.write(f"Error for {bg_id} with {obj_name}: {error}")
                n_skipped += 1
                continue

            # Buffer the composite and optional outputs under the same key.
            key = f"{bg_id}_with_{Path(obj_name).stem}"
            _add_image(writers["composites"], key, _encode_image(composite, "JPEG", quality=config["jpeg_quality"]))
            if config["save_masks"]:
                _add_image(writers["masks"], key, _encode_image(mask, "PNG"))
            if config["save_silhouettes"]:
                _add_image(writers["silhouettes"], key, _encode_image(silhouette, "PNG"))

            # Record the source images, semantic rank, and object placement.
            metadata_rows.append({
                "filename": f"{key}.jpg",
                "background_id": bg_id,
                "background_class_hash": bg_row["class_hash"],
                "background_class_label": bg_row["class_label"],
                "background_category": bg_row["background_category"],
                "background_subcategory": bg_row["background_subcategory"],
                "object_image_name": obj_name,
                "object_class_hash": class_hash,
                "object_class_label": obj_class_label,
                "dissimilarity_rank": rank,
                "dissimilarity_score": round(dissimilarity, 6),
                **placement,
                "image_seed": img_seed,
            })

    return metadata_rows, n_skipped


def generate_images(backgrounds, data, writers, config):
    """Generate composites one background at a time and collect their metadata."""
    metadata_rows = []
    n_skipped = 0
    progress = tqdm(backgrounds.iterrows(), total=len(backgrounds), desc="Backgrounds")

    for _, bg_row in progress:
        rows, skipped = generate_background_composites(bg_row, data, writers, config)
        n_skipped += skipped
        if rows is None:
            continue
        metadata_rows.extend(rows)
        progress.set_postfix(composites=len(metadata_rows), skipped=n_skipped)

    return pd.DataFrame(metadata_rows), n_skipped


def save_outputs(metadata, config, writers):
    """Write metadata and configuration, flush image buffers, and close files."""
    _write_h5_metadata(writers["composites"]["file"], metadata)
    writers["composites"]["file"].attrs["config"] = json.dumps(config)
    for writer in writers.values():
        _flush_image_batch(writer)
        writer["file"].close()

    output_dir = Path(config["output_dir"])
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def print_summary(metadata, n_skipped, start_time):
    """Print the image count, elapsed time, and number of skipped entries."""
    minutes = (time.time() - start_time) / 60
    print(f"Created {len(metadata):,} composites in {minutes:.1f} min ({n_skipped} skipped)")


def generate_composites(
    source_dir,
    similarity_matrix_path,
    class_index_path,
    output_dir,
    bg_per_subcategory=10,
    objects_per_class=4,
    scale_min=0.50,
    scale_max=0.60,
    margin=0.05,
    min_object_dim=100,
    jpeg_quality=95,
    seed=42,
    save_masks=False,
    save_silhouettes=False,
    h5_batch_size=500,
):
    """
    Generate object-background composites across the semantic axis.

    Parameters
    ----------
    source_dir : str or pathlib.Path
        Dataset source directory containing the backgrounds, the object library and metadata.
    similarity_matrix_path : str or pathlib.Path
        NumPy file containing class dissimilarities in ImageNet index order.
    class_index_path : str or pathlib.Path
        JSON mapping ImageNet indices to class hashes and labels.
    output_dir : str or pathlib.Path
        Destination for HDF5 images, metadata.csv, and config.json.
    bg_per_subcategory : int
        Maximum backgrounds sampled per category/subcategory group.
    objects_per_class : int
        Maximum object instances sampled per class and background.
    scale_min, scale_max : float
        Bounds for object size relative to the background, before size limits.
    margin : float
        Fractional background margin used when sampling object positions.
    min_object_dim : int
        Minimum object side length before the background-size cap.
    jpeg_quality : int
        JPEG encoding quality for composites.
    seed : int
        Global seed for background sampling and per-image seeds.
    save_masks, save_silhouettes : bool
        Save full-image binary masks and transformed RGBA cutouts as PNGs.
    h5_batch_size : int
        Number of images buffered before writing to each HDF5 file.

    Returns
    -------
    metadata : pandas.DataFrame
        One row per composite, in HDF5 image order.
    """

    # Record the dataset paths and image-generation settings.
    config = {
        "source_dir": str(source_dir),
        "similarity_matrix_path": str(similarity_matrix_path),
        "class_index_path": str(class_index_path),
        "output_dir": str(output_dir),
        "bg_per_subcategory": bg_per_subcategory,
        "objects_per_class": objects_per_class,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "margin": margin,
        "min_object_dim": min_object_dim,
        "jpeg_quality": jpeg_quality,
        "seed": seed,
        "save_masks": save_masks,
        "save_silhouettes": save_silhouettes,
    }

    # Load the class data and select backgrounds.
    data = load_data(source_dir, similarity_matrix_path, class_index_path)
    backgrounds = sample_backgrounds(data["bg_metadata"], bg_per_subcategory, seed)

    # Generate images and write them in batches.
    writers = open_output_files(output_dir, save_masks, save_silhouettes, h5_batch_size)
    start_time = time.time()
    metadata, n_skipped = generate_images(backgrounds, data, writers, config)

    # Save the metadata and configuration, then report the totals.
    save_outputs(metadata, config, writers)
    print_summary(metadata, n_skipped, start_time)
    return metadata


def load_composite_image(h5_path, index):
    """Return the key and RGB image at an HDF5 row index."""
    with h5py.File(h5_path, "r") as f:
        key = f["keys"][index]
        img_bytes = f["images"][index]
    key = key.decode() if isinstance(key, bytes) else key
    return key, Image.open(io.BytesIO(img_bytes.tobytes())).convert("RGB")


def build_key_index(h5_path):
    """Map composite keys to their row indices in the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        keys = f["keys"][:]
    return {(k.decode() if isinstance(k, bytes) else k): i for i, k in enumerate(keys)}


def load_composite_by_key(h5_path, key, key_index=None):
    """Return an RGB composite by key, or None if absent; optionally reuse an index."""
    key_index = build_key_index(h5_path) if key_index is None else key_index
    if key not in key_index:
        return None
    return load_composite_image(h5_path, key_index[key])[1]


def load_h5_metadata(h5_path):
    """Load HDF5 metadata columns into a DataFrame."""
    with h5py.File(h5_path, "r") as f:
        metadata = {}
        for column in f["metadata"]:
            values = f["metadata"][column][:]
            if values.dtype.kind in ("O", "S", "U"):
                values = np.array([v.decode() if isinstance(v, bytes) else v for v in values])
            metadata[column] = values
    return pd.DataFrame(metadata)
