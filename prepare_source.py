"""Rebuild the ImageNet-OOC1k source folder from ImageNet and ImageNet-S.

The source folder ships with our metadata, the background patches and a manifest. This script adds the
originals, the object library and the masks from your ImageNet and ImageNet-S copies, and turns the patches into
the inpainted backgrounds. Every file is checked against the SHA-256 in the manifest (masks by their pixels, since
PNG compression differs between zlib builds).

Run it from the repository root:

    python prepare_source.py --imagenet-val /path/to/ILSVRC2012_img_val.tar \
                             --imagenet-s /path/to/ImageNetS919/validation-segmentation
"""

import argparse
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard
from PIL import Image
from tqdm.auto import tqdm

SOURCE_DIR = Path(__file__).resolve().parent / "ImageNet-OOC1k" / "source"


def get_sha256(data):
    """Return the SHA-256 hex digest of some bytes."""
    return hashlib.sha256(data).hexdigest()


def index_files(path, suffix):
    """Map file names to loaders for a folder (searched recursively), a .tar or a .zip file."""

    # Collect (name, loader) pairs from the given container.
    path = Path(path)
    if path.is_dir():
        entries = [(p.name, p.read_bytes) for p in path.rglob(f"*{suffix}")]
    elif path.suffix == ".tar":
        archive = tarfile.open(path)
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
        entries = [(Path(m.name).name, lambda m=m: archive.extractfile(m).read()) for m in members]
    elif path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        names = [n for n in archive.namelist() if n.endswith(suffix)]
        entries = [(Path(n).name, lambda n=n: archive.read(n)) for n in names]
    else:
        raise ValueError(f"Expected a folder, .tar or .zip file, got {path}")

    # Names that occur twice are marked as ambiguous (None).
    index = {}
    for name, loader in entries:
        index[name] = None if name in index else loader
    return index


def load_source(index, name, where):
    """Return the bytes of one input file, failing clearly if it is missing or ambiguous."""
    if name not in index:
        raise FileNotFoundError(f"{name} not found in {where}")
    if index[name] is None:
        raise ValueError(f"{name} occurs more than once in {where}; point to a folder that holds it only once")
    return index[name]()


def make_mask(segmentation):
    """Turn ImageNet-S annotation bytes into our binary mask PNG (255 where the class id R + 256 * G is not 0)."""
    seg = np.asarray(Image.open(io.BytesIO(segmentation)).convert("RGB")).astype(np.int32)
    mask = np.where(seg[..., 0] + 256 * seg[..., 1] != 0, 255, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    return buffer.getvalue()


def apply_patch(original, patch):
    """Rebuild a background from the original ImageNet file and its zstd patch."""
    reference = zstandard.ZstdCompressionDict(original, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    return zstandard.ZstdDecompressor(dict_data=reference).decompress(patch)


def get_pixel_sha256(data):
    """Return the SHA-256 of the decoded pixels of an image file."""
    return get_sha256(np.asarray(Image.open(io.BytesIO(data))).tobytes())


def matches_reference(data, row):
    """Check rebuilt bytes against the manifest. Masks are compared by pixels, the rest byte for byte."""
    if row.kind == "mask":
        return get_pixel_sha256(data) == row.pixel_sha256
    return get_sha256(data) == row.target_sha256


def prepare_source(imagenet_val, imagenet_s, source_dir=SOURCE_DIR):
    """
    Add the originals, object library, masks and backgrounds to the source folder.

    Files that already exist with the right content are skipped, so the script can be run again after an error.

    Parameters
    ----------
    imagenet_val : str or Path
        ImageNet validation images: ILSVRC2012_img_val.tar or an extracted folder (flat or one folder per class).
    imagenet_s : str or Path
        ImageNet-S-919 validation segmentations: the validation-segmentation folder or a zip that contains it.
    source_dir : str or Path
        The source folder with manifest.csv, the metadata and background_patches/.

    Returns
    -------
    manifest : pd.DataFrame
        One row per rebuilt file.
    """

    # Read the manifest and index the inputs.
    source_dir = Path(source_dir)
    manifest = pd.read_csv(source_dir / "manifest.csv")
    print("Indexing ImageNet and ImageNet-S (this can take a minute)")
    imagenet = index_files(imagenet_val, ".JPEG")
    segmentations = index_files(imagenet_s, ".png")

    n_skipped = 0
    for row in tqdm(manifest.itertuples(), total=len(manifest), desc="Rebuilding source"):
        target_path = source_dir / row.path
        if target_path.exists() and matches_reference(target_path.read_bytes(), row):
            n_skipped += 1
            continue

        # Load and check the input file.
        where = imagenet_s if row.kind == "mask" else imagenet_val
        source = load_source(segmentations if row.kind == "mask" else imagenet, row.source_name, where)
        if get_sha256(source) != row.source_sha256:
            raise ValueError(f"{row.source_name} in {where} is not the expected file (SHA-256 differs). "
                             "Use the original ILSVRC2012 validation images (not the face-blurred version) "
                             "and the official ImageNet-S annotations.")

        # Rebuild the file: masks from the annotation, backgrounds from the patch, the rest is copied.
        if row.kind == "mask":
            data = make_mask(source)
        elif row.kind == "background":
            data = apply_patch(source, (source_dir / row.patch_path).read_bytes())
        else:
            data = source

        # Check the result before writing it.
        if not matches_reference(data, row):
            raise ValueError(f"Rebuilt {row.path} does not match the reference (SHA-256 differs).")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)

    print(f"Rebuilt {len(manifest) - n_skipped:,} files in {source_dir} ({n_skipped:,} were already there)")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the ImageNet-OOC1k source folder from ImageNet and ImageNet-S.")
    parser.add_argument("--imagenet-val", required=True, help="ILSVRC2012_img_val.tar or a folder with the validation images")
    parser.add_argument("--imagenet-s", required=True, help="ImageNetS919 validation-segmentation folder or zip")
    parser.add_argument("--source-dir", default=SOURCE_DIR, help=f"source folder to complete (default: {SOURCE_DIR})")
    args = parser.parse_args()
    prepare_source(args.imagenet_val, args.imagenet_s, args.source_dir)
