"""Generate object-only images by replacing the background around each object."""

import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# Mask threshold, flat background values, and image locations inside the source directory.
MASK_THRESHOLD = 127
FLAT_BACKGROUNDS = {"black": 0, "white": 255}
IMAGE_EXTENSIONS = (".JPEG", ".jpeg", ".jpg", ".png", ".PNG")

IMAGE_SETS = {
    "main": ("originals", "masks"),
    "library": ("object_library/images", "object_library/masks"),
}


def find_original(mask_path, masks_dir, originals_dir):
    """Return the original image with the same relative path as a mask, or None."""
    rel_path = mask_path.relative_to(masks_dir)
    for ext in IMAGE_EXTENSIONS:
        candidate = originals_dir / rel_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def load_image_and_mask(original_path, mask_path, mask_threshold):
    """Load an RGB image and its binary object mask with a trailing channel axis."""
    original = np.array(Image.open(original_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Match the mask dimensions to the image, keeping the mask binary.
    if mask.shape[:2] != original.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize((original.shape[1], original.shape[0]), resample=Image.NEAREST))

    mask = (mask > mask_threshold).astype(np.float32)
    return original, mask[:, :, np.newaxis]


def make_seed(filename):
    """Derive a seed from a file name, so every image has its own noise pattern."""
    return int(hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16], 16)


def make_pink_noise(height, width, rng):
    """Generate one channel of pink (1/f) noise, scaled to [0, 255]."""

    # Build a 1/f amplitude spectrum without a constant (DC) component.
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]
    freq_x = np.fft.fftfreq(width)[np.newaxis, :]
    freq_magnitude = np.sqrt(freq_y**2 + freq_x**2)
    freq_magnitude[0, 0] = 1.0
    amplitude = 1.0 / freq_magnitude
    amplitude[0, 0] = 0.0

    # Combine it with random phases and transform back to image space.
    phase = rng.uniform(0, 2 * np.pi, size=(height, width))
    noise = np.fft.ifft2(amplitude * np.exp(1j * phase)).real

    # Rescale the noise to the 8-bit range.
    noise -= noise.min()
    if noise.max() > 0:
        noise /= noise.max()
    return (noise * 255).astype(np.uint8)


def make_background(background, height, width, seed_name):
    """Return a flat grey value or an RGB pink-noise image for one object."""
    if background in FLAT_BACKGROUNDS:
        return FLAT_BACKGROUNDS[background]

    # Draw the red, green, and blue channels in turn from one seeded generator.
    rng = np.random.default_rng(make_seed(seed_name))
    noise = np.stack([make_pink_noise(height, width, rng) for _ in range(3)], axis=-1)
    return noise.astype(np.float32)


def make_object_only_image(original, mask, background):
    """Keep the object pixels and fill everything else with the background."""
    result = original.astype(np.float32) * mask + background * (1.0 - mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_object_only_images(source_dir, output_dir, backgrounds=("black", "white", "pink_noise"), image_sets=("main", "library"), mask_threshold=MASK_THRESHOLD):
    """
    Place every object of the main images and the object library on neutral backgrounds.

    Parameters
    ----------
    source_dir : str or pathlib.Path
        Dataset source directory containing the original images, the object library and their masks.
    output_dir : str or pathlib.Path
        Destination, written as `<image set>/<background>/` with the folder structure of the masks.
    backgrounds : tuple of str
        Backgrounds to generate, any of "black", "white" and "pink_noise".
    image_sets : tuple of str
        Image sets to process, any of "main" and "library".
    mask_threshold : int
        Mask values above this threshold belong to the object.

    Returns
    -------
    counts : dict
        Number of images created per image set.
    """

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    counts = {}
    start_time = time.time()

    for image_set in image_sets:
        # Find every mask of this image set; each one defines one object-only image.
        originals_dir = source_dir / IMAGE_SETS[image_set][0]
        masks_dir = source_dir / IMAGE_SETS[image_set][1]
        mask_paths = sorted(masks_dir.rglob("*.png"))

        n_created = 0
        n_skipped = 0
        for mask_path in tqdm(mask_paths, desc=f"Object-only ({image_set})"):
            original_path = find_original(mask_path, masks_dir, originals_dir)
            if original_path is None:
                n_skipped += 1
                continue

            # Load the object once and save it on every requested background.
            original, mask = load_image_and_mask(original_path, mask_path, mask_threshold)
            for background in backgrounds:
                fill = make_background(background, original.shape[0], original.shape[1], mask_path.stem)
                out_path = output_dir / image_set / background / mask_path.relative_to(masks_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(make_object_only_image(original, mask, fill)).save(out_path.with_suffix(".png"))
            n_created += 1

        counts[image_set] = n_created
        print(f"{image_set}: {n_created:,} objects on {len(backgrounds)} backgrounds ({n_skipped} masks without image)")

    minutes = (time.time() - start_time) / 60
    print(f"Created {sum(counts.values()) * len(backgrounds):,} object-only images in {minutes:.1f} min")
    return counts
