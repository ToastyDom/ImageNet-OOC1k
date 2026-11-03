# How Far Is Too Far?
## Object Recognition Declines Monotonically with Semantic Distance

Code and dataset resources for our **NeurIPS 2026 Spotlight paper** by [Domenic Bersch](https://toastydom.github.io/), Hai Van Tran, and [Gemma Roig, Goethe University Frankfurt](https://www.cvai.cs.uni-frankfurt.de/).

Vision models recognise objects partly through their surroundings. But how does recognition change as an object and its background become less related?

**ImageNet-OOC1k** lets us study this relationship across a graded range of object–background combinations. We separate objects from ImageNet images, inpaint their original backgrounds, and recombine them at different visual-semantic distances. A **consensus visual-semantic axis**, built from the classifier heads of ten ImageNet models, measures the relationship between classes.

We evaluate 35 models and examine their internal representations to understand how object and background evidence contribute to recognition.

<p align="center">
  <img src="assets/MainFigure.png" width="90%" alt="Overview of ImageNet-OOC1k, controlled object–background combinations, and recognition accuracy across visual-semantic distance">
</p>

## Contents

- [Main findings](#main-findings)
- [Dataset](#dataset)
- [Installation](#installation)
- [Getting the data](#getting-the-data)
- [Running the notebooks](#running-the-notebooks)
- [Acknowledgements](#acknowledgements)
- [Other resources](#other-resources)
- [Citation](#citation)

## Main findings

- **Recognition declines as objects and backgrounds become less related.** Across all 35 evaluated models, top-1 accuracy drops by approximately 9–19 percentage points from the most related to the least related combinations.
- **Backgrounds carry structured class information.** With the target object removed, models still predict its class above chance. Their incorrect predictions also tend to fall within related semantic categories.
- **Vision Transformers can retain object evidence despite an incorrect prediction.** In these “found-but-lost” cases, patch-level predictions identify the object, but the final prediction selects another class. Spatial re-ranking recovers a substantial fraction of these errors, improving accuracy by 5.7 percentage points on the diagnostic subset.
- **The decline persists across additional axes and generation controls.** It also appears under discriminative axes built from independently pretrained representations and remains after accounting for measured placement, scale, background, cutout, and boundary properties.

## Dataset

ImageNet-OOC1k is derived from [ImageNet](https://image-net.org) validation images and [ImageNet-S](https://github.com/LUSSeg/ImageNet-S) segmentation masks.

| Component | Description |
|---|---|
| Backgrounds | 1,137 inpainted scenes, comprising 558 nature, 533 human-related, and 46 miscellaneous backgrounds |
| Object library | 12,321 segmented objects spanning 918 ImageNet classes |
| Possible pairings | Approximately 14 million object-instance–background combinations |
| Evaluated composites | 953,862 images generated using 260 backgrounds and the available object-class pool |
| Mechanistic subset | 75,000 composites for patch-level analyses of five Vision Transformers |
| Controls | Original images, backgrounds alone, and objects alone on black, white, and `1/f` noise backgrounds |

The dataset decouples object identity from background context. Composite specifications record the object, background, placement, scale, and other generation parameters.

We distribute metadata and reconstruction files rather than complete ImageNet-derived images. To recreate the dataset, you need your own copies of ImageNet and the ImageNet-S annotations. The steps below reconstruct the source images; notebook 02 then generates the composites.

## Installation

Clone the repository and create the environment.

```bash
git clone https://github.com/ToastyDom/ImageNet-OOC1k.git
cd ImageNet-OOC1k
conda env create -f environment.yml
conda activate ooc1k
```

The environment includes Python 3.11, PyTorch 2.10, timm 1.0.26, and the repository's `ooc1k` package. Pretrained model weights are downloaded when the models are first loaded.

## Getting the data

### 1. Download the reconstruction files

Download our source files from **[LINK TBA]** and extract them into `ImageNet-OOC1k/source/` inside the repository.

This directory should contain:

- `metadata.csv`
- `metadata_objects.csv`
- `manifest.csv`
- `versions.json`
- `background_patches/`

The background patches contain the inpainted regions used to reconstruct our backgrounds from the original ImageNet images.

### 2. Download the ImageNet validation images

Download `ILSVRC2012_img_val.tar` (approximately 6.3 GB) from [ImageNet](https://image-net.org/download-images).

Use the **original validation images**, rather than the face-blurred version. The reconstruction script accepts either the tar archive or an extracted directory.

### 3. Download the ImageNet-S masks

Download the **ImageNet-S-919 validation segmentations** from [ImageNet-S](https://github.com/LUSSeg/ImageNet-S).

You need the `ImageNetS919/validation-segmentation` directory, or a zip archive containing it.

### 4. Reconstruct the source dataset

From the repository root, run:

```bash
python prepare_source.py \
    --imagenet-val /path/to/ILSVRC2012_img_val.tar \
    --imagenet-s /path/to/ImageNetS919/validation-segmentation
```

Replace the paths with the locations of your downloaded files.

The script extracts the 1,137 original images and 12,321 object-library images, builds the object masks from the ImageNet-S annotations, and reconstructs the inpainted backgrounds. Checksums verify that the reconstructed files match the expected dataset.

You can then generate the composites and control images with notebook 02.

> Notebook 08 also requires ImageNet **training** images to fit the linear probes. These are only needed if you want to rerun that analysis.

## Running the notebooks

The notebooks follow the analysis from axis construction and dataset generation to model evaluation and diagnostic controls. Open them from the repository root and follow their instructions in order. Figures are saved to `outputs/figures/`.

| Notebook | Purpose | Paper |
|---|---|---|
| `01_semantic_axes` | Construct the consensus visual-semantic axis, individual classifier-head axes, and CLIP-text and WordNet comparisons | §4, Appendix B |
| `02_dataset_generation` | Generate composites, object-only controls, and the mechanistic subset | §3, Appendix A |
| `03_model_inference` | Run the 35 evaluated models and save their predictions | Behavioral experiments |
| `04_object_background_controls` | Evaluate objects and backgrounds separately and analyse background-only predictions | §5.1, Appendix C.1 |
| `05_semantic_distance` | Measure accuracy across visual-semantic distance and compare individual backgrounds | §5.2, Appendices C.2–C.3 |
| `06_patch_extraction` | Extract patch-level predictions and characterise found-but-lost errors across five Vision Transformers | §6, Appendices D and F |
| `07_spatial_reranking` | Test recoverability using spatial re-ranking | §6.3, Appendices E and F |
| `08_visual_semantic_axes` | Test model overlap and compare independently constructed probe, retrieval, linguistic, and taxonomic axes | Appendix C.4 |
| `09_generation_artifacts` | Test whether measured generation properties explain the accuracy decline | Appendix A.2 |

### Compute requirements

The dataset occupies approximately **98 GB**. The reported model inference and mechanistic extraction runs took approximately **97 GPU-hours on one NVIDIA A100 with 80 GB memory**. Running only the six main models reduces the inference workload.

### Generating your own variants

Use `02_dataset_generation.ipynb` to follow the generation pipeline and create your own dataset variants. The notebook brings together composite generation, object-only controls, and sampling of the mechanistic subset.

The underlying composite-generation code is in `ooc1k/dataset_generation/composites.py`. To explore other placement, scaling, or blending procedures, adjust the corresponding settings or implementation, regenerate the composites through notebook 02, and rerun inference.

## Acknowledgements

TBA

## Other resources

Our dataset and analyses build on the following resources:

- **[ImageNet](https://image-net.org)** provides the source images and class labels.
- **[ImageNet-S](https://github.com/LUSSeg/ImageNet-S)** provides the segmentation annotations used to construct the object masks.
- **[IOPaint](https://github.com/Sanster/IOPaint)** and the MAT model were used for background inpainting.
- **[timm](https://github.com/huggingface/pytorch-image-models)** provides pretrained vision models and inference utilities.
- **[OpenAI CLIP](https://github.com/openai/CLIP)** provides the image–text model used in our evaluations.

ImageNet-derived components remain subject to the source datasets' access and redistribution terms. If you use ImageNet or ImageNet-S, please also cite their original publications using the citation information on their project pages.

The code license is provided in [LICENSE](LICENSE).

## Citation

If you use ImageNet-OOC1k or this code, please cite:

```bibtex
TBA
```