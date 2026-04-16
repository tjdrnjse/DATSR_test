# Reference-based Image Super-Resolution with Deformable Attention Transformer (ECCV 2022)

[Jiezhang Cao](https://www.jiezhangcao.com/), [Jingyun Liang](https://jingyunliang.github.io), [Kai Zhang](https://cszn.github.io/), [Yawei Li](https://ofsoundof.github.io/),  [Yulun Zhang](http://yulunzhang.com/), [Wenguan Wang](https://sites.google.com/view/wenguanwang), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)

Computer Vision Lab, ETH Zurich.

---

[arxiv](https://arxiv.org/abs/2207.11938)
**|**
[supplementary](https://github.com/caojiezhang/DATSR/releases)
**|**
[pretrained models](https://github.com/caojiezhang/DATSR/releases)
**|**
[visual results](https://github.com/caojiezhang/DATSR/releases)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2207.11938)
[![GitHub Stars](https://img.shields.io/github/stars/caojiezhang/DATSR?style=social)](https://github.com/caojiezhang/DATSR)
[![download](https://img.shields.io/github/downloads/caojiezhang/DATSR/total.svg)](https://github.com/caojiezhang/DATSR/releases)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=caojiezhang/DATSR)

This repository is the official PyTorch implementation of "Reference-based Image Super-Resolution with Deformable Attention Transformer"
([arxiv](https://arxiv.org/abs/2207.11938), [supp](https://github.com/caojiezhang/DATSR/releases/download/v0.0/supplementary.pdf), [pretrained models](https://github.com/caojiezhang/DATSR/releases), [visual results](https://github.com/caojiezhang/DATSR/releases)). 

---

Reference-based image super-resolution (RefSR) aims to exploit auxiliary reference (Ref) images to super-resolve low-resolution (LR) images. Recently, RefSR has been attracting great attention as it provides an alternative way to surpass single image SR. However, addressing the RefSR problem has two critical challenges: (i) It is difficult to match the correspondence between LR and Ref images when they are significantly different; (ii) How to transfer the relevant texture from Ref images to compensate the details for LR images is very challenging. To address these issues of RefSR, this paper proposes a deformable attention Transformer, namely DATSR, with multiple scales, each of which consists of a texture feature encoder (TFE) module, a reference-based deformable attention (RDA) module and a residual feature aggregation (RFA) module. Specifically, TFE first extracts image transformation (e.g., brightness) insensitive features for LR and Ref images, RDA then can exploit multiple relevant textures to compensate more information for LR features, and RFA lastly aggregates LR features and relevant textures to get a more visually pleasant result. Extensive experiments demonstrate that our DATSR achieves state-of-the-art performance on benchmark datasets quantitatively and qualitatively.

<p align="center">
  <img width="1000" src="assets/framework.png">
</p>

#### Contents

1. [Requirements](#Requirements)
1. [Quick Testing](#Quick-Testing)
1. [Training](#Training)
1. [Results](#Results)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)


## TODO
- [ ] Add pretrained model
- [ ] Add results of test set


## Requirements
> - Python 3.10
> - PyTorch 2.1.0 + torchvision 0.16.0
> - CUDA 11.8 or CUDA 12.1 (CPU-only inference also supported)

### Core Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.1.0 | Main framework |
| torchvision | 0.16.0 | Replaces C++ DCNv2 (`deform_conv2d`) |
| timm | >= 0.6.12 | Swin Transformer layers |
| mmcv | >= 2.0.0 | Runner utilities |
| pyyaml | >= 6.0 | Config parsing |
| numpy | >= 1.24 | Array ops |
| pillow | >= 9.0 | Image I/O |
| opencv-python | >= 4.7 | Image processing |
| tqdm | >= 4.65 | Progress bars |

## Quick Testing
Following commands will download [pretrained models](https://github.com/caojiezhang/DATSR/releases) and [test datasets](https://github.com/caojiezhang/DATSR/releases). 

1. Clone Repo and Install Dependencies
    ```bash
   git clone https://github.com/caojiezhang/DATSR.git
   cd DATSR
   # PyTorch 2.1.0 (CUDA 11.8)
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
   pip install timm mmcv pyyaml numpy pillow opencv-python tqdm
   ```


## Dataset 

- Train Set: [CUFED Dataset](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I)
- Test Set: [WR-SR Dataset](https://drive.google.com/drive/folders/16UKRu-7jgCYcndOlGYBmo5Pp0_Mq71hP?usp=sharing), [CUFED5 Dataset](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)

Please refer to [Datasets.md](datasets/DATASETS.md) for pre-processing and more details.

## Get Started

### Pretrained Models
Downloading the pretrained models from this [link](https://github.com/caojiezhang/DATSR/releases) and put them under `experiments/pretrained_model folder`.

### Test

We provide quick test code with the pretrained model.

```bash
# Run test code for models trained using only **reconstruction loss**.
PYTHONPATH="./:${PYTHONPATH}" python datsr/test.py -opt "options/test/test_restoration_mse.yml"

# Run test code for models trained using **GAN loss**.
PYTHONPATH="./:${PYTHONPATH}" python datsr/test.py -opt "options/test/test_restoration.yml"
```

## Training

Train restoration network
```bash
# Train the restoration network with only mse loss
PYTHONPATH="./:${PYTHONPATH}" python datsr/train.py -opt "options/train/train_restoration_mse.yml"

# Train the restoration network with all loss
PYTHONPATH="./:${PYTHONPATH}" python datsr/train.py -opt "options/train/train_restoration_gan.yml"
```

## Pipeline Overview

This repository provides three fully independent pipelines:

| Pipeline | Script | Purpose | GT Required |
|----------|--------|---------|-------------|
| **Train** | `datsr/train.py` | Train restoration network | Yes |
| **Test** | `datsr/test.py` | Benchmark evaluation (PSNR/SSIM/LPIPS) | Yes |
| **Inference** | `inference.py` | In-the-wild SR, no ground-truth needed | No |

`test.py` and `inference.py` are completely separate — tiling logic lives only in `inference.py`.

---

## Standalone Inference (No GT Required)

For real-world / in-the-wild images where you only have LR and Ref pairs (no HR ground-truth), use the standalone inference pipeline.

### Input Data Layout

```
data/inference/
  LR/
    0001.png
    0002.png
    ...
  Ref/
    0001.png   <-- must share the exact same filename as the LR counterpart
    0002.png
    ...
```

**Name-Matching Rule:** The pipeline pairs LR and Ref images by filename stem (extension-agnostic). If a LR file has no Ref with the same stem, it is skipped with a warning.

### Quick Start

```bash
python inference.py -opt options/inference/inference_datsr_tiling.yml
```

Results are saved to the path specified by `output_dir` in the YAML (default: `results/inference/SR/`).

### YAML Configuration

Edit `options/inference/inference_datsr_tiling.yml`:

```yaml
# Input / Output paths
dataroot_lr:  data/inference/LR
dataroot_ref: data/inference/Ref
output_dir:   results/inference/SR

# Pretrained weights
pretrain_model_g:          experiments/pretrained_model/restoration_mse.pth
pretrain_model_extractor:  experiments/pretrained_model.pth

# Tiling settings
tiling:
  lr_tile_size: 64          # LR tile side length (px). Smaller = less VRAM. Recommended: 64-128
  lr_overlap_pixels: 16     # Overlap between tiles. Stride = lr_tile_size - lr_overlap_pixels
  ref_search_margin: 64     # Extra margin (px) added around each Ref crop
  padding_mode: "reflect"   # Boundary padding: "reflect" | "replicate" | "constant"
  blending_method: "gaussian"  # Blend weights: "gaussian" | "linear"
  gaussian_sigma: 0.5       # Gaussian window sigma as fraction of half-width

# Runtime
gpu_ids: [0]   # Use [] for CPU-only
fp16: false    # Half-precision inference (GPU only)
```

| Option | Description |
|--------|-------------|
| `lr_tile_size` | Square tile size in LR space. Reduce to save VRAM. |
| `lr_overlap_pixels` | Wider overlap reduces seam artifacts at tile boundaries. |
| `ref_search_margin` | Larger margin gives the correspondence module more search area. |
| `blending_method` | `gaussian`: center-weighted smooth blend. `linear`: tent/triangular blend. |

## Visual Results

For more results on the benchmarks, you can directly download our DATSR results from [here](https://github.com/caojiezhang/DATSR/releases).

![result](assets/rec_comp.png)

![result](assets/gan_comp.png)

## Citation
  ```
  @inproceedings{cao2022datsr,
    title={Reference-based Image Super-Resolution with Deformable Attention Transformer},
    author={Cao, Jiezhang and Liang, Jingyun and Zhang, Kai and Li, Yawei and Zhang, Yulun and Wang, Wenguan and Van Gool, Luc},
    booktitle={European conference on computer vision},
    year={2022}
  }
  ```

## License and Acknowledgement
This project is released under the CC-BY-NC license. We refer to codes from [C2-Matching](https://github.com/yumingj/C2-Matching) and [BasicSR](https://github.com/xinntao/BasicSR). Thanks for their awesome works. The majority of DATSR is licensed under CC-BY-NC, however portions of the project are available under separate license terms: C2-Matching is licensed under the MIT License, BasicSR are licensed under the Apache 2.0 license.
