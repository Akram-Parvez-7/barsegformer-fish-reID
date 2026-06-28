# CichlidReID: A Segmentation-Guided Foundation Model Framework for Non-Invasive Individual Fish Re-Identification

A two-stage computer vision pipeline for closed-set individual re-identification of cichlid fish using segmentation-guided visual features.

**Author:** Akram Shaik

---

## Overview

This repository contains the full implementation of a fish re-identification framework built around two components:

1. **BarSegFormer** -- a custom SegFormer-based semantic segmentation model that detects the vertical bar patterns on cichlid fish bodies.
2. **CichlidID** -- a modified DINOv2 (ViT-S/14) model for closed-set individual identification, with three input variants evaluated.

The pipeline works as follows:

```
Input Image (RGB)
      │
      ▼
 BarSegFormer ──► Binary Bar Mask
      │
      ▼
  RGB + Mask (4-channel)
      │
      ▼
  CichlidID (DINOv2) ──► Individual Identity (1 of 25)
```

---

## Repository Structure

```
barsegformer-fish-reID/
├── README.md
├── .gitignore
├── environment_segmentation.yml        # Conda env for BarSegFormer
├── requirements_cichlidid.txt          # Pip requirements for CichlidID
│
├── barsegformer/                       # Stage 1: bar segmentation model
│   ├── README.md
│   └── SegFormer/                      # Modified SegFormer codebase
│
└── cichlidID/                          # Stage 2: individual re-ID model
    ├── README.md                       # Covers all three input variants
    ├── rgb_only/                       # Variant A: 3-channel RGB baseline
    ├── four_channel_segformer/         # Variant B: RGB + baseline SegFormer masks
    └── four_channel_barsegformer/      # Variant C: RGB + BarSegFormer masks
```

---

## Components

### BarSegFormer

A modified SegFormer-B1 with four custom additions: a bar-aware stem with parallel horizontal/vertical convolutions, local depthwise convolutions after each encoder stage, a U-shaped skip-connection decoder, and edge supervision during training.

- Best checkpoint: `iter_18000.pth` -- **88.51% mIoU**
- Reached baseline SegFormer's final performance by iteration 2,000
- [Setup and training instructions](barsegformer/README.md)

For the **baseline SegFormer-B1** (no modifications), refer to the [official MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation). Training configuration used here is `segformer.b1.512x512.ade.160k.py` with SGD lr=0.01 for 20,000 iterations.

### CichlidID

DINOv2 ViT-S/14 fine-tuned for 25-class closed-set identification. Three input variants were evaluated as an ablation study. All three share the same codebase structure (`train.py`, `model.py`, `dataset.py`, `evaluate.py`) with modifications to the input pipeline.

- [Setup and usage instructions for all variants](cichlidID/README.md)

---

## Dataset

The dataset consists of **1,071 images** of **25 individual cichlid fish** across F3 and F4-Cross generations, housed in 5 tanks. Each image is paired with a binary segmentation mask (`_seg.png`) annotating the vertical bar pattern.

| Split | Images |
|---|---|
| Segmentation train | 428 |
| Segmentation val | 105 |
| Re-ID train | 863 |
| Re-ID val | 208 |

The dataset is not included in this repository.

---

## Setup

### BarSegFormer

```bash
conda env create -f environment_segmentation.yml
conda activate segformer
```

### CichlidID

```bash
python -m venv CichlidID_env
CichlidID_env\Scripts\activate      # Windows
pip install -r requirements_cichlidid.txt
```

Full instructions in each component's README.

---

## Hardware

Developed and tested on:

- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- RAM: 16GB
- OS: Windows 11
- CUDA: 12.1

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{shaik2026cichlidreID,
  author = {Akram Shaik},
  title  = {Segmentation-guided foundation model framework for non-invasive individual fish re-identification},
  school = {University of Helsinki},
  year   = {2026}
}
```

---

## Acknowledgements

This work builds on:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) -- OpenMMLab
- [DINOv2](https://github.com/facebookresearch/dinov2) -- Meta AI Research
- [SegFormer](https://github.com/NVlabs/SegFormer) -- NVIDIA
