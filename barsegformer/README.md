# BarSegFormer

A bar-aware semantic segmentation model for detecting vertical bar patterns on cichlid fish. Built as a custom modification of SegFormer-B1 using the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework.

> For the unmodified baseline SegFormer-B1, see the [official MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation). The instructions below cover BarSegFormer only.

---

## Architecture

BarSegFormer introduces four modifications to the standard SegFormer-B1:

| Component | Location | Description |
|---|---|---|
| **BarAwareStem** | `stripe_stem.py` | Replaces stage-1 patch embedding with parallel 3×3, 7×1, and 1×7 convolutions to capture horizontal and vertical bar structure |
| **LocalDWConv** | `mix_transformer.py` | 3×3 depthwise conv with residual connection appended after each of the four encoder stages |
| **U-shaped Decoder** | `barSeg_head.py` | Replaces the MLP decoder with skip connections, progressive upsampling, and 3×3 refinement convolutions |
| **Edge Supervision** | `barSeg_head.py` | Sobel-generated edge maps with auxiliary weighted BCE loss during training; edge head removed at inference |

**Combined loss:**
```
L_total = CrossEntropy(pred, gt) + 0.4 × BCE(edge_pred, edge_gt)
```

---

## Setup

### Prerequisites

- Conda
- CUDA 12.1

### Environment

From the repo root:

```bash
conda env create -f ../environment_segmentation.yml
conda activate segformer
```

Then install the local MMSegmentation package:

```bash
cd barsegformer
pip install -e .
```

---

## Data Preparation

Place your data under `barsegformer/data/my_dataset/`:

```
data/
└── my_dataset/
    ├── train_images/      # .jpg
    ├── train_masks/       # .png  (binary: pixel values 0 or 1)
    ├── val_images/        # .jpg
    └── val_masks/         # .png
```

> **Important:** Masks must be strictly binary (pixel values 0 or 1, not 0/255). If your masks were exported from Roboflow as JPEGs renamed to PNG, run a threshold pass before training: pixels > 127 → 1, else → 0.

For running inference over the full dataset (to generate masks for CichlidID), structure your data as:

```
data/main_data/
├── F3/
│   └── Tank_X/
│       └── Fish_N/
│           └── *.jpg
└── F4-Cross/
    └── ...
```

---

## Training

```bash
cd barsegformer

python tools/train.py \
  local_configs/barsegformer/barsegformer.b1.512x512.py \
  --launcher none \
  --work-dir work_dirs/barsegformer
```

### Configuration

| Setting | Value |
|---|---|
| Backbone | SegFormer-B1 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Iterations | 20,000 |
| Eval/checkpoint interval | 2,000 iters |
| Input size | 512×512 |
| num_workers | 0 (Windows) |

---

## Inference

Run segmentation over the full dataset to produce masks for the CichlidID pipeline:

```bash
python inference.py \
  --config local_configs/barsegformer/barsegformer.b1.512x512.py \
  --checkpoint work_dirs/barsegformer/iter_18000.pth \
  --input-dir data/main_data \
  --output-dir data/main_data/segmentations
```

Output masks mirror the input folder tree with a `_seg.png` suffix appended to each filename.

---

## Results

| Model | Best Iter | mIoU | Skin IoU | Bars IoU | mAcc | aAcc |
|---|---|---|---|---|---|---|
| SegFormer-B1 (baseline) | 16,000 | 87.14% | 97.31% | 76.97% | 92.00% | 97.53% |
| **BarSegFormer** | **18,000** | **88.51%** | — | — | — | — |

BarSegFormer reached the baseline's final mIoU by iteration 2,000, demonstrating significantly faster convergence.

---

## Windows-Specific Notes

- **`--launcher none`** is required. `torch.distributed.launch` is not supported on Windows.
- **SyncBN:** `segformer_head.py` hardcodes `SyncBN`, which is incompatible with single-GPU Windows training. Replace with `BN` directly in the file.
- **AdamW NaN loss:** SGD with lr=0.01 is stable on single-GPU setups. AdamW at default learning rates causes NaN loss.
- **Distributed backend:** Set `dist_params = dict(backend='gloo')` in the config.
- **DataLoader:** Set `num_workers=0` in the config to avoid multiprocessing errors.
