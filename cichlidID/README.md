# CichlidID

Closed-set individual re-identification of cichlid fish using a modified DINOv2 (ViT-S/14) backbone fine-tuned for 25 individuals.

Three input variants are provided as an ablation study. All share the same code structure -- the differences are in the input pipeline (how the image and mask are loaded and combined in `dataset.py` and `model.py`).

---

## Variants

| Folder | Input | Mask source |
|---|---|---|
| `rgb_only/` | 3-channel RGB | -- |
| `four_channel_segformer/` | 4-channel RGB + mask | Baseline SegFormer-B1 |
| `four_channel_barsegformer/` | 4-channel RGB + mask | BarSegFormer |

---

## Architecture

The base model is DINOv2 ViT-S/14 fine-tuned with a linear classification head for 25-class identification.

For the two 4-channel variants, the patch embedding projection is expanded from 3 to 4 input channels. The weights for the 4th channel are initialised to zero so that pretrained RGB features are fully preserved at the start of fine-tuning.

```
rgb_only:
  [R, G, B]  →  DINOv2 ViT-S/14  →  [CLS] token  →  Linear(768, 25)  →  Identity

four_channel variants:
  [R, G, B, Mask]  →  Modified Patch Embed (3→4 ch)  →  DINOv2 ViT-S/14  →  [CLS] token  →  Linear(768, 25)  →  Identity
```

---

## Scripts

Each variant folder contains the same set of scripts:

| Script | Purpose |
|---|---|
| `train.py` | Train the model |
| `evaluate.py` | Evaluate on the validation set, outputs accuracy and confusion matrix |
| `model.py` | Model definition (DINOv2 + classification head; 4-ch patch embed for non-RGB variants) |
| `dataset.py` | Dataset and dataloader (handles RGB or RGB+mask pairing) |
| `preprocess.py` | Resizes images and masks, outputs to `data_224/` |
| `specs.py` | Training hyperparameters and paths |
| `find_confident.py` | Filter predictions by confidence threshold |
| `show_fish_prediction.py` | Visualise per-image predictions |
| `time_check.py` | Benchmark inference time |
| `label_map.json` | Maps folder names to class indices |

---

## Setup

From the repo root:

```bash
python -m venv CichlidID_env

# Windows
CichlidID_env\Scripts\activate

# Linux / macOS
source CichlidID_env/bin/activate

pip install -r ../requirements_cichlidid.txt
```

---

## Data Preparation

### Step 1 -- Run BarSegFormer inference

First, run inference using BarSegFormer (or baseline SegFormer) on your fish images following the instructions in [../barsegformer/README.md](../barsegformer/README.md). This produces a `segmentations/` folder mirroring the image folder structure.

### Step 2 -- Organise the data folder

Place your images and segmentations under `data/` inside the variant folder you want to train. The structure should be:

```
data/
├── F3/
│   ├── Tank_1/
│   │   ├── Fish_1/
│   │   │   ├── img_001.jpg
│   │   │   ├── img_002.jpg
│   │   │   └── ...
│   │   ├── Fish_2/
│   │   └── ...
│   └── Tank_2/
│       └── ...
├── F4-Cross/
│   └── ...  (same structure as F3)
└── segmentations/
    ├── F3/
    │   ├── Tank_1/
    │   │   ├── Fish_1/
    │   │   │   ├── img_001_seg.png
    │   │   │   ├── img_002_seg.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── F4-Cross/
        └── ...
```

The `segmentations/` folder mirrors the exact same F3/F4-Cross/Tank/Fish structure as the image folders, with `_seg.png` masks.

For `rgb_only/`, only the `F3/` and `F4-Cross/` image folders are needed -- no `segmentations/` folder required.

### Step 3 -- Preprocess

Run `preprocess.py` to resize images and masks. This creates a `data_224/` folder with the same structure as `data/` but with all images resized to 224×224:

```bash
python preprocess.py
```

The model trains on `data_224/`, not `data/` directly.

**Train/Val split:** 80/20, stratified by individual, `seed=42` -- 863 train / 208 val.

---

## Training

Navigate into the variant folder you want to train, edit `specs.py` to set your data paths and hyperparameters, then run:

```bash
cd cichlidID/rgb_only
python train.py

cd cichlidID/four_channel_segformer
python train.py

cd cichlidID/four_channel_barsegformer
python train.py
```

---

## Evaluation

```bash
cd <variant_folder>
python evaluate.py
```

Outputs top-1 accuracy and a per-class confusion matrix.

---

## Visualisation

To visualise predictions on individual images:

```bash
python show_fish_prediction.py
```

To filter and inspect high-confidence predictions:

```bash
python find_confident.py
```

---

## Results

| Metric | RGB Only | RGB + SegFormer Mask | RGB + BarSegFormer Mask |
|---|---|---|---|
| Top-1 Accuracy | 82.21% | **84.62%** | 79.81% |
| Top-3 Accuracy | 99.52% | 99.04% | 95.67% |
| Top-5 Accuracy | 100.00% | 100.00% | 98.08% |
| Macro Precision | 0.83 | **0.86** | 0.82 |
| Macro Recall | 0.82 | **0.84** | 0.79 |
| Macro F1-score | 0.82 | **0.84** | 0.79 |

The `four_channel_segformer` variant achieved the best performance across all metrics. `four_channel_barsegformer` underperformed relative to the other two variants, likely due to domain shift -- BarSegFormer was trained on a different fish dataset, so its masks on the re-ID data are less precise than those from the baseline SegFormer.

---

## Notes

- `num_workers=0` is required on Windows.
- The zero-initialisation of the 4th channel weights in the patch embedding is critical -- without it, fine-tuning from the pretrained DINOv2 checkpoint is unstable.
- `data_224/` is generated by `preprocess.py` and should not be committed to the repository.
