import os
import glob
import torch
import numpy as np
from PIL import Image
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmseg.datasets import build_dataloader
import mmcv

# Setup
config_file = 'local_configs/segformer/B1/barSegFormer.b1.512x512.ade.20k.py'
checkpoint_file = 'work_dirs/barSegFormer.b1.512x512.ade.20k/iter_18000.pth'
data_root = 'data/main_data'
output_root = 'data/main_data/segmentations'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Build model
cfg = Config.fromfile(config_file)
cfg.model.pretrained = None
cfg.model.train_cfg = None
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint_file, map_location='cpu')
model = model.to(device)
model.eval()

# Image preprocessing pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
pipeline = Compose(test_pipeline)

# Find all jpg/png images recursively, skip CR3
image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

image_paths = []
for root, dirs, files in os.walk(data_root):
    # Skip the segmentations output folder
    if 'segmentations' in root:
        continue
    for f in files:
        if f.endswith(image_extensions):
            image_paths.append(os.path.join(root, f))

print(f"Found {len(image_paths)} images")

# Run inference
for img_path in image_paths:
    # Build output path mirroring input structure
    rel_path = os.path.relpath(img_path, data_root)
    rel_dir = os.path.dirname(rel_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename + '_seg.png')
    
    # Skip if already processed
    if os.path.exists(out_path):
        print(f"Skipping (already done): {rel_path}")
        continue
    
    # Load and preprocess
    data = dict(img_info=dict(filename=img_path), img_prefix='')
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [device])[0]
    
    # Inference
    with torch.no_grad():
        result = model(return_loss=False, **data)
    
    # Save segmentation mask (0=skin, 1=bars)
    # Save as visible image: bars=white, skin=black
    seg_map = result[0].astype(np.uint8)
    visible = (seg_map * 255).astype(np.uint8)
    Image.fromarray(visible).save(out_path)
    
    #print(f"Done: {rel_path}")

print("All done!")