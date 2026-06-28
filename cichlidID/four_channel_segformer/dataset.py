import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json

class CichlidDataset(Dataset):
    def __init__(self, samples, label_map, transform=None, mask_transform=None):
        """
        samples: list of (jpg_path, png_path, label) tuples
        label_map: dict mapping identity string to integer label
        """
        self.samples = samples
        self.label_map = label_map
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpg_path, png_path, label = self.samples[idx]

        # Load RGB image
        image = Image.open(jpg_path).convert("RGB")

        # Load mask (bar segmentation)
        mask = Image.open(png_path).convert("L")  # grayscale

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Concatenate along channel dim → 4-channel tensor
        combined = torch.cat([image, mask], dim=0)

        return combined, label


def build_dataset(data_dir, val_split=0.2, image_size=224, seed=42):
    """
    Walks DATA/ and builds paired (JPG, PNG, label) samples.
    Splits by fish identity to avoid leakage.
    """
    data_dir = Path(data_dir)
    seg_dir = data_dir / "segmentations"

    # Collect all identities and their images
    # identity_dict: { "F3/Tank_A2(A2-Extra F3)/Fish_1": [jpg_path, ...], ... }
    identity_dict = {}

    for type_dir in ["F3", "F4-Cross"]:
        type_path = data_dir / type_dir
        if not type_path.exists():
            continue
        for tank_dir in sorted(type_path.iterdir()):
            if not tank_dir.is_dir():
                continue
            for fish_dir in sorted(tank_dir.iterdir()):
                if not fish_dir.is_dir():
                    continue
                identity = f"{type_dir}/{tank_dir.name}/{fish_dir.name}"
                jpgs = sorted(fish_dir.glob("*.JPG"))
                if len(jpgs) == 0:
                    continue
                identity_dict[identity] = jpgs

    # Build label map: identity string → integer
    identity_list = sorted(identity_dict.keys())
    label_map = {identity: idx for idx, identity in enumerate(identity_list)}

    # Save label map for inference later
    with open(data_dir.parent / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Found {len(identity_list)} identities:")
    for identity, idx in label_map.items():
        print(f"  [{idx:02d}] {identity} — {len(identity_dict[identity])} images")

    # Build samples list: (jpg_path, png_path, label)
    all_samples = []
    missing_masks = 0

    for identity, jpgs in identity_dict.items():
        label = label_map[identity]
        parts = identity.split("/")  # [type, tank, fish]

        for jpg_path in jpgs:
            png_path = seg_dir / parts[0] / parts[1] / parts[2] / (jpg_path.stem + "_seg.png")
            if not png_path.exists():
                missing_masks += 1
                continue
            all_samples.append((jpg_path, png_path, label))

    print(f"\nTotal samples: {len(all_samples)}")
    if missing_masks > 0:
        print(f"Warning: {missing_masks} JPGs had no matching segmentation mask")

    # Train/val split — split by identity to avoid leakage
    import random
    random.seed(seed)

    train_samples, val_samples = [], []

    for identity, jpgs in identity_dict.items():
        label = label_map[identity]
        parts = identity.split("/")

        identity_samples = []
        for jpg_path in jpgs:
            png_path = seg_dir / parts[0] / parts[1] / parts[2] / (jpg_path.stem + "_seg.png")
            if png_path.exists():
                identity_samples.append((jpg_path, png_path, label))

        random.shuffle(identity_samples)
        n_val = max(1, int(len(identity_samples) * val_split))
        val_samples.extend(identity_samples[:n_val])
        train_samples.extend(identity_samples[n_val:])

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # Transforms
    # Note: mask uses same spatial transforms but no color jitter
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    val_image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = CichlidDataset(train_samples, label_map,
                                   transform=image_transform,
                                   mask_transform=mask_transform)

    val_dataset = CichlidDataset(val_samples, label_map,
                                 transform=val_image_transform,
                                 mask_transform=val_mask_transform)

    return train_dataset, val_dataset, label_map


if __name__ == "__main__":
    train_ds, val_ds, label_map = build_dataset("data")

    print(f"\nSample tensor shape: {train_ds[0][0].shape}")  # Should be [4, 224, 224]
    print(f"Sample label: {train_ds[0][1]}")