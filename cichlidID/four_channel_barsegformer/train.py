import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from pathlib import Path
import random
import numpy as np
import json
import time
from dataset import build_dataset, CichlidDataset
from model import CichlidIDModel


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Synchronized transforms (image + mask flipped together) ───────────────────
class SyncTransform:
    """Applies identical spatial augmentations to both RGB and mask."""
    def __init__(self, image_size=224, is_train=True):
        self.image_size = image_size
        self.is_train = is_train

        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        )
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, mask):
        # Resize both
        # image = TF.resize(image, (self.image_size, self.image_size))
        # mask = TF.resize(mask, (self.image_size, self.image_size))

        if self.is_train:
            # Horizontal flip — same decision for both
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation ±15 degrees
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Color jitter on RGB only
            image = self.color_jitter(image)

        # To tensor
        image = TF.to_tensor(image)   # [3, H, W]
        mask = TF.to_tensor(mask)     # [1, H, W]

        # Normalize RGB only
        image = self.normalize(image)

        # Concatenate → 4-channel
        combined = torch.cat([image, mask], dim=0)  # [4, H, W]
        return combined


# ── Dataset with sync transforms ──────────────────────────────────────────────
class CichlidDatasetSync(CichlidDataset):
    def __init__(self, samples, label_map, sync_transform=None):
        super().__init__(samples, label_map, transform=None, mask_transform=None)
        self.sync_transform = sync_transform

    def __getitem__(self, idx):
        from PIL import Image
        jpg_path, png_path, label = self.samples[idx]
        image = Image.open(jpg_path).convert("RGB")
        mask = Image.open(png_path).convert("L")

        if self.sync_transform:
            combined = self.sync_transform(image, mask)
        else:
            combined = torch.cat([
                TF.to_tensor(TF.resize(image, (224, 224))),
                TF.to_tensor(TF.resize(mask, (224, 224)))
            ], dim=0)

        return combined, label


# ── Training one epoch ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    # Add this line
    print(f"Model device: {next(model.parameters()).device}")

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ── Validation ────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(42)

    # Config
    DATA_DIR = "data_224"
    SAVE_DIR = Path("checkpoints")
    SAVE_DIR.mkdir(exist_ok=True)
    NUM_CLASSES = 25
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # early stopping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Build datasets
    train_ds_raw, val_ds_raw, label_map = build_dataset(DATA_DIR)

    train_sync = SyncTransform(image_size=224, is_train=True)
    val_sync = SyncTransform(image_size=224, is_train=False)

    train_ds = CichlidDatasetSync(train_ds_raw.samples, label_map, sync_transform=train_sync)
    val_ds = CichlidDatasetSync(val_ds_raw.samples, label_map, sync_transform=val_sync)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Model
    model = CichlidIDModel(num_classes=NUM_CLASSES).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>5} {train_loss:>11.4f} {train_acc:>10.4f} "
              f"{val_loss:>9.4f} {val_acc:>8.4f} {elapsed:>5.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "label_map": label_map,
            }, SAVE_DIR / "best_model.pth")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} — no improvement for {PATIENCE} epochs")
                break

    # Save training history
    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {SAVE_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()