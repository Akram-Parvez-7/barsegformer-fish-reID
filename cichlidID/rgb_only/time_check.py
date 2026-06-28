import time
from dataset import build_dataset
from train import CichlidDatasetSync, SyncTransform

train_ds_raw, val_ds_raw, label_map = build_dataset("data_224")
train_sync = SyncTransform(image_size=224, is_train=True)
train_ds = CichlidDatasetSync(train_ds_raw.samples, label_map, sync_transform=train_sync)

t0 = time.time()
for i in range(50):
    _ = train_ds[i]
print(f"50 samples in {time.time()-t0:.2f}s")