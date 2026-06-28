import torch
import torch.nn.functional as F
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from dataset import build_dataset
from train import CichlidDatasetSync, SyncTransform
from model import CichlidIDModel


def load_model(checkpoint_path, num_classes=25, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CichlidIDModel(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["label_map"]


def predict_single(model, image_path, mask_path, label_map, device):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    transform = SyncTransform(image_size=224, is_train=False)
    tensor = transform(image, mask).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

    id_to_name = {v: k for k, v in label_map.items()}
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

    return ranked, id_to_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map = load_model("checkpoints/best_model.pth", device=device)

    # Build val set to pick a sample from
    _, val_ds_raw, _ = build_dataset("data_224")

    # Pick the first sample from the val set — change index to try others
    SAMPLE_INDEX = 173
    jpg_path, png_path, true_label = val_ds_raw.samples[SAMPLE_INDEX]

    id_to_name = {v: k for k, v in label_map.items()}
    true_identity = id_to_name[true_label]

    print(f"Image:          {jpg_path}")
    print(f"True identity:  {true_identity}")
    print(f"\nRanked predictions:")
    print(f"{'Rank':<6} {'Identity':<45} {'Confidence':>10}")
    print("-" * 63)

    ranked, id_to_name = predict_single(model, jpg_path, png_path, label_map, device)

    for rank, (idx, prob) in enumerate(ranked, 1):
        marker = " <-- correct" if idx == true_label else ""
        print(f"{rank:<6} {id_to_name[idx]:<45} {prob*100:>9.2f}%{marker}")


if __name__ == "__main__":
    main()