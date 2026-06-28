import torch
import torch.nn.functional as F
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map = load_model("checkpoints/best_model.pth", device=device)
    _, val_ds_raw, _ = build_dataset("data_224")
    id_to_name = {v: k for k, v in label_map.items()}
    transform = SyncTransform(image_size=224, is_train=False)

    from PIL import Image
    print(f"{'Idx':<5} {'True identity':<40} {'Top confidence':>14} {'Correct?':>9}")
    print("-" * 72)

    for i, (jpg_path, png_path, true_label) in enumerate(val_ds_raw.samples):
        image = Image.open(jpg_path).convert("RGB")
        mask = Image.open(png_path).convert("L")
        tensor = transform(image, mask).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(tensor), dim=1).squeeze().cpu()

        top_conf, top_idx = probs.max(0)
        correct = top_idx.item() == true_label

        if correct and top_conf.item() > 0.70:
            print(f"{i:<5} {id_to_name[true_label]:<40} {top_conf.item()*100:>13.2f}% {'✓':>9}")

if __name__ == "__main__":
    main()