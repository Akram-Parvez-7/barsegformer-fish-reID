import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import build_dataset
from train import CichlidDatasetSync, SyncTransform
from model import CichlidIDModel


def load_model(checkpoint_path, num_classes=25, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CichlidIDModel(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})")
    return model, checkpoint["label_map"]


def evaluate(model, loader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


def top_k_accuracy(probs, labels, k=3):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(labels[i] in top_k[i] for i in range(len(labels)))
    return correct / len(labels)


def plot_confusion_matrix(labels, preds, label_map, save_path):
    # Reverse label map: int -> short name
    id_to_name = {}
    for identity, idx in label_map.items():
        parts = identity.split("/")
        # Shorten: "F3/Tank_A2(...)/Fish_1" -> "A2-F3/F1"
        tank = parts[1].split("(")[1].replace(")", "").replace(" ", "_")
        fish = parts[2].replace("Fish_", "F")
        id_to_name[idx] = f"{tank}/{fish}"

    names = [id_to_name[i] for i in range(len(label_map))]
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linewidths=0.5)
    plt.title("Confusion Matrix — Cichlid Individual ID", fontsize=14, pad=20)
    plt.ylabel("True Identity", fontsize=11)
    plt.xlabel("Predicted Identity", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_fish_accuracy(labels, preds, label_map, save_path):
    id_to_name = {}
    for identity, idx in label_map.items():
        parts = identity.split("/")
        tank = parts[1].split("(")[1].replace(")", "").replace(" ", "_")
        fish = parts[2].replace("Fish_", "F")
        id_to_name[idx] = f"{tank}/{fish}"

    per_class_correct = {}
    per_class_total = {}
    for true, pred in zip(labels, preds):
        per_class_total[true] = per_class_total.get(true, 0) + 1
        if true == pred:
            per_class_correct[true] = per_class_correct.get(true, 0) + 1

    names = [id_to_name[i] for i in range(len(label_map))]
    accs = [per_class_correct.get(i, 0) / per_class_total.get(i, 1)
            for i in range(len(label_map))]

    colors = ["#2ecc71" if a >= 0.8 else "#e67e22" if a >= 0.5 else "#e74c3c"
              for a in accs]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, accs, color=colors, edgecolor="white", linewidth=0.5)
    plt.axhline(y=np.mean(accs), color="navy", linestyle="--",
                linewidth=1.5, label=f"Mean: {np.mean(accs):.3f}")
    plt.title("Per-Fish Identification Accuracy", fontsize=14)
    plt.ylabel("Accuracy", fontsize=11)
    plt.xlabel("Fish Identity", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-fish accuracy plot saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "checkpoints/best_model.pth"
    DATA_DIR = "data_224"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load model
    model, label_map = load_model(CHECKPOINT, num_classes=25, device=device)

    # Build val dataset
    _, val_ds_raw, _ = build_dataset(DATA_DIR)
    val_sync = SyncTransform(image_size=224, is_train=False)
    val_ds = CichlidDatasetSync(val_ds_raw.samples, label_map, sync_transform=val_sync)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"\nEvaluating on {len(val_ds)} validation samples...\n")

    # Run evaluation
    preds, labels, probs = evaluate(model, val_loader, device)

    # Metrics
    top1 = (preds == labels).mean()
    top3 = top_k_accuracy(probs, labels, k=3)
    top5 = top_k_accuracy(probs, labels, k=5)

    print("=" * 50)
    print(f"  Top-1 Accuracy: {top1:.4f} ({top1*100:.2f}%)")
    print(f"  Top-3 Accuracy: {top3:.4f} ({top3*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5:.4f} ({top5*100:.2f}%)")
    print("=" * 50)

    # Per class report
    id_to_name = {}
    for identity, idx in label_map.items():
        parts = identity.split("/")
        tank = parts[1].split("(")[1].replace(")", "").replace(" ", "_")
        fish = parts[2].replace("Fish_", "F")
        id_to_name[idx] = f"{tank}/{fish}"

    target_names = [id_to_name[i] for i in range(len(label_map))]
    report = classification_report(labels, preds, target_names=target_names)
    print("\nPer-class Classification Report:")
    print(report)

    # Save report
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Top-1 Accuracy: {top1:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3:.4f}\n")
        f.write(f"Top-5 Accuracy: {top5:.4f}\n\n")
        f.write(report)

    # Plots
    plot_confusion_matrix(labels, preds, label_map,
                          RESULTS_DIR / "confusion_matrix.png")
    plot_per_fish_accuracy(labels, preds, label_map,
                           RESULTS_DIR / "per_fish_accuracy.png")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()