import torch
from torch.utils.data import DataLoader
from src.utils import SuperTuxDataset
from src.pipeline import load_model, evaluate, visualize_predictions

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = SuperTuxDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    model = load_model("model.th", device=device)
    val_loss, val_acc, val_iou = evaluate(model, val_loader, criterion, device)
    print(f"Eval - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | mIoU: {val_iou:.4f}")
    visualize_predictions(model, val_ds, out_dir="outputs", num_samples=6, device=device)

if __name__ == "__main__":
    main()
