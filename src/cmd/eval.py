import argparse
import torch
from torch.utils.data import DataLoader
from src.utils import SuperTuxDataset
from src.pipeline import load_model, evaluate, visualize_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sam2", choices=["sam2", "yolo"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = SuperTuxDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    weights_path = "model.th" if args.model == "sam2" else "model_yolo.th"
    out_dir = "outputs" if args.model == "sam2" else "outputs_yolo"

    model = load_model(weights_path, model_type=args.model, device=device)
    val_loss, val_acc, val_iou = evaluate(model, val_loader, criterion, device)
    print(f"Eval [{args.model}] - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | mIoU: {val_iou:.4f}")
    visualize_predictions(model, val_ds, out_dir=out_dir, num_samples=6, device=device)

if __name__ == "__main__":
    main()
