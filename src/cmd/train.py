import argparse
import torch
from torch.utils.data import DataLoader
from src.model import SAM2Segmenter, YOLOSegmenter
from src.utils import SuperTuxDataset, get_class_weights
from src.pipeline import train_pipeline, save_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sam2", choices=["sam2", "yolo"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SuperTuxDataset(split="train")
    val_ds = SuperTuxDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    weights = get_class_weights(train_ds).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    model = SAM2Segmenter().to(device) if args.model == "sam2" else YOLOSegmenter().to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    ckpt_path = "checkpoint.pt" if args.model == "sam2" else "checkpoint_yolo.pt"
    out_path = "model.th" if args.model == "sam2" else "model_yolo.th"

    train_pipeline(model, train_loader, val_loader, optimizer, criterion, epochs=5, device=device, ckpt_path=ckpt_path)
    save_model(model, out_path)
    print(f"Training finished. Saved model to {out_path}")

if __name__ == "__main__":
    main()
