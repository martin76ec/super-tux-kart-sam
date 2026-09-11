import torch
from torch.utils.data import DataLoader
from src.model import SAM2Segmenter
from src.utils import SuperTuxDataset, get_class_weights
from src.pipeline import train_pipeline, save_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SuperTuxDataset(split="train")
    val_ds = SuperTuxDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    weights = get_class_weights(train_ds).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    model = SAM2Segmenter().to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    train_pipeline(model, train_loader, val_loader, optimizer, criterion, epochs=5, device=device)
    save_model(model, "model.th")
    print("Training finished. Saved model to model.th")

if __name__ == "__main__":
    main()
