import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils import SuperTuxDataset
from src.pipeline import load_model, visualize_predictions

LABELS = ["Background", "Track", "Kart", "Pickup", "Nitro", "Bomb", "Projectile"]

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

    total_inter = np.zeros(7)
    total_union = np.zeros(7)
    total_pred = np.zeros(7)
    total_target = np.zeros(7)
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device if "cuda" in device else "cpu"):
                out = model(x)
                loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(1)
            p_f = pred.view(-1).cpu().numpy()
            t_f = y.view(-1).cpu().numpy()
            total_correct += (p_f == t_f).sum()
            total_pixels += len(p_f)
            for c in range(7):
                p_c = (p_f == c)
                t_c = (t_f == c)
                total_inter[c] += (p_c & t_c).sum()
                total_union[c] += (p_c | t_c).sum()
                total_pred[c] += p_c.sum()
                total_target[c] += t_c.sum()

    ious = total_inter / np.maximum(total_union, 1)
    prec = total_inter / np.maximum(total_pred, 1)
    rec = total_inter / np.maximum(total_target, 1)
    acc = total_correct / total_pixels
    val_loss = total_loss / len(val_loader)

    print(f"\n==================== Evaluation [{args.model.upper()}] ====================")
    print(f"Val Loss: {val_loss:.4f} | Overall Accuracy: {acc*100:.2f}% | mIoU: {ious.mean():.4f}")
    print(f"{'Class':<14} | {'IoU':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 48)
    for i in range(7):
        print(f"{LABELS[i]:<14} | {ious[i]:<8.4f} | {prec[i]:<10.4f} | {rec[i]:<8.4f}")
    print("=" * 48 + "\n")

    visualize_predictions(model, val_ds, out_dir=out_dir, num_samples=6, device=device)

if __name__ == "__main__":
    main()
