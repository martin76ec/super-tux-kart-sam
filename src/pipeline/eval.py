import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils.dataset import compute_metrics

def draw_bboxes(img_bgr, pred):
    labels = ["bg", "track", "kart", "pickup", "nitro", "bomb", "proj"]
    palette = [(0, 0, 0), (0, 140, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]
    out = img_bgr.copy()
    for c in range(2, 7):
        mask = (pred == c).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = 100 if c == 2 else 25
        for cnt in cnts:
            if cv2.contourArea(cnt) >= min_a:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(out, (x, y), (x + w, y + h), palette[c], 2)
                cv2.putText(out, labels[c], (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, palette[c], 1, cv2.LINE_AA)
    return out

def evaluate(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss, total_acc, total_iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device if "cuda" in device else "cpu"):
                out = model(x)
                loss = criterion(out, y)
            acc, iou = compute_metrics(out.argmax(1), y)
            total_loss += loss.item()
            total_acc += acc
            total_iou += iou
    n = len(loader)
    return total_loss / n, total_acc / n, total_iou / n

def visualize_predictions(model, dataset, out_dir="outputs", num_samples=6, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    step = max(1, len(dataset) // num_samples)
    for idx in range(0, len(dataset), step)[:num_samples]:
        x, y = dataset[idx]
        with torch.no_grad():
            with torch.amp.autocast(device if "cuda" in device else "cpu"):
                pred = model(x.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()
        inv_norm = x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb = np.clip(inv_norm, 0, 1)
        bgr = (rgb[:, :, ::-1] * 255).astype(np.uint8)
        bbox_bgr = draw_bboxes(bgr, pred)
        bbox_rgb = bbox_bgr[:, :, ::-1]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Input Frame")
        axes[0].axis("off")
        axes[1].imshow(y.numpy(), cmap="tab10", vmin=0, vmax=6)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=6)
        axes[2].set_title("Prediction Mask")
        axes[2].axis("off")
        axes[3].imshow(bbox_rgb)
        axes[3].set_title("Detected Bounding Boxes")
        axes[3].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{idx:04d}.png"))
        plt.close()
    print(f"Saved {num_samples} visualizations to {out_dir}/")
