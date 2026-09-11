import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_dir="seg_dataset", split="train", val_ratio=0.2, seed=42):
        self.img_dir = os.path.join(dataset_dir, "images")
        self.mask_dir = os.path.join(dataset_dir, "masks")
        imgs = sorted(os.listdir(self.img_dir))
        masks = set(os.listdir(self.mask_dir))
        pairs = [(f, f.replace("_frame_", "_mask_combined_")) for f in imgs if f.replace("_frame_", "_mask_combined_") in masks]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(pairs))
        split_pt = int(len(pairs) * (1.0 - val_ratio))
        chosen_idx = idx[:split_pt] if split == "train" else idx[split_pt:]
        self.pairs = [pairs[i] for i in chosen_idx]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name))
        img_t = self.transform(img)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
        return img_t, mask_t

def get_class_weights(dataset, num_classes=7):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, mask_name in dataset.pairs:
        arr = np.array(Image.open(os.path.join(dataset.mask_dir, mask_name)))
        for c in range(num_classes):
            counts[c] += np.sum(arr == c)
    weights = (counts.sum() / (counts + 1000.0)) ** 0.25
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def compute_metrics(pred, target, num_classes=7):
    pred_f = pred.view(-1)
    target_f = target.view(-1)
    acc = (pred_f == target_f).float().mean().item()
    ious = []
    for c in range(num_classes):
        p_c = pred_f == c
        t_c = target_f == c
        intersection = (p_c & t_c).sum().float().item()
        union = (p_c | t_c).sum().float().item()
        if union > 0:
            ious.append(intersection / union)
    miou = float(np.mean(ious)) if ious else 0.0
    return acc, miou

def explore_dataset(dataset_dir="seg_dataset", num_classes=7):
    mask_dir = os.path.join(dataset_dir, "masks")
    files = sorted(os.listdir(mask_dir))
    counts = np.zeros(num_classes, dtype=np.int64)
    for f in files:
        arr = np.array(Image.open(os.path.join(mask_dir, f)))
        for c in range(num_classes):
            counts[c] += np.sum(arr == c)
    return {"total_samples": len(files), "class_counts": counts.tolist()}
