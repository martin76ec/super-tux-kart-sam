import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class YOLOSegmenter(nn.Module):
    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()
        yolo = YOLO("yolov8n-seg.pt")
        self.backbone = yolo.model.model[:10]
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = []
        cur = x
        for i, layer in enumerate(self.backbone):
            cur = layer(cur)
            if i in [4, 6, 9]:
                feats.append(cur)
        f0, f1, f2 = feats
        f1_up = F.interpolate(f1, size=f0.shape[2:], mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        cat = torch.cat([f0, f1_up, f2_up], dim=1)
        out = self.head(cat)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
