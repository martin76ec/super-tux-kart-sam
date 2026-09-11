import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model

class SAM2Segmenter(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        for p in self.sam.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(352, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x_in = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False) if (h, w) != (1024, 1024) else x
        f0, f1, f2 = self.sam.get_image_embeddings(x_in)
        f1_up = F.interpolate(f1, size=f0.shape[2:], mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f0, f1_up, f2_up], dim=1)
        out = self.head(feat)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
