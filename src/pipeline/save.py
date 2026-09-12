import torch
from src.model.model import SAM2Segmenter
from src.model.yolo import YOLOSegmenter

def save_model(model, path="model.th"):
    torch.save(model.state_dict(), path)

def load_model(path="model.th", model_type="sam2", device="cpu"):
    model = SAM2Segmenter() if model_type == "sam2" else YOLOSegmenter()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
