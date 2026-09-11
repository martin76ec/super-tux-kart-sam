import torch
from src.model.model import SAM2Segmenter

def save_model(model, path="model.th"):
    torch.save(model.state_dict(), path)

def load_model(path="model.th", device="cpu"):
    model = SAM2Segmenter()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
