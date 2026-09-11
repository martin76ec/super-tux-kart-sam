import json
import subprocess
import time
import cv2
import numpy as np
import torch
from src.pipeline import load_model

PALETTE = np.array([
    [0, 0, 0],
    [0, 140, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 0, 255],
    [255, 255, 0]
], dtype=np.uint8)

def get_window_geom(target="supertuxkart"):
    try:
        out = subprocess.check_output(["hyprctl", "clients", "-j"], timeout=0.5)
        clients = json.loads(out)
        for c in clients:
            if target in c.get("class", "").lower() or target in c.get("initialClass", "").lower() or target in c.get("title", "").lower():
                x, y = c["at"]
                w, h = c["size"]
                if w > 50 and h > 50:
                    return f"{x},{y} {w}x{h}"
    except Exception:
        pass
    return None

def capture_frame(geom=None):
    cmd = ["grim", "-t", "ppm"]
    if geom:
        cmd += ["-g", geom]
    cmd.append("-")
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=0.2)
        if proc.returncode == 0 and proc.stdout:
            return cv2.imdecode(np.frombuffer(proc.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("model.th", device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    mode = "overlay"
    alpha = 0.5
    geom = get_window_geom()
    print("Live mode started. Press 'm' to toggle mode, 'q' to quit.")
    while True:
        t0 = time.time()
        if geom is None:
            geom = get_window_geom()
        frame = capture_frame(geom)
        if frame is None:
            geom = None
            time.sleep(0.1)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        tensor = (tensor - mean) / std
        with torch.no_grad():
            with torch.amp.autocast(device if "cuda" in device else "cpu"):
                pred = model(tensor).argmax(1).squeeze(0).cpu().numpy()
        mask_bgr = PALETTE[pred]
        fps = 1.0 / max(time.time() - t0, 1e-4)
        if mode == "overlay":
            disp = cv2.addWeighted(frame, 1.0 - alpha, mask_bgr, alpha, 0)
        elif mode == "side":
            disp = np.hstack([frame, mask_bgr])
        else:
            disp = mask_bgr
        cv2.putText(disp, f"FPS: {fps:.1f} | Mode: {mode}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("SuperTuxKart SAM Live", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("m"):
            mode = "side" if mode == "overlay" else "mask" if mode == "side" else "overlay"
        elif key == ord("r"):
            geom = get_window_geom()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
