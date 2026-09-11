import os
import torch

def train_epoch(model, loader, optimizer, criterion, device="cuda"):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device if "cuda" in device else "cpu"):
            out = model(x)
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_pipeline(model, train_loader, val_loader, optimizer, criterion, epochs=5, device="cuda", ckpt_path="checkpoint.pt"):
    start_epoch = 0
    start_batch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0)
        start_batch = ckpt.get("batch", 0)
        print(f"Resumed from epoch {start_epoch}, batch {start_batch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            if epoch == start_epoch and i < start_batch:
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device if "cuda" in device else "cpu"):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 25 == 0 or (i + 1) == len(train_loader):
                torch.save({"epoch": epoch, "batch": i + 1, "model": model.state_dict(), "opt": optimizer.state_dict(), "loss": loss.item()}, ckpt_path)
                print(f"Epoch {epoch+1}/{epochs} [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        start_batch = 0
        from src.pipeline.eval import evaluate
        val_loss, val_acc, val_iou = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Eval -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | mIoU: {val_iou:.4f}")
        torch.save({"epoch": epoch + 1, "batch": 0, "model": model.state_dict(), "opt": optimizer.state_dict(), "loss": val_loss}, ckpt_path)
    return model
