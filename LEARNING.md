# Key Learnings, Technical Notes & Metrics

## 1. Experimental Metrics & Benchmark Comparison

Validation set: 300 held-out frames (20% split) at native 400x400 resolution.

### Global Performance Summary

| Model / Experiment | Backbone | Resolution | Epochs | Val Loss | Overall Accuracy | Mean IoU (mIoU) | Inference Latency | FPS (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab 2 Baseline** | Custom U-Net | 128x128 | 15 | 0.2200 | ~88.0% | ~0.4200 | ~8.5 ms | ~118 FPS |
| **YOLO Transfer** | CSPDarknet (YOLOv8n) | 400x400 | 5 | 0.1871 | 92.86% | 0.5622 | **~1.0 ms** | **>1000 FPS** |
| **SAM 2.1 Transfer** | Hiera-Tiny (SAM 2.1) | 400x400 | 5 | **0.1328** | **95.28%** | **0.6204** | ~27.8 ms | ~36 FPS |

---

### Per-Class Metrics Breakdown (Validation Set)

#### SAM 2.1 Tiny (`model.th`)
- Overall Pixel Accuracy: **95.28%**
- Mean IoU (mIoU): **0.6204**

| Class | Class ID | IoU | Precision | Recall | Qualitative Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Background** | 0 | 0.9128 | 0.9837 | 0.9268 | Crisp sky/mountain borders, zero false sky artifacts |
| **Track** | 1 | 0.9087 | 0.9244 | 0.9817 | Continuous road boundary, handles curves and inclines |
| **Kart** | 2 | 0.9098 | 0.9421 | 0.9636 | Tight contour matching, wheel/chassis integrity |
| **Pickup** | 3 | 0.6466 | 0.7183 | 0.8664 | Reliably detects distant item gift boxes |
| **Nitro** | 4 | 0.5453 | 0.5823 | 0.8955 | Captures small green nitro bottles on track |
| **Bomb** | 5 | 0.4115 | 0.6791 | 0.5109 | Segments round obstacle shape with high precision |
| **Projectile** | 6 | 0.0081 | 0.0776 | 0.0089 | Extremely rare class (<0.01% of total pixels) |

#### YOLOv8n (`model_yolo.th`)
- Overall Pixel Accuracy: **92.86%**
- Mean IoU (mIoU): **0.5622**

| Class | Class ID | IoU | Precision | Recall | Qualitative Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Background** | 0 | 0.8692 | 0.9878 | 0.8787 | Good separation, slightly coarse horizon edges |
| **Track** | 1 | 0.8688 | 0.8806 | 0.9849 | High recall on road surface |
| **Kart** | 2 | 0.8277 | 0.8988 | 0.9128 | Solid detections, slightly dilated boundaries |
| **Pickup** | 3 | 0.4961 | 0.5308 | 0.8836 | Good box coverage, lower precision at distance |
| **Nitro** | 4 | 0.3930 | 0.5830 | 0.5466 | Detects close-range bottles |
| **Bomb** | 5 | 0.3279 | 0.4144 | 0.6110 | Sensitive to spherical shape |
| **Projectile** | 6 | 0.1525 | 1.0000 | 0.1525 | 100% precision when activated, low recall |

---

## 2. From-Scratch Training vs. Foundation Model Transfer Learning

- **Lab 2 (Custom U-Net trained from scratch):**
  - Required downsampling input frames from 400x400 to 128x128 to meet compute constraints.
  - Required 15 epochs to converge to ~0.37 - 0.48 mIoU.
  - Suffered from false positives on small objects (e.g. spurious pickup activations in the sky) and dilated boundaries around karts.
- **Final Project (SAM 2 & YOLO Transfer Learning):**
  - Retained native 400x400 resolution.
  - Reached 0.6204 mIoU and 95.28% pixel accuracy in only 5 epochs.
  - Pre-trained visual representations provide sharp object boundaries and strong spatial priors for small objects at a distance.

---

## 3. Architecture Comparison: Vision Transformers vs. Modern ConvNets

- **SAM 2.1 (Hiera ViT):**
  - Highest segmentation fidelity (+0.0582 mIoU over YOLO, +0.20 over U-Net).
  - Multi-scale window attention captures fine details like fences, kart antennas, and distant pickups.
  - Runs at ~36 FPS (27.8 ms latency per frame) on RTX 4060 Ti with PyTorch AMP.
- **YOLOv8 (CSPDarknet + C2f):**
  - Extreme throughput (>1000 FPS offline, ~1.0 ms latency per frame).
  - Faster training (~8 seconds per epoch vs. ~48 seconds for SAM 2).
  - Best suited when FPS is the primary objective or on lower-power edge devices.

---

## 4. Class Imbalance in Semantic Segmentation

- **Dataset Distribution:**
  - Background (Class 0): 122,844,679 pixels (51.2%)
  - Track (Class 1): 111,387,780 pixels (46.4%)
  - Kart (Class 2): 5,304,448 pixels (2.21%)
  - Pickups (Class 3): 286,119 pixels (0.12%)
  - Nitro (Class 4): 46,899 pixels (0.02%)
  - Bomb (Class 5): 100,058 pixels (0.04%)
  - Projectile (Class 6): 30,017 pixels (0.01%)
- **Impact of Naive Loss:**
  - Standard unweighted Cross-Entropy causes the network to collapse toward background and track, ignoring items and karts.
- **Weighting Strategy:**
  - Direct inverse frequency (1 / count) introduces extreme gradients that destabilize training on dominant classes.
  - Smoothed power-law weighting:
    `weight = (N_total / (N_class + 1000)) ** 0.25`
  - Normalized so mean weight equals 1.0. This prevents vanishing gradients on road and sky while giving items sufficient learning signal.

---

## 5. Categorical Data Integrity

- Images are continuous RGB signals normalized to floating-point tensors with ImageNet statistics.
- Segmentation masks represent discrete categorical labels (0 to 6).
- Standard image resizing (bilinear/bicubic interpolation) corrupts class IDs by creating fractional pixel values (e.g. 1.4 between track and kart).
- Target tensors must remain discrete integer types (`torch.long`) with nearest-neighbor resampling when spatial transformations are required.

---

## 6. Instance Detection from Semantic Masks

- **Heuristic Extraction via `cv2.findContours`:**
  - Fast post-processing step (<0.5 ms per frame).
  - Works well for isolated, separated foreground objects (karts and items).
- **Limitations:**
  - Merger Problem: When two karts collide or overlap, their pixel masks form one contiguous blob, resulting in a single merged bounding box.
  - Fragmentation Problem: When an object is partially occluded by a thin structure (e.g. a barrier), the mask splits into two disconnected contours.
- **Advanced Alternatives:**
  - Distance Transform + Watershed segmentation (`cv2.watershed`) to separate touching instances via centroid peaks.
  - Top-down instance segmentation (Mask R-CNN, YOLO-seg) that predicts individual bounding box proposals prior to masking.

---

## 7. Training Pipeline and Systems Engineering

- **Automatic Mixed Precision (AMP):**
  - FP16/BF16 tensor cores on RTX 4060 Ti reduced memory footprint to ~2.3 GB VRAM and tripled iteration speed.
- **Fault-Tolerant Step Breakpoints:**
  - Saving checkpoints every 25 batches and at epoch boundaries enables mid-epoch recovery from interruptions or system restarts.
- **Native Wayland Window Capture:**
  - Querying `hyprctl clients -j` and piping `grim` captures into OpenCV enabled real-time, low-overhead inference directly over the running game client.
