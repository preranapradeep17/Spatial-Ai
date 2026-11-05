# Computer Vision Training Guide: Custom Detector from Scratch

## 1. Educational Objectives (CS4231)
In accordance with **CS4231 Fundamentals of Computer Vision**, this module builds a single-stage convolutional object detector from random initialization without using pretrained weights (such as Ultralytics YOLO or torchvision checkpoints).

## 2. Architecture Specifications
- **Input**: $(B, 3, 480, 640)$ normalized RGB tensor.
- **Backbone**:
  - Layer 1: Conv2D(3 $\to$ 32, kernel=3, stride=2, pad=1), BatchNorm2D, LeakyReLU(0.1) $\to (B, 32, 240, 320)$
  - Layer 2: Conv2D(32 $\to$ 64, kernel=3, stride=2, pad=1), BatchNorm2D, LeakyReLU(0.1) $\to (B, 64, 120, 160)$
  - Layer 3: Conv2D(64 $\to$ 128, kernel=3, stride=2, pad=1), BatchNorm2D, LeakyReLU(0.1) $\to (B, 128, 60, 80)$
  - Layer 4: Conv2D(128 $\to$ 256, kernel=3, stride=2, pad=1), BatchNorm2D, LeakyReLU(0.1) $\to (B, 256, 30, 40)$
  - Layer 5: Conv2D(256 $\to$ 256, kernel=3, stride=2, pad=1), BatchNorm2D, LeakyReLU(0.1) $\to (B, 256, 15, 20)$
- **Prediction Head**:
  - Conv2D(256 $\to$ 11, kernel=1, stride=1) $\to (B, 11, 15, 20)$
  - Reshaped to $(B, 15, 20, 11)$ where $11 = 1 \text{ (objectness)} + 4 \text{ (bbox: } t_x, t_y, t_w, t_h) + 6 \text{ (class logits)}$.

## 3. Target Encoding & Loss Formulation
For each grid cell $(i, j)$ with cell size $S_w = \frac{640}{20} = 32$, $S_h = \frac{480}{15} = 32$:
1. If an object center falls within cell $(i, j)$:
   - Objectness target: $1.0$
   - Center offsets: $t_x = \frac{u_{center}}{S_w} - j$, $t_y = \frac{v_{center}}{S_h} - i \quad (\in [0, 1])$
   - Box scale targets: $t_w = \log(\frac{w}{S_w})$, $t_h = \log(\frac{h}{S_h})$
   - Class target: one-hot vector across the 6 classes
2. Loss Components:
   $$\mathcal{L}_{total} = \lambda_{obj} \mathcal{L}_{obj} + \lambda_{box} \mathcal{L}_{box} + \lambda_{cls} \mathcal{L}_{cls}$$
   - $\mathcal{L}_{obj}$: Binary Cross-Entropy with positive/negative weighting.
   - $\mathcal{L}_{box}$: Smooth L1 or CIoU loss on positive cells only.
   - $\mathcal{L}_{cls}$: Cross-Entropy on positive cells only.

## 4. Hardware Optimization & Reproducibility
- Automatic device routing to Apple Silicon Metal Performance Shaders (`mps`) or CPU fallback.
- Fixed random seeds for dataset splits and weight initializations.
