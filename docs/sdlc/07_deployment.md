# SDLC Stage 7: Deployment and Reproducibility

**Project Name**: Spatial Quest (SpatialAI)  
**Document Version**: 1.0.0  
**Target Runtime**: Local Workstation (Zero Cloud Dependencies)  
**Hardware Support**: Apple Silicon MPS / NVIDIA CUDA / CPU Fallback  

---

## 1. System Requirements & Environment Setup

Spatial Quest is engineered to operate 100% locally on standard consumer developer workstations without requiring external cloud accounts, API keys, or paid inference subscriptions.

### Minimum Hardware
- **CPU**: 4-core x86_64 or Apple Silicon (M1/M2/M3/M4).
- **RAM**: 8 GB minimum (16 GB recommended).
- **Disk Space**: ~2 GB for repository, checkpoints, and benchmark dataset.
- **Display**: Any modern web browser supporting WebGL (Chrome, Safari, Firefox, Edge).

### Software Environment
- **Python**: 3.9+ (Python 3.12 verified).
- **PyTorch**: 2.8+ (PyTorch 2.11.0 with Apple Silicon MPS acceleration verified).
- **FastAPI / Uvicorn**: 0.128+ / 0.34+.

---

## 2. Installation Steps

### Step 1: Clone Repository & Open Directory
```bash
git clone <repository_url>
cd "Spatial AI"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```
*Key packages installed*: `torch`, `numpy`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `pillow`, `pyyaml`, `scikit-learn`, `requests`.

---

## 3. Starting the Game

### Launch FastAPI Local Server
From the project root:
```bash
python run.py
```
*Or directly via Uvicorn*:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Primary 3D Game
Open your web browser and navigate to:
```
http://localhost:8000
```
The game client loads immediately with:
1. Interactive 3D Sector 4 Research Facility (5 distinct rooms).
2. Character movement controls (<kbd>W, A, S, D</kbd> / Touch D-Pad / Mouse Look).
3. Companion Drone Dialogue Terminal on the left.
4. **Live System Inspector** accessible via the top navigation bar (`🔍 SYSTEM INSPECTOR`).

---

## 4. Checkpoints & Model Assets

The repository includes pre-trained checkpoints trained from scratch on our local datasets:
- **Vision Model**: `models/vision/spatial_detector_best.pt` (2.16M params, CNN detector).
- **Language Model**: `models/language_model/transformer_best.pt` (856K params, decoder Transformer).
- **Tokenizer Vocabulary**: `models/language_model/tokenizer.json` (384 BPE subwords).

### Automated Checkpoint Verification
To verify checkpoint integrity:
```bash
python -c "import torch; print('Vision Model:', torch.load('models/vision/spatial_detector_best.pt', map_location='cpu')['epoch']); print('LM Model:', torch.load('models/language_model/transformer_best.pt', map_location='cpu')['epoch'])"
```

---

## 5. Running the Automated Test Suite

Run all 51 automated unit and integration tests:
```bash
pytest -v
```
All 51 tests will pass in under 15 seconds.

---

## 6. Reproducing Training and Evaluation

### 6.1 Retraining the Custom Detector from Scratch
To retrain `SpatialDetector` on the SUN RGB-D benchmark subset:
```bash
python scripts/train_detector.py
```
Outputs saved to `models/vision/` and training curves saved to `results/plots/detector_loss_curve.png`.

### 6.2 Evaluating the Detector on Real-World Ground Truth
To measure real mAP@0.5 and 3D metric localization error:
```bash
python scripts/evaluate_detector.py
```
Outputs saved to `results/metrics/real_benchmark_eval.json`.

### 6.3 Retraining the Custom Transformer from Scratch
To retrain `DecoderTransformer` on domain investigation text:
```bash
python scripts/train_language_model.py
```
Outputs saved to `models/language_model/transformer_best.pt` and training curves to `results/plots/transformer_loss_curve.png`.

---

## 7. Recovering from Missing Files

If any checkpoint or dataset file is accidentally removed:
1. **Regenerate Synthetic Test Data**:
   ```bash
   python scripts/generate_synthetic_dataset.py
   ```
2. **Verify Dataset Integrity**:
   ```bash
   python scripts/validate_dataset.py
   ```
3. **Reset Game Progression**:
   Click the `🔄 RESET` button in the game header or execute:
   ```bash
   curl -X POST http://localhost:8000/game/reset -H "Content-Type: application/json" -d '{"seed": 42}'
   ```
