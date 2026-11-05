# Evaluation Methodology & Empirical Results for SpatialAI
**File:** `docs/evaluation.md`  
**Status:** Complete & Measured  

---

## 1. Computer Vision Metrics (CS4231)

### 1.1 Mathematical Formulations
- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{\text{Area}(B_{pred} \cap B_{gt})}{\text{Area}(B_{pred} \cup B_{gt})}$$
  A prediction is a True Positive (TP) if $\text{IoU} \ge 0.5$ and class prediction matches ground truth.

- **Precision & Recall**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

- **Mean Average Precision (mAP@0.5)**:
  $$\text{mAP@0.5} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c, \quad \text{where } \text{AP}_c = \int_0^1 p(r) \, dr$$

- **3D Euclidean Localization Error**:
  $$\text{Error}_{3D} = \sqrt{(X_{est} - X_{gt})^2 + (Y_{est} - Y_{gt})^2 + (Z_{est} - Z_{gt})^2}$$

### 1.2 Measured Computer Vision & Localization Benchmark
| Evaluation Task | Benchmark / Split | Measured Metric | Numerical Value | Academic Findings / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CNN Detector Training** | Synthetic + Real 10 Epochs | Multi-task Loss | $19.03 \to 0.876$ | Smooth convergence across BCE, SmoothL1, and Cross-Entropy |
| **Object Detection mAP@0.5** | Real SUN RGB-D Test Split | mAP@0.5 | **0.0000** | Rigorous verification showed zero-score is mathematically genuine: 2.16M CNN trained from random weights on 5 frames cannot generalize; raw scores peak below standard 0.20 threshold; at 0.01 threshold, predictions collapse to `sofa` due to class imbalance. See `results/plots/detector_test_predictions_vs_gt.png`. |
| **Detector Inference Latency** | Apple Silicon MPS ($224 \times 224$) | Mean Latency | **9.58 ms** (~104 FPS) | Fast forward pass suitable for real-time 3D perception loops. |
| **Calibrated 3D Localization (Oracle 2D)** | SUN RGB-D Benchmark (40 objects) | Mean Error | **0.423 m (42.3 cm)** | Calibrated pinhole back-projection with sensor tilt transform achieves sub-half-meter precision when given oracle ground-truth 2D bounding boxes and physical depth maps. |
| **Calibrated 3D Localization (Oracle 2D)** | SUN RGB-D Benchmark (40 objects) | Median Error | **0.358 m (35.8 cm)** | Bookshelf: 0.168m, Door: 0.254m, Cabinet: 0.251m, Chair: 0.474m. |
| **End-to-End 3D Localization (Predicted 2D)** | SUN RGB-D Benchmark | Recall-Gated Error | **Gated by Recall** | When chained to predicted boxes from the un-pretrained detector, 3D localization is bounded by detector recall (0.0% at 0.20 threshold). Decoupling confirms geometric math is correct while vision backbone needs scaled training data. |

---

## 2. Language Model & Agent Metrics (CS3235)

### 2.1 Mathematical Formulations
- **Perplexity (PPL)**:
  $$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i}) \right)$$

- **Valid Tool Call Syntax Rate**:
  $$\text{Rate}_{Valid} = \frac{\text{Outputs parsed cleanly as valid tool calls}}{\text{Total Generated Prompts}} \times 100\%$$

- **Intent Routing Accuracy**:
  $$\text{Accuracy}_{Intent} = \frac{\text{Queries routed to correct target tool}}{\text{Total Test Queries}} \times 100\%$$

### 2.2 Measured Language Model & Agent Benchmark
| Evaluation Task | Benchmark / Split | Measured Metric | Numerical Value | Academic Findings / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Transformer Pretraining** | Multi-Turn Corpus (30 Epochs) | Cross-Entropy Loss | $5.717 \to \mathbf{1.2610}$ | Pure tensor math Pre-LN causal self-attention on MPS |
| **Transformer Perplexity** | Multi-Turn Corpus (30 Epochs) | Perplexity | $304.07 \to \mathbf{3.53}$ | Sharp convergence on 13 standardized tools and dialogue flows |
| **Structured Tool Syntax Rate** | Test Queries (50 diverse inputs) | Syntax Valid Rate | **100.0%** | Validated via Pydantic schemas; fallback explicitly logged in telemetry |
| **Intent Routing Accuracy** | Test Queries (50 diverse inputs) | Routing Accuracy | **94.0%** | Supports anaphora resolution ("near it", "storage room instead") |
| **Local RAG Retrieval** | Facility Emergency Manuals | Top-1 Accuracy | **100.0%** | TF-IDF cosine similarity retrieves exact emergency codes (e.g. 7391) |

---

## 3. End-to-End System & Regression Suite
- **Total Automated Unit & Integration Tests**: **51 tests**
- **Test Pass Rate**: **100% (51/51 passing)**
- **Test Suite Execution Time**: **~11.9 seconds** on Apple Silicon MPS
- **External API Dependencies**: **0** (Zero commercial cloud APIs or hosted models)
