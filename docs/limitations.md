# SpatialAI: System Limitations & Scope Boundaries
**File:** `docs/limitations.md`  
**Status:** Comprehensive Academic Review  

---

## 1. Computer Vision & 3D Spatial Perception Limitations
- **Real-World Indoor Clutter & Domain Transfer**:
  - The custom single-stage CNN detector is trained from random initialization with 2,162,731 parameters. On an honest 20-sample micro-split of real SUN RGB-D benchmark frames, mAP@0.5 evaluates to 0.0000 due to severe class imbalance, small sample count, and absence of pretrained ImageNet weights.
  - In the 3D WebGL / synthetic domain, the detector operates reliably, but transferring to raw real-world webcams without fine-tuning requires active domain adaptation or larger training splits.
- **Occlusions & Depth Discontinuities**:
  - While our 40% central patch median filtering algorithm suppresses edge boundary artifacts, severe foreground occlusions (>80%) can still cause background depth bleeding into the object centroid estimate.
- **Pinhole Camera Intrinsics Assumption**:
  - The spatial back-projection engine assumes a rectified, undistorted pinhole camera model ($f_x, f_y, c_x, c_y$). Non-linear lens radial and tangential distortions from fisheye sensors are assumed to be pre-rectified before ingestion.

---

## 2. Language Model & Agent Reasoning Limitations
- **Compact Domain-Specific Parameter Scale**:
  - The custom decoder-only Transformer contains 856,832 parameters (4 layers, 4 heads, $d_{model}=256, d_{ff}=1024$) and a custom 437-token vocabulary. It is specialized for spatial reasoning, tool dispatching, and facility lore; it is not a general-purpose foundation LLM and cannot answer open-domain general trivia.
- **Out-of-Vocabulary (OOV) Terms**:
  - Words entirely absent from the 437-token subword vocabulary are mapped to `<unk>`. While our deterministic intent router provides a robust fallback, extreme linguistic drift or complex compound sentences may degrade parsing performance.
- **Context Window Length**:
  - The causal Transformer operates with a maximum context window of 128 tokens. Very long multi-turn conversational logs are truncated to the most recent bounded turns to remain within memory constraints.

---

## 3. Computer Graphics & Game Environment Boundaries
- **Planar 2.5D Facility Layout**:
  - The 5 facility rooms (Reception, Research Office, Library / Archives, Investigation Laboratory, Storage Room & Vault) are arranged across a single horizontal ground plane ($Y=0$). Multi-story staircases, vertical elevator shafts, and complex 3D vertical puzzles are out of scope.
- **Static Environmental Geometry**:
  - Furniture, archways, and room walls are static colliders. While doors, wall safes, and clue datapads change state upon valid interaction, dynamic fluid simulations or deformable soft bodies are not modeled.
- **Turn-Based Agent Execution**:
  - The AI companion queries the backend asynchronously on a turn-by-turn basis. Continuous real-time visual tracking of moving dynamic adversaries operates at discrete inference intervals (~10 ms per perception frame) rather than continuous high-frequency visual servoing.
