# SpatialAI: Project Specification & Architecture

## 1. Executive Summary
**SpatialAI** is a from-scratch academic multimodal AI system for 3D spatial perception and natural-language interaction in a virtual laboratory environment. It integrates:
1. **Computer Vision (CS4231)**: A custom single-stage CNN object detector (anchor-free grid prediction) built and trained from random initialization, coupled with calibrated RGB-D pinhole back-projection and spatial scene-graph generation.
2. **Language Models & Agentic AI (CS3235)**: A custom tokenizer and decoder-only Transformer built from raw tensor mathematics (multi-head causal self-attention, layer normalization, residual blocks), trained first on domain autoregressive language modeling and then on supervised instruction/tool-call generation.
3. **Computer Graphics & Virtual Reality**: A 3D virtual laboratory in Unity 6, exposing RGB rendering, linear depth maps, camera intrinsics/extrinsics, and an avatar navigation controller.

---

## 2. Core Architectural Principles
1. **From-Scratch Educational Ownership**:
   - Zero reliance on pre-trained YOLO/ResNet/CLIP models.
   - Zero reliance on pre-trained GPT/Llama weights or commercial LLM APIs in the core loop.
   - All modules expose inspectable weights, tensor shapes, and loss curves.
2. **Empirical Measurement**:
   - Genuine test-set evaluation using mAP@0.5, IoU, exact-match tool call accuracy, and 3D Euclidean localization error.
3. **Decoupled Perception and Geometry**:
   - Perception predicts 2D bounding boxes and class probabilities.
   - Localization calculates 3D coordinates via calibrated depth back-projection:
     $$X_{cam} = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y_{cam} = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z_{cam} = Z_{depth}$$
   - Spatial transformation maps camera coordinates to Unity world coordinates:
     $$P_{world} = R_{cam \to world} \cdot P_{cam} + T_{cam \to world}$$
4. **Safety & Tool Registry**:
   - The LLM emits structured JSON tool calls that are strictly validated against Pydantic schemas before execution.

---

## 3. Supported Laboratory Object Classes
The initial vocabulary consists of 6 primary object classes commonly found in laboratory scenes:
1. `table` (Class ID 0)
2. `chair` (Class ID 1)
3. `laptop` (Class ID 2)
4. `monitor` (Class ID 3)
5. `door` / `exit` (Class ID 4)
6. `fire_extinguisher` (Class ID 5)

---

## 4. Hardware and Environment Specifications
- **Operating System**: macOS 14.7.1 (Darwin arm64)
- **Primary Compute Device**: Apple Silicon GPU via PyTorch Metal Performance Shaders (`mps`), with transparent fallback to `cpu` or `cuda`.
- **Python**: 3.9+ with PyTorch 2.8.0, NumPy 2.0.2, SciPy 1.13.1, Pillow 11.3.0, FastAPI 0.128.8, Uvicorn 0.39.0.
- **Unity Version**: Unity 6 (`6000.6.0f1-arm64`) using Universal Render Pipeline (URP) or Built-in Render Pipeline.

---

## 5. Development Phases
- **Phase 0**: Workspace inspection, hardware verification, architecture sign-off, syllabus matrix creation.
- **Phase 1**: Minimal working scaffold, API skeleton (`/health`, `/perception`, `/scene`, `/agent/query`), Unity C# scripts, synthetic dataset generator smoke test, automated unit tests.
- **Phase 2**: Full dataset generation pipeline (RGB, depth, 2D bboxes, 3D poses, camera intrinsics/extrinsics), validation checks, train/val/test splits.
- **Phase 3**: Custom CNN object detector from scratch: forward pass, loss functions (objectness, box regression, classification), training on MPS/CPU, mAP@0.5 evaluation.
- **Phase 4**: 3D spatial vision & scene graph: robust depth sampling, back-projection, spatial relationships (`left_of`, `right_of`, `in_front_of`, `on`, `near`), JSON scene graph.
- **Phase 5**: Custom Transformer language model from scratch: custom tokenizer, embeddings, multi-head causal attention, autoregressive pre-training, supervised tool-call fine-tuning.
- **Phase 6**: Agentic AI & Local RAG: Tool registry (`get_visible_objects`, `locate_object`, `calculate_distance`, `find_nearest_exit`, `search_safety_manual`), TF-IDF retriever, multi-turn reasoning loop.
- **Phase 7**: End-to-end integration with Unity: live camera feed, real-time perception, agent reasoning, safe avatar navigation.
- **Phase 8**: Rigorous evaluation, visualization dashboard, viva presentation guide, comprehensive academic documentation.
