# SpatialAI — Agent & Engineering Guidelines (AGENTS.md)

## 1. Project Philosophy & Educational Objectives
SpatialAI is an academic, from-scratch integration of:
1. **Computer Vision (CS4231)**: Custom CNN-based single-stage detector built and trained from random weights, depth-based 3D localization, and scene graph synthesis.
2. **Language Models & Agentic AI (CS3235)**: Custom tokenizer, decoder-only Transformer implemented from tensor math, causal pre-training, supervised instruction/tool-call tuning, and multi-step agent loop with RAG.
3. **Computer Graphics & Virtual Reality**: Unity 3D virtual laboratory providing RGB-D visual input, camera pose/intrinsics, and receiving safe, validated avatar/action commands.

### Strict Inviolable Rules
1. **No Pretrained Black-Boxes in the Core Inference Path**:
   - Do **NOT** use pretrained YOLO (Ultralytics), ResNet, or MobileNet weights.
   - Do **NOT** use pretrained GPT, Llama, Hugging Face pretrained models, or OpenAI/Anthropic/Google hosted LLM APIs for the core agent loop.
   - PyTorch tensor operations and autograd are used, but the architecture, initialization, forward pass, loss functions, training loops, tokenizer, and decoding are built from scratch.
2. **No Fabricated Metrics or Faked Inference**:
   - Never mock or hardcode model predictions while claiming they are trained model outputs.
   - All evaluation numbers (loss, mAP, IoU, exact-match accuracy) must be measured on real held-out test splits.
   - An untrained model must be identified as untrained.
3. **Traceability to Official Course Syllabi**:
   - Every topic from CS4231 (Computer Vision) and CS3235 (Working with LLMs) is tracked in `docs/syllabus_alignment.md` with explicit status and test verification.
4. **Safety & Determinism**:
   - Never execute arbitrary code from the language model.
   - Tools are validated through strict Pydantic schemas and registered in a secure whitelist.

---

## 2. Directory Structure Conventions
- `backend/`: FastAPI server, perception modules, 3D spatial calculations, from-scratch Transformer and tokenizer, tool execution, and local RAG.
- `configs/`: YAML configuration files (`vision.yaml`, `language_model.yaml`, `agent.yaml`).
- `data/`: Dataset storage (`data/generated/`, `data/corpora/`, `data/documents/`).
- `models/`: Checkpoints saved from our own training runs (`models/vision/`, `models/language_model/`).
- `unity/`: C# scripts, shaders, and scene instructions for Unity 6 (`unity/SpatialAIUnity/`).
- `scripts/`: Data generation, dataset validation, training, and evaluation scripts.
- `tests/`: Automated unit and integration tests.
- `docs/`: Comprehensive technical, academic, and evaluation documentation.

---

## 3. Development Phases & Gates
- **Phase 0: Workspace & Plan**: Environment verification, hardware acceleration detection (Apple Silicon MPS / CUDA / CPU fallback), syllabus matrix initialization.
- **Phase 1: Minimal Working Scaffold**: Repository setup, FastAPI skeleton (`/health`, `/perception`, `/scene`, `/agent/query`), Unity C# scripts, synthetic data generation smoke test, and unit tests.
- **Phase 2: Dataset Pipeline**: Unity-based and synthetic generator for RGB-D + 2D bounding boxes + 3D ground truth + camera intrinsics; dataset validation and split verification.
- **Phase 3: Computer Vision (From Scratch)**: CNN feature backbone, grid prediction heads (objectness, bbox, class logits), custom loss (BCE/Focal + CIoU/SmoothL1), training loop, mAP@0.5 evaluation.
- **Phase 4: 3D Vision & Spatial Localization**: Pinhole back-projection ($X = (u-c_x)Z/f_x, Y = (v-c_y)Z/f_y, Z = depth$), camera-to-world transform, spatial relationship calculation, JSON scene graph.
- **Phase 5: Language Model (From Scratch)**: Character/subword custom tokenizer, decoder-only Transformer (multi-head causal self-attention, layer norm, residual connections), autoregressive pretraining, instruction/tool-call fine-tuning.
- **Phase 6: Agent Loop & Local RAG**: Tool registry, structured JSON output validation, max-step execution guard, TF-IDF cosine-similarity document retriever.
- **Phase 7: End-to-End Unity Integration**: Live RGB-D streaming, perception update, agent natural-language queries, safe avatar waypoint navigation.
- **Phase 8: Evaluation & Documentation**: Measured metric tables, viva demonstration guide, project report.

---

## 4. Definition of Done
A phase or feature is marked **Done** only when:
1. Implementation uses standard, inspectable algorithms and math.
2. Unit and integration tests pass with 100% automated verification.
3. Code is documented with tensor shapes, docstrings, and type hints.
4. `docs/syllabus_alignment.md` is updated with exact code locations and evaluation evidence.
