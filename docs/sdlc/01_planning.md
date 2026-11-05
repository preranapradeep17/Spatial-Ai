# SDLC Stage 1: Planning and Feasibility Study

**Project Name**: Spatial Quest (SpatialAI)  
**Academic Alignment**: CS4231 (Computer Vision) & CS3235 (Working with LLMs) & Computer Graphics  
**Document Version**: 1.0.0  
**Status**: Approved & Active  

---

## 1. Problem Statement
Modern commercial AI systems frequently rely on opaque third-party cloud APIs (such as OpenAI/Anthropic/Google LLMs or pre-trained Ultralytics YOLO models), creating a disconnect in computer science education between theoretical mathematical foundations and practical deployment. In academia, students must understand:
1. How convolutional feature extractors perform spatial downsampling, grid assignment, and bounding-box regression from random tensor weights.
2. How depth maps and pinhole camera intrinsics project 2D image detections into 3D metric Euclidean coordinates.
3. How autoregressive language models tokenize input sequences, compute multi-head causal self-attention, and generate structured tool-calling schemas without commercial black-box APIs.
4. How 3D computer graphics engines represent virtual geometry, coordinate transformations, lighting, and camera projections.

**Spatial Quest** addresses this educational problem by implementing an end-to-end, fully transparent, playable 3D AI mystery-hunt game where visual perception, language understanding, tool execution, and 3D spatial simulation are built from first principles in verifiable Python and WebGL/Three.js code.

---

## 2. Project Objectives
1. **From-Scratch Computer Vision**: Train and evaluate a custom CNN detector (`SpatialDetector`, 2.16M parameters) from random weight initialization on real-world indoor scenes (SUN RGB-D dataset). Perform pinhole camera backprojection to calculate metric $(X, Y, Z)$ positions and build directed spatial scene graphs.
2. **From-Scratch Language Model & Agent**: Implement a subword BPE tokenizer and a 4-layer decoder-only Transformer (`DecoderTransformer`, 856K parameters) using raw PyTorch tensor operations. Implement causal pretraining, instruction tuning for structured `<tool_call>` generation, a ReAct multi-step agent loop, and a local TF-IDF document retriever (RAG).
3. **Playable 3D Graphics & Spatial Environment**: Provide a rich 3D virtual research facility featuring **five distinct locations** (Reception / Central Hub, Research Office, Library / Archives, Investigation Laboratory, Storage Room), real-time camera projections, dynamic lighting, player movement, collision boundaries, and interactive clue objects.
4. **Transparent Live System Inspector**: Deliver a collapsible 3-tab side panel in the game UI displaying real-time telemetry, intermediate tensor shapes, confidence values, camera intrinsics/transforms, tokenized IDs, tool calls, and retrieval similarity scores tied by a correlation ID.
5. **Replayable Mystery Gameplay**: Implement three distinct mystery hunt scenarios (Mystery A: Missing Research Key, Mystery B: Missing Research File, Mystery C: Laboratory Mystery) with reproducible seed-based randomization, item inventory tracking, and dynamic hint generation.
6. **Strict Academic Honesty**: Measure and report real empirical metrics on held-out test splits. Strictly prohibit hardcoded fake outputs or commercial cloud AI APIs.

---

## 3. Target Users
1. **Academic Evaluators & Professors**: Reviewing code adherence to CS4231 and CS3235 course outcomes, inspecting tensor dimensions, loss curves, and mAP/accuracy metrics.
2. **Students & Peer Researchers**: Studying the end-to-end pipeline of computer vision, language modeling, and computer graphics without commercial abstractions.
3. **Players / Investigators**: Navigating the 3D research facility, issuing natural-language instructions to the AI companion, gathering clues, and solving spatial puzzles.

---

## 4. Scope and Out-of-Scope Features

### In Scope
- Custom single-stage CNN detector architecture (`SpatialDetector`) with positive/negative grid target assignment, multi-task loss (BCE objectness, Smooth L1 bbox, CrossEntropy class logits), and NMS.
- Metric depth estimation and 3D world backprojection:
  $$X = \frac{(u - c_x) Z}{f_x}, \quad Y = \frac{(v - c_y) Z}{f_y}, \quad Z = \text{depth}$$
- Directed scene graph construction with spatial relations (`left_of`, `right_of`, `above`, `near`, `distance_to`).
- Custom BPE subword tokenizer and vocabulary management.
- Custom decoder-only Transformer architecture with multi-head causal self-attention, Pre-LayerNorm, residual connections, and position embeddings.
- Multi-step ReAct agent loop supporting 13 validated tools with strict Pydantic schemas.
- Local TF-IDF chunking and cosine similarity RAG over facility incident logs.
- Primary 3D browser runtime at `http://localhost:8000` with 5 distinct rooms, dynamic lighting, 3rd/1st/top-down cameras, player avatar, and drone companion.
- Live 3-tab System Inspector (Computer Vision, Computer Graphics, LLM / Agents) with correlation telemetry.
- Replayable mystery-hunt scenarios with seed randomization and inventory management.
- Automated pytest test suite covering all modules.
- Preservation of Unity 6 C# scripts in `unity/SpatialAIUnity/`.

### Out of Scope
- Commercial foundation model APIs (OpenAI, Anthropic, Google Gemini, HuggingFace Inference API).
- Pretrained weights for vision (YOLOv8, ResNet-50, MobileNet, DETR) or language (GPT-2, Llama, BERT).
- Multiplayer networking or external persistent databases.
- Commercial game engines as mandatory client runtimes (the WebGL/Three.js browser game is the standalone primary runtime, with Unity 6 as a secondary demonstration target).

---

## 5. Technical Feasibility Analysis
- **Compute Feasibility**: The custom CNN (2.16M parameters) and Transformer (856K parameters) are deliberately sized for local inference on consumer workstations. Training and inference run comfortably on Apple Silicon MPS (Metal Performance Shaders) or CPU fallback in milliseconds to seconds per forward pass.
- **Memory Feasibility**: Total model checkpoint footprint is under 30 MB. RAM requirements for the FastAPI backend and Three.js client remain under 1.5 GB.
- **Dataset Feasibility**: The curated SUN RGB-D benchmark subset (RGB images, 16-bit depth arrays, and 3D bounding-box annotations) is maintained locally without external download dependencies.

---

## 6. Dataset and Compute Requirements
- **Dataset**: Real-world SUN RGB-D indoor benchmark (Song et al., CVPR 2015) subset in `data/sunrgbd/` with canonical classes: `table`, `chair`, `door`, `bookshelf`, `sofa`, `cabinet`.
- **Corpora**: Domain-specific investigation text and tool-calling demonstration dialogues in `data/corpora/investigation_corpus.txt`.
- **Documents**: 6 structured facility safety and maintenance documents in `data/documents/facility_documents.json`.
- **Compute**: Local Python 3.9+ with PyTorch 2.8+ using MPS hardware acceleration on Apple Silicon (or CPU fallback).

---

## 7. Main Risks and Mitigation Strategies

| Risk ID | Description | Severity | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | Limited real dataset size yields lower mAP on small classes | Medium | High | Report genuine measured metrics honestly; emphasize pipeline verification, depth backprojection accuracy (0.42m), and domain transfer analysis rather than faking 99% accuracy. |
| **RSK-02** | Compact 856K Transformer generates non-standard JSON syntax | Medium | Medium | Implement robust regex/JSON parser with deterministic semantic fallback in the ReAct loop to ensure zero player lockout. |
| **RSK-03** | Complex 3D scene causes WebGL rendering performance drops | Low | Low | Use low-poly procedural geometries, efficient Three.js mesh instancing, and standard WebGL renderer settings. |
| **RSK-04** | Dual runtimes (Browser vs Unity) fall out of synchronization | Medium | Low | Maintain canonical coordinate and object registry in `backend/game/facility.py` shared by both the backend API and client viewports. |

---

## 8. Required Hardware & Software
- **Operating System**: macOS (Darwin 24.3.0 / Apple Silicon M-series) or Linux / Windows with Python 3.9+.
- **Software Dependencies**: PyTorch 2.8+, FastAPI 0.128+, Uvicorn 0.34+, NumPy 2.0+, Pillow 11.0+, PyYAML 6.0+, pytest 8.4+.
- **Web Client**: Modern web browser (Chrome, Safari, Edge, Firefox) supporting WebGL and ES6.
- **Secondary Client**: Unity 6 (optional, preserved in `unity/SpatialAIUnity/`).

---

## 9. Development Milestones
- **M1: Baseline Verification & Planning** (SDLC Stages 1–3, current state audit, requirements matrix, architecture design).
- **M2: Facility Layout Expansion** (Implement 5 distinct rooms: Reception, Office, Archive/Library, Lab, Storage Room).
- **M3: Replayable Mystery System** (Seed-based mystery hunt generator, inventory tracking, clue state machine, dynamic hints).
- **M4: Live System Inspector & Telemetry** (Build 3-tab collapsible drawer, backend telemetry recorder, correlated trace IDs).
- **M5: Tool Registry & Agent Expansion** (`get_inventory`, `get_mission_objectives`, `request_hint`, fix test assertions).
- **M6: Verification, Integration & Final Testing** (Full regression suite, end-to-end user scenario testing, SDLC documentation completion).

---

## 10. Primary Runtime Identification
The **primary, standalone playable runtime** is the browser-based 3D game served directly by the FastAPI backend at `http://localhost:8000`. It requires zero external game engine installation, opens instantly in any browser, and provides full 3D interactive character movement, drone escort, and clue inspection. The Unity 6 project in `unity/SpatialAIUnity/` is preserved as an academic graphics extension and secondary demonstration runtime.
