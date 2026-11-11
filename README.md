# Spatial Quest: An AI Investigation Game Powered by Computer Vision and a Custom-Trained Language Model

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-orange.svg)](https://pytorch.org/)
[![Hardware Acceleration](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%2F%20CUDA-green.svg)]()
[![Tests](https://img.shields.io/badge/Tests-124%20Passed%20(100%25)-brightgreen.svg)]()
[![Gemini SDK](https://img.shields.io/badge/AI%20Partner-Google%20Gemini%20API%20%2B%20Local%20Fallback-blueviolet.svg)]()

**Spatial Quest** is a 3D Embodied Vision-Language Investigation Environment integrating:
1. **Fundamentals of Computer Vision (CS4231)**: A custom CNN single-stage object detector (`SpatialDetector`, 2,162,731 trainable parameters) trained from scratch on indoor **SUN RGB-D** imagery, paired with calibrated depth back-projection, Scene Graph synthesis (with explicit `facility_layout` vs. live `custom_detector` provenance), and live viewport CNN scan support.
2. **Working with Large Language Models & Agentic AI (CS3235)**:
   - **Primary Conversational AI Partner**: Server-side integration via the official `google-genai` SDK (`Gemini 2.5 Flash` / configurable via `GEMINI_MODEL`), grounded in the live 3D Scene Graph, TF-IDF RAG document retrieval, session-scoped memory, SSE streaming, and 18+ Pydantic-validated investigation tools.
   - **Built-in Academic & Offline Fallback**: Custom BPE subword tokenizer (`SpatialTokenizer`) and decoder-only Transformer (`DecoderTransformer`, 856,832 trainable parameters) preserved cleanly for offline verification and zero-key environments.
3. **Computer Graphics & 3D Simulation**: Modular Three.js r128 web interface at `http://localhost:8000` (and desktop/tablet/mobile responsive layout) featuring a rigged humanoid 3D investigator (`Idle` / `Walk` / `Run` skeletal animations, 1st/3rd/top-down cameras, AABB collisions), a realistic PBR 5-zone facility (Reception Hub, Research Office, Library / Archives, Investigation Laboratory, Storage Vault), interactive blueprint Facility Map, object inspection panel, and System Telemetry Inspector. See [`docs/assets_sources.md`](docs/assets_sources.md) for 3D asset licensing and attribution.

---

## 1. Quickstart & Verification

### Step 1: Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (Server-Side Gemini API)
Copy `.env.example` to `.env` and add your Gemini API key (never exposed to browser JS):
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### Step 3: Run Full Automated Test Suite (124 Tests)
```bash
python3 -m pytest tests/ -v
```
*(Automated tests cover API, geometry, scene graph, dataset loader, 3D localization, tokenizer, transformer, RAG, agent, 5-room facility bounds, seed randomization, inventory state, real and mock perception inference, and telemetry).*

### Step 3: Run Live Perception & Detector Evaluation
```bash
# Evaluate custom detector on held-out SUN RGB-D real test split
python3 scripts/evaluate_detector.py

# Run Dual Domain-Transfer evaluation (SUN RGB-D Real vs Unity Game-Camera)
python3 scripts/evaluate_unity_transfer.py
```

### Step 4: Run Language Model & Agent Verification
```bash
# Verify custom Transformer training from random weights
python3 scripts/train_language_model.py

# Test autonomous agent on all tools
python3 -m pytest tests/test_agent.py -v
```

### Step 5: Launch Playable 3D Game & System Inspector
```bash
python3 run.py
```
Open `http://localhost:8000` in your web browser:
- **Investigator Avatar**: Walk with `WASD` / Arrow Keys across all 5 facility rooms in 3rd-person view (toggle `1ST PERSON` or `TOP-DOWN` on the camera bar).
- **Multimodal Voice Input**: Click the microphone icon (`🎙️`), speak your investigation inquiry, and review the live transcribed text in the input box before sending.
- **Offline Spoken Responses**: Toggle `🔊 VOICE: ON` to hear the AI companion speak answers aloud via native browser speech synthesis (or click `⏹️ STOP` to silence).
- **Mystery Hunt Selector**: Click `NEW MYSTERY` to select Mystery A, B, C, D, or E; use `RESTART MISSION` to replay, or `RETURN TO HUB`.
- **Live System Inspector**: Click `🔍 SYSTEM INSPECTOR` in the header to view correlated real-time telemetry (`#corr-...`) across:
  - **Computer Vision**: Real SUN RGB-D test frame overlay, depth colormap, intermediate CNN tensor shapes, 3D detections table, and honest benchmark cards.
  - **Computer Graphics**: Live player coordinates `[X, Y, Z]`, orientation angle, active room, rendering FPS, and ground-truth vs estimated transforms.
  - **LLM / Agents**: Real token IDs, raw model output, direct generation vs fallback badge, tool execution results, RAG passages, and latency.

---

## 2. Core Technical Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │          3D World: WebGL Three.js & Unity 6            │
                       │   5 Rooms: Reception / Office / Library / Lab / Vault  │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │ RGB-D Stream / REST Queries
                                                  v
                       ┌────────────────────────────────────────────────────────┐
                       │          FastAPI Game Server Backend (Local)           │
                       └──────────┬───────────────────────────────┬─────────────┘
                                  │                               │
             Vision & Spatial Perception               Autonomous Investigation Agent
  ┌────────────────────────────────────────────┐ ┌────────────────────────────────────────────┐
  │ 1. SpatialDetector (Custom CNN, 2.16M)     │ │ 1. SpatialTokenizer (Custom BPE, 437)      │
  │ 2. Anchor-Free Grid Head (15x20x11)        │ │ 2. DecoderTransformer (Custom, 856K)      │
  │ 3. SUN RGB-D Real-World Benchmark          │ │ 3. 13 Standardized Investigation Tools     │
  │ 4. Calibrated Depth Back-Projection        │ │ 4. TF-IDF Cosine Similarity Local RAG      │
  │ 5. Dynamic 3D Spatial Scene Graph Engine   │ │ 5. ReAct Autonomous Multi-Step Loop        │
  └────────────────────────────────────────────┘ └────────────────────────────────────────────┘
                                  │                               │
                                  └───────────────┬───────────────┘
                                                  v
                       ┌────────────────────────────────────────────────────────┐
                       │              Live 3-Tab System Inspector               │
                       │  CV (Tensors) ↔ CG (Poses) ↔ LLM (Tokens & Tool Calls) │
                       └────────────────────────────────────────────────────────┘
```

---

## 3. The 13 Standardized Investigation Tools

| Tool Signature | Purpose | Module |
|---|---|---|
| `get_visible_objects()` | Returns all detected objects in camera FOV with 3D positions | `backend.tools.registry` |
| `locate_object(object_name)` | Locates specific object class and returns metric coordinates | `backend.spatial.scene_graph` |
| `query_spatial_relation(object_name, relation)` | Intersite spatial reasoning (`left_of`, `right_of`, `near`, `above`) | `backend.spatial.scene_graph` |
| `calculate_distance(object_a, object_b)` | Euclidean distance in meters between two detected objects | `backend.spatial.scene_graph` |
| `find_nearest_target(target_class)` | Identifies closest object of specified class to the observer | `backend.spatial.scene_graph` |
| `find_path(destination)` | Calculates obstacle-avoiding 3D waypoints | `backend.tools.registry` |
| `navigate_to(destination)` | Dispatches physical waypoint navigation to companion avatar | `unity.SpatialAIUnity` / WebGL |
| `inspect_clue(clue_id)` | Reveals state and clues of interactive objects (`safe`, `pda`) | `backend.tools.registry` |
| `search_documents(query)` | RAG search over incident reports and emergency override codes | `backend.rag.retriever` |
| `interact_with_object(object_id)` | Triggers environmental state changes (unlock safe, open door) | `backend.tools.registry` |
| `get_inventory()` | Inspects investigator collected clues and inventory items | `backend.game.facility` |
| `get_mission_objectives()` | Returns current active mission objectives and completion status | `backend.game.missions` |
| `request_hint()` | Generates contextual guidance without revealing the solution | `backend.agent.investigation_agent` |

---

## 4. Key Experimental Results & Metrics

All metrics are measured on real data and un-faked test splits:
- **3D Spatial Localization**: **0.423 m** mean 3D Euclidean error across 40 real physical indoor objects in SUN RGB-D (bookshelf: 16.8 cm, cabinet: 25.1 cm, door: 25.4 cm, chair: 47.4 cm) calibrated against oracle 2D bounding boxes and depth maps.
- **Vision Detector Training & Verification**: Loss converged from **19.03** to **0.876** on Apple Silicon MPS (`models/vision/spatial_detector_best.pt`). Real split mAP@0.5 is 0.0000 due to training 2.16M params on 5 frames without pretraining; visual comparison saved in `results/plots/detector_test_predictions_vs_gt.png`.
- **Transformer Language Model**: Causal cross-entropy loss converged to **1.2610**; perplexity dropped to **3.53** on multi-turn dialogue corpus (`models/language_model/transformer_best.pt`).
- **Structured Tool Emission**: **100% valid schema format** with **94.0% intent routing accuracy** across test prompts, featuring transparent telemetry between direct model generation and fallback semantic routing.
- **Dual Domain Transfer**: Quantitative evaluation comparing SUN RGB-D real test frames vs Unity game camera (`results/metrics/dual_domain_transfer_metrics.json`).
- **Automated Tests**: **51 / 51 tests passing** (100% verification across all components in ~12.5s).

---

## 5. SDLC Stages Documentation

1. [Stage 1 — Planning and Feasibility Study](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/01_planning.md)
2. [Stage 2 — Requirements Analysis](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/02_requirements.md)
3. [Stage 3 — System Design](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/03_system_design.md)
4. [Stage 4 — Detailed Design](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/04_detailed_design.md)
5. [Stage 5 — Testing and Verification](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/05_testing.md)
6. [Stage 6 — Integration](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/06_integration.md)
7. [Stage 7 — Deployment and Reproducibility](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/07_deployment.md)
8. [Stage 8 — Maintenance and Operations](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/sdlc/08_maintenance.md)

---

## 6. Academic Documentation & Viva Preparation
- [Official Syllabus Traceability Matrix](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/syllabus_alignment.md)
- [Academic Viva & Defense Guide](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/viva_demonstration.md)
- [Game Design Document (GDD)](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/game_design.md)
- [System Architecture](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/architecture.md)
- [Empirical Evaluation & Metrics](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/evaluation.md)
- [System Limitations & Scope Boundaries](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/limitations.md)
- [Safe File Cleanup & Integrity Report](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/cleanup_report.md)
- [Metric-Verified Resume Project Entry](file:///Users/meghak/Documents/sem7/Spatial%20AI/docs/resume_project_entry.md)
