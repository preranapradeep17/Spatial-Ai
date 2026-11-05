# SDLC Stage 2: Requirements Analysis

**Project Name**: Spatial Quest (SpatialAI)  
**Document Version**: 1.0.0  
**Traceability**: CS4231 & CS3235 Official Syllabus Alignment  

---

## 1. Requirements Overview & Taxonomy
Requirements are categorized with unique identifiers:
- **CV-xxx**: Computer Vision & Perception
- **LLM-xxx**: Language Model, Tokenizer & Agentic Reasoning
- **CG-xxx**: Computer Graphics & 3D Environment
- **GAME-xxx**: Gameplay, Mystery Hunt & Player Experience
- **DATA-xxx**: Data Ingestion, Formatting & Checkpoints
- **SAFE-xxx**: Safety, Validation & Guardrails
- **PERF-xxx**: Performance & Resource Budgets
- **TEST-xxx**: Acceptance Criteria & Verification Procedures

---

## 2. Requirements Traceability Matrix

### 2.1 Computer Vision Requirements (CV)

| Req ID | Description | Component | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **CV-001** | Implement a custom CNN-based object detector (`SpatialDetector`) trained from random weight initialization without pretrained backbones. | `backend/perception/model.py` | `tests/test_vision_model.py::test_model_forward_pass` |
| **CV-002** | Support 6 canonical indoor object classes: `table`, `chair`, `door`, `bookshelf`, `sofa`, `cabinet`. | `backend/perception/model.py`, `configs/vision.yaml` | `tests/test_vision_model.py::test_loss_computation` |
| **CV-003** | Implement multi-task loss combining Binary Cross-Entropy objectness, Smooth L1 bounding box regression, and Cross-Entropy classification. | `backend/perception/losses.py` | `tests/test_vision_model.py::test_loss_computation` |
| **CV-004** | Implement pinhole camera backprojection to convert 2D bounding boxes and depth into camera-space metric 3D coordinates: $X=(u-c_x)Z/f_x, Y=(v-c_y)Z/f_y$. | `backend/spatial/geometry.py` | `tests/test_geometry.py::test_pinhole_backprojection` |
| **CV-005** | Implement tilt matrix correction to transform camera-space coordinates to facility world-space coordinates. | `backend/spatial/localization.py` | `tests/test_localization.py::test_benchmark_3d_localization` |
| **CV-006** | Construct a directed 3D spatial scene graph with geometric relationships (`left_of`, `right_of`, `above`, `near`, `distance_to`). | `backend/spatial/scene_graph.py` | `tests/test_scene_graph.py::test_scene_graph_building` |
| **CV-007** | Evaluate precision, recall, IoU, and mAP@0.5 on held-out test splits without hardcoded or fabricated scores. | `backend/perception/metrics.py`, `scripts/evaluate_detector.py` | `scripts/evaluate_detector.py` run report |

### 2.2 Language Model & Agentic AI Requirements (LLM)

| Req ID | Description | Component | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **LLM-001** | Implement a custom subword BPE tokenizer from scratch with special tokens (`<tool_call>`, `</tool_call>`, `<thought>`, `</thought>`). | `backend/language_model/tokenizer.py` | `tests/test_tokenizer.py::test_tokenizer_train_and_encode_decode` |
| **LLM-002** | Implement a decoder-only Transformer from tensor mathematics with scaled dot-product attention, causal masking, Pre-LN, and FFN blocks. | `backend/language_model/transformer.py` | `tests/test_transformer.py::test_causal_self_attention_shapes_and_mask` |
| **LLM-003** | Support autoregressive token generation with temperature sampling and stop-token conditions. | `backend/language_model/transformer.py` | `tests/test_transformer.py::test_transformer_autoregressive_generation` |
| **LLM-004** | Implement a multi-step ReAct agent loop capable of parsing structured tool calls and feeding tool observations back into context. | `backend/agent/investigation_agent.py` | `tests/test_agent.py::test_agent_visible_objects_query` |
| **LLM-005** | Support configurable logical agent roles (Scout / Investigation Agent, Navigator / Spatial Agent, Archivist / Knowledge Agent) sharing the base Transformer. | `backend/game/missions.py`, `backend/agent/investigation_agent.py` | `tests/test_missions.py::test_agent_switch` |
| **LLM-006** | Implement a local TF-IDF document chunking, indexing, and cosine-similarity retrieval system (RAG) over facility records. | `backend/rag/retriever.py` | `tests/test_rag.py::test_tfidf_retriever_chunking_and_indexing` |
| **LLM-007** | Maintain bounded conversation history without using conversational claims as ground-truth authority over inventory or mission state. | `backend/agent/investigation_agent.py` | `tests/test_agent.py::test_agent_distance_query` |

### 2.3 Computer Graphics & 3D Environment Requirements (CG)

| Req ID | Description | Component | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **CG-001** | Render a real-time 3D virtual research facility in the browser using Three.js / WebGL. | `frontend/index.html` | Browser rendering at `http://localhost:8000` |
| **CG-002** | Provide **five distinct facility locations**: Reception / Central Hub, Research Office, Library / Archives, Investigation Laboratory, Storage Room. | `backend/game/facility.py`, `frontend/index.html` | Scene traversal & zone trigger verification |
| **CG-003** | Provide distinct lighting, floor textures/grids, and furniture layouts for each location. | `frontend/index.html` | Visual inspection & zone lighting tests |
| **CG-004** | Implement third-person, first-person, and top-down camera modes with perspective projection. | `frontend/index.html` | Camera mode button triggers and projection update |
| **CG-005** | Provide responsive player movement (WASD / D-Pad), mouse-drag look controls, and facility boundary collisions. | `frontend/index.html` | Interactive character traversal across all 5 rooms |
| **CG-006** | Render a visual companion drone avatar with smooth hover animation and waypoint transit navigation. | `frontend/index.html` | Drone transit trigger upon navigation command |
| **CG-007** | Preserve Unity 6 project integration in `unity/SpatialAIUnity/` as an academic graphics extension. | `unity/SpatialAIUnity/` | Unity C# script audit and action receiver tests |

### 2.4 Gameplay & Mystery Hunt Requirements (GAME)

| Req ID | Description | Component | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **GAME-001** | Implement Mystery A ("Missing Research Key"): player searches rooms, inspects clues, locates the key, and unlocks the target. | `backend/game/missions.py` | `tests/test_missions.py::test_mystery_a_progression` |
| **GAME-002** | Implement Mystery B ("Missing Research File"): player searches archives and office, retrieves notes, and recovers the file. | `backend/game/missions.py` | `tests/test_missions.py::test_mystery_b_progression` |
| **GAME-003** | Implement Mystery C ("Laboratory Mystery"): player interprets chemical lab notes, analyzes spatial relations, and reaches the objective. | `backend/game/missions.py` | `tests/test_missions.py::test_mystery_c_progression` |
| **GAME-004** | Support controlled seed-based randomization to vary clues, room assignments, and puzzle targets between sessions. | `backend/game/missions.py` | Session generation with different seeds produces valid variants |
| **GAME-005** | Track player inventory (collected keys, files, datapads) with a dedicated query tool (`get_inventory`). | `backend/game/missions.py`, `backend/tools/registry.py` | Inventory inspection tests |
| **GAME-006** | Provide dynamic hints without revealing solutions prematurely via `request_hint`. | `backend/tools/registry.py` | Hint generation test |
| **GAME-007** | Display an interactive "How to Play" tutorial modal covering movement, camera, clues, commands, and inspector access. | `frontend/index.html` | Modal display and dismissal verification |

### 2.5 Live System Inspector Requirements (INSP)

| Req ID | Description | Component | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **INSP-001** | Implement a collapsible side drawer with 3 distinct tabs: Computer Vision, Computer Graphics / 3D, and LLM / Agents. | `frontend/index.html` | Inspector toggle and tab switching |
| **INSP-002** | Computer Vision tab: display input image, detection boxes, confidence, class logits, model architecture, tensor shapes, and depth map. | `frontend/index.html`, `backend/api/routes.py` | CV telemetry rendering |
| **INSP-003** | Computer Graphics tab: display active room, player coordinates, camera properties, render dimensions, and model-estimated vs ground-truth positions. | `frontend/index.html` | CG telemetry rendering |
| **INSP-004** | LLM / Agent tab: display user query, tokenization with token IDs, generated JSON, parsed intent, tool name/args, tool result, and RAG retrieval scores. | `frontend/index.html`, `backend/api/routes.py` | LLM telemetry rendering |
| **INSP-005** | Implement a telemetry correlation engine linking CV, CG, and LLM events under a shared correlation ID per user turn. | `backend/api/routes.py`, `backend/agent/investigation_agent.py` | `/telemetry/latest` endpoint test |

### 2.6 Safety & Guardrail Requirements (SAFE)

| Req ID | Description | Component | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **SAFE-001** | Prohibit execution of arbitrary code from language model output; enforce whitelist tool dispatch. | `backend/tools/registry.py` | `tests/test_agent.py` |
| **SAFE-002** | Validate all tool arguments against strict Pydantic schemas before invocation. | `backend/api/schemas.py`, `backend/tools/registry.py` | Invalid argument rejection tests |
| **SAFE-003** | Ensure game state confirmation before assistant claims an action succeeded. | `backend/agent/investigation_agent.py` | Confirmed action verification |

### 2.7 Performance & Reliability Requirements (PERF)

| Req ID | Description | Target Metric | Verification Procedure |
| :--- | :--- | :--- | :--- |
| **PERF-001** | FastAPI backend health check latency | $< 50$ ms | `curl http://localhost:8000/health` |
| **PERF-002** | Three.js rendering frame rate | $\ge 45$ FPS on modern displays | Chrome DevTools FPS meter |
| **PERF-003** | Transformer inference latency | $< 1.5$ s per turn on Apple Silicon MPS | Turn execution timestamp diff |
| **PERF-004** | Model checkpoint size footprint | $< 35$ MB total disk storage | `ls -lh models/*` |

---

## 3. Acceptance Criteria (TEST)

- **TEST-001**: 100% of automated unit and integration tests pass via `pytest tests/`.
- **TEST-002**: Player can walk through all 5 rooms, observe distinct visual themes, and encounter interactive clues.
- **TEST-003**: Player can execute natural-language queries to detect objects, query spatial relations, calculate distances, and navigate to waypoints.
- **TEST-004**: System Inspector can be opened/closed seamlessly, displaying live CV, CG, and LLM telemetry.
- **TEST-005**: All three mysteries (A, B, C) can be started, progressed, and completed with inventory tracking.
- **TEST-006**: Zero commercial cloud AI API dependencies present in the runtime codebase.
