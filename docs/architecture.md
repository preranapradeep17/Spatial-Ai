# SpatialAI Architecture & System Design

## 1. System Overview & Core Loop
SpatialAI bridges computer vision, language modeling, and 3D computer graphics into a closed-loop spatial agent.

```
+-----------------------------------------------------------------------------------+
|                        3D World: WebGL Three.js & Unity 6                         |
|  5 Distinct Facility Rooms: Reception, Office, Library, Laboratory, Storage Vault  |
|           RGB Camera Capture [640x480]  +  Linear Depth Map [640x480]             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v (HTTP POST /perception)
+-----------------------------------------------------------------------------------+
|                        SpatialAI Python Perception Server                         |
|                                                                                   |
|  1. Custom CNN Detector (From Scratch, 2.16M params):                             |
|     - Input: RGB (3, 480, 640)                                                    |
|     - Architecture: Conv2D Backbones + LeakyReLU + Stride-32 Downsampling         |
|     - Grid Prediction Head: (15, 20, 11) -> [Obj, x, y, w, h, 6 Class Logits]     |
|                                                                                   |
|  2. Robust Depth Sampling:                                                        |
|     - Median filter over inner 40% bounding box patch                             |
|                                                                                   |
|  3. Pinhole Back-Projection & World Coordinate Mapping:                           |
|     X_cam = (u - cx)*Z / fx                                                       |
|     Y_cam = (v - cy)*Z / fy                                                       |
|     P_world = R_cam * P_optical + T_cam                                           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             3D Dynamic Scene Graph                                |
|  - Nodes: Detected objects with 3D world coordinates [X, Y, Z] & confidence       |
|  - Edges: Geometric spatial relations (ON, LEFT_OF, RIGHT_OF, NEAR)                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v (Spatial Context)
+-----------------------------------------------------------------------------------+
|                     Language Model & Agent Reasoning Loop                         |
|                                                                                   |
|  User Natural Language Query ("Where is the nearest door?")                       |
|                             |                                                     |
|                             v                                                     |
|  Custom Tokenizer -> Custom Decoder-Only Transformer (From Scratch, 856K params)  |
|                             |                                                     |
|                             v                                                     |
|  Structured Tool Call Emission:                                                   |
|  <tool_call>{"name": "find_nearest_target", "arguments": {"target_class": "door"}}</tool_call>
|                             |                                                     |
|                             v                                                     |
|  Pydantic Validation & Whitelisted 13-Tool Registry:                              |
|  - get_visible_objects()          - calculate_distance(A, B)                      |
|  - locate_object(name)            - find_nearest_target(class)                    |
|  - query_spatial_relation(obj, r) - search_documents(query)                       |
|  - find_path(destination)         - inspect_clue(id)                              |
|  - navigate_to(destination)       - interact_with_object(id)                      |
|  - get_inventory()                - get_mission_objectives()                      |
|  - request_hint()                                                                 |
|                             |                                                     |
|                             v                                                     |
|  Tool Execution -> Observation Feedback -> Grounded Natural Language Response     |
+-----------------------------------------+-----------------------------------------+
                                          |
               +--------------------------+--------------------------+
               |                                                     |
               v                                                     v
+-----------------------------+       +---------------------------------------------+
|    3D Avatar / Companion    |       |        Live 3-Tab System Inspector          |
|  Avatar smoothly navigates  |       |  Correlated telemetry via correlation_id:   |
|  along waypoints to target  |       |  1. CV: Tensors, intrinsics, 3D detections  |
|  room/object coordinate     |       |  2. CG: Poses, camera frustum, ground truth |
|                             |       |  3. LLM: Tokens, params, tool call, RAG sim |
+-----------------------------+       +---------------------------------------------+
```

---

## 2. Component Decoupling
Each module is engineered to run and be tested independently:
- **Perception**: Evaluated on SUN RGB-D benchmark and synthetic datasets without frontend running.
- **Geometry**: Verified using analytical unit tests with known synthetic ground truths (`tests/test_geometry.py`).
- **Language Model**: Pre-trained and evaluated on domain corpora without perception running (`tests/test_transformer.py`).
- **Agent & Tools**: Evaluated against in-memory mock or real scene graphs (`tests/test_agent.py`, `tests/test_api.py`).
- **3D World & System Inspector**: Operates both in browser WebGL (Three.js) at `http://localhost:8000` and in Unity 6 C# without requiring cloud AI APIs.
