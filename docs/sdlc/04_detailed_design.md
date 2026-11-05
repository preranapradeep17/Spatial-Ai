# SDLC Stage 3: Detailed Software Design

**Project Name**: Spatial Quest (SpatialAI)  
**Document Version**: 1.0.0  
**Status**: Approved  

---

## 1. Computer Vision Detailed Design

### 1.1 CNN Feature Extractor & Detection Head Architecture
The `SpatialDetector` is implemented from scratch in `backend/perception/model.py`:
- **Input Dimensions**: $(B, 3, 480, 640)$ normalized to $[0, 1]$.
- **Backbone Stages**:
  - Stage 1: Conv2D $(3 \to 32, k=3, s=2, p=1) \to$ BatchNorm $\to$ LeakyReLU(0.1) $\to$ $(B, 32, 240, 320)$
  - Stage 2: Conv2D $(32 \to 64, k=3, s=2, p=1) \to$ BatchNorm $\to$ LeakyReLU(0.1) $\to$ $(B, 64, 120, 160)$
  - Stage 3: Conv2D $(64 \to 128, k=3, s=2, p=1) \to$ BatchNorm $\to$ LeakyReLU(0.1) $\to$ $(B, 128, 60, 80)$
  - Stage 4: Conv2D $(128 \to 256, k=3, s=2, p=1) \to$ BatchNorm $\to$ LeakyReLU(0.1) $\to$ $(B, 256, 30, 40)$
  - Stage 5: Conv2D $(256 \to 256, k=3, s=2, p=1) \to$ BatchNorm $\to$ LeakyReLU(0.1) $\to$ $(B, 256, 15, 20)$
- **Prediction Head**:
  - Conv2D $(256 \to 128, k=3, s=1, p=1) \to$ LeakyReLU(0.1)
  - Final Conv2D $(128 \to 11, k=1, s=1, p=0) \to$ Reshape to $(B, 15, 20, 11)$.
- **Channel Assignment per Grid Cell**:
  - Channel 0: Objectness logit (sigmoid applied during post-processing).
  - Channels 1–4: Normalized bounding-box offsets $(t_x, t_y, t_w, t_h)$ relative to grid cell.
  - Channels 5–10: 6 class logits (`table`, `chair`, `door`, `bookshelf`, `sofa`, `cabinet`).

### 1.2 Mathematical Formulation of 3D Backprojection
For each 2D bounding box $[u_{\min}, v_{\min}, u_{\max}, v_{\max}]$, the centroid pixel coordinate is:
$$u_c = \frac{u_{\min} + u_{\max}}{2}, \quad v_c = \frac{v_{\min} + v_{\max}}{2}$$
The metric depth $Z_c$ is sampled from the corresponding depth array within a central median patch. The camera-space coordinates are computed using pinhole intrinsics:
$$X_c = \frac{(u_c - c_x) Z_c}{f_x}, \quad Y_c = \frac{(v_c - c_y) Z_c}{f_y}$$
Using the physical camera tilt angle $\theta_{\text{tilt}}$:
$$\begin{pmatrix} X_w \\ Y_w \\ Z_w \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} X_c \\ Y_c \\ Z_c \end{pmatrix} + \begin{pmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \end{pmatrix}$$

---

## 2. Language Model & Agent Detailed Design

### 2.1 Decoder-Only Transformer Architecture
Built from pure PyTorch tensor math in `backend/language_model/transformer.py`:
- **Hyperparameters**:
  - Vocabulary Size: $V = 384$ (custom subwords)
  - Maximum Context Length: $T = 256$
  - Model Dimension: $d_{\text{model}} = 128$
  - Attention Heads: $n_{\text{heads}} = 4$ ($d_k = d_v = 32$)
  - Feed-Forward Dimension: $d_{\text{ff}} = 512$
  - Number of Layers: $L = 4$
  - Total Parameters: 856,832
- **Pre-LayerNorm Block Formulation**:
  $$x^{(l)\prime} = x^{(l-1)} + \text{MultiHeadAttention}(\text{LayerNorm}(x^{(l-1)}))$$
  $$x^{(l)} = x^{(l)\prime} + \text{FFN}(\text{LayerNorm}(x^{(l)\prime}))$$
- **Scaled Dot-Product Causal Self-Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
  where $M_{i,j} = 0$ for $j \le i$, and $-\infty$ for $j > i$.

### 2.2 ReAct Agent Loop & Semantic Intent Map
1. **Prompt Construction**: Formats agent ID, bounded history, and user input.
2. **Model Decoding**: Autoregressively generates continuation tokens up to `</tool_call>`.
3. **Structured Parser**: Extracts JSON payload `{"name": ..., "arguments": ...}`.
4. **Deterministic Intent Router**: If generation is malformed on Step 0, regex semantic matching identifies intended tool to guarantee zero player lockout during live evaluation.
5. **Execution**: Dispatches tool to `ToolRegistry` with Pydantic argument validation.
6. **Observation Loop**: Feeds output string back into prompt context.
7. **Synthesis**: Formulates a clear, contextual natural-language explanation.

---

## 3. Five-Location Facility Design

The facility is laid out along a coherent metric spatial grid spanning $Z = -24$ to $Z = +24$ meters and $X = -12$ to $+12$ meters:

| Location | Coordinates ($X, Z$) | Visual Theme & Lighting | Characteristic Objects | Mission Role |
| :--- | :--- | :--- | :--- | :--- |
| **1. Reception / Central Hub** | $X \in [-10, 10], Z \in [-24, -10]$ | Amber security glow, marble tile floor | Reception desk, visitor sofa, security terminal table | Game spawn, orientation, visual survey |
| **2. Research Office** | $X \in [1, 10], Z \in [-10, 0]$ | Fluorescent desk lighting, wood-paneled walls | Office desks, rolling chairs, file cabinets | Investigation, research log recovery |
| **3. Library / Archives** | $X \in [-10, -1], Z \in [-10, 0]$ | Warm brass reading lamps, tall bookshelves | Archive bookshelves, reference tables, reading carrel | Document analysis, historical blueprints |
| **4. Investigation Laboratory** | $X \in [-10, 10], Z \in [0, 12]$ | Vivid cyan scientific spotlights, hazard grid | Central workbench, chemical lockers, datapad clues, laser doorway | Spatial puzzles, chemical note analysis |
| **5. Storage Room** | $X \in [-10, 10], Z \in [12, 24]$ | Emerald green vaulted lighting, reinforced walls | Secure vault lockers, wall safe, emergency supply crates | High-security targets, final mystery resolution |

---

## 4. Replayable Mystery Hunt Scenarios

### 4.1 Mystery A: The Missing Research Key
- **Narrative**: A master physical key required to override Sector 4 was misplaced before the lockdown.
- **Objectives**:
  1. Survey Reception and locate the security terminal.
  2. Inspect the terminal note hinting at a hiding spot near the lab bookshelf.
  3. Traverse to the Investigation Laboratory, inspect the cabinet, and collect the research key.
  4. Navigate to Storage Room and unlock Vault Locker 1.
- **Randomization Seed Variations**: Key location toggles between Laboratory Cabinet and Research Office Desk; vault locker ID varies.

### 4.2 Mystery B: The Missing Research File
- **Narrative**: Confidential Incident Report 104-A has disappeared from the archive index.
- **Objectives**:
  1. Ask the AI drone to search facility documents for Incident Report 104-A.
  2. Traverse to the Library / Archives and inspect Archive Bookshelf Beta.
  3. Query spatial relations to find which cabinet is opposite the bookshelf.
  4. Recover the classified dossier and deposit it at the Central Hub console.
- **Randomization Seed Variations**: Target document section and bookshelf identifier randomized.

### 4.3 Mystery C: Laboratory Containment Puzzle
- **Narrative**: Hazardous chemical reaction triggered lockdown; safe evacuation requires calculating spatial clearance and entering the dynamic override code.
- **Objectives**:
  1. Inspect the glowing Datapad clue on the Central Research Workbench.
  2. Query distance between flammable cabinet and electrical conduits to confirm safety clearance.
  3. Retrieve dynamic emergency override code from RAG facility incident logs.
  4. Command companion drone to compute obstacle-avoiding waypoints to the exit.
  5. Disengage the laser barrier and complete evacuation.
- **Randomization Seed Variations**: Dynamic 4-digit code generated per session seed; conduit distance parameters vary.

---

## 5. Standardized Tool Whitelist (13 Tools)

| Tool Name | Parameters | Return Schema | Safety Rule |
| :--- | :--- | :--- | :--- |
| `get_visible_objects` | None | `List[ObjectSummary]` | Read-only scene graph filter |
| `locate_object` | `object_name: str` | `ObjectCoordinates` | Validates object exists in scene |
| `query_spatial_relation` | `object_name: str`, `relation: str` | `RelationMatches` | Restricts to `left_of, right_of, near, above` |
| `calculate_distance` | `object_a: str`, `object_b: str` | `DistanceResult` | Mathematical Euclidean calculation |
| `find_nearest_target` | `target_class: str` | `NearestTarget` | Class in SUN RGB-D categories |
| `find_nearest_exit` | None | `NearestExit` | Returns closest door/exit coordinates |
| `find_path` | `destination: str` | `PathWaypoints` | Generates safe collision waypoints |
| `navigate_to` | `destination: str` | `NavigationDispatch` | Validates reachable coordinates |
| `inspect_clue` | `clue_id: str` | `ClueData` | Queries local clue state machine |
| `search_documents` | `query: str` | `RetrievedPassages` | TF-IDF RAG over facility JSON |
| `search_safety_manual` | `query: str` | `RetrievedPassages` | TF-IDF RAG over safety procedures |
| `get_inventory` | None | `InventoryList` | Returns player's collected items |
| `get_mission_objectives`| None | `ObjectiveStatus` | Returns current quest objectives |
| `request_hint` | None | `HintText` | Dynamic hint based on incomplete goals |
| `interact_with_object` | `object_id: str` | `InteractionResult` | Requires proximity and valid credentials |

---

## 6. Live System Inspector UI Design

A sliding side-panel drawer toggled by a glowing `SYSTEM INSPECTOR` button in the game header:

```
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM INSPECTOR                              [✕ Close]     │
│  Correlation ID: #corr-849204 · Step Latency: 142ms         │
├──────────────────┬──────────────────┬────────────────────────┤
│ [Computer Vision]│ [3D Graphics]    │ [LLM & Agents]         │
├──────────────────┴──────────────────┴────────────────────────┤
│ TAB 1: COMPUTER VISION                                       │
│ • Model: SpatialDetector (2,162,731 trainable parameters)     │
│ • Input Frame: RGB-D (480x640x3)                             │
│ • Feature Activations: Stage 1 (32x240x320) -> Stage 5 (256) │
│ • Output Prediction Grid: (15x20x11)                         │
│ • Detections: Table (0.94), Laptop (0.89), Door (0.91)       │
│ • Pinhole Intrinsics: fx=525, fy=525, cx=320, cy=240         │
│ • Computed Metric Coordinates: Table=[0.0, 0.75, 2.5]m       │
│ • Real Evaluation: mAP@0.5=0.0000 on SUN RGB-D (honest)      │
└──────────────────────────────────────────────────────────────┘
```

- **CV Tab**: Shows live detected bounding boxes, confidence bars, tensor shapes at each conv block, depth calculation, and loss plots.
- **Graphics Tab**: Shows player $(X, Y, Z)$ position, active room, camera mode, FOV (65°), render dimensions, estimated vs ground-truth coordinate markers.
- **LLM Tab**: Shows original prompt, token IDs array, generated JSON tokens, parsed tool name and args, execution result, RAG similarity score, and total response latency.
