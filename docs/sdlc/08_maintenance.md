# SDLC Stage 8: Maintenance, Operations and Release Readiness

**Project Name**: Spatial Quest (SpatialAI)  
**Document Version**: 1.0.0  
**Status**: Release Candidate Ready (RC-1)  

---

## 1. Known Limitations

Spatial Quest is built with full academic transparency. The known limitations and engineering boundaries are:

1. **Compact Vocabulary Footprint**: The custom BPE tokenizer features $V = 384$ tokens. While optimized for indoor investigation, spatial reasoning, and tool execution, out-of-domain conversational queries will fall back to character/subword pieces or the deterministic semantic intent router.
2. **Limited Dataset Scale**: The curated SUN RGB-D benchmark subset contains 6 canonical indoor classes (`table`, `chair`, `door`, `bookshelf`, `sofa`, `cabinet`). Expanding to unusual categories requires adding annotations to `data/sunrgbd/annotations/`.
3. **Single-Stage Grid Resolution**: The detector utilizes a $15 \times 20$ grid cell prediction head on $480 \times 640$ frames. Extremely small or overlapping objects in the same grid cell may be suppressed by Non-Maximum Suppression (NMS).
4. **Single-Player Simulation**: The current architecture is designed as a standalone local investigation experience without persistent multi-user networking.

---

## 2. Troubleshooting Guide

| Issue / Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| **Backend Offline Badge in UI** | FastAPI server not started on port 8000 | Run `python run.py` and verify `http://localhost:8000/health` returns status "ok". |
| **PyTorch MPS Fallback Warning** | Running on non-Apple Silicon or older macOS | System automatically falls back to CPU without crashing. Check device via `GET /health`. |
| **Door 3 Does Not Unlock** | Player attempted unlock without retrieving override | Use quick prompt `[🔐 Door 3 override code (RAG)]` to extract code "7391" from logs first. |
| **Mission Locked Error (400)** | Attempting to start mission ahead of sequence | Complete current mission objectives before starting subsequent missions. |
| **WebGL Canvas Not Rendering** | Browser hardware acceleration disabled | Enable WebGL in browser flags or test with Google Chrome. |

---

## 3. Dataset Updates and Model Retraining

### Adding New SUN RGB-D Samples
1. Place RGB images into `data/sunrgbd/rgb/`.
2. Place corresponding 16-bit depth arrays into `data/sunrgbd/depth/`.
3. Add 2D and 3D bounding box annotations to `data/sunrgbd/annotations/`.
4. Validate dataset format:
   ```bash
   python scripts/validate_dataset.py
   ```
5. Retrain the detector:
   ```bash
   python scripts/train_detector.py
   ```

### Updating Investigation Corpora & RAG Documents
1. Add new incident logs or manuals to `data/documents/facility_documents.json`.
2. Re-index is automatic on next query via `global_retriever.load_documents_from_json()`.

---

## 4. Extensibility Guide

### 4.1 Adding a New Facility Location
1. Open `backend/game/facility.py`.
2. Append a new room definition to `ROOM_DEFINITIONS` with `id`, `name`, `bounds`, `description`, and `spawn_z`.
3. Update `get_room_for_position()` to check the coordinate bounds.
4. Add corresponding 3D meshes in `frontend/index.html` inside `buildSector4Facility()`.

### 4.2 Adding a New Mystery Mission
1. Open `backend/game/missions.py`.
2. Append a new mission dictionary in `_mission_catalog(seed)`.
3. Define `objectives` using `_objective(id, description, points, required_tools)`.
4. The mission automatically appears in the game client's mission pill list.

### 4.3 Adding or Removing Logical Companion Agents
1. In `backend/game/missions.py`, edit the `COMPANION_AGENTS` dictionary.
2. Add a new agent persona:
   ```python
   "analyst": {
       "id": "analyst",
       "name": "Data Analyst Drone",
       "specialty": "Incident log correlation and RAG query analysis",
       "preferred_tools": ["search_documents", "inspect_clue"],
       "zone": "Sector 4 Data Center",
   }
   ```
3. No duplicate model checkpoints are required; all logical agents share the same from-scratch Transformer.

---

## 5. Regression Testing Protocol

Whenever modifications are made to perception, spatial geometry, language modeling, or game state:
```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest -v
```
All 51 tests must pass before approving commits.

---

## 6. Release-Readiness Report

| Category | Requirement Range | Verification | Status |
| :--- | :--- | :--- | :--- |
| **Computer Vision** | CV-001 – CV-007 | `SpatialDetector` (2.16M params), pinhole backprojection, 3D scene graph, real benchmark eval. | **READY** |
| **Language Model & Agents** | LLM-001 – LLM-007 | `SpatialTokenizer`, `DecoderTransformer` (856K params), ReAct loop, 13 tools, RAG. | **READY** |
| **Computer Graphics** | CG-001 – CG-007 | 5 distinct 3D rooms, dynamic lighting, player avatar, drone escort, Three.js runtime. | **READY** |
| **Gameplay & Mysteries** | GAME-001 – GAME-007 | Mysteries A, B, C, seed-based randomization, inventory, clue states, dynamic hints. | **READY** |
| **Live System Inspector** | INSP-001 – INSP-005 | 3-tab collapsible drawer, correlation IDs, CV tensor shapes, CG transforms, LLM token traces. | **READY** |
| **Safety & Guardrails** | SAFE-001 – SAFE-003 | Whitelist tool dispatch, Pydantic schemas, confirmed game execution. | **READY** |
| **Performance** | PERF-001 – PERF-004 | Fast response times, < 30MB model disk footprint, 60 FPS WebGL rendering. | **READY** |
| **Acceptance Criteria** | TEST-001 – TEST-006 | 50/50 automated tests passed, zero external AI APIs, 100% locally playable. | **READY** |

**Conclusion**: Spatial Quest meets all SDLC Stage 1 through Stage 8 quality gates and is release-ready for academic viva demonstrations and project reviews.
