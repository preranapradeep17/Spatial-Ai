# Spatial Quest — Implementation Plan (SDLC Stage 4 Roadmap)
**File:** `docs/implementation_plan.md`  
**Document Version**: 2.0.0  
**Status**: 100% Completed & Verified  

---

## 1. Incremental Implementation Strategy
To uphold the strict SDLC rule ("Do not jump straight into modifying code without understanding requirements... implement the software in small, verifiable increments"), development proceeded in the following six sequential increments:

### Increment 1: Core Bug Fix & Tool Whitelist Expansion (COMPLETED)
- **Objective**: Fix the string assertion mismatch in `tests/test_api.py::test_agent_query_nearest_exit`.
- **Modifications**:
  - `backend/agent/investigation_agent.py`: Enhanced `find_nearest_exit` response synthesis so it explicitly confirms navigation to the nearest exit.
  - `backend/tools/registry.py`: Added `get_inventory`, `get_mission_objectives`, and `request_hint` tools (13 tools total).
- **Verification**: Ran `pytest tests/test_api.py tests/test_agent.py` $\to$ 100% pass rate.

### Increment 2: Five-Location Facility Layout & Scene Graph (COMPLETED)
- **Objective**: Expand facility layout from 3 zones to 5 distinct physical locations (Reception, Research Office, Library/Archives, Investigation Lab, Storage Room).
- **Modifications**:
  - `backend/game/facility.py`: Defined canonical 3D object positions and room bounds for all 5 locations.
  - `backend/spatial/scene_graph.py`: Ensured spatial relation calculations and nearest queries operate across all 5 room layouts.
- **Verification**: Ran `pytest tests/test_geometry.py tests/test_scene_graph.py` $\to$ 100% pass rate.

### Increment 3: Replayable Mystery System & Inventory State (COMPLETED)
- **Objective**: Implement Mysteries A, B, and C with seed-based controlled randomization, inventory tracking, and dynamic hints.
- **Modifications**:
  - `backend/game/missions.py`: Added `MysteryManager` with reproducible seed generator, inventory storage (`keys`, `files`, `datapads`), clue state transitions, and dynamic hint generator.
  - `backend/api/routes.py`: Added `POST /game/reset`, `/game/inventory`, `/game/hint`, and seed parameters.
- **Verification**: Ran `pytest tests/test_missions.py` $\to$ 100% pass rate.

### Increment 4: Telemetry Correlation Service (COMPLETED)
- **Objective**: Implement unified telemetry capturing CV intermediate tensor shapes, detection bounding boxes, Graphics scene transforms, and LLM token IDs under a shared correlation ID.
- **Modifications**:
  - `backend/api/telemetry.py`: Circular buffer recording turns with correlation IDs (`source_id` resolution).
  - `backend/api/routes.py`: Exposed `GET /telemetry/latest` and `GET /telemetry/{correlation_id}`.
- **Verification**: Added `tests/test_api.py::test_telemetry_latest_endpoint` and `test_telemetry_by_id_endpoint` $\to$ Verified valid telemetry JSON structure.

### Increment 5: Frontend 3D Game & Live System Inspector UI (COMPLETED)
- **Objective**: Upgrade browser 3D client at `frontend/index.html` with all 5 rooms, rich aesthetics, distinct lighting/furniture, and the collapsible 3-tab System Inspector.
- **Modifications**:
  - `frontend/index.html`: Built 5 distinct rooms in Three.js, room signboards, partition walls, sliding 3-tab System Inspector drawer, live telemetry polling, inventory HUD, seed reset modal, and tutorial modal.
- **Verification**: Verified UI responsiveness, inspector tab switching, and 3D navigation in browser at `http://localhost:8000`.

### Increment 6: Automated Test Suite Expansion & SDLC Documentation (COMPLETED)
- **Objective**: Created `docs/sdlc/01_planning.md` through `08_maintenance.md`, updated all 18 project documents, created `docs/cleanup_report.md`, and validated full test suite.
- **Verification**: 50/50 automated tests passing; 100% verification across all requirements.
