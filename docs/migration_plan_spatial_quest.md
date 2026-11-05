# Spatial Quest: Architectural Migration & Phased Implementation Plan

## 1. Executive Summary
This document specifies the migration plan to evolve **SpatialAI** into **Spatial Quest: An AI Investigation Game Powered by Computer Vision and a Custom-Trained Language Model**.

The evolution strengthens academic research rigor by:
1. Replacing synthetic Unity data with **SUN RGB-D** as the primary Computer Vision training and evaluation benchmark.
2. Formulating a formal **Domain Transfer Experiment** (Real-World Benchmark vs. Unity Game-Camera Evaluation).
3. Wrapping the real-time perception, 3D scene graph, and custom Transformer agent in a mission-driven 3D research facility investigation game.
4. Preserving and reusing all completed, tested modules from Phase 0 and Phase 1.

---

## 2. Existing Repository State & Preservation Map

| Existing Component | Current Status | Spatial Quest Role & Action |
| :--- | :--- | :--- |
| `backend/perception/device.py` | Complete (MPS/CUDA/CPU) | **Preserve as-is**: Used across all training and inference scripts. |
| `backend/spatial/geometry.py` | Complete (Pinhole math, depth sampling) | **Preserve & Expand**: Will now be benchmarked against SUN RGB-D 3D ground truth. |
| `backend/spatial/scene_graph.py` | Complete (Relations & distances) | **Preserve & Expand**: Supports new investigation categories and game entity relationships. |
| `backend/tools/registry.py` | Complete (8 core tools) | **Expand**: Update tool signatures to the 10 standardized game investigation tools. |
| `backend/api/routes.py` & `schemas.py` | Complete (5 endpoints) | **Preserve & Expand**: Add game mission telemetry and oracle/debug toggle. |
| `scripts/generate_synthetic_dataset.py` | Complete (Headless generator) | **Preserve**: Repurposed for **Evaluation B (Domain Transfer Benchmark)** and offline CI tests. |
| `scripts/validate_dataset.py` | Complete (Shape & bound checks) | **Preserve & Generalize**: Reused to validate both SUN RGB-D converted annotations and synthetic frames. |
| `unity/Assets/Scripts/` | Complete (Scene generator, capture, chat) | **Adapt**: Adapt into multi-room facility, mission objectives, and investigation terminal. |
| `tests/` | 18 passing tests | **Preserve & Extend**: Ensure 100% test pass rate throughout all migrations. |

---

## 3. Real-World Dataset Pipeline (SUN RGB-D)

### Selected Real-World Category Set (6 Core Classes)
Mapped directly from SUN RGB-D indoor annotations:
1. `table` (Class ID 0)
2. `chair` (Class ID 1)
3. `door` (Class ID 2)
4. `bookshelf` (Class ID 3)
5. `sofa` (Class ID 4)
6. `cabinet` (Class ID 5)

### Safe Storage Management (Under 12 GiB Free Disk Limit)
1. **Toolbox Download**: Download `SUNRGBDtoolbox.zip` (~543 MB).
2. **Metadata Extraction**: Parse official sensor calibration matrices, 2D bboxes, 3D bboxes, and official train/test split IDs into clean JSON formats.
3. **Curated Benchmark Subset**: Download a curated, representative mini-subset (~500–1,000 RGB-D images, ~1.5–2.0 GB) focused on indoor laboratory, office, and classroom scenes.
4. **Safety Margin**: Retains >9 GB of free disk space on the local drive for checkpoints, logs, and OS cache.

---

## 4. Standardized Tool Signatures for Spatial Quest Agent
All tools will use uniform signatures across schemas, models, and Unity:
1. `get_visible_objects() -> List[Dict]`
2. `locate_object(object_name: str) -> Dict`
3. `query_spatial_relation(object_name: str, relation: str) -> Dict`
4. `calculate_distance(object_a: str, object_b: str) -> Dict`
5. `find_nearest_target(target_class: str) -> Dict`
6. `find_path(destination: str) -> Dict`
7. `navigate_to(destination: str) -> Dict`
8. `inspect_clue(clue_id: str) -> Dict`
9. `search_documents(query: str) -> Dict`
10. `interact_with_object(object_id: str) -> Dict`

---

## 5. Domain Transfer Experiment Design
- **Evaluation A: Real-World Benchmark**:
  Custom detector trained on SUN RGB-D training split $\to$ evaluated on SUN RGB-D held-out test split.
  Reports mAP@0.5, per-class PR curves, IoU, and 3D Euclidean localization error against physical sensor ground truth.
- **Evaluation B: Unity Game-Camera Evaluation**:
  Same trained custom detector evaluated on Unity facility screenshots.
  Measures domain gap, texture/lighting sensitivity, and failure modes.
  No synthetic images will be silently injected into the real-world benchmark.

---

## 6. Phased Implementation Roadmap
- **Phase 1**: Workspace inspection, storage verification, and migration plan (Complete).
- **Phase 2**: Dataset research, comparison table, and recommendation in `docs/dataset_research.md` (Complete).
- **Phase 3**: Implement SUN RGB-D dataset loader, annotation converter, and small real-data smoke test.
- **Phase 4**: Adapt custom CNN detector and loss heads to SUN RGB-D format; train real checkpoint.
- **Phase 5**: Verify calibrated 3D back-projection against SUN RGB-D 3D bounding box ground truth.
- **Phase 6**: Adapt Unity environment into the 3-room research facility with mission manager and terminal.
- **Phase 7**: Implement custom Transformer and agent loop with the 10 standardized investigation tools.
- **Phase 8**: Execute dual evaluation (Real-world benchmark vs. Unity game-camera transfer).
- **Phase 9**: Finalize documentation, syllabus matrix, viva guide, and resume project entry.
