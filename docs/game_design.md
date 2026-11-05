# Spatial Quest — Game Design Document (GDD)

## 1. Executive Summary & Gameplay Concept
**Spatial Quest: An AI Investigation Game Powered by Computer Vision and a Custom-Trained Language Model** is a 3D investigative mystery game where the player explores a locked-down advanced research facility.

The core gameplay loop is:
$$\text{EXPLORE} \longrightarrow \text{ASK} \longrightarrow \text{INVESTIGATE} \longrightarrow \text{SOLVE} \longrightarrow \text{PROGRESS} \longrightarrow \text{REPLAY}$$

Accompanying the player is an **AI Companion** (conversational software assistant with optional avatar/drone visualization) equipped with:
1. A **From-Scratch Computer Vision Detector** trained on the official real-world **SUN RGB-D** benchmark to detect indoor furniture, fixtures, doors, and obstacles (2.16M parameters, He normal initialization).
2. A **From-Scratch Decoder-Only Transformer Language Model** implementing causal multi-head self-attention, trained from raw tensor math on facility corpora, spatial reasoning, and structured `<tool_call>` emission across 13 tools.
3. A **Calibrated 3D Depth-Backprojection Engine** mapping 2D bounding boxes to metric world coordinates ($X = (u - c_x)Z / f_x$, $Y = (v - c_y)Z / f_y$, $Z = \text{depth}$) with sensor pose transformations.
4. A **Local RAG Retrieval Engine** using TF-IDF and cosine similarity to search facility incident logs, maintenance manuals, and emergency access codes.
5. A **Live System Inspector** providing real-time telemetry across Computer Vision, Computer Graphics, and LLM/Agents tabs.

---

## 2. Five Distinct Facility Locations
The facility is architecturally partitioned into five distinct rooms, each with unique layouts, lighting themes, furniture, clues, and spatial boundaries:

```
+-------------------------------------------------------------------------+
|                  Location 5: Storage Room & Vault                       |
|         Z in [10, 24] | High Archive Racks, Wall Safe, Exit Door        |
|               Emergency Red Warning Beacon & Vault Keypad               |
+-------------------------------------------------------------------------+
                                    | (Laser Security Gate / Door 3)
+-------------------------------------------------------------------------+
|               Location 4: Investigation Laboratory                      |
|         Z in [0, 10] | Research Workbench, Stools, Datapad Clue         |
|               Cool Cyan Fluorescent Cleanroom Illumination              |
+-------------------------------------------------------------------------+
                                    | (Corridor Archway)
+-----------------------------------+-------------------------------------+
|   Location 3: Library / Archives  |    Location 2: Research Office      |
|    Z in [-10, 0], X in [-16, 0]   |     Z in [-10, 0], X in [0, 16]     |
|   Oak Bookshelves, Reading Carrel |   Executive Desk, Swivel Chair,     |
|       Warm Amber Lamp Glow        |   Filing Cabinet, Research Dossier  |
+-----------------------------------+-------------------------------------+
                                    | (Reception Doors)
+-------------------------------------------------------------------------+
|                  Location 1: Reception / Central Hub                    |
|         Z in [-24, -10] | Reception Desk, Visitor Sofa, Console         |
|                     Neutral White Daylight Illumination                 |
+-------------------------------------------------------------------------+
```

---

## 3. Replayable Mystery Archetypes
Spatial Quest features 3 procedural mystery archetypes with seed-based randomization (`POST /game/reset` with integer seed):

| Mystery Archetype | Narrative Scenario | Starting Room | Target Clue / Solution | Unlocking Condition |
| :--- | :--- | :--- | :--- | :--- |
| **Mystery A: Missing Research Key** | The chief scientist's physical keycard was misplaced during lockdown. Investigate the Reception and Office to locate the key. | Reception / Central Hub | `research_key` on office desk | Recover key to unlock the Laboratory security gate. |
| **Mystery B: Missing Research File** | A confidential laboratory incident dossier went missing from the central archives. Search bookshelves and filing cabinets. | Library / Archives | `research_file` in archive filing cabinet | Read document via RAG to uncover the vault emergency code. |
| **Mystery C: Laboratory Mystery** | An urgent handwritten note on a datapad references a hazardous chemical reaction. Decipher the note and reach the Storage Vault. | Investigation Laboratory | `datapad_note` on workbench $\to$ Safe code `7391` | Unlock the vault wall safe and trigger the emergency exit. |

Every generated mission instance validates path reachability and clue availability, ensuring 100% solvability without universal hardcoded override cheats.

---

## 4. Validated Tool Registry (The 13 Standardized Tools)

| Tool Signature | Purpose | Module |
|---|---|---|
| `get_visible_objects()` | Returns all detected objects in camera FOV with estimated 3D positions | `backend.tools.registry` |
| `locate_object(object_name)` | Locates specific object class and returns metric coordinates | `backend.spatial.scene_graph` |
| `query_spatial_relation(object_name, relation)` | Intersite spatial reasoning (`left_of`, `right_of`, `near`, `above`) | `backend.spatial.scene_graph` |
| `calculate_distance(object_a, object_b)` | Euclidean distance in meters between two detected objects | `backend.spatial.scene_graph` |
| `find_nearest_target(target_class)` | Identifies closest object of specified class to the observer | `backend.spatial.scene_graph` |
| `find_path(destination)` | Calculates obstacle-avoiding 3D waypoints | `backend.tools.registry` |
| `navigate_to(destination)` | Dispatches physical waypoint navigation to avatar companion | `unity.SpatialAIUnity` / WebGL |
| `inspect_clue(clue_id)` | Reveals state and clues of interactive objects (`safe`, `pda`) | `backend.tools.registry` |
| `search_documents(query)` | RAG search over incident reports and emergency override codes | `backend.rag.retriever` |
| `interact_with_object(object_id)` | Triggers environmental state changes (unlock safe, open door) | `backend.tools.registry` |
| `get_inventory()` | Inspects investigator collected clues and inventory items | `backend.game.facility` |
| `get_mission_objectives()` | Returns current active mission objectives and completion status | `backend.game.missions` |
| `request_hint()` | Generates contextual guidance without spoiling final solution | `backend.agent.investigation_agent` |

---

## 5. Scoring, Progression & HUD
- **Investigation Score**: Points awarded for completing analytical queries, finding clues, and unlocking doors. Tracked by `backend/game/missions.py` and shown in the browser HUD.
- **Mission Reset & Seed Randomization**: `POST /game/reset` resets inventory, door locks, and re-seeds target clue placements deterministically.
- **Companion Roles**: Investigation Agent (generalist), Spatial Agent (spatial/path), Knowledge Agent (RAG/clues). All powered by our single from-scratch Transformer checkpoint without duplicated memory overhead.
- **Interactive Chat HUD**: Communicates with the FastAPI backend via `POST /agent/query` and renders assistant thoughts, tool executions, and avatar waypoints.
- **Live System Inspector**: Accessible at any time via the `🔍 SYSTEM INSPECTOR` button, providing simultaneous insight into Computer Vision, Computer Graphics, and LLM/Agent internal state.
