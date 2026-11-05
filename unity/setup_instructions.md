# Unity 6 Setup Instructions for SpatialAI

This guide provides step-by-step instructions to configure and run the 3D Virtual Laboratory in Unity.

---

## 1. Prerequisites
- **Unity Hub** installed (detected at `/Applications/Unity Hub.app`).
- **Unity Editor 6000.6.0f1 (arm64)** installed (detected at `/Applications/Unity/Hub/Editor/6000.6.0f1-arm64/Unity.app`).
- **Python Backend** running locally (`http://localhost:8000`).

---

## 2. Project Setup in Unity Editor

### Option A: Create a New Project or Open SpatialAIUnity
1. Open **Unity Hub**.
2. Click **Add** $\to$ **Add project from disk**.
3. Select `unity/SpatialAIUnity` (or create a new 3D Core project named `SpatialAIUnity` under `unity/`).
4. Ensure the selected Editor Version is `6000.6.0f1`.
5. Open the project.

### Option B: Using Existing CG Lab Project
If preferred, the scripts in `unity/SpatialAIUnity/Assets/Scripts/` can simply be copied or symlinked into your existing Unity project (`lab9-unity-interactive-3d-scene/Assets/Scripts/SpatialAI`).

---

## 3. Assembling the Virtual Laboratory Scene
1. In the Project window, create a new scene: **File $\to$ New Scene $\to$ Basic (Built-in or URP)**. Save as `Assets/Scenes/LabEnvironment.unity`.
2. Create an empty GameObject in the Hierarchy:
   - Rename it to `LabManager`.
   - Add Component $\to$ **LabSceneGenerator** (`SpatialAI.Scene.LabSceneGenerator`).
3. In the Inspector for `LabManager`:
   - Click the three dots (context menu) $\to$ **Regenerate Laboratory** (or simply press Play).
   - This procedurally generates the laboratory room, floor, walls, lighting, and all 6 target object classes:
     1. Table
     2. Chair
     3. Laptop
     4. Monitor
     5. Door / Exit
     6. Fire Extinguisher

---

## 4. Capturing Synthetic Datasets from Unity
1. Select the **Main Camera** in the Hierarchy.
2. Add Component $\to$ **SyntheticDataCapture** (`SpatialAI.Perception.SyntheticDataCapture`).
3. Set the capture resolution (default: $640 \times 480$) and number of frames (e.g. 100).
4. Right-click the component header $\to$ **Execute Capture Sequence**.
5. The dataset will be saved to `DatasetExport/` containing:
   - `rgb/frame_XXXXX.png`
   - `depth/frame_XXXXX.depth`
   - `annotations/frame_XXXXX.json`

---

## 5. Live Runtime Interaction Loop
1. With the Python backend running (`python3 -m uvicorn backend.app.main:app --port 8000`):
2. Attach **RuntimePerceptionSender** to the Main Camera.
3. Attach **UnityActionReceiver** and **ChatUIController** to an Avatar GameObject or the Main Camera.
4. Enter **Play Mode** in Unity.
5. Use the GUI Chat Box to submit queries such as:
   - *"Where is the nearest exit?"*
   - *"How far is the fire extinguisher from the door?"*
   - *"What should I do during a fire?"*
6. Observe live tool execution and avatar navigation response.
