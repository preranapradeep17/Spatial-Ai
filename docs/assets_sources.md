# Spatial Quest — 3D Assets, Textures & Attribution

All 3D assets, loader scripts, and procedural textures used in **Spatial Quest** are self-hosted locally in `frontend/assets/` so the application operates cleanly offline or on LAN without broken external CDN requests.

## 1. Human Investigator Character Model
- **Local File**: `frontend/assets/investigator.glb`
- **Source Asset**: `Soldier.glb` from the official Three.js examples archive (`mrdoob/three.js` tag `r128`, `examples/models/gltf/Soldier.glb`)
- **Rigging & Animations**: Standard Mixamo humanoid skeleton (`mixamorig:Hips` root, spine, neck, head, shoulders, arms, hands, legs, feet) with embedded skeletal keyframe clips:
  - `Idle`
  - `Walk`
  - `Run`
  - `TPose`
- **Customization Applied at Runtime (`frontend/assets/investigator3d.js`)**:
  - Material shading adjusted to a dark tactical investigative suit (slate charcoal jacket & trousers, muted metallic visor, leather boots, utility scanner light mounted on the right hand bone).
  - Full animation blending via `THREE.AnimationMixer` (`Idle`, `Walk`, `Run`, smooth turn interpolation, and brief inspect/reach gesture).
  - Articulated fallback humanoid rig provided automatically only when opened over `file://` where browser CORS prevents fetching `.glb` binaries.
- **License / Attribution**:
  - Three.js examples repository: MIT License (`Copyright © 2010-2021 three.js authors`).
  - Character asset bundled under the Three.js r128 examples distribution for educational & non-commercial simulation use.

## 2. Three.js Runtime & GLTF Loader
- **Local Files**:
  - `frontend/three.min.js` (Three.js r128, MIT License)
  - `frontend/assets/GLTFLoader.js` (official Three.js `examples/js/loaders/GLTFLoader.js` r128, MIT License)

## 3. Procedural PBR Facility Materials (`frontend/assets/facility3d.js`)
- Floors, architectural walls, ceiling grid tiles, glass partitions, oak/walnut laboratory tables, brushed aluminum cabinetry, and monitor screens use runtime HTML5 Canvas procedural albedo, roughness, and bump maps with zero external image dependencies.

## 4. Computer Vision Training & Evaluation Dataset
- Real-world RGB-D indoor benchmark samples come from the **SUN RGB-D** dataset (Song, Lichtenberg, & Xiao, CVPR 2015), stored locally when evaluated via `backend/perception/sun_rgbd_loader.py` and `/cv/benchmark_sample`.

