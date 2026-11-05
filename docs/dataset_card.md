# SpatialAI Virtual Laboratory Dataset Card

## 1. Dataset Summary
The SpatialAI Laboratory Dataset is a synthetic multimodal dataset containing synchronized RGB images, 32-bit floating-point linear depth arrays, and rich 2D/3D ground-truth annotations designed for spatial perception and object detection.

## 2. Supported Classes (6 Classes)
1. `table` (Class ID 0)
2. `chair` (Class ID 1)
3. `laptop` (Class ID 2)
4. `monitor` (Class ID 3)
5. `door` (Class ID 4, representing exits)
6. `fire_extinguisher` (Class ID 5)

## 3. Data Modalities & Specifications
- **RGB Images**: $640 \times 480$ pixels, 24-bit sRGB PNG format.
- **Depth Maps**: $640 \times 480$ array, stored as float32 NumPy arrays (`.npy`), representing radial or linear orthogonal distance from the camera in meters ($0.1\text{m} - 15.0\text{m}$).
- **Annotations**: JSON metadata file per frame containing:
  - `frame_id`: unique string identifier.
  - `camera`: focal lengths ($f_x, f_y$), principal point ($c_x, c_y$), image dimensions, camera world position, camera orientation quaternion.
  - `objects`: list of detected objects with class ID, class name, 2D bounding box `[xmin, ymin, xmax, ymax]`, 3D world coordinates $[X, Y, Z]$, 3D camera-relative coordinates, Euclidean distance, and visibility flag.

## 4. Split Structure
- **Train Split**: 70% of generated scene layouts.
- **Validation Split**: 15% of layouts.
- **Test Split**: 15% strictly held-out room layouts with distinct object arrangements and camera trajectories.

## 5. Usage & Verification
Validation script `scripts/validate_dataset.py` checks bounds, depth sanity, back-projection geometry, and class balance.
Visual overlays are rendered to `results/predictions/validation_debug/` for visual verification.
