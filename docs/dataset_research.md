# Spatial Quest: Public Dataset Research & Feasibility Analysis

> **Document Purpose**: Systematic comparative evaluation of candidate real-world datasets (**SUN RGB-D**, **Open Images V7**, and **ScanNet**) to replace synthetic Unity imagery as the primary Computer Vision benchmark, in accordance with the *Spatial Quest* project evolution.

---

## 1. Local Hardware & Storage Constraints
Before evaluating dataset scale, the local host environment was inspected:
- **Filesystem Mount**: `/System/Volumes/Data` (macOS 14.7.1 arm64)
- **Total Filesystem Capacity**: 228 GiB
- **Currently Used**: 183 GiB (94% utilization)
- **Available Free Disk Space**: **~12.0 GiB**

> [!WARNING]
> **Storage Risk**: With only ~12 GiB of remaining free disk space, downloading massive public datasets in full (e.g., ScanNet ~1.2 TB, Open Images V7 >500 GB, or uncompressed SUN RGB-D ~20 GB) would immediately exhaust disk storage, causing system instability.
> 
> Therefore, any selected dataset must be acquired via an official, curated, reproducible mini-split (e.g., ~1.0–2.5 GB) preserving official train/test IDs, or filtered for indoor classes before download.

---

## 2. Comparative Analysis of Candidate Datasets

| Evaluation Criterion | SUN RGB-D | Open Images V7 | ScanNet (v2) |
| :--- | :--- | :--- | :--- |
| **Primary URL** | [rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu/) | [storage.googleapis.com/.../v7.html](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | [github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet) |
| **Official Citation** | Song, Lichtenberg, Xiao, *"SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite"*, CVPR 2015. | Kuznetsova et al., *"The Open Images Dataset V4 & V7"*, IJCV 2020. | Dai et al., *"ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes"*, CVPR 2017. |
| **Data Modalities** | Synchronized RGB + Registered Depth maps + Camera Intrinsics ($K$) | High-resolution RGB photos (Flickr) | RGB-D video streams + Camera Trajectory poses + 3D Meshes |
| **RGB Availability** | **Yes** ($10,335$ indoor frames) | **Yes** (~9 million images) | **Yes** (~2.5 million frames across 1,513 scans) |
| **True Depth Availability** | **Yes** (Captured via Intel RealSense, Asus Xtion, Kinect v1 & v2) | **NO** (No sensor depth or camera intrinsics) | **Yes** (Captured via iPad Structure Sensor) |
| **2D Bounding Boxes** | **Yes** ($10,335$ images densely annotated) | **Yes** (16 million boxes, 600 classes) | **Yes** (Projected from 3D annotations) |
| **3D Bounding Boxes** | **Yes** (64,595 3D oriented bounding boxes with yaw) | **NO** (No 3D spatial boxes) | **Yes** (Axis-aligned & oriented 3D bounding boxes) |
| **Spatial / Semantic Relations**| Implicit from 3D layout & spatial metrics | Explicit 2D relationships (3.3M pairs: "on", "holds") | Implicit from 3D spatial mesh topology |
| **Official Splits** | 5,285 Training / 5,050 Testing images | Train (7.3M) / Validation (41k) / Test (125k) | 1,201 Train / 312 Validation / 100 Hidden Test scenes |
| **Full Download Size** | ~10.5 GB compressed (~15–20 GB uncompressed) | >500 GB to several Terabytes | ~1.2 TB raw scans (~50 GB subsampled frames) |
| **Access & Licensing** | **Free Academic Use** (Princeton Vision Lab; direct HTTP zip download) | **Creative Commons** (CC BY 2.0 / CC BY 4.0; public GCP buckets) | **Gated Academic Agreement** (Requires signed application email to Stanford/TUM) |
| **Compatibility with Existing 3D Back-Projection** | **Native**: Direct optical frame ($f_x, f_y, c_x, c_y$) and metric depth arrays | **Incompatible for 3D**: Lacks sensor depth and focal lengths | **Compatible**: Requires parsing sensor stream `.sens` bundles |
| **From-Scratch CNN Training Feasibility** | **High**: Manageable image count, standardized resolutions, aligned depth | **Low on Laptop**: Massive data volume; requires distributed clusters | **Medium**: Frame extraction requires custom C++/Python tool |

---

## 3. Deep-Dive on Evaluated Datasets

### A. SUN RGB-D (Primary Recommendation)
- **Overview**: Constructed by Princeton Vision & Robotics Lab to unify NYU Depth v2, Berkeley B3DO, and SUN3D with newly acquired Intel RealSense and Kinect v2 data.
- **Relevant Object Categories**:
  - The dataset features standard indoor benchmarks:
    - **19-Class Semantic Benchmark**: `chair`, `table`, `desk`, `sofa`, `bed`, `bookshelf`, `door`, `window`, `cabinet`, `counter`, `blinds`, `curtain`, `dresser`, `pillow`, `mirror`, `floor`, `wall`.
    - **Candidate Core Classes for Spatial Quest Facility**:
      1. `chair` (~18,000 instances)
      2. `table` / `desk` (~9,000 instances)
      3. `door` (~2,500 instances)
      4. `bookshelf` / `bookcase` (~2,100 instances)
      5. `sofa` (~3,000 instances)
      6. `cabinet` (~3,500 instances)
- **3D Ground Truth**:
  - Each annotated object includes 3D bounding box center $(X, Y, Z)$ in camera coordinate frame (metric units), 3D extent (width, height, depth), and 3D orientation rotation matrix/yaw angle.
  - Matches our calibrated pinhole back-projection pipeline:
    $$X_{cam} = \frac{(u - c_x) Z}{f_x}, \quad Y_{cam} = \frac{(v - c_y) Z}{f_y}, \quad Z_{cam} = Z_{depth}$$
- **Download Structure**:
  - Official toolbox: `https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip` (543 MB). Contains 2D/3D annotations, sensor calibration matrices, camera extrinsics/intrinsics, and official train/test split mappings.
  - Imagery: Full image zip is ~10.5 GB. Given our ~12 GB disk space, we can download the metadata toolbox first, extract official split indices, and download a curated, representative benchmark subset (e.g. 500–1,000 images, ~1.5 GB) containing our candidate classes, leaving >10 GB of disk safety margin.

### B. Open Images V7 (Evaluated Supplementary Source)
- **Strengths**: Contains rich 2D relationship triplets (e.g., `chair` on `floor`, `laptop` on `table`).
- **Deficiencies for SpatialAI/Spatial Quest**:
  - Completely lacks registered depth sensor data.
  - Lacks camera focal lengths and sensor calibration parameters.
  - Calculating 3D camera-space coordinates $(X, Y, Z)$ is impossible without relying on pretrained monocular depth neural networks (e.g., MiDaS/DPT), which violates the project's strict requirement of zero pretrained black-box models in the core perception path.
- **Verdict**: Retain Open Images taxonomy as an optional linguistic reference for relationship schemas, but reject as the primary 3D perception dataset.

### C. ScanNet v2 (Evaluated Future Extension)
- **Strengths**: Rich 3D scans of indoor environments with high geometric fidelity and multi-view consistency.
- **Deficiencies for Current Phase**:
  - **Gated Approval**: Access requires signing a Terms of Service agreement and submitting an institutional email request to Princeton/Stanford/TUM administrators.
  - **Massive Storage Footprint**: Uncompressed raw `.sens` files span 1.2 Terabytes; even a minimal single scene spans 1.5–3 GB. Storing multiple train/val scenes would immediately exceed our machine's 12 GiB free disk space.
- **Verdict**: Defer to later research extensions when dedicated external storage is attached.

---

## 4. Formal Recommendation: SUN RGB-D Curated Benchmark Subset

We formally recommend **SUN RGB-D** as the primary real-world benchmark dataset for *Spatial Quest*:
1. **Calibrated Multimodal Input**: Supplies authentic RGB images and physical sensor depth maps with known intrinsic matrices ($f_x, f_y, c_x, c_y$).
2. **True 3D Ground Truth**: Provides 3D bounding box centroids, enabling rigorous evaluation of our depth back-projection and 3D Euclidean error against real physical objects.
3. **Indoor Facility Alignment**: Rich coverage of `chair`, `table`, `desk`, `door`, `bookshelf`, and `sofa` fits the research facility mystery theme.
4. **Feasibility on 12 GiB Laptop**: An official curated subset of 500–1,000 real RGB-D frames (occupying ~1.5–2.0 GB) guarantees full experimental validity without risking host disk saturation.

---

## 5. Proposed Class Mapping for Spatial Quest

To bridge the real-world dataset, the custom from-scratch detector, and the Unity investigation game:

| Class ID | Target Category | SUN RGB-D Label Equivalents | Game Investigation Role |
| :---: | :--- | :--- | :--- |
| `0` | `table` | `table`, `desk`, `coffee_table` | Clue examination surface, holding documents/laptops |
| `1` | `chair` | `chair`, `office_chair`, `armchair` | Obstacle and spatial reference point |
| `2` | `door` | `door` | Mission navigation target, exit waypoint |
| `3` | `bookshelf` | `bookshelf`, `bookcase`, `shelves` | Storage location for facility research logs |
| `4` | `sofa` | `sofa`, `couch` | Waiting area furniture, spatial relationship target |
| `5` | `cabinet` | `cabinet`, `night_stand`, `cupboard` | Storage unit for hidden evidence |

*Note on Unsupported Classes*: If a game narrative item (e.g. an emergency keycard or specialized specimen tube) is required by the plot, it will be handled as an interactive game-logic entity. It will not be falsely claimed as a trained CV detection category.

---

## 6. Official Citations & Terms of Use
- **Citation**:
  ```bibtex
  @inproceedings{song2015sunrgbd,
    title={SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite},
    author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={567--576},
    year={2015}
  }
  ```
- **License / Terms**: Non-commercial academic research use. Data originates from Princeton University, University of Washington, and NYU.
