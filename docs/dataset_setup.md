# SUN RGB-D Real-World Dataset Setup & Validation Guide

## 1. Overview & Provenance
In accordance with the **Spatial Quest** specification, the primary Computer Vision training and evaluation benchmark uses authentic physical sensor data from the official **SUN RGB-D** benchmark suite:
- **Reference**: Song, Lichtenberg, Xiao, *"SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite"*, IEEE CVPR 2015.
- **Official URL**: [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)
- **License**: Free for non-commercial academic research use.

---

## 2. Real-World Modalities & Metric Depth
Each sample in `data/sunrgbd/` contains:
1. **RGB Image**: Authentic physical indoor photograph from real sensors (Intel RealSense, Asus Xtion, Microsoft Kinect v1 & v2).
2. **Depth Array (`.npy`)**: 32-bit floating-point array decoded using Princeton's official bitshift formula:
   $$\text{depth} = (\text{raw\_png} \gg 3) \mid (\text{raw\_png} \ll 13)$$
   $$\text{depth\_meters} = \frac{\text{depth}}{1000.0} \quad (\text{clamped at } 8.0\text{m})$$
3. **Camera Intrinsics**: Calibrated $3 \times 3$ intrinsic matrix ($f_x, f_y, c_x, c_y$).
4. **2D Modal Bounding Boxes**: Updated PASCAL-style modal boxes around visible object contours.
5. **3D Oriented Centroids**: Real-world $(X, Y, Z)$ centroid coordinates in metric camera space.

---

## 3. Class Mapping & Split Segregation

### A. 6 Target Indoor Investigation Classes
| Class ID | Target Class | SUN RGB-D Source Classes |
| :---: | :--- | :--- |
| `0` | `table` | `table`, `desk`, `coffee_table`, `dining_table`, `endtable` |
| `1` | `chair` | `chair`, `office_chair`, `armchair` |
| `2` | `door` | `door` |
| `3` | `bookshelf` | `bookshelf`, `bookcase`, `shelf`, `shelves` |
| `4` | `sofa` | `sofa`, `couch` |
| `5` | `cabinet` | `cabinet`, `night_stand`, `cupboard`, `dresser` |

### B. Official Split Preservation
- Splits are strictly partitioned according to Princeton's official `allsplit.mat` (5,285 training frames vs 5,050 testing frames).
- Zero data leakage between train and test splits is enforced and verified via automated test [`tests/test_sunrgbd_loader.py::test_sunrgbd_dataset_splits`](file:///Users/meghak/Documents/sem7/Spatial%20AI%20/tests/test_sunrgbd_loader.py).

---

## 4. Verification & Inspection
1. **Manifest File**: [`data/sunrgbd/manifest.json`](file:///Users/meghak/Documents/sem7/Spatial%20AI%20/data/sunrgbd/manifest.json) tracks original frame IDs, splits, and object counts.
2. **Visual Inspection**: Inspection composites rendered side-by-side [RGB with Bounding Boxes | Depth Colormap] are available in [`results/predictions/sunrgbd_debug/`](file:///Users/meghak/Documents/sem7/Spatial%20AI%20/results/predictions/sunrgbd_debug/).
3. **Automated Verification**:
   ```bash
   python3 -m pytest tests/test_sunrgbd_loader.py -v
   ```
