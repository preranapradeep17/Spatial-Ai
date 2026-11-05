"""SUN RGB-D Subset Downloader & Annotation Converter for Spatial Quest.

Selects real indoor frames from the official SUN RGB-D benchmark containing our target classes
(table, chair, door, bookshelf, sofa, cabinet), fetches RGB images and registered depth maps
directly via HTTP range requests from Princeton's official archive without downloading the full 6.4 GB zip,
decodes the official 16-bit depth bitshift, and converts annotations into Spatial Quest format
preserving official train/test splits.
"""
import os
import json
import zlib
import urllib.request
import argparse
import scipy.io
import numpy as np
from PIL import Image
import io

SUNRGBD_ZIP_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"

CLASS_MAP = {
    # 0: table / desk
    "table": 0, "desk": 0, "coffee_table": 0, "dining_table": 0, "endtable": 0,
    # 1: chair
    "chair": 1, "office_chair": 1, "armchair": 1,
    # 2: door
    "door": 2,
    # 3: bookshelf
    "bookshelf": 3, "bookcase": 3, "shelf": 3, "shelves": 3,
    # 4: sofa
    "sofa": 4, "couch": 4,
    # 5: cabinet
    "cabinet": 5, "night_stand": 5, "nightstand": 5, "cupboard": 5, "dresser": 5
}

TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]

def parse_zip64_central_directory_slice(cd_bytes: bytes) -> dict:
    """Parses ZIP Central Directory headers to build a lookup table: filename -> (offset, c_size, u_size)."""
    file_index = {}
    pos = 0
    cd_len = len(cd_bytes)

    while pos < cd_len - 46:
        if cd_bytes[pos:pos+4] != b'PK\x01\x02':
            idx = cd_bytes.find(b'PK\x01\x02', pos)
            if idx == -1:
                break
            pos = idx

        comp_method = int.from_bytes(cd_bytes[pos+10:pos+12], 'little')
        c_size = int.from_bytes(cd_bytes[pos+20:pos+24], 'little')
        u_size = int.from_bytes(cd_bytes[pos+24:pos+28], 'little')
        fn_len = int.from_bytes(cd_bytes[pos+28:pos+30], 'little')
        extra_len = int.from_bytes(cd_bytes[pos+30:pos+32], 'little')
        comment_len = int.from_bytes(cd_bytes[pos+32:pos+34], 'little')
        local_offset = int.from_bytes(cd_bytes[pos+42:pos+46], 'little')

        fn = cd_bytes[pos+46:pos+46+fn_len].decode('utf-8', errors='ignore')

        # Check for ZIP64 extra field if fields overflowed
        extra = cd_bytes[pos+46+fn_len:pos+46+fn_len+extra_len]
        if local_offset == 0xFFFFFFFF or c_size == 0xFFFFFFFF or u_size == 0xFFFFFFFF:
            epos = 0
            while epos < len(extra) - 4:
                tag = int.from_bytes(extra[epos:epos+2], 'little')
                sz = int.from_bytes(extra[epos+2:epos+4], 'little')
                if tag == 1:
                    dpos = epos + 4
                    if u_size == 0xFFFFFFFF:
                        u_size = int.from_bytes(extra[dpos:dpos+8], 'little')
                        dpos += 8
                    if c_size == 0xFFFFFFFF:
                        c_size = int.from_bytes(extra[dpos:dpos+8], 'little')
                        dpos += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = int.from_bytes(extra[dpos:dpos+8], 'little')
                        dpos += 8
                    break
                epos += 4 + sz

        file_index[fn] = (local_offset, c_size, u_size, comp_method)
        pos += 46 + fn_len + extra_len + comment_len

    return file_index

def fetch_zip_file_bytes(url: str, local_offset: int, c_size: int, comp_method: int) -> bytes:
    """Fetches and decompresses a single file from a remote zip via HTTP range request."""
    # 1. Read local header to get data offset
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={local_offset}-{local_offset + 100}'})
    with urllib.request.urlopen(req) as resp:
        hdr = resp.read()

    fn_len = int.from_bytes(hdr[26:28], 'little')
    extra_len = int.from_bytes(hdr[28:30], 'little')
    data_start = local_offset + 30 + fn_len + extra_len

    # 2. Fetch compressed payload
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={data_start}-{data_start + c_size - 1}'})
    with urllib.request.urlopen(req) as resp:
        comp_data = resp.read()

    if comp_method == 8:
        return zlib.decompress(comp_data, -15)
    elif comp_method == 0:
        return comp_data
    else:
        raise ValueError(f"Unsupported zip compression method: {comp_method}")

def decode_sunrgbd_depth(raw_png_bytes: bytes) -> np.ndarray:
    """Decodes 16-bit SUN RGB-D depth PNG using Princeton's official bit-shift and metric scaling."""
    depth_img = Image.open(io.BytesIO(raw_png_bytes))
    depth_raw = np.array(depth_img, dtype=np.uint16)

    # Official formula: bitor(bitshift(depthVis,-3), bitshift(depthVis,16-3)) / 1000.0
    depth_unpacked = np.bitwise_or(np.right_shift(depth_raw, 3), np.left_shift(depth_raw, 16 - 3))
    depth_meters = depth_unpacked.astype(np.float32) / 1000.0
    depth_meters[depth_meters > 8.0] = 8.0
    return depth_meters

def download_sunrgbd_subset(
    output_dir: str = "data/sunrgbd",
    num_samples: int = 10,
    seed: int = 42
) -> dict:
    """Downloads a curated subset of real SUN RGB-D frames, converting annotations and preserving splits."""
    np.random.seed(seed)

    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # Load 2D, 3D metadata and split mappings
    mat2d_path = os.path.join(meta_dir, "SUNRGBDMeta2DBB_v2.mat")
    mat3d_path = os.path.join(meta_dir, "SUNRGBDMeta3DBB_v2.mat")
    split_path = os.path.join(meta_dir, "allsplit.mat")

    if not os.path.exists(mat2d_path):
        print("Downloading SUNRGBDMeta2DBB_v2.mat...")
        urllib.request.urlretrieve("https://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat", mat2d_path)
    if not os.path.exists(mat3d_path):
        print("Downloading SUNRGBDMeta3DBB_v2.mat...")
        urllib.request.urlretrieve("https://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat", mat3d_path)

    mat2d = scipy.io.loadmat(mat2d_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta2DBB"]
    mat3d = scipy.io.loadmat(mat3d_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]
    all_split = scipy.io.loadmat(split_path, squeeze_me=True)
    train_seqs = set(all_split["alltrain"])
    test_seqs = set(all_split["alltest"])

    # Load Central Directory index
    cd_file = os.path.join(meta_dir, "sunrgbd_cd.bin")
    if not os.path.exists(cd_file):
        raise FileNotFoundError(f"Central directory cache not found: {cd_file}")

    with open(cd_file, "rb") as f:
        cd_bytes = f.read()
    file_index = parse_zip64_central_directory_slice(cd_bytes)
    print(f"Parsed {len(file_index)} file entries from central directory.")

    # Find candidates that contain our classes and whose RGB and depth files are in our file_index
    candidates = []
    for i in range(len(mat2d)):
        entry2d = mat2d[i]
        seq = getattr(entry2d, "sequenceName", "")

        # Find relative rgb and depth paths inside zip
        # E.g., rgbpath: /n/fs/sun3d/data/SUNRGBD/kv1/...
        raw_rgb = getattr(entry2d, "rgbpath", "")
        raw_depth = getattr(entry2d, "depthpath", "")

        if "SUNRGBD/" in raw_rgb:
            rel_rgb = raw_rgb[raw_rgb.index("SUNRGBD/"):].replace("//", "/")
        else:
            continue
        if "SUNRGBD/" in raw_depth:
            rel_depth = raw_depth[raw_depth.index("SUNRGBD/"):].replace("//", "/")
        else:
            continue

        if rel_rgb in file_index and rel_depth in file_index:
            # Check if has relevant objects
            boxes2d = getattr(entry2d, "groundtruth2DBB", [])
            if not isinstance(boxes2d, (list, np.ndarray)):
                boxes2d = [boxes2d]

            has_target = False
            for b in boxes2d:
                cname = getattr(b, "classname", "").lower().strip()
                if cname in CLASS_MAP:
                    has_target = True
                    break

            if has_target:
                split_tag = "train" if any(seq in s for s in train_seqs) else "test"
                candidates.append((i, rel_rgb, rel_depth, split_tag))

    print(f"Found {len(candidates)} valid candidate scenes containing target classes.")
    if len(candidates) == 0:
        raise RuntimeError("No candidate scenes found in current central directory slice.")

    # Balance train and test samples
    train_cands = [c for c in candidates if c[3] == "train"]
    test_cands = [c for c in candidates if c[3] == "test"]

    n_train = num_samples // 2
    n_test = num_samples - n_train
    selected = train_cands[:n_train] + test_cands[:n_test]

    manifest = {
        "dataset_name": "SUN RGB-D Curated Benchmark Subset",
        "citation": "Song, Lichtenberg, Xiao, CVPR 2015",
        "classes": TARGET_CLASSES,
        "class_mapping": CLASS_MAP,
        "samples": []
    }

    print(f"Downloading {len(selected)} real samples ({len([s for s in selected if s[3]=='train'])} train, {len([s for s in selected if s[3]=='test'])} test)...")

    for s_idx, (frame_idx, rel_rgb, rel_depth, split_tag) in enumerate(selected):
        entry2d = mat2d[frame_idx]
        entry3d = mat3d[frame_idx]

        # 1. Fetch RGB JPEG
        off_rgb, c_rgb, u_rgb, m_rgb = file_index[rel_rgb]
        rgb_bytes = fetch_zip_file_bytes(SUNRGBD_ZIP_URL, off_rgb, c_rgb, m_rgb)
        pil_rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        orig_w, orig_h = pil_rgb.size

        # 2. Fetch Depth PNG
        off_depth, c_depth, u_depth, m_depth = file_index[rel_depth]
        depth_raw_bytes = fetch_zip_file_bytes(SUNRGBD_ZIP_URL, off_depth, c_depth, m_depth)
        depth_m = decode_sunrgbd_depth(depth_raw_bytes)

        frame_id = f"sunrgbd_{s_idx:05d}"
        rgb_filename = f"rgb/{frame_id}.jpg"
        depth_filename = f"depth/{frame_id}.npy"
        annot_filename = f"annotations/{frame_id}.json"

        # Save image and depth array
        pil_rgb.save(os.path.join(output_dir, rgb_filename))
        np.save(os.path.join(output_dir, depth_filename), depth_m)

        # 3. Parse Camera Intrinsics
        K = getattr(entry3d, "K", np.eye(3))
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # 4. Parse Objects (2D and 3D)
        boxes2d = getattr(entry2d, "groundtruth2DBB", [])
        if not isinstance(boxes2d, (list, np.ndarray)):
            boxes2d = [boxes2d]

        boxes3d = getattr(entry3d, "groundtruth3DBB", [])
        if not isinstance(boxes3d, (list, np.ndarray)):
            boxes3d = [boxes3d]

        objects_data = []
        for obj_idx, b in enumerate(boxes2d):
            raw_cname = getattr(b, "classname", "").lower().strip()
            if raw_cname not in CLASS_MAP:
                continue

            target_class_id = CLASS_MAP[raw_cname]
            target_class_name = TARGET_CLASSES[target_class_id]

            # [xmin, ymin, w, h] in 2DBB
            bbox_raw = getattr(b, "gtBb2D", [0, 0, 0, 0])
            xmin = float(bbox_raw[0])
            ymin = float(bbox_raw[1])
            w = float(bbox_raw[2])
            h = float(bbox_raw[3])
            xmax = xmin + w
            ymax = ymin + h

            # Match 3D box if available
            pos_cam_3d = None
            coeffs_3d = None
            if hasattr(b, "has3dbox") and b.has3dbox and obj_idx < len(boxes3d):
                b3 = boxes3d[obj_idx]
                if hasattr(b3, "centroid"):
                    pos_cam_3d = [round(float(v), 3) for v in b3.centroid]
                if hasattr(b3, "coeffs"):
                    coeffs_3d = [round(float(v), 3) for v in b3.coeffs]

            objects_data.append({
                "object_id": f"{target_class_name}_{obj_idx:02d}",
                "class_id": target_class_id,
                "class_name": target_class_name,
                "source_classname": raw_cname,
                "bbox_2d": {
                    "xmin": round(xmin, 1),
                    "ymin": round(ymin, 1),
                    "xmax": round(xmax, 1),
                    "ymax": round(ymax, 1)
                },
                "position_camera_3d": pos_cam_3d,
                "dimensions_3d": coeffs_3d,
                "is_visible": True
            })

        annotation = {
            "frame_id": frame_id,
            "original_sequence": getattr(entry2d, "sequenceName", ""),
            "split": split_tag,
            "image_filename": rgb_filename,
            "depth_filename": depth_filename,
            "camera": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "image_width": orig_w,
                "image_height": orig_h,
                "sensor_type": getattr(entry2d, "sensorType", "unknown")
            },
            "objects": objects_data
        }

        with open(os.path.join(output_dir, annot_filename), "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

        manifest["samples"].append({
            "frame_id": frame_id,
            "split": split_tag,
            "original_sequence": getattr(entry2d, "sequenceName", ""),
            "num_objects": len(objects_data),
            "image": rgb_filename,
            "depth": depth_filename
        })

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully processed {len(selected)} real SUN RGB-D samples into {output_dir}.")
    print(f"Manifest written to {manifest_path}.")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real SUN RGB-D benchmark subset.")
    parser.add_argument("--output_dir", type=str, default="data/sunrgbd")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_sunrgbd_subset(args.output_dir, args.num_samples, args.seed)
