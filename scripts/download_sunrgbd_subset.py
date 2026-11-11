"""SUN RGB-D Subset Downloader & Annotation Converter for Spatial Quest.

Selects real indoor frames from the official SUN RGB-D benchmark containing our target classes
(table, chair, door, bookshelf, sofa, cabinet), fetches RGB images and registered depth maps
directly via HTTP range requests from Princeton's official archive without downloading the full 6.4 GB zip,
decodes the official 16-bit depth bitshift, and converts annotations into Spatial Quest format
with strict sequence-disjoint train / val / test splits and zero data leakage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
import shutil
import struct
import time
import urllib.request
import zlib

import httpx
import numpy as np
from PIL import Image
import scipy.io

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
    "cabinet": 5, "night_stand": 5, "nightstand": 5, "cupboard": 5, "dresser": 5,
}

TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]


def _http_get_range(client: httpx.Client, url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Fetches a byte range with retries and exponential backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            r = client.get(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
            if r.status_code in (200, 206):
                return r.content
            last_err = RuntimeError(f"HTTP {r.status_code} for range {start}-{end}")
        except Exception as exc:
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Range fetch failed after {retries} attempts: {last_err}")


def ensure_central_directory(url: str, cd_path: str) -> bytes:
    """Downloads and caches ONLY the ZIP64 Central Directory via HTTP byte-range requests."""
    if os.path.exists(cd_path) and os.path.getsize(cd_path) > 60_000_000:
        with open(cd_path, "rb") as f:
            return f.read()

    print("Fetching SUNRGBD.zip central directory metadata via HTTP Range...")
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        probe = client.get(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-1"})
        cr = probe.headers.get("Content-Range", "")
        if "/" in cr:
            total_len = int(cr.split("/")[-1])
        else:
            total_len = int(probe.headers["Content-Length"])

        tail_start = max(0, total_len - 65536)
        tail = _http_get_range(client, url, tail_start, total_len - 1)
        idx67 = tail.rfind(b"PK\x06\x07")
        if idx67 == -1:
            raise RuntimeError("Could not locate ZIP64 EOCD Locator in SUNRGBD.zip")
        _, eocd64_off, _ = struct.unpack("<IQI", tail[idx67 + 4:idx67 + 20])
        eocd64 = _http_get_range(client, url, eocd64_off, eocd64_off + 55)
        cd_size, cd_offset = struct.unpack("<QQ", eocd64[40:56])

        chunks = []
        chunk_bytes = 4 * 1024 * 1024
        pos = cd_offset
        end = cd_offset + cd_size - 1
        while pos <= end:
            chunk_end = min(end, pos + chunk_bytes - 1)
            chunks.append(_http_get_range(client, url, pos, chunk_end))
            pos = chunk_end + 1
    cd_bytes = b"".join(chunks)
    tmp_path = cd_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(cd_bytes)
    os.replace(tmp_path, cd_path)
    print(f"Cached Central Directory ({len(cd_bytes) / (1024 * 1024):.1f} MB) -> {cd_path}")
    return cd_bytes


def parse_zip64_central_directory_slice(cd_bytes: bytes) -> dict:
    """Parses ZIP Central Directory headers: filename -> (offset, c_size, u_size, comp_method)."""
    file_index = {}
    pos = 0
    cd_len = len(cd_bytes)

    while pos < cd_len - 46:
        if cd_bytes[pos:pos + 4] != b"PK\x01\x02":
            idx = cd_bytes.find(b"PK\x01\x02", pos)
            if idx == -1:
                break
            pos = idx

        comp_method = int.from_bytes(cd_bytes[pos + 10:pos + 12], "little")
        c_size = int.from_bytes(cd_bytes[pos + 20:pos + 24], "little")
        u_size = int.from_bytes(cd_bytes[pos + 24:pos + 28], "little")
        fn_len = int.from_bytes(cd_bytes[pos + 28:pos + 30], "little")
        extra_len = int.from_bytes(cd_bytes[pos + 30:pos + 32], "little")
        comment_len = int.from_bytes(cd_bytes[pos + 32:pos + 34], "little")
        local_offset = int.from_bytes(cd_bytes[pos + 42:pos + 46], "little")

        fn = cd_bytes[pos + 46:pos + 46 + fn_len].decode("utf-8", errors="ignore")
        extra = cd_bytes[pos + 46 + fn_len:pos + 46 + fn_len + extra_len]
        if local_offset == 0xFFFFFFFF or c_size == 0xFFFFFFFF or u_size == 0xFFFFFFFF:
            epos = 0
            while epos < len(extra) - 4:
                tag = int.from_bytes(extra[epos:epos + 2], "little")
                sz = int.from_bytes(extra[epos + 2:epos + 4], "little")
                if tag == 1:
                    dpos = epos + 4
                    if u_size == 0xFFFFFFFF:
                        u_size = int.from_bytes(extra[dpos:dpos + 8], "little")
                        dpos += 8
                    if c_size == 0xFFFFFFFF:
                        c_size = int.from_bytes(extra[dpos:dpos + 8], "little")
                        dpos += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = int.from_bytes(extra[dpos:dpos + 8], "little")
                        dpos += 8
                    break
                epos += 4 + sz

        file_index[fn] = (local_offset, c_size, u_size, comp_method)
        pos += 46 + fn_len + extra_len + comment_len

    return file_index


def fetch_zip_file_bytes(client: httpx.Client, url: str, local_offset: int, c_size: int, comp_method: int) -> bytes:
    """Fetches and decompresses a single zip entry via HTTP range request."""
    hdr = _http_get_range(client, url, local_offset, local_offset + 120)
    fn_len = int.from_bytes(hdr[26:28], "little")
    extra_len = int.from_bytes(hdr[28:30], "little")
    data_start = local_offset + 30 + fn_len + extra_len

    comp_data = _http_get_range(client, url, data_start, data_start + c_size - 1)
    if comp_method == 8:
        return zlib.decompress(comp_data, -15)
    if comp_method == 0:
        return comp_data
    raise ValueError(f"Unsupported zip compression method: {comp_method}")


def decode_sunrgbd_depth(raw_png_bytes: bytes) -> np.ndarray:
    """Decodes 16-bit SUN RGB-D depth PNG using Princeton's official bit-shift and metric scaling."""
    depth_img = Image.open(io.BytesIO(raw_png_bytes))
    depth_raw = np.array(depth_img, dtype=np.uint16)
    depth_unpacked = np.bitwise_or(np.right_shift(depth_raw, 3), np.left_shift(depth_raw, 16 - 3))
    depth_meters = depth_unpacked.astype(np.float32) / 1000.0
    depth_meters[depth_meters > 8.0] = 8.0
    return depth_meters


def _scene_group_key(seq_name: str) -> str:
    """Groups SUN RGB-D sequence strings by room/video session to prevent sequence leakage across splits."""
    parts = [p for p in seq_name.strip("/").split("/") if p]
    if "sun3ddata" in parts and len(parts) >= 5:
        return "/".join(parts[:5])
    return seq_name.strip("/")


def _stratified_select(cands: list, target_n: int, rng: np.random.Generator) -> list:
    """Selects up to target_n scene-diverse frames prioritizing balanced class representation."""
    if len(cands) <= target_n:
        return list(cands)
    chosen = []
    used_groups = set()
    class_totals = np.zeros(len(TARGET_CLASSES), dtype=np.int64)
    pool = list(cands)
    rng.shuffle(pool)

    while len(chosen) < target_n and pool:
        inv_freq = 1.0 / np.maximum(1.0, class_totals.astype(np.float64))
        best_idx = -1
        best_score = -1.0
        for idx, c in enumerate(pool[: min(len(pool), 800)]):
            if c["scene_group"] in used_groups:
                continue
            score = sum(inv_freq[cid] for cid in c["present_classes"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx == -1:
            break
        picked = pool.pop(best_idx)
        chosen.append(picked)
        used_groups.add(picked["scene_group"])
        for cid in picked["present_classes"]:
            class_totals[cid] += 1

    while len(chosen) < target_n and pool:
        picked = pool.pop(0)
        chosen.append(picked)
    return chosen


def download_sunrgbd_subset(
    output_dir: str = "data/sunrgbd",
    num_samples: int = 160,
    seed: int = 42,
) -> dict:
    """Downloads a sequence-disjoint curated subset of real SUN RGB-D frames (train / val / test)."""
    rng = np.random.default_rng(seed)

    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    stage_dir = os.path.join(output_dir, ".staging")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    for sub in ("rgb", "depth", "annotations"):
        os.makedirs(os.path.join(stage_dir, sub), exist_ok=True)

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
    train_seqs = set(str(s).strip("/") for s in all_split["alltrain"])
    test_seqs = set(str(s).strip("/") for s in all_split["alltest"])

    cd_file = os.path.join(meta_dir, "sunrgbd_cd.bin")
    cd_bytes = ensure_central_directory(SUNRGBD_ZIP_URL, cd_file)
    file_index = parse_zip64_central_directory_slice(cd_bytes)
    print(f"Parsed {len(file_index)} file entries from central directory.")

    train_norm_suffixes = {s.split("SUNRGBD/")[-1] for s in train_seqs}
    test_norm_suffixes = {s.split("SUNRGBD/")[-1] for s in test_seqs}

    train_val_cands = []
    test_cands = []
    for i in range(len(mat2d)):
        entry2d = mat2d[i]
        seq = str(getattr(entry2d, "sequenceName", "")).strip("/")
        raw_rgb = getattr(entry2d, "rgbpath", "")
        raw_depth = getattr(entry2d, "depthpath", "")
        if "SUNRGBD/" not in raw_rgb or "SUNRGBD/" not in raw_depth:
            continue
        rel_rgb = raw_rgb[raw_rgb.index("SUNRGBD/"):].replace("//", "/")
        rel_depth = raw_depth[raw_depth.index("SUNRGBD/"):].replace("//", "/")
        if rel_rgb not in file_index or rel_depth not in file_index:
            continue

        boxes2d = getattr(entry2d, "groundtruth2DBB", [])
        if not isinstance(boxes2d, (list, np.ndarray)):
            boxes2d = [boxes2d]

        present_classes = set()
        for b in boxes2d:
            cname = str(getattr(b, "classname", "")).lower().strip()
            bb = getattr(b, "gtBb2D", [0, 0, 0, 0])
            if cname in CLASS_MAP and len(bb) == 4 and float(bb[2]) >= 12.0 and float(bb[3]) >= 12.0:
                present_classes.add(CLASS_MAP[cname])
        if not present_classes:
            continue

        seq_key = seq.split("SUNRGBD/")[-1]
        is_test = seq_key in test_norm_suffixes
        is_train = seq_key in train_norm_suffixes
        if not is_test and not is_train:
            continue

        cand = {
            "idx": i,
            "seq": seq,
            "scene_group": _scene_group_key(seq),
            "rel_rgb": rel_rgb,
            "rel_depth": rel_depth,
            "present_classes": sorted(present_classes),
        }
        if is_test:
            test_cands.append(cand)
        else:
            train_val_cands.append(cand)

    print(f"Valid candidates -> official train+val pool: {len(train_val_cands)}, official test pool: {len(test_cands)}")
    if not train_val_cands or not test_cands:
        raise RuntimeError("Insufficient SUN RGB-D candidates matching train/test index.")

    n_test = max(15, round(num_samples * 0.25))
    n_val = max(12, round(num_samples * 0.15))
    n_train = max(20, num_samples - n_test - n_val)

    all_groups = sorted({c["scene_group"] for c in train_val_cands})
    rng.shuffle(all_groups)
    val_group_cutoff = max(1, int(round(len(all_groups) * 0.20)))
    val_groups = set(all_groups[:val_group_cutoff])
    train_groups = set(all_groups[val_group_cutoff:])

    test_groups = {c["scene_group"] for c in test_cands}
    val_groups -= test_groups
    train_groups -= test_groups

    train_pool = [c for c in train_val_cands if c["scene_group"] in train_groups]
    val_pool = [c for c in train_val_cands if c["scene_group"] in val_groups]

    selected_train = _stratified_select(train_pool, n_train, rng)
    selected_val = _stratified_select(val_pool, n_val, rng)
    selected_test = _stratified_select(test_cands, n_test, rng)

    tagged_selected = (
        [("train", c) for c in selected_train]
        + [("val", c) for c in selected_val]
        + [("test", c) for c in selected_test]
    )

    assert not ({c["scene_group"] for sp, c in tagged_selected if sp == "train"} & {c["scene_group"] for sp, c in tagged_selected if sp == "val"})
    assert not ({c["scene_group"] for sp, c in tagged_selected if sp == "train"} & {c["scene_group"] for sp, c in tagged_selected if sp == "test"})
    assert not ({c["scene_group"] for sp, c in tagged_selected if sp == "val"} & {c["scene_group"] for sp, c in tagged_selected if sp == "test"})

    print(
        f"Downloading {len(tagged_selected)} real SUN RGB-D RGB-D scenes "
        f"(train={len(selected_train)}, val={len(selected_val)}, test={len(selected_test)})..."
    )

    records = []
    with httpx.Client(timeout=40.0, follow_redirects=True, limits=httpx.Limits(max_connections=6)) as http_client:
        def _download_one(task_item):
            s_idx, split_tag, cand = task_item
            frame_idx = cand["idx"]
            entry2d = mat2d[frame_idx]
            entry3d = mat3d[frame_idx]

            off_rgb, c_rgb, _, m_rgb = file_index[cand["rel_rgb"]]
            rgb_bytes = fetch_zip_file_bytes(http_client, SUNRGBD_ZIP_URL, off_rgb, c_rgb, m_rgb)
            pil_rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            orig_w, orig_h = pil_rgb.size
            img_md5 = hashlib.md5(rgb_bytes).hexdigest()

            off_depth, c_depth, _, m_depth = file_index[cand["rel_depth"]]
            depth_raw_bytes = fetch_zip_file_bytes(http_client, SUNRGBD_ZIP_URL, off_depth, c_depth, m_depth)
            depth_m = decode_sunrgbd_depth(depth_raw_bytes)

            frame_id = f"sunrgbd_{s_idx:05d}"
            rgb_filename = f"rgb/{frame_id}.jpg"
            depth_filename = f"depth/{frame_id}.npy"
            annot_filename = f"annotations/{frame_id}.json"

            pil_rgb.save(os.path.join(stage_dir, rgb_filename), quality=92)
            np.save(os.path.join(stage_dir, depth_filename), depth_m)

            K = getattr(entry3d, "K", np.eye(3))
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            boxes2d = getattr(entry2d, "groundtruth2DBB", [])
            if not isinstance(boxes2d, (list, np.ndarray)):
                boxes2d = [boxes2d]
            boxes3d = getattr(entry3d, "groundtruth3DBB", [])
            if not isinstance(boxes3d, (list, np.ndarray)):
                boxes3d = [boxes3d]

            objects_data = []
            for obj_idx, b in enumerate(boxes2d):
                raw_cname = str(getattr(b, "classname", "")).lower().strip()
                if raw_cname not in CLASS_MAP:
                    continue
                cid = CLASS_MAP[raw_cname]
                cname = TARGET_CLASSES[cid]
                bb = getattr(b, "gtBb2D", [0, 0, 0, 0])
                xmin = max(0.0, float(bb[0]))
                ymin = max(0.0, float(bb[1]))
                xmax = min(float(orig_w), xmin + float(bb[2]))
                ymax = min(float(orig_h), ymin + float(bb[3]))
                if xmax - xmin < 12.0 or ymax - ymin < 12.0:
                    continue

                pos_cam_3d = None
                coeffs_3d = None
                if hasattr(b, "has3dbox") and b.has3dbox and obj_idx < len(boxes3d):
                    b3 = boxes3d[obj_idx]
                    if hasattr(b3, "centroid"):
                        pos_cam_3d = [round(float(v), 3) for v in b3.centroid]
                    if hasattr(b3, "coeffs"):
                        coeffs_3d = [round(float(v), 3) for v in b3.coeffs]

                objects_data.append({
                    "object_id": f"{cname}_{obj_idx:02d}",
                    "class_id": cid,
                    "class_name": cname,
                    "source_classname": raw_cname,
                    "bbox_2d": {
                        "xmin": round(xmin, 1),
                        "ymin": round(ymin, 1),
                        "xmax": round(xmax, 1),
                        "ymax": round(ymax, 1),
                    },
                    "position_camera_3d": pos_cam_3d,
                    "dimensions_3d": coeffs_3d,
                    "is_visible": True,
                })

            annotation = {
                "frame_id": frame_id,
                "original_sequence": getattr(entry2d, "sequenceName", ""),
                "scene_group": cand["scene_group"],
                "split": split_tag,
                "image_md5": img_md5,
                "image_filename": rgb_filename,
                "depth_filename": depth_filename,
                "camera": {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "image_width": orig_w,
                    "image_height": orig_h,
                    "sensor_type": getattr(entry2d, "sensorType", "unknown"),
                },
                "objects": objects_data,
            }
            with open(os.path.join(stage_dir, annot_filename), "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2)

            return s_idx, {
                "frame_id": frame_id,
                "split": split_tag,
                "original_sequence": annotation["original_sequence"],
                "scene_group": cand["scene_group"],
                "image_md5": img_md5,
                "num_objects": len(objects_data),
                "classes_present": sorted({o["class_name"] for o in objects_data}),
                "image": rgb_filename,
                "depth": depth_filename,
            }

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(_download_one, (idx, sp, cand))
                for idx, (sp, cand) in enumerate(tagged_selected)
            ]
            completed = 0
            for fut in as_completed(futures):
                records.append(fut.result())
                completed += 1
                if completed % 25 == 0 or completed == len(futures):
                    print(f"  Fetched {completed}/{len(futures)} SUN RGB-D scenes...")

    # Promote staging cleanly into output_dir
    for sub in ("rgb", "depth", "annotations"):
        dst = os.path.join(output_dir, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.replace(os.path.join(stage_dir, sub), dst)
    shutil.rmtree(stage_dir, ignore_errors=True)

    records.sort(key=lambda x: x[0])
    samples_list = [r[1] for r in records if r[1]["num_objects"] > 0]

    hashes_by_split = {"train": set(), "val": set(), "test": set()}
    groups_by_split = {"train": set(), "val": set(), "test": set()}
    split_class_counts = {sp: {c: 0 for c in TARGET_CLASSES} for sp in ("train", "val", "test")}
    for s in samples_list:
        hashes_by_split[s["split"]].add(s["image_md5"])
        groups_by_split[s["split"]].add(s["scene_group"])
        ann_path = os.path.join(output_dir, "annotations", f"{s['frame_id']}.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        for o in ann["objects"]:
            split_class_counts[s["split"]][o["class_name"]] += 1

    leakage = {
        "train_val_group_overlap": len(groups_by_split["train"] & groups_by_split["val"]),
        "train_test_group_overlap": len(groups_by_split["train"] & groups_by_split["test"]),
        "val_test_group_overlap": len(groups_by_split["val"] & groups_by_split["test"]),
        "train_val_md5_overlap": len(hashes_by_split["train"] & hashes_by_split["val"]),
        "train_test_md5_overlap": len(hashes_by_split["train"] & hashes_by_split["test"]),
        "val_test_md5_overlap": len(hashes_by_split["val"] & hashes_by_split["test"]),
    }
    assert all(v == 0 for v in leakage.values()), f"Leakage detected: {leakage}"

    manifest = {
        "dataset_name": "SUN RGB-D Curated Benchmark Subset (Expanded Train/Val/Test)",
        "citation": "Song, Lichtenberg, Xiao, CVPR 2015",
        "classes": TARGET_CLASSES,
        "class_mapping": CLASS_MAP,
        "split_counts": {sp: len([s for s in samples_list if s["split"] == sp]) for sp in ("train", "val", "test")},
        "split_class_counts": split_class_counts,
        "leakage_audit": leakage,
        "samples": samples_list,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("SUN RGB-D Expanded Subset Summary:")
    print(json.dumps({
        "split_counts": manifest["split_counts"],
        "split_class_counts": split_class_counts,
        "leakage_audit": leakage,
    }, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real SUN RGB-D train/val/test benchmark subset.")
    parser.add_argument("--output_dir", type=str, default="data/sunrgbd")
    parser.add_argument("--num_samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download_sunrgbd_subset(args.output_dir, args.num_samples, args.seed)
