"""Captures authentic WebGL RGB-D training, validation, and held-out test frames from Three.js Sector 4.

Computes exact 2D bounding boxes via Three.js camera projection of 3D mesh bounding boxes,
raycast wall-occlusion filtering, and synchronized RG16 measured WebGL linear depth maps.
Strictly separates camera positions across train / val / test with zero spatial/viewpoint overlap.
"""
import argparse
import asyncio
import base64
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
from PIL import Image
import websockets

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]
SERVER_PORT = 18872
CDP_PORT = 19232


def build_viewpoint_specs():
    """Generates spatially disjoint viewpoints across all 5 Sector 4 rooms for train, val, and test."""
    train_views = []
    val_views = []
    test_views = []

    def add_orbit_views(targets, cam_points, split_list, modes=("1st", "1st", "3rd"), unlock_d3=False, max_dist=11.5):
        for t_idx, (tx, tz) in enumerate(targets):
            for c_idx, (cx, cz) in enumerate(cam_points):
                dx = tx - cx
                dz = tz - cz
                dist = math.hypot(dx, dz)
                if dist < 1.3 or dist > max_dist:
                    continue
                base_yaw = math.atan2(-dx, -dz)
                mode = modes[(t_idx + c_idx) % len(modes)]
                pitch = -0.11 if mode == "1st" else -0.20
                for yaw_off in (-0.16, 0.0, 0.16):
                    split_list.append({
                        "x": round(cx, 2),
                        "z": round(cz, 2),
                        "yaw": round(base_yaw + yaw_off, 3),
                        "pitch": pitch,
                        "mode": mode,
                        "door3_unlocked": unlock_d3 and ((t_idx + c_idx) % 2 == 1),
                    })

    # 1. Reception / Central Hub (door, sofa, tables, chairs)
    rec_targets = [(0.0, -14.8), (-6.5, -16.0), (6.5, -14.0), (0.0, -13.8)]
    rec_train_pts = [
        (0.0, -18.5), (-2.6, -18.2), (2.6, -18.2), (-3.8, -14.5), (3.8, -14.5),
        (-3.2, -11.5), (3.2, -11.5), (0.0, -11.4), (-9.0, -14.0), (9.0, -14.0),
        (-4.0, -18.6), (4.0, -18.6), (-6.5, -12.0), (6.5, -11.6), (-2.0, -17.2), (2.0, -17.2),
    ]
    rec_val_pts = [(-4.8, -17.4), (4.8, -17.4), (-1.2, -12.0)]
    rec_test_pts = [(-1.8, -18.8), (2.2, -12.2), (-8.5, -17.5), (8.6, -17.0), (0.0, -17.2)]
    add_orbit_views(rec_targets, rec_train_pts, train_views)
    add_orbit_views(rec_targets, rec_val_pts, val_views, modes=("1st", "3rd"))
    add_orbit_views(rec_targets, rec_test_pts, test_views, modes=("1st", "3rd"))

    # 2. Research Office (table_office_desk, chair_office, cabinet_office)
    off_targets = [(6.0, -5.0), (9.0, -5.0), (6.0, -4.0)]
    off_train_pts = [
        (3.4, -7.8), (3.4, -5.0), (3.4, -2.2), (5.8, -8.2), (6.2, -1.8),
        (9.5, -8.0), (9.5, -2.0), (10.8, -5.0), (7.6, -7.8), (4.2, -7.2),
    ]
    off_val_pts = [(5.2, -2.5), (7.8, -2.0)]
    off_test_pts = [(3.7, -5.0), (4.0, -2.8), (8.2, -1.8), (10.5, -7.5)]
    add_orbit_views(off_targets, off_train_pts, train_views)
    add_orbit_views(off_targets, off_val_pts, val_views, modes=("1st", "3rd"))
    add_orbit_views(off_targets, off_test_pts, test_views, modes=("1st", "3rd"))

    # 3. Library / Archives (bookshelf_library, reading table, reading chair)
    lib_targets = [(-8.5, -5.0), (-4.5, -5.0), (-4.5, -4.0)]
    lib_train_pts = [
        (-3.2, -7.8), (-3.2, -5.0), (-3.2, -2.2), (-6.2, -8.2), (-6.2, -1.8),
        (-10.5, -8.0), (-10.5, -2.0), (-6.5, -5.0), (-8.5, -8.4), (-8.5, -1.8),
    ]
    lib_val_pts = [(-4.8, -7.6), (-8.8, -6.5)]
    lib_test_pts = [(-3.6, -6.8), (-6.8, -2.4), (-10.6, -5.0)]
    add_orbit_views(lib_targets, lib_train_pts, train_views)
    add_orbit_views(lib_targets, lib_val_pts, val_views, modes=("1st", "3rd"))
    add_orbit_views(lib_targets, lib_test_pts, test_views, modes=("1st", "3rd"))

    # 4. Investigation Laboratory + Door 3 emphasis
    lab_targets = [(0.0, 0.0), (8.5, 2.0), (-8.5, 2.0), (0.0, 8.0), (-1.4, 0.6)]
    lab_train_pts = [
        (0.0, -3.8), (2.2, 3.8), (-2.5, 3.8), (4.2, 5.5), (-4.2, 5.5),
        (5.2, 2.0), (-5.2, 2.0), (0.0, 4.6), (3.5, 1.5), (-3.8, 1.5),
        (6.0, 6.2), (-6.0, 6.2), (-1.6, 5.8), (1.6, 5.8),
    ]
    lab_val_pts = [(-2.0, 1.8), (3.0, 5.8), (-3.2, 2.5)]
    lab_test_pts = [(1.9, 2.2), (4.8, 4.4), (-5.0, 4.2), (0.0, 3.4)]
    add_orbit_views(lab_targets, lab_train_pts, train_views, unlock_d3=True)
    add_orbit_views(lab_targets, lab_val_pts, val_views, modes=("1st", "3rd"), unlock_d3=True)
    add_orbit_views(lab_targets, lab_test_pts, test_views, modes=("1st", "3rd"), unlock_d3=True)

    # Dedicated Door 3 + Emergency Exit close-up angles (improves door AP & prevents door false positives)
    add_orbit_views([(0.0, 8.0)], [(0.0, 4.0), (-1.2, 4.4), (1.2, 4.4), (0.0, 12.2)], train_views, modes=("1st", "3rd"), unlock_d3=False)
    add_orbit_views([(0.0, 8.0)], [(-0.8, 5.2), (0.8, 5.2)], val_views, modes=("1st", "3rd"), unlock_d3=False)
    add_orbit_views([(0.0, 22.0)], [(0.0, 17.5), (-1.5, 18.0), (1.5, 18.0)], train_views, modes=("1st", "3rd"))
    add_orbit_views([(0.0, 22.0)], [(-1.0, 19.2), (1.2, 16.5)], val_views, modes=("1st", "3rd"))

    # 5. Storage Room / Vault
    sto_targets = [(-8.5, 14.0), (-8.5, 17.5), (8.5, 15.0), (0.0, 22.0), (0.0, 8.0)]
    sto_train_pts = [
        (0.0, 11.5), (0.0, 15.5), (-3.8, 12.0), (-3.8, 16.5), (3.8, 12.0),
        (3.8, 16.5), (0.0, 18.5), (-4.5, 19.5), (4.5, 19.5),
    ]
    sto_val_pts = [(-3.0, 18.0), (2.8, 14.2)]
    sto_test_pts = [(-2.6, 14.2), (3.2, 18.0), (0.0, 12.5)]
    add_orbit_views(sto_targets, sto_train_pts, train_views, unlock_d3=True)
    add_orbit_views(sto_targets, sto_val_pts, val_views, modes=("1st", "3rd"), unlock_d3=True)
    add_orbit_views(sto_targets, sto_test_pts, test_views, modes=("1st", "3rd"), unlock_d3=True)

    # Ensure strictly zero (x, z) coordinate overlap across splits
    train_xz = {(v["x"], v["z"]) for v in train_views}
    val_xz = {(v["x"], v["z"]) for v in val_views} - train_xz
    test_xz = {(v["x"], v["z"]) for v in test_views} - train_xz - val_xz
    val_views = [v for v in val_views if (v["x"], v["z"]) in val_xz]
    test_views = [v for v in test_views if (v["x"], v["z"]) in test_xz]

    def subsample(views, limit):
        if len(views) <= limit:
            return views
        step = len(views) / float(limit)
        return [views[int(i * step)] for i in range(limit)]

    return subsample(train_views, 240), subsample(val_views, 56), subsample(test_views, 56)


JS_CAPTURE_FN = r"""
window.__captureThreeJSGroundTruth = function(spec) {
  const st = window.__SPATIAL_STATE__;
  if (!st || !st.investigator || !st.facility || !st.camera || !st.renderer || !st.scene) {
    return null;
  }
  const inv = st.investigator;
  const fac = st.facility;
  const cam = st.camera;
  const renderer = st.renderer;
  const scene = st.scene;

  fac.setDoor3Unlocked(Boolean(spec.door3_unlocked));
  inv.position.set(spec.x, 0.0, spec.z);
  inv.cameraYaw = spec.yaw;
  inv.facingYaw = spec.yaw;
  inv.cameraPitch = spec.pitch;
  inv.setCameraMode(spec.mode);

  const forwardX = -Math.sin(inv.cameraYaw);
  const forwardZ = -Math.cos(inv.cameraYaw);
  if (spec.mode === "1st") {
    cam.position.set(inv.position.x, 1.66, inv.position.z);
    cam.lookAt(
      inv.position.x + forwardX * 6.0,
      1.66 + Math.sin(inv.cameraPitch) * 3.8,
      inv.position.z + forwardZ * 6.0
    );
  } else {
    const camDist = 3.8;
    const camHeight = 2.4 - Math.sin(inv.cameraPitch) * 1.4;
    cam.position.set(
      inv.position.x - forwardX * camDist,
      Math.max(1.2, camHeight),
      inv.position.z - forwardZ * camDist
    );
    cam.lookAt(
      inv.position.x + forwardX * 2.5,
      1.35 + Math.sin(inv.cameraPitch) * 1.6,
      inv.position.z + forwardZ * 2.5
    );
  }
  inv.root.position.copy(inv.position);
  inv.root.rotation.y = inv.facingYaw;
  scene.updateMatrixWorld(true);
  cam.updateMatrixWorld(true);
  renderer.render(scene, cam);

  const cw = renderer.domElement.width || 640;
  const ch = renderer.domElement.height || 480;
  const srcW = Math.min(cw, Math.round((ch * 4) / 3));
  const srcH = Math.min(ch, Math.round((cw * 3) / 4));
  const srcX = Math.max(0, Math.round((cw - srcW) * 0.5));
  const srcY = Math.max(0, Math.round((ch - srcH) * 0.5));

  const off = document.createElement("canvas");
  off.width = 640;
  off.height = 480;
  const octx = off.getContext("2d");
  octx.drawImage(renderer.domElement, srcX, srcY, srcW, srcH, 0, 0, 640, 480);

  // Synchronized RG16 linear WebGL depth pass aligned to exact crop
  const depthRes = window.SpatialDepthCapture
    ? window.SpatialDepthCapture.captureDepthPngBase64(renderer, scene, cam, srcX, srcY, srcW, srcH, 640, 480)
    : null;

  const fovRad = THREE.MathUtils.degToRad(cam.fov * 0.5);
  const fyCanvas = (ch * 0.5) / Math.tan(fovRad);
  const fy = fyCanvas * (480.0 / Math.max(1, srcH));
  const fx = fy;

  const rayMeshes = [];
  fac.group.traverse((obj) => {
    if (obj.isMesh && (!obj.material || !obj.material.transparent || obj.material.opacity > 0.5)) {
      rayMeshes.push(obj);
    }
  });

  const targetClasses = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"];
  const raycaster = new THREE.Raycaster();
  const invView = cam.matrixWorldInverse;
  const objs = [];

  for (const item of fac.interactables) {
    const clsIdx = targetClasses.indexOf(item.className);
    if (clsIdx === -1) continue;

    const box = new THREE.Box3().setFromObject(item.meshGroup);
    const center = box.getCenter(new THREE.Vector3());
    const distToCam = cam.position.distanceTo(center);
    if (distToCam > 12.5 || distToCam < 0.7) continue;

    let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
    let inFrontCount = 0;
    const xs = [box.min.x, (box.min.x + box.max.x) * 0.5, box.max.x];
    const ys = [box.min.y, (box.min.y + box.max.y) * 0.5, box.max.y];
    const zs = [box.min.z, (box.min.z + box.max.z) * 0.5, box.max.z];

    for (const x of xs) {
      for (const y of ys) {
        for (const z of zs) {
          const p = new THREE.Vector3(x, y, z);
          const pView = p.clone().applyMatrix4(invView);
          // Near-plane guard: ONLY project sample points strictly in front of camera
          if (pView.z >= -0.35) continue;
          inFrontCount++;
          const proj = p.clone().project(cam);
          const sxCanvas = (proj.x * 0.5 + 0.5) * cw;
          const syCanvas = (-proj.y * 0.5 + 0.5) * ch;
          const sx = (sxCanvas - srcX) * (640.0 / srcW);
          const sy = (syCanvas - srcY) * (480.0 / srcH);
          if (sx < xmin) xmin = sx;
          if (sx > xmax) xmax = sx;
          if (sy < ymin) ymin = sy;
          if (sy > ymax) ymax = sy;
        }
      }
    }
    if (inFrontCount < 18) continue;

    xmin = Math.max(0, Math.min(640, xmin));
    xmax = Math.max(0, Math.min(640, xmax));
    ymin = Math.max(0, Math.min(480, ymin));
    ymax = Math.max(0, Math.min(480, ymax));

    const bw = xmax - xmin;
    const bh = ymax - ymin;
    if (bw < 22 || bh < 22 || bw * bh < 650) continue;

    const samplePts = [center];
    for (const fxF of [0.25, 0.75]) {
      for (const fyF of [0.25, 0.75]) {
        for (const fzF of [0.25, 0.75]) {
          samplePts.push(new THREE.Vector3(
            box.min.x + (box.max.x - box.min.x) * fxF,
            box.min.y + (box.max.y - box.min.y) * fyF,
            box.min.z + (box.max.z - box.min.z) * fzF
          ));
        }
      }
    }

    let visibleHits = 0;
    for (const sp of samplePts) {
      const dir = sp.clone().sub(cam.position);
      const targetDist = dir.length();
      dir.normalize();
      raycaster.set(cam.position, dir);
      const hits = raycaster.intersectObjects(rayMeshes, false);
      if (hits.length > 0) {
        const h0 = hits[0];
        if ((h0.object && h0.object.userData && h0.object.userData.interactId === item.id) || h0.distance >= targetDist - 0.45) {
          visibleHits++;
        }
      } else {
        visibleHits++;
      }
    }
    if (visibleHits < 2) continue;

    const cView = center.clone().applyMatrix4(invView);
    objs.push({
      object_id: item.id,
      class_id: clsIdx,
      class_name: item.className,
      bbox_2d: {
        xmin: Math.round(xmin * 10) / 10,
        ymin: Math.round(ymin * 10) / 10,
        xmax: Math.round(xmax * 10) / 10,
        ymax: Math.round(ymax * 10) / 10
      },
      position_camera_3d: [
        Math.round(cView.x * 1000) / 1000,
        Math.round(-cView.y * 1000) / 1000,
        Math.round(-cView.z * 1000) / 1000
      ],
      position_world_3d: [
        Math.round(center.x * 1000) / 1000,
        Math.round(center.y * 1000) / 1000,
        Math.round(center.z * 1000) / 1000
      ]
    });
  }

  return {
    fx: Math.round(fx * 100) / 100,
    fy: Math.round(fy * 100) / 100,
    cx: 320.0,
    cy: 240.0,
    camera_pose: {
      position: [Math.round(cam.position.x * 1000) / 1000, Math.round(cam.position.y * 1000) / 1000, Math.round(cam.position.z * 1000) / 1000],
      rotation: [Math.round(cam.quaternion.x * 100000) / 100000, Math.round(cam.quaternion.y * 100000) / 100000, Math.round(cam.quaternion.z * 100000) / 100000, Math.round(cam.quaternion.w * 100000) / 100000]
    },
    objects: objs,
    image_b64: off.toDataURL("image/jpeg", 0.92).split(",")[1],
    depth_b64: depthRes ? depthRes.base64 : null
  };
};
"""


def _decode_rg16_png_to_meters(depth_b64: str) -> np.ndarray:
    raw = base64.b64decode(depth_b64)
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    arr = np.array(im, dtype=np.uint16)
    mm = (arr[..., 0] << 8) | arr[..., 1]
    dm = mm.astype(np.float32) / 1000.0
    dm[(mm <= 100) | (mm >= 60000)] = 0.0
    return dm


async def capture_dataset(out_dir: str):
    train_views, val_views, test_views = build_viewpoint_specs()
    stage_dir = os.path.join(out_dir, ".staging")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    for sub in ("rgb", "depth", "annotations"):
        os.makedirs(os.path.join(stage_dir, sub), exist_ok=True)

    async with httpx.AsyncClient(timeout=5.0) as client:
        tlist = (await client.get(f"http://127.0.0.1:{CDP_PORT}/json/list")).json()
        page_ws = next(t["webSocketDebuggerUrl"] for t in tlist if t.get("type") == "page")

    async with websockets.connect(page_ws, max_size=25 * 1024 * 1024) as ws:
        mid = 1

        async def call(method, params=None):
            nonlocal mid
            mid += 1
            await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method}: {msg['error']}")
                    return msg.get("result", {})

        async def eval_js(expr):
            res = await call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
            if "exceptionDetails" in res:
                raise RuntimeError(f"JS Exception: {res['exceptionDetails']}")
            return res.get("result", {}).get("value")

        await call("Page.enable")
        await call("Runtime.enable")
        await call("Emulation.setDeviceMetricsOverride", {
            "width": 1284, "height": 532, "deviceScaleFactor": 1, "mobile": False
        })
        await call("Page.navigate", {"url": f"http://127.0.0.1:{SERVER_PORT}/"})

        for _ in range(30):
            ready = await eval_js("(() => { const el = document.getElementById('loading-screen'); return el ? el.classList.contains('hidden') : false; })()")
            if ready:
                break
            await asyncio.sleep(0.35)
        await asyncio.sleep(1.0)
        await eval_js(JS_CAPTURE_FN)

        class_counts = {sp: {c: 0 for c in TARGET_CLASSES} for sp in ("train", "val", "test")}
        frame_idx = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        xz_by_split = {"train": set(), "val": set(), "test": set()}

        for split_name, views in [("train", train_views), ("val", val_views), ("test", test_views)]:
            for v_i, spec in enumerate(views):
                if v_i == len(views) // 2:
                    await call("Emulation.setDeviceMetricsOverride", {
                        "width": 1440, "height": 860, "deviceScaleFactor": 1, "mobile": False
                    })
                    await asyncio.sleep(0.12)
                elif v_i == 0:
                    await call("Emulation.setDeviceMetricsOverride", {
                        "width": 1284, "height": 532, "deviceScaleFactor": 1, "mobile": False
                    })
                    await asyncio.sleep(0.12)

                data = await eval_js(f"window.__captureThreeJSGroundTruth({json.dumps(spec)})")
                if not data or not data["objects"]:
                    continue

                frame_id = f"threejs_{frame_idx:05d}"
                img_rel = f"rgb/{frame_id}.jpg"
                with open(os.path.join(stage_dir, img_rel), "wb") as f:
                    f.write(base64.b64decode(data["image_b64"]))

                depth_rel = None
                if data.get("depth_b64"):
                    depth_rel = f"depth/{frame_id}.npy"
                    depth_m = _decode_rg16_png_to_meters(data["depth_b64"])
                    np.save(os.path.join(stage_dir, depth_rel), depth_m)

                ann = {
                    "frame_id": frame_id,
                    "split": split_name,
                    "viewpoint_xz": [spec["x"], spec["z"]],
                    "view_mode": spec["mode"],
                    "image_filename": img_rel,
                    "depth_filename": depth_rel,
                    "camera_pose": data.get("camera_pose"),
                    "camera": {"fx": data["fx"], "fy": data["fy"], "cx": 320.0, "cy": 240.0},
                    "objects": data["objects"],
                }
                with open(os.path.join(stage_dir, "annotations", f"{frame_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(ann, f, indent=2)

                for obj in data["objects"]:
                    class_counts[split_name][obj["class_name"]] += 1
                split_counts[split_name] += 1
                xz_by_split[split_name].add((spec["x"], spec["z"]))
                frame_idx += 1

        leakage = {
            "train_val_xz_overlap": len(xz_by_split["train"] & xz_by_split["val"]),
            "train_test_xz_overlap": len(xz_by_split["train"] & xz_by_split["test"]),
            "val_test_xz_overlap": len(xz_by_split["val"] & xz_by_split["test"]),
        }
        assert all(v == 0 for v in leakage.values()), f"Viewpoint leakage detected: {leakage}"

        os.makedirs(out_dir, exist_ok=True)
        for sub in ("rgb", "depth", "annotations"):
            dst = os.path.join(out_dir, sub)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.replace(os.path.join(stage_dir, sub), dst)
        shutil.rmtree(stage_dir, ignore_errors=True)

        manifest = {
            "dataset": "Spatial Quest Three.js Sector 4 Ground-Truth RGB-D Captures",
            "classes": TARGET_CLASSES,
            "splits": split_counts,
            "split_class_counts": class_counts,
            "leakage_audit": leakage,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("CAPTURE COMPLETE:", json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Capture real Three.js RGB-D dataset with projected ground truth.")
    parser.add_argument("--out-dir", default="data/generated/threejs_sector4")
    args = parser.parse_args()

    srv = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    user_dir = tempfile.mkdtemp(prefix="threejs_capture_")
    chrome = None
    try:
        for _ in range(30):
            try:
                r = httpx.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.3)

        chrome = subprocess.Popen(
            [
                CHROME_BIN,
                "--headless=new",
                "--no-sandbox",
                "--enable-webgl",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                f"--remote-debugging-port={CDP_PORT}",
                f"--user-data-dir={user_dir}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(25):
            try:
                r = httpx.get(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=1.0)
                if r.status_code == 200 and r.json():
                    break
            except Exception:
                time.sleep(0.3)

        asyncio.run(capture_dataset(args.out_dir))
    finally:
        if chrome:
            chrome.terminate()
            chrome.wait(timeout=5)
        srv.terminate()
        srv.wait(timeout=5)


if __name__ == "__main__":
    main()
