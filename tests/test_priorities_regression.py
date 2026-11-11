"""Comprehensive regression tests for Spatial Quest Priorities 1-4."""
import base64
import io
import os
import unittest

import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.telemetry import load_cv_benchmarks
from backend.app.main import app
from backend.game.facility import seed_facility_scene
from backend.game.missions import GameSession
from backend.perception.model import SpatialDetector
from backend.perception.sun_rgbd_loader import SUNRGBDDataset
from backend.tools.registry import global_tool_registry


client = TestClient(app)


class TestPrioritiesRegression(unittest.TestCase):
    """Verifies real CV predictions, provenance separation, formatting, and strict mission gates."""

    def setUp(self):
        seed_facility_scene()
        global_tool_registry._environment_state["door_3"] = "locked"
        global_tool_registry._environment_state["ventilation"] = "offline"
        global_tool_registry._environment_state["power_grid"] = "emergency_mode"

    def test_trained_detector_produces_non_zero_threejs_detections(self):
        ckpt_path = "models/vision/spatial_detector_best.pt"
        self.assertTrue(os.path.exists(ckpt_path), "Trained detector checkpoint must exist")

        dataset = SUNRGBDDataset(root_dir="data/generated/threejs_sector4", split="test")
        self.assertGreater(len(dataset), 10)

        model = SpatialDetector()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        total_preds = 0
        with torch.no_grad():
            for i in range(min(10, len(dataset))):
                img_tensor = dataset[i]["rgb"]
                grid_pred = model(img_tensor.unsqueeze(0))
                preds = model.decode_predictions(grid_pred, conf_threshold=0.22)[0]
                total_preds += len(preds)

        self.assertGreater(
            total_preds,
            0,
            "Trained SpatialDetector must emit real detections on Three.js held-out test images",
        )

    def test_perception_threejs_endpoint_and_provenance_separation(self):
        rgb_path = "data/generated/threejs_sector4/rgb/threejs_00000.jpg"
        self.assertTrue(os.path.exists(rgb_path))
        with Image.open(rgb_path) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        res = client.post(
            "/perception",
            json={
                "image_base64": b64,
                "coordinate_system": "threejs",
                "camera_pose": {"position": [0.0, 1.5, -15.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                "camera_intrinsics": {"fx": 415.69, "fy": 415.69, "cx": 320.0, "cy": 240.0},
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["source"], "custom_detector")
        self.assertGreater(len(data["detections"]), 0)

        tel = client.get("/telemetry/latest").json()
        self.assertIn("seeded_layout_preview", tel["cv"])
        self.assertIsNotNone(tel["cv"].get("last_live_scan"))
        self.assertEqual(tel["cv"]["last_live_scan"]["source"], "custom_detector")
        bench = load_cv_benchmarks()
        self.assertGreater(bench["threejs_test_mAP_50"], 0.20)
        self.assertGreater(bench["threejs_test_precision"], 0.15)
        self.assertGreater(bench["sunrgbd_test_mAP_50"], 0.0)

    def test_mission_objectives_strict_target_and_result_gating(self):
        session = GameSession()
        session.select_mystery("mission_a")

        # Locating a chair must NOT complete scout_02 ("Locate the table in the facility")
        c_chair = session.record_tools(
            tool_calls=[{"tool": "locate_object", "arguments": {"object_name": "chair"}}],
            tool_results=[
                {
                    "tool": "locate_object",
                    "success": True,
                    "result": {"found": True, "class_name": "chair", "object_id": "chair_01"},
                }
            ],
        )
        self.assertNotIn("scout_02", c_chair)

        # Text-only reply without tools must NEVER advance objectives
        c_text = session.record_tools(tool_calls=[], tool_results=[])
        self.assertEqual(c_text, [])

        # Locating table DOES complete scout_02
        c_table = session.record_tools(
            tool_calls=[{"tool": "locate_object", "arguments": {"object_name": "table"}}],
            tool_results=[
                {
                    "tool": "locate_object",
                    "success": True,
                    "result": {"found": True, "class_name": "table", "object_id": "table_01"},
                }
            ],
        )
        self.assertIn("scout_02", c_table)

        # Switch to mission_d (Door 3 cipher mystery) and execute REAL tool with wrong passcode -> access denied
        session.select_mystery("mission_d")
        bad_door_res = global_tool_registry.execute(
            "interact_with_object",
            {"object_id": "door_3", "action": "unlock", "passcode": "1111"},
        )
        c_bad_door = session.record_tools(
            tool_calls=[{"tool": "interact_with_object", "arguments": {"object_id": "door_3", "action": "unlock", "passcode": "1111"}}],
            tool_results=[bad_door_res],
        )
        self.assertNotIn("clue_03", c_bad_door)

        # Verify environment_state is synchronized in to_dict()
        payload = session.to_dict()
        self.assertIn("environment_state", payload)
        self.assertEqual(payload["environment_state"].get("door_3"), "locked")

    def test_dual_nms_suppresses_cross_class_duplicates(self):
        boxes = [
            {"xmin": 100.0, "ymin": 100.0, "xmax": 300.0, "ymax": 300.0, "confidence": 0.88, "class_id": 0},
            {"xmin": 104.0, "ymin": 104.0, "xmax": 298.0, "ymax": 296.0, "confidence": 0.79, "class_id": 1},
            {"xmin": 420.0, "ymin": 120.0, "xmax": 560.0, "ymax": 320.0, "confidence": 0.65, "class_id": 1},
        ]
        kept = SpatialDetector._apply_nms(boxes, iou_thresh=0.45, cross_class_iou_thresh=0.55)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["class_id"], 0)
        self.assertEqual(kept[1]["class_id"], 1)
        self.assertEqual(kept[1]["xmin"], 420.0)

    def test_webgl_rg16_measured_depth_vs_estimated_depth_and_live_tool_isolation(self):
        import numpy as np
        from backend.language_model.llm_provider import build_spatial_context_snapshot

        rgb_path = "data/generated/threejs_sector4/rgb/threejs_00000.jpg"
        with Image.open(rgb_path) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
        rgb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Build lossless RG16 PNG representing constant 3.25m measured WebGL depth (3250mm -> hi=12, lo=178)
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        rgba[..., 0] = 3250 >> 8
        rgba[..., 1] = 3250 & 0xFF
        rgba[..., 3] = 255
        d_buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(d_buf, format="PNG")
        depth_b64 = base64.b64encode(d_buf.getvalue()).decode("ascii")

        res_measured = client.post(
            "/perception",
            json={
                "image_base64": rgb_b64,
                "depth_base64": depth_b64,
                "depth_encoding": "webgl_rg16_mm",
                "coordinate_system": "threejs",
                "camera_pose": {"position": [0.0, 1.5, -15.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                "camera_intrinsics": {"fx": 415.69, "fy": 415.69, "cx": 320.0, "cy": 240.0},
            },
        ).json()
        self.assertGreater(len(res_measured["detections"]), 0)
        self.assertEqual(res_measured["detections"][0]["depth_source"], "measured_webgl_depth")
        self.assertAlmostEqual(res_measured["detections"][0]["depth_estimate"], 3.25, delta=0.05)

        # Verify live_only=True returns strictly custom_detector live detections (no facility_layout seeded objects)
        live_tool = global_tool_registry.execute("get_visible_objects", {"live_only": True})
        self.assertTrue(live_tool["success"])
        live_list = live_tool["result"]
        self.assertGreater(len(live_list), 0)
        self.assertTrue(
            all(
                it["is_live_cnn"] is True and it["source"] == "custom_detector" and it["data_source"] == "spatial_detector_cnn"
                for it in live_list
            )
        )
        self.assertTrue(all(it["id"].startswith("cnn_") for it in live_list))

        snapshot, _ = build_spatial_context_snapshot("What did the live camera scan detect?", agent_id="scout")
        self.assertIn("LIVE CNN PERCEPTION DETECTIONS (custom_detector)", snapshot)
        self.assertIn("measured_webgl_depth", snapshot)

        # Verify omitting depth_base64 explicitly reports estimated_bbox_height
        res_est = client.post(
            "/perception",
            json={
                "image_base64": rgb_b64,
                "coordinate_system": "threejs",
                "camera_pose": {"position": [0.0, 1.5, -15.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                "camera_intrinsics": {"fx": 415.69, "fy": 415.69, "cx": 320.0, "cy": 240.0},
            },
        ).json()
        self.assertGreater(len(res_est["detections"]), 0)
        self.assertEqual(res_est["detections"][0]["depth_source"], "estimated_bbox_height")

    def test_expanded_sunrgbd_splits_disjoint_zero_leakage(self):
        train_ds = SUNRGBDDataset("data/sunrgbd", split="train")
        val_ds = SUNRGBDDataset("data/sunrgbd", split="val")
        test_ds = SUNRGBDDataset("data/sunrgbd", split="test")
        self.assertGreaterEqual(len(train_ds), 50)
        self.assertGreaterEqual(len(val_ds), 15)
        self.assertGreaterEqual(len(test_ds), 20)

        train_groups = {s.get("scene_group", s["frame_id"]) for s in train_ds.samples}
        val_groups = {s.get("scene_group", s["frame_id"]) for s in val_ds.samples}
        test_groups = {s.get("scene_group", s["frame_id"]) for s in test_ds.samples}
        self.assertEqual(len(train_groups & val_groups), 0)
        self.assertEqual(len(train_groups & test_groups), 0)
        self.assertEqual(len(val_groups & test_groups), 0)


if __name__ == "__main__":
    unittest.main()
