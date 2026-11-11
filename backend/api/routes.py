import os
import io
import json
import base64
from fastapi import APIRouter, HTTPException, Body
from typing import Optional, Dict, Any
from fastapi.responses import StreamingResponse
import time
from PIL import Image
import numpy as np
import torch
from backend.api.schemas import (
    HealthResponse,
    PerceptionRequest,
    PerceptionResponse,
    DetectedObject,
    BoundingBox2D,
    SceneGraphResponse,
    AgentQueryRequest,
    AgentQueryResponse,
    UnityActionResult,
    ToolCall,
    GameStartRequest,
    GameResetRequest,
    GameAgentRequest,
    PlayerSyncRequest,
)
from backend.perception.device import get_device_info, get_optimal_device
from backend.perception.model import SpatialDetector
from backend.spatial.geometry import (
    backproject_pixel_to_camera_point,
    camera_to_world_transform,
    quaternion_to_rotation_matrix,
    sample_robust_depth_from_box,
)
from backend.spatial.scene_graph import global_scene_graph
from backend.tools.registry import global_tool_registry
from backend.agent.investigation_agent import global_investigation_agent
from backend.language_model.llm_provider import global_llm_provider
from backend.game.facility import seed_facility_scene
from backend.game.missions import global_game_session
from backend.api.telemetry import global_telemetry, load_cv_benchmarks

router = APIRouter()
START_TIME = time.time()

# Diagnostic flags — true when from-scratch checkpoints are present on disk
VISION_MODEL_LOADED = os.path.exists("models/vision/spatial_detector_best.pt")
LANGUAGE_MODEL_LOADED = os.path.exists("models/language_model/transformer_best.pt")

_detector_singleton = None
_detector_mtime = None
_TYPICAL_CLASS_HEIGHT_M = {
    "table": 0.76,
    "chair": 0.90,
    "door": 2.20,
    "bookshelf": 2.10,
    "sofa": 0.85,
    "cabinet": 1.80,
}

def _get_loaded_detector():
    """Lazily loads and hot-reloads the custom SpatialDetector trained checkpoint and calibrated thresholds."""
    global _detector_singleton, _detector_mtime
    ckpt_path = "models/vision/spatial_detector_best.pt"
    if os.path.exists(ckpt_path):
        mtime = os.path.getmtime(ckpt_path)
        if _detector_singleton is None or _detector_mtime != mtime:
            try:
                device = get_optimal_device()
                model = SpatialDetector()
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)
                model.eval()
                class_thresholds = ckpt.get("class_thresholds")
                _detector_singleton = (model, device, class_thresholds)
                _detector_mtime = mtime
            except Exception as e:
                print(f"[Warning] Failed to load SpatialDetector: {e}")
                _detector_singleton = None
    return _detector_singleton

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Returns system status, active compute device, model states, and scene count."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        device_info=get_device_info(),
        vision_model_loaded=VISION_MODEL_LOADED,
        language_model_loaded=LANGUAGE_MODEL_LOADED,
        scene_objects_count=len(global_scene_graph.get_all_objects()),
        uptime_seconds=round(time.time() - START_TIME, 2)
    )

@router.post("/perception", response_model=PerceptionResponse)
def process_perception(request: PerceptionRequest):
    """Processes camera visual inputs (RGB-D) or test payload and updates scene graph."""
    t0 = time.time()

    if request.mock_mode or not VISION_MODEL_LOADED:
        # Explicit test mode: provides calibrated synthetic laboratory detection samples
        # for pipeline verification without requiring a pre-existing trained checkpoint.
        sample_detections = [
            DetectedObject(
                object_id="table_01",
                class_id=0,
                class_name="table",
                confidence=0.94,
                bbox=BoundingBox2D(xmin=120.0, ymin=200.0, xmax=520.0, ymax=440.0),
                estimated_3d_position=[0.0, 0.75, 2.5],
                depth_estimate=2.5
            ),
            DetectedObject(
                object_id="laptop_01",
                class_id=2,
                class_name="laptop",
                confidence=0.89,
                bbox=BoundingBox2D(xmin=260.0, ymin=220.0, xmax=380.0, ymax=310.0),
                estimated_3d_position=[-0.1, 0.85, 2.3],
                depth_estimate=2.3
            ),
            DetectedObject(
                object_id="chair_01",
                class_id=1,
                class_name="chair",
                confidence=0.82,
                bbox=BoundingBox2D(xmin=40.0, ymin=240.0, xmax=160.0, ymax=420.0),
                estimated_3d_position=[-0.9, 0.5, 2.2],
                depth_estimate=2.2
            ),
            DetectedObject(
                object_id="door_01",
                class_id=4,
                class_name="door",
                confidence=0.91,
                bbox=BoundingBox2D(xmin=500.0, ymin=100.0, xmax=620.0, ymax=450.0),
                estimated_3d_position=[2.8, 1.2, 4.5],
                depth_estimate=4.5
            ),
            DetectedObject(
                object_id="fire_extinguisher_01",
                class_id=5,
                class_name="fire_extinguisher",
                confidence=0.88,
                bbox=BoundingBox2D(xmin=540.0, ymin=260.0, xmax=600.0, ymax=360.0),
                estimated_3d_position=[2.5, 0.9, 4.1],
                depth_estimate=4.1
            )
        ]
        global_scene_graph.update_objects(sample_detections, source="test_mock_pipeline")
        return PerceptionResponse(
            detections=sample_detections,
            inference_time_ms=round((time.time() - t0) * 1000.0, 2),
            source="test_mock_pipeline"
        )

    # Real forward inference with trained SpatialDetector
    loaded = _get_loaded_detector()
    if loaded is None or not request.image_base64:
        # If no image is provided, return empty detection response
        return PerceptionResponse(
            detections=[],
            inference_time_ms=round((time.time() - t0) * 1000.0, 2),
            source="custom_detector"
        )

    model, device, class_thresholds = loaded
    try:
        raw_bytes = base64.b64decode(request.image_base64)
        # Use Image.BILINEAR to match SUNRGBDDataset preprocessing identically
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((640, 480), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        x_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            grid_pred = model(x_tensor)
            decoded_batches = model.decode_predictions(
                grid_pred,
                conf_threshold=0.32,
                nms_iou_threshold=0.45,
                class_thresholds=class_thresholds,
                cross_class_iou_threshold=0.55,
            )
            preds = decoded_batches[0] if decoded_batches else []

        # Parse optional depth array cleanly (WebGL RG16 millimeter PNG vs standard depth PNG)
        depth_arr = None
        measured_depth_tag = "measured_rgbd_sensor"
        if request.depth_base64:
            try:
                depth_raw = base64.b64decode(request.depth_base64)
                if (request.depth_encoding or "").lower() == "webgl_rg16_mm":
                    d_pil = Image.open(io.BytesIO(depth_raw)).convert("RGBA").resize((640, 480), Image.NEAREST)
                    d_rgba = np.array(d_pil, dtype=np.uint16)
                    mm = (d_rgba[..., 0] << 8) | d_rgba[..., 1]
                    depth_arr = mm.astype(np.float32) / 1000.0
                    depth_arr[(mm <= 100) | (mm >= 60000)] = 0.0
                    measured_depth_tag = "measured_webgl_depth"
                else:
                    d_pil = Image.open(io.BytesIO(depth_raw)).resize((640, 480), Image.NEAREST)
                    depth_arr = np.array(d_pil, dtype=np.float32)
            except Exception:
                depth_arr = None

        fx = request.camera_intrinsics.fx if request.camera_intrinsics else 525.0
        fy = request.camera_intrinsics.fy if request.camera_intrinsics else 525.0
        cx = request.camera_intrinsics.cx if request.camera_intrinsics else 320.0
        cy = request.camera_intrinsics.cy if request.camera_intrinsics else 240.0
        cam_pos = request.camera_pose.position if request.camera_pose else [0.0, 1.5, 0.0]
        cam_rot = request.camera_pose.rotation if request.camera_pose else [0.0, 0.0, 0.0, 1.0]
        coord_sys = (request.coordinate_system or "unity").lower().strip()

        detected_objects = []
        for idx, pred in enumerate(preds):
            u_c = (pred["xmin"] + pred["xmax"]) / 2.0
            v_c = (pred["ymin"] + pred["ymax"]) / 2.0
            box_h = max(16.0, float(pred["ymax"] - pred["ymin"]))

            depth_val = None
            depth_src = "estimated_bbox_height"
            if depth_arr is not None:
                depth_val = sample_robust_depth_from_box(
                    depth_arr,
                    pred["xmin"],
                    pred["ymin"],
                    pred["xmax"],
                    pred["ymax"],
                    sampling_ratio=0.55,
                    reject_background=True,
                )
                if depth_val is not None:
                    depth_src = measured_depth_tag
            if depth_val is None:
                # Pinhole apparent-height depth fallback only when measured depth unavailable for bbox
                nominal_h = _TYPICAL_CLASS_HEIGHT_M.get(pred["class_name"], 1.0)
                depth_val = float(np.clip((nominal_h * abs(fy)) / box_h, 0.8, 9.5))

            cam_point = backproject_pixel_to_camera_point(u_c, v_c, depth_val, fx, fy, cx, cy)
            if coord_sys == "threejs":
                R = quaternion_to_rotation_matrix(cam_rot)
                p_local = np.array([cam_point[0], -cam_point[1], -cam_point[2]], dtype=np.float32)
                world_pos = (R @ p_local + np.array(cam_pos, dtype=np.float32)).tolist()
            else:
                world_pos = camera_to_world_transform(cam_point, cam_pos, cam_rot)

            obj = DetectedObject(
                object_id=f"cnn_{pred['class_name']}_{idx+1:02d}",
                class_id=pred["class_id"],
                class_name=pred["class_name"],
                confidence=round(pred["confidence"], 3),
                bbox=BoundingBox2D(
                    xmin=round(pred["xmin"], 1),
                    ymin=round(pred["ymin"], 1),
                    xmax=round(pred["xmax"], 1),
                    ymax=round(pred["ymax"], 1)
                ),
                estimated_3d_position=[round(p, 2) for p in world_pos],
                depth_estimate=round(depth_val, 2),
                depth_source=depth_src,
                source="custom_detector",
            )
            detected_objects.append(obj)

        global_scene_graph.merge_live_detections(detected_objects, source="custom_detector")
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        global_telemetry.record_cv_scan(
            detections=detected_objects,
            inference_time_ms=elapsed_ms,
            intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            camera_position=cam_pos,
        )

        return PerceptionResponse(
            detections=detected_objects,
            inference_time_ms=elapsed_ms,
            source="custom_detector"
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}") from exc

@router.get("/scene", response_model=SceneGraphResponse)
def get_scene_graph():
    """Returns the current 3D spatial scene graph with nodes and computed relations."""
    if not global_scene_graph.get_all_objects():
        seed_facility_scene()
    return global_scene_graph.to_response()

def _prepare_agent_turn(request: AgentQueryRequest) -> str:
    """Seeds scene graph if needed, switches mission if specified, and syncs pose."""
    if not global_scene_graph.get_all_objects():
        seed_facility_scene()

    agent_id = request.agent_id or global_game_session.active_agent_id
    if request.mission_id:
        try:
            global_game_session.start_mission(request.mission_id)
        except ValueError:
            pass

    if request.player_position is not None or request.active_room is not None:
        global_game_session.update_player_state(
            position=request.player_position,
            room=request.active_room,
        )
    return agent_id


def _finalize_agent_turn(
    request: AgentQueryRequest,
    agent_id: str,
    result: dict,
    t_start: float,
) -> AgentQueryResponse:
    """Updates mission objectives, records telemetry, and builds response."""
    global_game_session.record_tools(result.get("tool_calls", []), result.get("tool_results", []))
    latency_ms = (time.time() - t_start) * 1000.0
    llm_mode = result.get("llm_mode", global_llm_provider.mode)
    model_name = result.get("model_name")

    corr_id = global_telemetry.record_turn(
        query=request.query,
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", []),
        final_response=result.get("final_response", ""),
        agent_id=agent_id,
        token_ids=result.get("token_ids"),
        latency_ms=latency_ms,
        rag_passages=result.get("rag_passages"),
        player_coords=request.player_position,
        active_room=request.active_room,
        used_fallback=result.get("used_fallback", False),
        model_raw_output=result.get("model_raw_output", ""),
        validation_status=result.get("validation_status", "PASSED_WHITELIST"),
        llm_mode=llm_mode,
        model_name=model_name,
    )

    tool_calls = [
        ToolCall(tool=tc.get("tool") or tc.get("name", "unknown"), arguments=tc.get("arguments", {}))
        for tc in result.get("tool_calls", [])
    ]

    return AgentQueryResponse(
        query=request.query,
        tool_calls=tool_calls,
        tool_results=result.get("tool_results", []),
        final_response=result.get("final_response", ""),
        unity_action=result.get("unity_action"),
        completed=True,
        game_state=global_game_session.to_dict(),
        correlation_id=corr_id,
        latency_ms=round(latency_ms, 2),
        validation_status=result.get("validation_status", "PASSED_WHITELIST"),
        model_output=result.get("model_raw_output", ""),
        used_fallback=bool(result.get("used_fallback", False)),
        llm_mode=llm_mode,
        model_name=model_name,
        session_id=request.session_id or "default",
        error=result.get("error"),
    )


@router.post("/agent/query", response_model=AgentQueryResponse)
def process_agent_query(request: AgentQueryRequest):
    """Processes a natural language query through the investigation agent and tool registry."""
    t_start = time.time()
    agent_id = _prepare_agent_turn(request)
    result = global_investigation_agent.run(
        request.query,
        agent_id=agent_id,
        mission_context=global_game_session.to_dict(),
        active_room=request.active_room,
        player_position=request.player_position,
        camera_heading=request.camera_heading,
        session_id=request.session_id or "default",
        selected_object_id=request.selected_object_id,
    )
    return _finalize_agent_turn(request, agent_id, result, t_start)


@router.post("/agent/query/stream")
def stream_agent_query(request: AgentQueryRequest):
    """Streams agent reasoning status, tool calls, response tokens, and final game state over SSE."""
    t_start = time.time()
    agent_id = _prepare_agent_turn(request)

    def _event_generator():
        final_result = None
        for evt in global_investigation_agent.stream_run(
            request.query,
            agent_id=agent_id,
            mission_context=global_game_session.to_dict(),
            active_room=request.active_room,
            player_position=request.player_position,
            camera_heading=request.camera_heading,
            session_id=request.session_id or "default",
            selected_object_id=request.selected_object_id,
        ):
            ev_name = evt.get("event", "message")
            ev_data = evt.get("data", {})
            if ev_name == "done":
                final_result = ev_data
                response_model = _finalize_agent_turn(request, agent_id, final_result, t_start)
                payload = response_model.model_dump() if hasattr(response_model, "model_dump") else response_model.dict()
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
            else:
                yield f"event: {ev_name}\ndata: {json.dumps(ev_data)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/clear")
def clear_agent_history(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Clears session-scoped AI investigation assistant conversation memory."""
    sid = (payload or {}).get("session_id", "default")
    global_llm_provider.memory.clear_session(sid)
    global_investigation_agent.conversation_history.clear()
    return {"success": True, "session_id": sid, "message": "Conversation history cleared."}


@router.get("/llm/status")
def get_llm_status():
    """Returns the current LLM provider status (Gemini API vs. offline fallback)."""
    return global_llm_provider.get_status()


@router.post("/llm/mode")
def set_llm_mode(payload: dict):
    """Switches the active LLM mode using server-side environment configuration only."""
    mode = payload.get("mode", "from_scratch")
    return global_llm_provider.set_mode(mode)

@router.get("/telemetry/latest")
def get_latest_telemetry():
    """Returns the most recent correlated telemetry across CV, CG, and LLM."""
    return global_telemetry.get_latest()

@router.get("/telemetry/{correlation_id}")
def get_telemetry_by_id(correlation_id: str):
    """Returns telemetry record for a specific correlation ID."""
    record = global_telemetry.get_by_correlation_id(correlation_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Correlation ID '{correlation_id}' not found.")
    return record

@router.get("/game/state")
def get_game_state():
    """Returns the current Spatial Quest mission, score, and companion drone."""
    if not global_scene_graph.get_all_objects():
        seed_facility_scene()
    return global_game_session.to_dict()

@router.post("/game/start")
def start_game_mission(request: GameStartRequest):
    """Starts or resumes a mission if it is unlocked."""
    try:
        return global_game_session.start_mission(request.mission_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/game/select-mystery")
def select_mystery_mission(request: GameStartRequest):
    """Directly selects and unlocks any available mystery template."""
    return global_game_session.select_mystery(request.mission_id)

@router.post("/game/reset")
def reset_game_session(request: GameResetRequest):
    """Resets the game session with a reproducible seed and resets lock/scene state."""
    seed = request.seed if request.seed is not None else 42
    global_game_session.set_seed(seed)
    global_game_session.score = 0
    global_game_session.unlocked_through_index = 10
    global_game_session.inventory.clear()
    global_game_session.collected_clues.clear()
    global_tool_registry._environment_state["door_3"] = "locked"
    global_tool_registry._clues_state["clue_safe_01"]["status"] = "locked"
    seed_facility_scene()
    global_game_session.start_mission("mission_a")
    return global_game_session.to_dict()

@router.post("/game/restart")
def restart_current_mission():
    """Restarts the active mission without wiping overall session seed."""
    return global_game_session.restart_current_mission()

@router.post("/game/hub")
def return_player_to_hub():
    """Teleports player focus back to Reception / Central Hub."""
    return global_game_session.return_to_hub()

@router.post("/game/agent")
def select_game_agent(request: GameAgentRequest):
    """Switches the active companion drone persona (same from-scratch Transformer)."""
    try:
        companion = global_game_session.select_agent(request.agent_id)
        state = global_game_session.to_dict()
        state["companion"] = companion
        return state
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/game/sync")
def sync_player_position_and_room(request: PlayerSyncRequest):
    """Synchronizes real-time 3D player coordinates and active room with backend game session."""
    global_game_session.update_player_state(request.player_position, request.active_room)
    return {
        "status": "synchronized",
        "player_position": global_game_session.player_position,
        "current_room": global_game_session.current_room
    }

@router.post("/unity/action-result")
def record_unity_action_result(result: UnityActionResult):
    """Receives execution confirmation callback from Unity avatar controller."""
    return {
        "status": "acknowledged",
        "action_id": result.action_id,
        "success": result.success
    }

@router.get("/cv/benchmark_sample")
def get_cv_benchmark_sample(index: int = 0):
    """Loads a real SUN RGB-D benchmark sample with RGB image, depth colormap, and real SpatialDetector CNN inference."""
    rgb_path = f"data/sunrgbd/rgb/sunrgbd_{index:05d}.jpg"
    depth_path = f"data/sunrgbd/depth/sunrgbd_{index:05d}.npy"
    if not os.path.exists(rgb_path):
        rgb_path = "data/sunrgbd/rgb/sunrgbd_00000.jpg"
        depth_path = "data/sunrgbd/depth/sunrgbd_00000.npy"
    if not os.path.exists(rgb_path):
        raise HTTPException(status_code=404, detail="SUN RGB-D sample files not found.")

    with open(rgb_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Generate depth colormap
    depth_b64 = ""
    if os.path.exists(depth_path):
        try:
            depth_arr = np.load(depth_path)
            v = depth_arr[depth_arr > 0]
            d_min = float(np.percentile(v, 5)) if len(v) else 0.5
            d_max = float(np.percentile(v, 95)) if len(v) else 6.0
            norm = np.clip((depth_arr - d_min) / max(d_max - d_min, 1e-4), 0.0, 1.0)
            r = np.clip(np.sin(norm * np.pi) * 255 + (norm > 0.5) * 120, 0, 255).astype(np.uint8)
            g = np.clip(np.sin(norm * np.pi * 0.8 + 0.3) * 230, 0, 255).astype(np.uint8)
            b = np.clip((1.0 - norm) * 240, 0, 255).astype(np.uint8)
            rgb_depth = np.stack([r, g, b], axis=-1)
            depth_img = Image.fromarray(rgb_depth).resize((320, 240))
            buf = io.BytesIO()
            depth_img.save(buf, format="JPEG", quality=85)
            depth_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            depth_b64 = ""

    loaded = _get_loaded_detector()
    detections = []
    t_start = time.time()
    if loaded is not None:
        model, device = loaded
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((640, 480))
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        x_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            grid_pred = model(x_tensor)
            decoded_batches = model.decode_predictions(grid_pred, conf_threshold=0.08, nms_iou_threshold=0.45)
            preds = decoded_batches[0] if decoded_batches else []
            for idx, p in enumerate(preds):
                u_c = (p["xmin"] + p["xmax"]) / 2.0
                v_c = (p["ymin"] + p["ymax"]) / 2.0
                depth_val = 2.4
                cam_point = backproject_pixel_to_camera_point(u_c, v_c, depth_val, 525.0, 525.0, 320.0, 240.0)
                detections.append({
                    "object_id": f"{p['class_name']}_{idx+1:02d}",
                    "class_name": p["class_name"],
                    "confidence": round(p["confidence"], 3),
                    "bbox": [round(p["xmin"], 1), round(p["ymin"], 1), round(p["xmax"], 1), round(p["ymax"], 1)],
                    "depth_m": round(depth_val, 2),
                    "estimated_3d_position": [round(c, 2) for c in cam_point]
                })

    inf_time_ms = round((time.time() - t_start) * 1000.0, 2)

    return {
        "sample_index": index,
        "source": f"SUN RGB-D Dataset (Song et al., CVPR 2015) - sample sunrgbd_{index:05d}",
        "resolution": [640, 480],
        "image_base64": img_b64,
        "depth_colormap_base64": depth_b64,
        "detections": detections,
        "inference_time_ms": inf_time_ms,
        "tensor_shapes": {
            "input": [1, 3, 480, 640],
            "stage1_conv": [1, 32, 240, 320],
            "stage2_conv": [1, 64, 120, 160],
            "stage3_conv": [1, 128, 60, 80],
            "stage4_conv": [1, 256, 30, 40],
            "stage5_conv": [1, 256, 15, 20],
            "head_prediction_grid": [1, 15, 20, 11]
        },
        "camera_intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0},
        "benchmark": load_cv_benchmarks()
    }
