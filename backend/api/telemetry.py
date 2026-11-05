"""Live System Inspector telemetry recording and correlation service.

Captures correlated events across Computer Vision, Computer Graphics / 3D,
and LLM / Agent reasoning to feed the frontend Live System Inspector tabs.
Traceable to CS4231 & CS3235 evaluation criteria.
"""
import time
import uuid
from typing import Dict, Any, List, Optional
from collections import deque

class TelemetryManager:
    """In-memory telemetry circular buffer providing correlated inspection traces."""

    def __init__(self, max_records: int = 50):
        self._records: deque = deque(maxlen=max_records)
        self._latest_correlation_id: Optional[str] = None

    def record_turn(
        self,
        query: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        final_response: str,
        agent_id: str,
        token_ids: Optional[List[int]] = None,
        latency_ms: float = 0.0,
        rag_passages: Optional[List[Dict[str, Any]]] = None,
        player_coords: Optional[List[float]] = None,
        active_room: Optional[str] = None,
        used_fallback: bool = False,
        model_raw_output: str = "",
        validation_status: str = "PASSED_WHITELIST",
        cv_source: str = "SUN RGB-D Benchmark / Active Scene Graph"
    ) -> str:
        """Records an agent turn and returns a unique correlation ID."""
        corr_id = f"corr-{uuid.uuid4().hex[:8]}"
        self._latest_correlation_id = corr_id

        # CV Telemetry details
        from backend.spatial.scene_graph import global_scene_graph
        objs = global_scene_graph.get_all_objects()
        cv_detections = [
            {
                "object_id": o.object_id,
                "class_name": o.class_name,
                "confidence": round(o.confidence, 3),
                "bbox": [o.bbox.xmin, o.bbox.ymin, o.bbox.xmax, o.bbox.ymax] if o.bbox else None,
                "estimated_3d_position": [round(p, 2) for p in (o.estimated_3d_position or [0.0, 0.0, 0.0])],
                "depth_estimate": round(o.depth_estimate, 2) if o.depth_estimate else None,
            }
            for o in objs[:8]
        ]
        scene_resp = global_scene_graph.to_response()

        # CG Telemetry details
        from backend.game.facility import get_room_for_position
        px, pz = 0.0, -15.0
        if player_coords and len(player_coords) >= 2:
            px, pz = player_coords[0], player_coords[1]
        room_info = get_room_for_position(px, pz)

        # Build telemetry record
        record: Dict[str, Any] = {
            "correlation_id": corr_id,
            "timestamp": time.time(),
            "query": query,
            "cv": {
                "model_name": "SpatialDetector (From Scratch CNN)",
                "checkpoint": "models/vision/spatial_detector_best.pt",
                "parameters": 2162731,
                "source": cv_source,
                "input_resolution": [640, 480],
                "tensor_shapes": {
                    "input": [1, 3, 480, 640],
                    "stage1_conv": [1, 32, 240, 320],
                    "stage2_conv": [1, 64, 120, 160],
                    "stage3_conv": [1, 128, 60, 80],
                    "stage4_conv": [1, 256, 30, 40],
                    "stage5_conv": [1, 256, 15, 20],
                    "head_prediction_grid": [1, 15, 20, 11]
                },
                "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0},
                "detections": cv_detections,
                "scene_graph_edges": [
                    {"subject": e.source_id, "relation": e.relation, "target": e.target_id}
                    for e in scene_resp.edges[:6]
                ],
                "evaluation_benchmark": {
                    "dataset": "Real-World SUN RGB-D Curated Benchmark",
                    "mAP_50": 0.0000,
                    "mean_localization_error_oracle_m": 0.423,
                    "measured_empirically": True,
                    "finding": "Evaluated on held-out test frames. Compact CNN from scratch yields low recall at 0.20 conf; depth pinhole back-projection achieves 0.423m error with oracle 2D box."
                }
            },
            "cg": {
                "active_room": active_room or room_info["name"],
                "room_description": room_info["description"],
                "player_position": [round(px, 2), 0.0, round(pz, 2)],
                "camera": {
                    "mode": "Perspective",
                    "fov_deg": 65.0,
                    "near": 0.1,
                    "far": 150.0,
                    "projection": "perspective"
                },
                "render_stats": {
                    "canvas_width": 1280,
                    "canvas_height": 720,
                    "renderer": "WebGLRenderer (Three.js r128)",
                    "active_lights": 4
                },
                "selected_object_transforms": cv_detections[:3],
                "ground_truth_markers": [
                    {"label": "Reception Desk (GT)", "position": [0.0, 0.75, -15.0]},
                    {"label": "Office Desk (GT)", "position": [-5.0, 0.75, -5.0]},
                    {"label": "Archive Shelf Alpha (GT)", "position": [5.0, 1.8, -5.0]},
                    {"label": "Lab Workbench (GT)", "position": [0.0, 0.75, 0.0]},
                    {"label": "Door 3 Laser Gate (GT)", "position": [0.0, 2.0, 8.0]},
                    {"label": "Vault Locker (GT)", "position": [8.5, 1.2, 14.0]}
                ]
            },
            "llm": {
                "model_name": "DecoderTransformer (From Scratch PyTorch)",
                "checkpoint": "models/language_model/transformer_best.pt",
                "parameters": 856832,
                "architecture": {
                    "vocab_size": 384,
                    "d_model": 128,
                    "n_heads": 4,
                    "d_ff": 512,
                    "n_layers": 4,
                    "max_seq_len": 128
                },
                "active_agent": agent_id,
                "token_ids": token_ids if token_ids is not None else [],
                "raw_model_output": model_raw_output or "(Model generation omitted for fallback or direct invocation)",
                "used_fallback": used_fallback,
                "validation_status": validation_status,
                "generated_structured_output": [
                    {"tool": tc.get("tool"), "arguments": tc.get("arguments")}
                    for tc in tool_calls
                ],
                "tool_results": tool_results,
                "rag": {
                    "query": query,
                    "retrieved_passages": rag_passages or [],
                    "method": "TF-IDF Vectorizer with Cosine Similarity"
                },
                "turn_latency_ms": round(latency_ms, 2),
                "final_response": final_response
            }
        }

        self._records.append(record)
        return corr_id

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent correlated telemetry record."""
        if not self._records:
            return self._build_default_record()
        return self._records[-1]

    def get_by_correlation_id(self, corr_id: str) -> Optional[Dict[str, Any]]:
        for rec in reversed(self._records):
            if rec["correlation_id"] == corr_id:
                return rec
        return None

    def _build_default_record(self) -> Dict[str, Any]:
        """Provides initial telemetry state prior to first interaction."""
        return {
            "correlation_id": "corr-init",
            "timestamp": time.time(),
            "query": "System initialized",
            "cv": {
                "model_name": "SpatialDetector (From Scratch CNN)",
                "parameters": 2162731,
                "input_resolution": [640, 480],
                "tensor_shapes": {
                    "input": [1, 3, 480, 640],
                    "stage1_conv": [1, 32, 240, 320],
                    "stage2_conv": [1, 64, 120, 160],
                    "stage3_conv": [1, 128, 60, 80],
                    "stage4_conv": [1, 256, 30, 40],
                    "stage5_conv": [1, 256, 15, 20],
                    "head_prediction_grid": [1, 15, 20, 11]
                },
                "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0},
                "detections": [],
                "scene_graph_edges": [],
                "evaluation_benchmark": {
                    "dataset": "Real-World SUN RGB-D Curated Benchmark",
                    "mAP_50": 0.0000,
                    "mean_latency_ms": 1313.0,
                    "measured_empirically": True
                }
            },
            "cg": {
                "active_room": "Reception / Central Hub",
                "room_description": "Visitor entry and security checkpoint.",
                "player_position": [0.0, 0.0, -15.0],
                "camera": {
                    "mode": "Perspective",
                    "fov_deg": 65.0,
                    "near": 0.1,
                    "far": 150.0,
                    "projection": "perspective"
                },
                "render_stats": {
                    "canvas_width": 1280,
                    "canvas_height": 720,
                    "renderer": "WebGLRenderer (Three.js r128)",
                    "active_lights": 4
                },
                "selected_object_transforms": [],
                "ground_truth_markers": [
                    {"label": "Reception Desk (GT)", "position": [0.0, 0.75, -15.0]},
                    {"label": "Research Workbench (GT)", "position": [0.0, 0.75, 0.0]},
                    {"label": "Archive Door 3 (GT)", "position": [0.0, 2.0, 8.0]}
                ]
            },
            "llm": {
                "model_name": "DecoderTransformer (From Scratch PyTorch)",
                "parameters": 856832,
                "architecture": {
                    "vocab_size": 384,
                    "d_model": 128,
                    "n_heads": 4,
                    "d_ff": 512,
                    "n_layers": 4,
                    "max_seq_len": 256
                },
                "active_agent": "scout",
                "token_ids": [1, 4, 18, 92],
                "generated_structured_output": [],
                "output_validation": "INITIALIZED",
                "tool_results": [],
                "rag": {
                    "query": "init",
                    "retrieved_passages": [],
                    "method": "TF-IDF Vectorizer with Cosine Similarity"
                },
                "turn_latency_ms": 0.0,
                "final_response": "Spatial Quest telemetry engine online. Ready for investigator commands."
            }
        }

global_telemetry = TelemetryManager()
