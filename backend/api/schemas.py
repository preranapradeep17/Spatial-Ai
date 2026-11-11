"""Pydantic API schemas for SpatialAI."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time

class BoundingBox2D(BaseModel):
    """Normalized or absolute 2D bounding box."""
    xmin: float = Field(..., description="Top-left X coordinate")
    ymin: float = Field(..., description="Top-left Y coordinate")
    xmax: float = Field(..., description="Bottom-right X coordinate")
    ymax: float = Field(..., description="Bottom-right Y coordinate")

class DetectedObject(BaseModel):
    """An object detected by the computer vision model or sensor."""
    object_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox2D
    estimated_3d_position: Optional[List[float]] = None  # [X, Y, Z] in world space
    depth_estimate: Optional[float] = None               # In meters
    depth_source: Optional[str] = None                   # "measured_webgl_depth", "measured_rgbd_sensor", "estimated_bbox_height", "seeded_layout"
    timestamp: float = Field(default_factory=time.time)
    source: Optional[str] = None                         # e.g. "facility_layout", "custom_detector", "test_mock_pipeline"

class CameraIntrinsics(BaseModel):
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

class CameraPose(BaseModel):
    position: List[float] = Field(default_factory=lambda: [0.0, 1.5, -3.0])
    rotation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # Quaternion [x, y, z, w]

class PerceptionRequest(BaseModel):
    """Request submitted by Unity or Three.js containing RGB and optional Depth information."""
    image_base64: Optional[str] = None
    depth_base64: Optional[str] = None
    depth_encoding: Optional[str] = None  # "webgl_rg16_mm" or standard grayscale/float png
    camera_intrinsics: Optional[CameraIntrinsics] = None
    camera_pose: Optional[CameraPose] = None
    coordinate_system: str = "unity"  # "unity" (+Z forward) or "threejs" (-Z forward)
    mock_mode: bool = False  # For isolated testing when models are training

class PerceptionResponse(BaseModel):
    detections: List[DetectedObject]
    timestamp: float = Field(default_factory=time.time)
    inference_time_ms: float
    source: str  # "custom_detector" or "test_mock"

class SceneGraphNode(BaseModel):
    id: str
    class_name: str
    position: List[float]  # [X, Y, Z]
    confidence: float
    depth: Optional[float] = None
    depth_source: Optional[str] = None
    source: Optional[str] = None

class SceneGraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str  # "left_of", "right_of", "in_front_of", "behind", "on", "near"
    distance_meters: Optional[float] = None

class SceneGraphResponse(BaseModel):
    nodes: List[SceneGraphNode]
    edges: List[SceneGraphEdge]
    last_updated: float
    source: str

class ToolCall(BaseModel):
    type: str = "tool_call"
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class AgentQueryRequest(BaseModel):
    query: str
    context_override: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    mission_id: Optional[str] = None
    player_position: Optional[List[float]] = None
    active_room: Optional[str] = None
    camera_heading: Optional[float] = None
    session_id: Optional[str] = "default"
    selected_object_id: Optional[str] = None

class AgentQueryResponse(BaseModel):
    query: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_response: str
    unity_action: Optional[Dict[str, Any]] = None
    completed: bool = True
    game_state: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    latency_ms: Optional[float] = None
    validation_status: Optional[str] = None
    model_output: Optional[str] = None
    used_fallback: bool = False
    llm_mode: Optional[str] = "gemini_api"
    model_name: Optional[str] = None
    session_id: Optional[str] = "default"
    error: Optional[str] = None

class PlayerSyncRequest(BaseModel):
    player_position: List[float] = Field(..., description="Current [X, Z] or [X, Y, Z] world position")
    active_room: Optional[str] = Field(None, description="Current room name or zone label")

class GameStartRequest(BaseModel):
    mission_id: str

class GameResetRequest(BaseModel):
    seed: Optional[int] = 42

class GameAgentRequest(BaseModel):
    agent_id: str

class UnityActionResult(BaseModel):
    action_id: str
    action_type: str
    success: bool
    message: Optional[str] = None
    avatar_position: Optional[List[float]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    device_info: Dict[str, Any]
    vision_model_loaded: bool
    language_model_loaded: bool
    scene_objects_count: int
    uptime_seconds: float
