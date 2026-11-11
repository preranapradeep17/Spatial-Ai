"""Pydantic validation schemas for all tools in the SpatialAI tool registry.

Ensures strict argument validation, type checking, and boundary constraints
traceable to CS3235 (Working with LLMs - Tool Calling, Function Execution & Safety).
"""
from typing import Optional, List, Dict, Any, Type
from pydantic import BaseModel, Field, field_validator


class BaseToolArgs(BaseModel):
    """Base class providing automatic whitespace stripping for all string fields."""
    model_config = {"extra": "ignore"}

    @classmethod
    def clean_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Strips whitespace from dictionary keys and string values."""
        cleaned = {}
        for k, v in data.items():
            k_clean = str(k).strip()
            if isinstance(v, str):
                cleaned[k_clean] = v.strip()
            elif isinstance(v, dict):
                cleaned[k_clean] = {str(k2).strip(): (v2.strip() if isinstance(v2, str) else v2) for k2, v2 in v.items()}
            else:
                cleaned[k_clean] = v
        return cleaned


class GetVisibleObjectsArgs(BaseToolArgs):
    room_name: Optional[str] = Field(None, description="Optional facility room name filter")
    player_position: Optional[List[float]] = Field(None, description="Optional player/camera 2D/3D coordinates")
    camera_heading: Optional[float] = Field(None, description="Optional camera/player heading in degrees")
    filter_fov: Optional[bool] = Field(False, description="Filter objects strictly within camera field of view")
    live_only: Optional[bool] = Field(False, description="When true, return ONLY live SpatialDetector CNN detections (custom_detector) and exclude pre-seeded facility layout objects")


class LocateObjectArgs(BaseToolArgs):
    object_name: str = Field(..., description="Name or class of object to locate")


class QuerySpatialRelationArgs(BaseToolArgs):
    object_name: str = Field(..., description="Primary reference object name (e.g., table, chair)")
    relation: str = Field("all", description="Spatial relation: left_of, right_of, near, above, behind, in_front_of, on, or all")
    target_object_name: Optional[str] = Field(None, description="Optional second object name when asking the relationship between two objects")


class CalculateDistanceArgs(BaseToolArgs):
    object_a: str = Field(..., description="First object name or class")
    object_b: str = Field(..., description="Second object name or class")


class FindNearestTargetArgs(BaseToolArgs):
    target_class: Optional[str] = Field("any", description="Target object class to locate nearest instance of (default 'any')")
    player_position: Optional[List[float]] = Field(None, description="Current [X, Z] or [X, Y, Z] player position")
    reference_object: Optional[str] = Field(None, description="Optional reference object to measure distance from (defaults to player position)")


class FindPathArgs(BaseToolArgs):
    destination: str = Field(..., description="Destination object or room")
    player_position: Optional[List[float]] = Field(None, description="Current [X, Z] or [X, Y, Z] player position")
    current_room: Optional[str] = Field(None, description="Current active room name")


class NavigateToArgs(BaseToolArgs):
    destination: str = Field(..., description="Destination object, landmark, or room")
    player_position: Optional[List[float]] = Field(None, description="Current player position")
    current_room: Optional[str] = Field(None, description="Current active room name")


class InspectClueArgs(BaseToolArgs):
    clue_id: str = Field(..., description="Clue identifier (e.g., clue_safe_01, clue_pda_02)")


class SearchDocumentsArgs(BaseToolArgs):
    query: str = Field(..., description="Natural language search query for facility documents")


class InteractWithObjectArgs(BaseToolArgs):
    object_id: str = Field(..., description="Target interactive object ID (e.g., door_3, clue_safe_01, cabinet)")
    passcode: Optional[str] = Field(None, description="Optional 4-digit security override code (e.g., 7391)")


class SearchSafetyManualArgs(BaseToolArgs):
    query: str = Field(..., description="Emergency or safety query to retrieve from manual")


class FindNearestExitArgs(BaseToolArgs):
    player_position: Optional[List[float]] = Field(None, description="Optional player position")


class InspectObjectCloselyArgs(BaseToolArgs):
    object_name: str = Field(..., description="Name of object for close-up visual inspection")


class RequestHintArgs(BaseToolArgs):
    topic: Optional[str] = Field("mission_progression", description="Optional hint topic or mission context")


class GetInventoryArgs(BaseToolArgs):
    pass


class GetMissionObjectivesArgs(BaseToolArgs):
    pass


class ExplainUncertaintyArgs(BaseToolArgs):
    query: Optional[str] = Field(None, description="Context of uncertainty")
    reason: Optional[str] = Field(None, description="Reason for uncertainty (e.g. not_detected, out_of_view)")


class GetGameGuideArgs(BaseToolArgs):
    topic: Optional[str] = Field("all", description="Guide section: all, introduction, commands, navigation, controls")


class ExplainUnreachableArgs(BaseToolArgs):
    query: Optional[str] = Field(None, description="Unreachable target query or context")
    destination: Optional[str] = Field(None, description="Target destination")


class GetCurrentLocationArgs(BaseToolArgs):
    player_position: Optional[List[float]] = Field(None, description="Player 3D coordinates")
    active_room: Optional[str] = Field(None, description="Active room name override")


TOOL_SCHEMA_MAP: Dict[str, Type[BaseToolArgs]] = {
    "get_visible_objects": GetVisibleObjectsArgs,
    "locate_object": LocateObjectArgs,
    "query_spatial_relation": QuerySpatialRelationArgs,
    "calculate_distance": CalculateDistanceArgs,
    "find_nearest_target": FindNearestTargetArgs,
    "find_path": FindPathArgs,
    "navigate_to": NavigateToArgs,
    "inspect_clue": InspectClueArgs,
    "search_documents": SearchDocumentsArgs,
    "interact_with_object": InteractWithObjectArgs,
    "search_safety_manual": SearchSafetyManualArgs,
    "find_nearest_exit": FindNearestExitArgs,
    "inspect_object_closely": InspectObjectCloselyArgs,
    "request_hint": RequestHintArgs,
    "get_inventory": GetInventoryArgs,
    "get_mission_objectives": GetMissionObjectivesArgs,
    "explain_uncertainty": ExplainUncertaintyArgs,
    "get_game_guide": GetGameGuideArgs,
    "explain_unreachable": ExplainUnreachableArgs,
    "get_current_location": GetCurrentLocationArgs,
}


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Validates raw tool arguments against the tool's Pydantic schema.
    
    Returns:
        (is_valid, validated_args_dict, error_message)
    """
    if tool_name not in TOOL_SCHEMA_MAP:
        return False, arguments, f"No schema registered for tool '{tool_name}'."
    
    schema_cls = TOOL_SCHEMA_MAP[tool_name]
    cleaned = schema_cls.clean_dict(arguments)
    try:
        model_inst = schema_cls(**cleaned)
        return True, model_inst.model_dump(exclude_unset=False), None
    except Exception as ex:
        return False, cleaned, str(ex)

