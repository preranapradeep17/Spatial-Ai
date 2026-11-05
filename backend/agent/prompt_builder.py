"""Canonical Structured Prompt Builder for Spatial Quest.

Unifies training prompt generation and runtime inference prompt construction.
Ensures identical structural formatting, field order, and token conditioning
across both datasets and interactive agent execution without distribution mismatch.
Traceable to CS3235 (Working with LLMs - Input Representation & Conditioning).
"""
from typing import List, Dict, Any, Optional, Tuple
import math


def format_visible_objects_block(
    objects: List[Any],
    provenance: str = "Scene Graph Registry",
    max_objects: int = 4
) -> str:
    """Formats visible or registered scene objects into consistent line items."""
    if not objects:
        return "- None currently visible"

    lines = []
    for obj in objects[:max_objects]:
        if hasattr(obj, "class_name") and hasattr(obj, "estimated_3d_position"):
            c_name = obj.class_name
            pos = obj.estimated_3d_position or [0.0, 0.0, 0.0]
        elif isinstance(obj, dict):
            c_name = obj.get("class_name", "object")
            pos = obj.get("position", [0.0, 0.0, 0.0])
        else:
            continue

        prov_str = f" ({provenance})" if provenance else ""
        lines.append(f"- {c_name}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]{prov_str}")

    return "\n".join(lines) if lines else "- None currently visible"


def format_spatial_relations_block(
    relations: List[Any],
    max_relations: int = 3
) -> str:
    """Formats pairwise geometric relations into consistent line items."""
    if not relations:
        return "- No immediate adjacency"

    lines = []
    for rel in relations[:max_relations]:
        if isinstance(rel, str):
            lines.append(f"- {rel}" if not rel.startswith("- ") else rel)
        elif isinstance(rel, dict):
            sub = rel.get("subject", "object")
            r_type = rel.get("relation", "near")
            obj = rel.get("object", "target")
            lines.append(f"- {sub} is {r_type} {obj}")
        elif hasattr(rel, "source_id") and hasattr(rel, "target_id"):
            s_name = rel.source_id.split("_")[0]
            t_name = rel.target_id.split("_")[0]
            lines.append(f"- {s_name} is {rel.relation} {t_name}")

    return "\n".join(lines) if lines else "- No immediate adjacency"


def format_conversation_history_block(
    history: Optional[List[Any]] = None,
    max_turns: int = 2
) -> str:
    """Formats recent conversational turns for multi-turn anaphora resolution."""
    if not history:
        return ""

    lines = []
    for turn in history[-max_turns:]:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            u, a = turn[0], str(turn[1]).split("\n")[0][:60]
            lines.append(f"User: {u}\nAssistant: {a}")
        elif isinstance(turn, dict):
            u = turn.get("user", "")
            a = str(turn.get("assistant", "")).split("\n")[0][:60]
            lines.append(f"User: {u}\nAssistant: {a}")

    if not lines:
        return ""
    return "\nHISTORY:\n" + "\n".join(lines)


def build_canonical_prompt(
    query: str,
    agent_mode: str = "scout",
    room_name: str = "Reception / Central Hub",
    player_position: Optional[List[float]] = None,
    visible_objects_str: str = "- None currently visible",
    spatial_relations_str: str = "- No immediate adjacency",
    active_mission_str: str = "Mystery A: Missing Research Key (Objectives: 0/3)",
    inventory_str: str = "None",
    history_str: str = "",
    prompt_format: str = "canonical"
) -> str:
    """Builds a structured prompt string for training and inference.

    Args:
        query: The user's natural language inquiry.
        agent_mode: Active companion persona ('scout', 'navigator', 'archivist').
        room_name: Active facility zone.
        player_position: Live [X, Y, Z] or [X, Z] observer stance.
        visible_objects_str: Pre-formatted visible objects block.
        spatial_relations_str: Pre-formatted spatial relationships block.
        active_mission_str: Mission title and objective counter.
        inventory_str: Comma-separated collected items or 'None'.
        history_str: Formatted conversational history block.
        prompt_format: 'canonical' (unified), 'original_training', or 'legacy_runtime'.

    Returns:
        Exact prompt string ending with 'ASSISTANT:'.
    """
    mode_clean = agent_mode.strip().upper()
    px, py, pz = 0.0, 0.0, -15.0
    if player_position:
        if len(player_position) == 2:
            px, pz = float(player_position[0]), float(player_position[1])
            py = 0.0
        elif len(player_position) >= 3:
            px, py, pz = float(player_position[0]), float(player_position[1]), float(player_position[2])

    # Strip any trailing 'ASSISTANT:' from history if accidentally embedded
    hist_clean = history_str.strip()
    if hist_clean and not hist_clean.startswith("\nHISTORY:"):
        hist_clean = "\n" + hist_clean

    if prompt_format == "original_training":
        # Format matching initial scripts/generate_lm_dataset.py pre-training
        return (
            f"SYSTEM: You are an intelligent investigation drone.\n"
            f"ROOM: {room_name}\n"
            f"VISIBLE OBJECTS:\n{visible_objects_str}\n"
            f"SPATIAL RELATIONS:\n{spatial_relations_str}"
            f"{hist_clean}\n"
            f"USER: {query}\n"
            f"ASSISTANT:"
        )

    elif prompt_format == "legacy_runtime":
        # Format matching initial backend/agent/investigation_agent.py runtime with long persona strings
        persona_map = {
            "SCOUT": "You are Scout, the Computer Vision specialist companion drone. Your role is visual survey, 3D object detection in camera FOV, bounding box verification, and depth estimation. You prioritize 'get_visible_objects', 'locate_object', 'inspect_object_closely', and 'explain_uncertainty'. Do not invent detections if objects are not visible.",
            "NAVIGATOR": "You are Navigator, the 3D Spatial and Navigation specialist companion drone. Your role is spatial localization, 3D geometry, turn-by-turn waypoint routing, metric Euclidean distance, and spatial relations. You prioritize 'get_current_location', 'navigate_to', 'find_path', 'calculate_distance', 'query_spatial_relation', and 'find_nearest_target'.",
            "ARCHIVIST": "You are Archivist, the RAG and Facility Records specialist companion drone. Your role is document retrieval, incident reports, safety manuals, clue investigation, and containment logs. You prioritize 'search_documents', 'search_safety_manual', 'inspect_clue', and 'get_mission_objectives'."
        }
        p_str = persona_map.get(mode_clean, persona_map["SCOUT"])
        return (
            f"SYSTEM: {p_str}\n"
            f"AGENT_MODE: {mode_clean}\n"
            f"ROOM: {room_name}\n"
            f"VISIBLE OBJECTS:\n{visible_objects_str}\n"
            f"SPATIAL RELATIONS:\n{spatial_relations_str}"
            f"{hist_clean}\n"
            f"USER: {query}\n"
            f"ASSISTANT:"
        )

    # Canonical unified prompt format
    return (
        f"SYSTEM: You are an autonomous facility companion drone.\n"
        f"AGENT_MODE: {mode_clean}\n"
        f"ROOM: {room_name}\n"
        f"PLAYER_POSITION: [{px:.1f}, {py:.1f}, {pz:.1f}]\n"
        f"VISIBLE OBJECTS:\n{visible_objects_str}\n"
        f"SPATIAL RELATIONS:\n{spatial_relations_str}\n"
        f"ACTIVE MISSION: {active_mission_str}\n"
        f"INVENTORY: {inventory_str}"
        f"{hist_clean}\n"
        f"USER: {query}\n"
        f"ASSISTANT:"
    )

