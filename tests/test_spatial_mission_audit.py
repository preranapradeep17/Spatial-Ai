"""Tests for Spatial Quest spatial reasoning and mission progression audit.

Covers:
1. Nearest-object selection (live player coordinates, Euclidean distance, room localization)
2. Distance calculations (geometric formula verification, candidate distance sorting)
3. Object-specific spatial relations (queries relative to specified objects like desk)
4. Missing objects vs registered objects (zero false uncertainty on registered objects)
5. Empty scene graphs (graceful zero-object handling without crashing)
6. Mission completion validation: blocked with missing inventory items
7. Mission completion validation: fulfilled when all objectives and required items exist
8. Consistency between mission progress, inventory state, and agent responses
"""
import pytest
import math
from backend.spatial.scene_graph import SceneGraphManager, global_scene_graph
from backend.tools.registry import global_tool_registry
from backend.game.facility import seed_facility_scene
from backend.game.missions import GameSession, global_game_session
from backend.agent.investigation_agent import global_investigation_agent
from backend.api.schemas import DetectedObject, BoundingBox2D


@pytest.fixture(autouse=True)
def setup_facility():
    global_game_session.return_to_hub()
    global_game_session.inventory.clear()
    global_game_session.collected_clues.clear()
    global_game_session.start_mission("mission_a")
    seed_facility_scene(source="facility_layout")


def test_nearest_object_selection_from_live_stance():
    """Live player coordinates must be used to calculate Euclidean distance and select nearest object."""
    # Stance 1: Player in Reception at [0.0, 0.0, -15.0]
    res_reception = global_tool_registry.execute("find_nearest_target", {
        "target_class": "any",
        "player_position": [0.0, 0.0, -15.0]
    })
    assert res_reception["success"] is True
    data_rec = res_reception["result"]
    assert data_rec["found"] is True
    assert data_rec["room"] == "Reception / Central Hub"
    assert data_rec["in_current_room"] is True
    assert data_rec["distance_meters"] < 2.5
    # Should identify table or chair in reception
    assert data_rec["class_name"] in ("table", "chair", "sofa")

    # Stance 2: Player moved to Research Office at [6.0, 0.0, -5.0]
    res_office = global_tool_registry.execute("find_nearest_target", {
        "target_class": "any",
        "player_position": [6.0, 0.0, -5.0]
    })
    assert res_office["success"] is True
    data_off = res_office["result"]
    assert data_off["found"] is True
    assert data_off["room"] == "Research Office"
    assert data_off["in_current_room"] is True
    assert data_off["distance_meters"] < 1.5
    assert data_off["class_name"] in ("table", "chair")


def test_candidate_distance_sorting_and_euclidean_math():
    """All candidate objects must be sorted ascending by exact 3D Euclidean distance."""
    ref = [0.0, 0.0, 0.0]
    ranked = global_scene_graph.get_objects_by_distance(reference_pos=ref, target_class="any")
    assert len(ranked) >= 5

    # Check ascending order
    distances = [d for _, d in ranked]
    assert distances == sorted(distances)

    # Check exact Euclidean formula
    for obj, d in ranked:
        p = obj.estimated_3d_position
        expected = round(math.sqrt(p[0]**2 + p[1]**2 + p[2]**2), 2)
        assert abs(d - expected) < 0.02


def test_find_nearest_table_intent_scout_mode():
    """'Find the nearest table' must route to find_nearest_target, NOT explain_uncertainty."""
    res = global_investigation_agent.run(
        "Find the nearest table.",
        agent_id="scout",
        player_position=[0.0, 0.0, -15.0],
        active_room="Reception / Central Hub"
    )
    assert len(res["tool_calls"]) > 0
    first_tool = res["tool_calls"][0]["tool"]
    assert first_tool == "find_nearest_target"
    assert res["tool_calls"][0]["arguments"]["target_class"] == "table"
    assert "closest table" in res["final_response"].lower()
    assert "reception" in res["final_response"].lower()
    assert "uncertainty" not in res["final_response"].lower()


def test_which_object_is_closest_to_me_query():
    """'Which object is closest to me?' and 'What is the nearest object?' must resolve without uncertainty."""
    res1 = global_investigation_agent.run(
        "Which object is closest to me?",
        agent_id="navigator",
        player_position=[0.0, 0.0, -15.0],
        active_room="Reception / Central Hub"
    )
    assert res1["tool_calls"][0]["tool"] == "find_nearest_target"
    assert "closest" in res1["final_response"].lower()

    res2 = global_investigation_agent.run(
        "What is the nearest object?",
        agent_id="navigator",
        player_position=[0.0, 0.0, -15.0],
        active_room="Reception / Central Hub"
    )
    assert res2["tool_calls"][0]["tool"] == "find_nearest_target"
    assert "closest" in res2["final_response"].lower()


def test_object_specific_spatial_queries_relative_to_desk():
    """'What is near the desk?' and 'What is closest to the desk?' compute distance relative to desk."""
    # 1. Spatial relation: what is near the desk?
    res_rel = global_investigation_agent.run(
        "What is near the desk?",
        agent_id="navigator",
        active_room="Research Office"
    )
    assert res_rel["tool_calls"][0]["tool"] == "query_spatial_relation"
    assert "chair" in res_rel["final_response"].lower()

    # 2. Nearest relative to desk
    res_near_desk = global_tool_registry.execute("find_nearest_target", {
        "target_class": "any",
        "reference_object": "desk"
    })
    assert res_near_desk["success"] is True
    data = res_near_desk["result"]
    assert data["found"] is True
    assert data["class_name"] == "chair"
    assert data["distance_meters"] in (1.04, 1.24)


def test_missing_objects_vs_registered_objects_no_false_uncertainty():
    """Registered objects must never trigger explain_uncertainty; only non-existent classes do."""
    # Table exists in facility -> must locate cleanly
    res_table = global_investigation_agent.run("Where is the table?", agent_id="scout")
    assert res_table["tool_calls"][0]["tool"] == "locate_object"
    assert "table" in res_table["final_response"].lower()
    assert "explain_uncertainty" not in [c["tool"] for c in res_table["tool_calls"]]

    # Laptop does not exist in facility -> explain uncertainty / not detected
    res_laptop = global_investigation_agent.run("Where is the laptop?", agent_id="scout")
    assert res_laptop["tool_calls"][0]["tool"] == "explain_uncertainty"
    assert "no laptop detected" in res_laptop["final_response"].lower()


def test_empty_scene_graph_graceful_handling():
    """An empty scene graph returns found=False with a clear message and does not crash."""
    empty_graph = SceneGraphManager()
    res = empty_graph.find_nearest("table", reference_pos=[0.0, 0.0, 0.0])
    assert res is None

    ranked = empty_graph.get_objects_by_distance(reference_pos=[0.0, 0.0, 0.0])
    assert ranked == []


def test_mission_a_completion_blocked_when_inventory_empty():
    """Mystery A must report incomplete and block unlocking Mission B when required inventory is missing."""
    session = GameSession(seed=42)
    assert len(session.inventory) == 0

    # User executes perceptual survey and localization tools
    session.record_tools(
        [
            {"tool": "get_visible_objects", "arguments": {"room_name": "Reception / Central Hub"}},
            {"tool": "locate_object", "arguments": {"object_name": "table"}}
        ],
        [{"success": True}, {"success": True}]
    )

    state = session.to_dict()
    # First two objectives complete, but scout_03 (key retrieval) is incomplete
    assert state["active_mission"]["objectives"][0]["is_completed"] is True
    assert state["active_mission"]["objectives"][1]["is_completed"] is True
    assert state["active_mission"]["objectives"][2]["is_completed"] is False

    # Mission must NOT be marked completed
    assert state["active_mission"]["completed"] is False
    assert state["missions"][0]["completed"] is False
    assert state["next_mission_id"] is None
    assert session.unlocked_through_index == 0

    # Procedural hint explains exactly what remains
    hint = session.get_hint()
    assert "Objective Incomplete" in hint
    assert "research_key" in hint


def test_mission_a_completion_with_all_objectives_and_inventory_fulfilled():
    """Mystery A completes and unlocks Mission B only after objectives AND research key are collected."""
    session = GameSession(seed=42)
    assert len(session.inventory) == 0

    # 1. Complete perception and localization
    session.record_tools(
        [
            {"tool": "get_visible_objects", "arguments": {}},
            {"tool": "locate_object", "arguments": {"object_name": "table"}}
        ],
        [{"success": True}, {"success": True}]
    )
    assert session.to_dict()["active_mission"]["completed"] is False

    # 2. Retrieve key by interacting with cabinet
    completed = session.record_tools(
        [{"tool": "interact_with_object", "arguments": {"object_id": "cabinet_lab"}}],
        [{"success": True, "result": {"object_id": "cabinet_lab", "action": "retrieve"}}]
    )
    assert "scout_03" in completed
    assert any(item["id"] == "research_key" for item in session.inventory)

    # 3. Mission state now reports completed
    state = session.to_dict()
    assert state["active_mission"]["completed"] is True
    assert state["missions"][0]["completed"] is True
    assert state["next_mission_id"] == "mission_b"
    assert session.unlocked_through_index >= 1

    # Hint confirms completion
    hint = session.get_hint()
    assert "All objectives" in hint


def test_consistency_between_mission_progress_inventory_and_agent():
    """Agent tool queries must report consistent state for objectives, inventory, and hints."""
    # Initially 0 items
    inv_res = global_investigation_agent.run("Show inventory", agent_id="archivist")
    assert inv_res["tool_calls"][0]["tool"] == "get_inventory"
    assert "inventory: 0 items" in inv_res["final_response"].lower() or "no items" in inv_res["final_response"].lower()

    # Hint explains key is missing
    hint_res = global_investigation_agent.run("Give me a hint", agent_id="archivist")
    assert hint_res["tool_calls"][0]["tool"] == "request_hint"
    assert "research_key" in hint_res["final_response"] or "key" in hint_res["final_response"].lower()

    # Collect key
    interact_res = global_investigation_agent.run("Interact with the cabinet", agent_id="archivist")
    assert interact_res["tool_calls"][0]["tool"] == "interact_with_object"

    # Inventory now reflects the key
    inv_res2 = global_investigation_agent.run("Show inventory", agent_id="archivist")
    assert "research key" in inv_res2["final_response"].lower()
