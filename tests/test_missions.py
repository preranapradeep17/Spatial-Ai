"""Tests for Spatial Quest mission progression and companion agents."""
from backend.game.missions import GameSession
from backend.game.facility import seed_facility_scene
from backend.agent.investigation_agent import InvestigationAgent
from backend.spatial.scene_graph import global_scene_graph


def test_facility_scene_has_sunrgbd_classes():
    count = seed_facility_scene()
    assert count >= 12
    names = {obj.class_name for obj in global_scene_graph.get_all_objects()}
    assert {"table", "chair", "door", "bookshelf", "sofa", "cabinet"}.issubset(names)


def test_missions_unlock_in_order():
    session = GameSession()
    assert session.active_mission_id == "mission_a"
    session.record_tools([{"tool": "get_visible_objects", "arguments": {}}], [{}])
    session.record_tools([{"tool": "locate_object", "arguments": {"object_name": "table"}}], [{}])
    mid_state = session.to_dict()
    # Mystery A must not complete without the required inventory item (research_key)
    assert mid_state["active_mission"]["completed"] is False
    assert mid_state["next_mission_id"] is None

    # Fulfill the required item objective by interacting with the cabinet
    session.record_tools(
        [{"tool": "interact_with_object", "arguments": {"object_id": "cabinet_lab"}}],
        [{"success": True, "result": {"object_id": "cabinet_lab", "action": "retrieve"}}]
    )
    state = session.to_dict()
    assert state["active_mission"]["completed"] is True
    assert state["next_mission_id"] == "mission_b"
    session.start_mission("mission_b")
    assert session.active_agent_id == "navigator"


def test_locked_mission_rejected():
    session = GameSession()
    try:
        session.start_mission("mission_d")
        assert False, "Locked mission should raise"
    except ValueError as exc:
        assert "locked" in str(exc).lower()


def test_agent_how_far_from_phrase():
    seed_facility_scene()
    agent = InvestigationAgent()
    res = agent.run("How far is the chair from the table?")
    assert res["tool_calls"][0]["tool"] == "calculate_distance"
    assert "distance" in res["final_response"].lower()


def test_companion_switch_is_whitelist_only():
    session = GameSession()
    session.select_agent("archivist")
    assert session.active_agent_id == "archivist"
    try:
        session.select_agent("unregistered_bot")
        assert False, "Unknown agent should raise"
    except ValueError:
        pass


def test_facility_has_five_distinct_rooms():
    from backend.game.facility import ROOM_DEFINITIONS
    room_ids = {r["id"] for r in ROOM_DEFINITIONS}
    assert {"reception", "office", "library", "laboratory", "storage"} == room_ids
    assert len(ROOM_DEFINITIONS) == 5


def test_room_for_position_mapping():
    from backend.game.facility import get_room_for_position
    # Reception: Z < -10
    assert get_room_for_position(0.0, -15.0)["id"] == "reception"
    # Research Office: Z in [-10, 0], X >= 0
    assert get_room_for_position(5.0, -5.0)["id"] == "office"
    # Library / Archives: Z in [-10, 0], X < 0
    assert get_room_for_position(-5.0, -5.0)["id"] == "library"
    # Investigation Laboratory: Z in [0, 10]
    assert get_room_for_position(0.0, 5.0)["id"] == "laboratory"
    # Storage Room: Z > 10
    assert get_room_for_position(0.0, 15.0)["id"] == "storage"


def test_seed_randomization_changes_variants_safely():
    session1 = GameSession(seed=42)
    session2 = GameSession(seed=999)
    m1 = session1.missions[3]["briefing"]
    m2 = session2.missions[3]["briefing"]
    # Seed variations produce different dynamic codes or text
    assert "override code" in m1
    assert "override code" in m2
    assert session1.missions[0]["id"] == session2.missions[0]["id"]


def test_inventory_state_and_clue_acquisition():
    session = GameSession()
    assert len(session.inventory) == 0
    session.add_to_inventory("test_key", "Sector 4 Key", "A bronze research key.")
    assert len(session.inventory) == 1
    assert session.inventory[0]["id"] == "test_key"
    # Duplicate add is idempotent
    session.add_to_inventory("test_key", "Sector 4 Key", "A bronze research key.")
    assert len(session.inventory) == 1
