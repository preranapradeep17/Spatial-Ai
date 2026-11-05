"""Unit tests for the autonomous investigation agent."""
import pytest
from backend.agent.investigation_agent import InvestigationAgent
from backend.spatial.scene_graph import global_scene_graph
from backend.api.schemas import DetectedObject, BoundingBox2D

@pytest.fixture(autouse=True)
def populate_test_scene():
    """Populates global_scene_graph with dummy objects for agent tests."""
    objs = [
        DetectedObject(
            object_id="table_1",
            class_id=0,
            class_name="table",
            confidence=0.92,
            bbox=BoundingBox2D(xmin=100, ymin=100, xmax=200, ymax=200),
            depth_estimate=2.15,
            estimated_3d_position=[0.12, -0.45, 2.15]
        ),
        DetectedObject(
            object_id="chair_1",
            class_id=1,
            class_name="chair",
            confidence=0.88,
            bbox=BoundingBox2D(xmin=50, ymin=100, xmax=100, ymax=200),
            depth_estimate=1.95,
            estimated_3d_position=[-0.85, -0.40, 1.95]
        ),
        DetectedObject(
            object_id="door_1",
            class_id=2,
            class_name="door",
            confidence=0.95,
            bbox=BoundingBox2D(xmin=250, ymin=50, xmax=350, ymax=400),
            depth_estimate=4.80,
            estimated_3d_position=[0.25, 0.05, 4.80]
        )
    ]
    global_scene_graph.update_objects(objs, source="test_fixture")

def test_agent_visible_objects_query():
    agent = InvestigationAgent()
    res = agent.run("What objects are visible around us?")
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "get_visible_objects"
    assert "table" in res["final_response"].lower()

def test_agent_distance_query():
    agent = InvestigationAgent()
    res = agent.run("What is the distance between the table and the chair?")
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "calculate_distance"
    assert "distance" in res["final_response"].lower()

def test_agent_find_path_query():
    agent = InvestigationAgent()
    res = agent.run("Calculate a path to the door")
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "find_path"
    assert "waypoints" in res["final_response"].lower()

def test_agent_rag_query():
    agent = InvestigationAgent()
    res = agent.run("What is the emergency override code for Door 3?")
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "search_documents"
    assert "7391" in res["final_response"] or "override" in res["final_response"].lower()

def test_agent_navigate_take_me_table():
    """Verify 'take me table' activates navigate_to and produces Google Maps-style turn-by-turn guidance."""
    agent = InvestigationAgent()
    res = agent.run("take me table", player_position=[0.0, -15.0])
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "navigate_to"
    assert "navigation command dispatched" in res["final_response"].lower()
    assert "gps navigation" in res["final_response"].lower() or "destination" in res["final_response"].lower()
    assert res.get("unity_action") is not None

def test_agent_how_to_play():
    """Verify 'how to play' query returns detailed gameplay guide."""
    agent = InvestigationAgent()
    res = agent.run("how to play")
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "get_game_guide"
    resp = res["final_response"].lower()
    assert "controls" in resp or "objective" in resp or "gameplay" in resp
