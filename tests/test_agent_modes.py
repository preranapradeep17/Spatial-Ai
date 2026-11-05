"""Automated tests for Spatial Quest AI Companion Modes: Scout, Navigator, and Archivist.

Validates that switching companion modes genuinely routes queries to distinct tools,
data sources, and response synthesizers according to their educational roles:
- Scout (Vision): SpatialDetector CNN detections and camera FOV surveys.
- Navigator (Spatial): 3D coordinates, room boundaries, Euclidean distances, and GPS turn-by-turn routes.
- Archivist (RAG): TF-IDF document retrieval, incident reports, and containment protocols.
"""
import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.agent.investigation_agent import InvestigationAgent
from backend.spatial.scene_graph import global_scene_graph
from backend.game.facility import seed_facility_scene
from backend.game.missions import global_game_session

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensures facility scene and game session are initialized."""
    seed_facility_scene()
    global_game_session.return_to_hub()
    global_game_session.update_player_state([0.0, -15.0], "1. Reception / Central Hub")


def test_describe_what_is_around_me_across_all_three_modes():
    """Verify that asking the exact same question in each mode uses distinct tools and data sources.
    
    Query: 'Describe what is around me.'
    - Scout: Uses get_visible_objects -> camera FOV detections, bounding boxes, depth.
    - Navigator: Uses get_current_location -> player coordinates, sector geometry, adjacent zones.
    - Archivist: Uses search_documents -> facility records, incident logs, containment status.
    """
    agent = InvestigationAgent()
    coords = [0.0, -15.0]
    room = "Reception / Central Hub"

    # 1. SCOUT MODE
    scout_res = agent.run("Describe what is around me.", agent_id="scout", player_position=coords, active_room=room)
    assert len(scout_res["tool_calls"]) > 0
    assert scout_res["tool_calls"][0]["tool"] == "get_visible_objects"
    assert "scout drone" in scout_res["final_response"].lower()
    assert "optical perception feed" in scout_res["final_response"].lower()
    assert "spatialdetector cnn" in scout_res["final_response"].lower()

    # 2. NAVIGATOR MODE
    nav_res = agent.run("Describe what is around me.", agent_id="navigator", player_position=coords, active_room=room)
    assert len(nav_res["tool_calls"]) > 0
    assert nav_res["tool_calls"][0]["tool"] == "get_current_location"
    assert "navigator drone" in nav_res["final_response"].lower()
    assert "3d spatial telemetry" in nav_res["final_response"].lower()
    assert "[0.0, 0.0, -15.0]" in nav_res["final_response"] or "reception" in nav_res["final_response"].lower()

    # 3. ARCHIVIST MODE
    arch_res = agent.run("Describe what is around me.", agent_id="archivist", player_position=coords, active_room=room)
    assert len(arch_res["tool_calls"]) > 0
    assert arch_res["tool_calls"][0]["tool"] == "search_documents"
    assert "archivist drone" in arch_res["final_response"].lower()
    assert "facility knowledge base" in arch_res["final_response"].lower()
    assert "archival records" in arch_res["final_response"].lower()

    # Verify that all 3 modes generated completely different responses
    assert scout_res["final_response"] != nav_res["final_response"]
    assert nav_res["final_response"] != arch_res["final_response"]
    assert scout_res["final_response"] != arch_res["final_response"]


def test_scout_mode_what_objects_can_you_see():
    """Verify Scout mode specializes in visual perception and object visibility."""
    agent = InvestigationAgent()
    res = agent.run("What objects can you see?", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "get_visible_objects"
    assert "scout drone" in res["final_response"].lower()
    assert "table" in res["final_response"].lower() or "objects" in res["final_response"].lower()
    assert "confidence" in res["final_response"].lower()


def test_navigator_mode_where_am_i_and_how_to_reach_library():
    """Verify Navigator mode specializes in spatial navigation and route guidance."""
    agent = InvestigationAgent()
    res = agent.run(
        "Where am I and how do I reach the Library?",
        agent_id="navigator",
        player_position=[0.0, -15.0],
        active_room="Reception / Central Hub"
    )
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "navigate_to"
    assert res["tool_calls"][0]["arguments"]["destination"].lower() == "library"
    assert "navigator drone" in res["final_response"].lower()
    assert "navigation command dispatched" in res["final_response"].lower()
    assert "library" in res["final_response"].lower()
    assert res.get("unity_action") is not None


def test_archivist_mode_summarise_incident_reports():
    """Verify Archivist mode specializes in RAG document retrieval and incident summaries."""
    agent = InvestigationAgent()
    res = agent.run(
        "Summarise the available incident reports.",
        agent_id="archivist",
        player_position=[0.0, -15.0],
        active_room="Reception / Central Hub"
    )
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "search_documents"
    assert "archivist drone" in res["final_response"].lower()
    assert len(res.get("rag_passages", [])) > 0
    assert "doc-001" in res["final_response"].lower() or "power surge" in res["final_response"].lower()


def test_scout_negative_observation_no_hallucination():
    """Verify Scout mode does not hallucinate absent objects."""
    agent = InvestigationAgent()
    res = agent.run(
        "Can you find the laptop?",
        agent_id="scout",
        player_position=[0.0, -15.0],
        active_room="Reception / Central Hub"
    )
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "explain_uncertainty"
    assert "scout drone" in res["final_response"].lower()
    assert "zero-hallucination" in res["final_response"].lower()


def test_api_companion_switching_and_query_flow():
    """Test full HTTP API flow: switching companion via POST /game/agent and verifying mode in /agent/query."""
    client = TestClient(app)

    # 1. Switch to Scout
    scout_sel = client.post("/game/agent", json={"agent_id": "scout"})
    assert scout_sel.status_code == 200
    assert scout_sel.json()["companion"]["id"] == "scout"

    # Query Scout
    scout_q = client.post("/agent/query", json={
        "query": "Describe what is around me.",
        "agent_id": "scout",
        "player_position": [0.0, -15.0],
        "active_room": "Reception / Central Hub"
    })
    assert scout_q.status_code == 200
    scout_data = scout_q.json()
    assert scout_data["tool_calls"][0]["tool"] == "get_visible_objects"
    assert "scout drone" in scout_data["final_response"].lower()

    # 2. Switch to Navigator
    nav_sel = client.post("/game/agent", json={"agent_id": "navigator"})
    assert nav_sel.status_code == 200
    assert nav_sel.json()["companion"]["id"] == "navigator"

    # Query Navigator
    nav_q = client.post("/agent/query", json={
        "query": "Describe what is around me.",
        "agent_id": "navigator",
        "player_position": [0.0, -15.0],
        "active_room": "Reception / Central Hub"
    })
    assert nav_q.status_code == 200
    nav_data = nav_q.json()
    assert nav_data["tool_calls"][0]["tool"] == "get_current_location"
    assert "navigator drone" in nav_data["final_response"].lower()

    # 3. Switch to Archivist
    arch_sel = client.post("/game/agent", json={"agent_id": "archivist"})
    assert arch_sel.status_code == 200
    assert arch_sel.json()["companion"]["id"] == "archivist"

    # Query Archivist
    arch_q = client.post("/agent/query", json={
        "query": "Describe what is around me.",
        "agent_id": "archivist",
        "player_position": [0.0, -15.0],
        "active_room": "Reception / Central Hub"
    })
    assert arch_q.status_code == 200
    arch_data = arch_q.json()
    assert arch_data["tool_calls"][0]["tool"] == "search_documents"
    assert "archivist drone" in arch_data["final_response"].lower()
