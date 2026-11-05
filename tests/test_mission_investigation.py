"""Automated regression tests for mission-aware investigation and missing-object queries in Spatial Quest.

Verifies:
1. 'Find the key' calls locate_object and investigates active mission clues rather than prematurely failing with generic CNN uncertainty.
2. 'Where was the key last reported?' retrieves mission records and maintenance logs.
3. 'Take me to the security table' navigates to the Reception security table.
4. 'Is the key near the lab cabinet?' checks spatial scene relations and cross-references cabinet drawer logs.
5. 'Can you confirm whether the key is visible?' evaluates optical FOV without fabricating detections.
6. 'What should I investigate next?' retrieves active mission objectives and recommended steps.
"""
import pytest
from backend.agent.investigation_agent import InvestigationAgent
from backend.game.facility import seed_facility_scene
from backend.game.missions import global_game_session

@pytest.fixture(autouse=True)
def setup_mystery_a():
    """Seeds facility layout and resets session to Mystery A in Reception."""
    seed_facility_scene()
    global_game_session.return_to_hub()
    global_game_session.select_mystery("mission_a")
    global_game_session.update_player_state([0.0, -15.0], "1. Reception / Central Hub")


def test_find_the_key_mission_aware_investigation():
    """Query 1: 'Find the key.'
    Must call locate_object and distinguish between visual FOV, scene graph, and mission records.
    Must NOT call explain_uncertainty.
    """
    agent = InvestigationAgent()
    res = agent.run("Find the key.", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "locate_object"
    assert res["tool_calls"][0]["arguments"]["object_name"] == "key"

    resp = res["final_response"]
    assert "not visible in the current camera frame" in resp.lower()
    assert "not registered" in resp.lower()
    assert "mission" in resp.lower()
    assert "security table" in resp.lower() or "cabinet" in resp.lower()


def test_where_was_the_key_last_reported():
    """Query 2: 'Where was the key last reported?'
    Retrieves incident records and documents citing the security table and lab cabinet.
    """
    agent = InvestigationAgent()
    res = agent.run("Where was the key last reported?", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "search_documents"

    resp = res["final_response"]
    assert "reported" in resp.lower()
    assert "cabinet" in resp.lower()
    assert "security table" in resp.lower() or "reception" in resp.lower()


def test_take_me_to_the_security_table():
    """Query 3: 'Take me to the security table.'
    Dispatches navigation directly to the Reception security table.
    """
    agent = InvestigationAgent()
    res = agent.run("Take me to the security table.", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "navigate_to"
    assert "security table" in res["tool_calls"][0]["arguments"]["destination"].lower()

    resp = res["final_response"]
    assert "navigation command dispatched" in resp.lower() or "already" in resp.lower() or "gps navigation" in resp.lower()
    assert res.get("unity_action") is not None


def test_is_the_key_near_the_lab_cabinet():
    """Query 4: 'Is the key near the lab cabinet?'
    Checks spatial relationships near cabinet and cross-references drawer records.
    """
    agent = InvestigationAgent()
    res = agent.run("Is the key near the lab cabinet?", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "query_spatial_relation"
    assert res["tool_calls"][0]["arguments"]["object_name"] == "cabinet"

    resp = res["final_response"]
    assert "scene graph" in resp.lower() or "cabinet" in resp.lower()
    assert "drawer" in resp.lower() or "doc-004" in resp.lower() or "maintenance log" in resp.lower()


def test_can_you_confirm_whether_the_key_is_visible():
    """Query 5: 'Can you confirm whether the key is visible?'
    Runs visual FOV survey and confirms zero fabrication without hallucinating the key.
    """
    agent = InvestigationAgent()
    res = agent.run("Can you confirm whether the key is visible?", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "get_visible_objects"

    resp = res["final_response"]
    assert "not visible" in resp.lower()
    assert "zero-fabrication" in resp.lower() or "not fabricate" in resp.lower()


def test_what_should_i_investigate_next():
    """Query 6: 'What should I investigate next?'
    Retrieves mission objectives and recommends actionable investigation steps.
    """
    agent = InvestigationAgent()
    res = agent.run("What should I investigate next?", agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    assert res["tool_calls"][0]["tool"] == "get_mission_objectives"

    resp = res["final_response"]
    assert "active mission objectives" in resp.lower() or "investigation plan" in resp.lower()
    assert "security table" in resp.lower()

