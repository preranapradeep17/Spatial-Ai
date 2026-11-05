"""Regression tests for player position and room synchronization in Spatial Quest navigation.

Verifies:
1. Standing inside Library [-5.8, -5.9] and querying 'Take me to the library' passes non-null
   coordinates and room name, returning 'already_at_destination' without unreachable errors.
2. Standing inside Library and querying 'Where am I currently?' returns exact player coordinates.
3. Standing inside Reception and querying 'Take me to the library' generates valid cross-room turns.
4. Standing inside Research Office and querying 'Take me to the Investigation Lab' routes smoothly.
5. Moving the player dynamically updates the arguments passed to navigation tools.
6. Consistent room name normalization across varied casings, abbreviations, and zone prefixes.
7. Real-time synchronization via POST /game/sync.
"""
import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.agent.investigation_agent import InvestigationAgent
from backend.game.facility import normalize_room_name, get_room_definition, is_room_identifier
from backend.game.missions import global_game_session

client = TestClient(app)


def test_room_name_normalization_variations():
    """Verify that room names in various forms normalize consistently."""
    library_variants = ["Library", "library", "3. Library / Archives", "Library / Archives", "archives", "Zone 3", "zone 3"]
    for var in library_variants:
        norm = normalize_room_name(var)
        assert norm == "Library / Archives", f"Failed for variant: {var}"
        assert is_room_identifier(var) is True

    office_variants = ["office", "Research Office", "2. Research Office", "workstations", "Zone 2"]
    for var in office_variants:
        assert normalize_room_name(var) == "Research Office"

    lab_variants = ["lab", "laboratory", "Investigation Laboratory", "4. Investigation Laboratory", "Zone 4"]
    for var in lab_variants:
        assert normalize_room_name(var) == "Investigation Laboratory"

    reception_variants = ["reception", "1. Reception / Central Hub", "Reception / Central Hub", "hub", "Zone 1"]
    for var in reception_variants:
        assert normalize_room_name(var) == "Reception / Central Hub"

    storage_variants = ["storage", "5. Storage Room & Vault", "Storage Room", "vault", "Zone 5"]
    for var in storage_variants:
        assert normalize_room_name(var) == "Storage Room"


def test_navigation_sync_inside_library():
    """Stand inside Library [-5.8, -5.9] and ask 'Take me to the library'.
    
    Verifies:
    - player_position is NOT null (is [-5.8, -5.9])
    - current_room is NOT null (is 'Library / Archives')
    - Does NOT return 'unreachable' error
    - Recognizes that player is already at the destination
    """
    agent = InvestigationAgent()
    res = agent.run("Take me to the library", player_position=[-5.8, -5.9], active_room="3. Library / Archives")
    
    assert len(res["tool_calls"]) > 0
    tc = res["tool_calls"][0]
    assert tc["tool"] == "navigate_to"
    
    args = tc["arguments"]
    assert args["player_position"] is not None, "player_position must not be null"
    assert args["player_position"] == [-5.8, -5.9]
    assert args["current_room"] == "Library / Archives", "current_room must be normalized"
    
    final_resp = res["final_response"].lower()
    assert "unreachable" not in final_resp
    assert "already inside" in final_resp
    assert "library" in final_resp


def test_location_query_inside_library():
    """Stand inside Library [-5.8, -5.9] and ask 'Where am I currently?'.
    
    Verifies that the response returns the actual player coordinates and room.
    """
    agent = InvestigationAgent()
    res = agent.run("Where am I currently?", player_position=[-5.8, -5.9], active_room="3. Library / Archives")
    
    assert len(res["tool_calls"]) > 0
    tc = res["tool_calls"][0]
    assert tc["tool"] == "get_current_location"
    
    args = tc["arguments"]
    assert args["player_position"] == [-5.8, -5.9]
    
    final_resp = res["final_response"]
    assert "Library / Archives" in final_resp
    assert "[-5.8, 0.0, -5.9]" in final_resp


def test_navigation_from_reception_to_library():
    """Stand inside Reception [0.0, -15.0] and ask 'Take me to the library'.
    
    Verifies turn-by-turn guidance is generated from Reception into Library.
    """
    agent = InvestigationAgent()
    res = agent.run("Take me to the library", player_position=[0.0, -15.0], active_room="1. Reception / Central Hub")
    
    assert len(res["tool_calls"]) > 0
    tc = res["tool_calls"][0]
    assert tc["tool"] == "navigate_to"
    assert tc["arguments"]["player_position"] == [0.0, -15.0]
    assert tc["arguments"]["current_room"] == "Reception / Central Hub"
    
    res_data = res["tool_results"][0]["result"]
    assert res_data["status"] == "command_dispatched"
    assert res_data["origin_room"] == "Reception / Central Hub"
    assert res_data["destination_room"] == "Library / Archives"
    assert len(res_data["turn_by_turn"]) >= 3
    assert len(res_data["waypoints"]) >= 2


def test_navigation_from_office_to_lab():
    """Stand inside Research Office [6.0, -5.0] and ask 'Take me to the Investigation Lab'.
    
    Verifies cross-room navigation from Office into Investigation Laboratory.
    """
    agent = InvestigationAgent()
    res = agent.run("Take me to the Investigation Lab", player_position=[6.0, -5.0], active_room="2. Research Office")
    
    assert len(res["tool_calls"]) > 0
    tc = res["tool_calls"][0]
    assert tc["tool"] == "navigate_to"
    assert tc["arguments"]["player_position"] == [6.0, -5.0]
    assert tc["arguments"]["current_room"] == "Research Office"
    
    res_data = res["tool_results"][0]["result"]
    assert res_data["status"] == "command_dispatched"
    assert res_data["origin_room"] == "Research Office"
    assert res_data["destination_room"] == "Investigation Laboratory"


def test_dynamic_coordinate_updates_across_moves():
    """Move player across distinct zones and verify tool arguments update dynamically."""
    agent = InvestigationAgent()
    checkpoints = [
        ([0.0, -15.0], "1. Reception / Central Hub", "Reception / Central Hub"),
        ([6.5, -4.5], "2. Research Office", "Research Office"),
        ([-5.8, -5.9], "3. Library / Archives", "Library / Archives"),
        ([0.0, 5.0], "4. Investigation Laboratory", "Investigation Laboratory"),
        ([0.0, 15.0], "5. Storage Room & Vault", "Storage Room"),
    ]
    for pos, input_room, expected_room in checkpoints:
        res = agent.run("Where am I currently?", player_position=pos, active_room=input_room)
        tc = res["tool_calls"][0]
        assert tc["arguments"]["player_position"] == pos
        assert tc["arguments"]["active_room"] == expected_room
        assert expected_room in res["final_response"]


def test_game_sync_api_endpoint():
    """Verify POST /game/sync updates backend session state in real time."""
    resp = client.post("/game/sync", json={
        "player_position": [-5.8, -5.9],
        "active_room": "3. Library / Archives"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "synchronized"
    assert data["player_position"] == [-5.8, 0.0, -5.9]
    assert data["current_room"] == "Library / Archives"
    assert global_game_session.current_room == "Library / Archives"

