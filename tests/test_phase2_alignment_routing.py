"""Phase 2 Validation Suite: LLM Input Alignment and Reliable Intent Routing.

Validates:
1. Shared canonical prompt equivalence between dataset generation and runtime inference.
2. Door vs Exit routing precedence ('Take me to the door' -> navigate_to vs 'Find nearest exit' -> find_nearest_exit).
3. Counting query routing ('How many chairs are here?' -> get_visible_objects).
4. Vision perception uncertainty routing ('Why can't you see the laptop?' -> explain_uncertainty).
5. Unreachable route query routing ('Why is the destination unreachable?' -> explain_unreachable).
6. Nearest object routing ('What is nearest to me?' -> find_nearest_target).
7. JSON and Tokenizer whitespace normalization resilience.
8. Authoritative player state propagation in API routes before agent execution.
"""
import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.agent.investigation_agent import InvestigationAgent, global_investigation_agent
from backend.agent.prompt_builder import (
    build_canonical_prompt,
    format_visible_objects_block,
    format_spatial_relations_block,
    format_conversation_history_block,
)
from backend.game.missions import global_game_session
from backend.game.facility import seed_facility_scene
from backend.spatial.scene_graph import global_scene_graph


@pytest.fixture(autouse=True)
def setup_environment():
    """Initializes facility scene layout and game session before each test."""
    seed_facility_scene()
    global_game_session.return_to_hub()
    global_game_session.start_mission("mission_a")


# -----------------------------------------------------------------------------
# 1. Training and Runtime Prompt Equivalence
# -----------------------------------------------------------------------------
def test_canonical_prompt_builder_equivalence():
    """Verify that prompt_builder formats canonical fields identically across training and inference."""
    prompt = build_canonical_prompt(
        query="Survey the room.",
        agent_mode="SCOUT",
        room_name="Reception / Central Hub",
        player_position=[0.0, 0.0, -15.0],
        visible_objects_str="- Table: [0.0, 0.8, -15.0]\n- Chair: [0.0, 0.5, -13.8]",
        spatial_relations_str="- Table is near Chair",
        active_mission_str="Mystery A: Missing Research Key (Objectives: 0/3)",
        inventory_str="None",
        history_str=""
    )

    # Verify all 9 required canonical fields exist in the generated prompt
    assert "SYSTEM: You are an autonomous facility companion drone." in prompt
    assert "AGENT_MODE: SCOUT" in prompt
    assert "ROOM: Reception / Central Hub" in prompt
    assert "PLAYER_POSITION: [0.0, 0.0, -15.0]" in prompt
    assert "VISIBLE OBJECTS:" in prompt
    assert "- Table: [0.0, 0.8, -15.0]" in prompt
    assert "SPATIAL RELATIONS:" in prompt
    assert "- Table is near Chair" in prompt
    assert "ACTIVE MISSION: Mystery A: Missing Research Key (Objectives: 0/3)" in prompt
    assert "INVENTORY: None" in prompt
    assert "USER: Survey the room." in prompt
    assert prompt.endswith("ASSISTANT:")


def test_runtime_agent_structured_prompt_matches_canonical():
    """Verify runtime agent uses build_canonical_prompt and matches expected canonical structure."""
    agent = InvestigationAgent(prompt_format="canonical")
    prompt = agent.build_structured_prompt(
        query="What is visible?",
        agent_id="scout",
        player_position=[0.0, -15.0],
        active_room="Reception / Central Hub"
    )

    assert "SYSTEM: You are an autonomous facility companion drone." in prompt
    assert "AGENT_MODE: SCOUT" in prompt
    assert "ROOM: Reception / Central Hub" in prompt
    assert "PLAYER_POSITION:" in prompt
    assert "VISIBLE OBJECTS:" in prompt
    assert "SPATIAL RELATIONS:" in prompt
    assert "ACTIVE MISSION:" in prompt
    assert "INVENTORY:" in prompt
    assert "USER: What is visible?" in prompt
    assert prompt.endswith("ASSISTANT:")


# -----------------------------------------------------------------------------
# 2. Door vs Exit Routing Precedence
# -----------------------------------------------------------------------------
def test_door_vs_exit_routing_precedence():
    """Verify 'Take me to the door' routes to navigate_to(destination='door')
    while 'Find the nearest exit' routes to find_nearest_exit()."""
    agent = InvestigationAgent()

    # Query 1: Physical door target navigation
    door_res = agent.run("Take me to the door.", agent_id="navigator", player_position=[0.0, -15.0])
    door_tools = [c["tool"] for c in door_res["tool_calls"]]
    assert "navigate_to" in door_tools, f"Expected navigate_to in {door_tools}"
    door_call = next(c for c in door_res["tool_calls"] if c["tool"] == "navigate_to")
    assert door_call["arguments"]["destination"].lower() == "door"

    # Query 2: Nearest exit search
    exit_res = agent.run("Find the nearest exit.", agent_id="navigator", player_position=[0.0, -15.0])
    exit_tools = [c["tool"] for c in exit_res["tool_calls"]]
    assert "find_nearest_exit" in exit_tools, f"Expected find_nearest_exit in {exit_tools}"

    # Query 3: Where is the exit?
    where_exit = agent.run("Where is the exit?", agent_id="navigator", player_position=[0.0, -15.0])
    where_tools = [c["tool"] for c in where_exit["tool_calls"]]
    assert "find_nearest_exit" in where_tools, f"Expected find_nearest_exit in {where_tools}"

    # Query 4: Take me to the nearest exit
    take_exit = agent.run("Take me to the nearest exit.", agent_id="navigator", player_position=[0.0, -15.0])
    take_tools = [c["tool"] for c in take_exit["tool_calls"]]
    assert "find_nearest_exit" in take_tools, f"Expected find_nearest_exit in {take_tools}"


# -----------------------------------------------------------------------------
# 3. Counting Queries
# -----------------------------------------------------------------------------
def test_counting_queries_route_to_get_visible_objects():
    """Verify 'How many chairs are here?' routes to get_visible_objects and outputs count result."""
    agent = InvestigationAgent()

    queries = [
        "How many chairs are here?",
        "How many tables are in this room?",
        "Count the chairs.",
    ]

    for q in queries:
        res = agent.run(q, agent_id="scout", player_position=[0.0, -15.0], active_room="Reception / Central Hub")
        tools = [c["tool"] for c in res["tool_calls"]]
        assert "get_visible_objects" in tools, f"Query '{q}' did not route to get_visible_objects!"
        assert "count assessment" in res["final_response"].lower() or "detected" in res["final_response"].lower()


# -----------------------------------------------------------------------------
# 4. Perception Uncertainty vs Reachability
# -----------------------------------------------------------------------------
def test_explain_uncertainty_vs_explain_unreachable():
    """Verify:
    - 'Why can't you see the laptop?' -> explain_uncertainty
    - 'Why is the destination unreachable?' -> explain_unreachable
    """
    agent = InvestigationAgent()

    # 4a. Perception uncertainty queries
    uncert_res = agent.run("Why can't you detect the object?", agent_id="scout")
    uncert_tools = [c["tool"] for c in uncert_res["tool_calls"]]
    assert "explain_uncertainty" in uncert_tools, f"Expected explain_uncertainty in {uncert_tools}"
    assert "explain_unreachable" not in uncert_tools

    uncert_res2 = agent.run("Why can't you see the laptop?", agent_id="scout")
    uncert_tools2 = [c["tool"] for c in uncert_res2["tool_calls"]]
    assert "explain_uncertainty" in uncert_tools2, f"Expected explain_uncertainty in {uncert_tools2}"
    assert "explain_unreachable" not in uncert_tools2

    # 4b. Reachability queries
    unreach_res = agent.run("Why is the destination unreachable?", agent_id="navigator")
    unreach_tools = [c["tool"] for c in unreach_res["tool_calls"]]
    assert "explain_unreachable" in unreach_tools, f"Expected explain_unreachable in {unreach_tools}"
    assert "explain_uncertainty" not in unreach_tools

    unreach_res2 = agent.run("Why is it blocked?", agent_id="navigator")
    unreach_tools2 = [c["tool"] for c in unreach_res2["tool_calls"]]
    assert "explain_unreachable" in unreach_tools2, f"Expected explain_unreachable in {unreach_tools2}"


# -----------------------------------------------------------------------------
# 5. Nearest-Object Queries vs Spatial Relations
# -----------------------------------------------------------------------------
def test_nearest_target_vs_spatial_relation_routing():
    """Verify:
    - 'What is nearest to me?' -> find_nearest_target
    - 'Find the nearest table' -> find_nearest_target(target_class='table')
    - 'What is next to the desk?' -> query_spatial_relation(object_name='table', relation='near')
    """
    agent = InvestigationAgent()

    # Nearest to player
    near_me = agent.run("What is nearest to me?", agent_id="navigator", player_position=[0.0, -15.0])
    tools_me = [c["tool"] for c in near_me["tool_calls"]]
    assert "find_nearest_target" in tools_me, f"Expected find_nearest_target in {tools_me}"

    # Nearest table
    near_tab = agent.run("Find the nearest table.", agent_id="navigator", player_position=[0.0, -15.0])
    tools_tab = [c["tool"] for c in near_tab["tool_calls"]]
    assert "find_nearest_target" in tools_tab, f"Expected find_nearest_target in {tools_tab}"
    tab_call = next(c for c in near_tab["tool_calls"] if c["tool"] == "find_nearest_target")
    assert tab_call["arguments"]["target_class"] == "table"

    # Spatial relation next to desk
    next_desk = agent.run("What is next to the desk?", agent_id="navigator")
    tools_desk = [c["tool"] for c in next_desk["tool_calls"]]
    assert "query_spatial_relation" in tools_desk, f"Expected query_spatial_relation in {tools_desk}"


# -----------------------------------------------------------------------------
# 6. JSON Whitespace Normalization Resilience
# -----------------------------------------------------------------------------
def test_tool_call_whitespace_and_bpe_normalization():
    """Verify parse_tool_call strips whitespace from keys and values and validates schemas."""
    agent = InvestigationAgent()

    # Raw model text with trailing spaces and BPE artifact inflation
    model_output = (
        '<thought>Locate table</thought>'
        '<tool_call>{"name ": "locate_object ", "arguments ": {"object_name ": "table "}}</tool_call>'
    )
    parsed = agent.parse_tool_call(model_output)
    assert parsed is not None
    assert parsed["name"] == "locate_object"
    assert parsed["arguments"]["object_name"] == "table"

    # Malformed unparseable junk should return None cleanly without crashing
    bad_output = '<tool_call>{"name": "broken_tool", "arguments": [1, 2, 3}</tool_call>'
    assert agent.parse_tool_call(bad_output) is None

    # Unwhitelisted tool should return None
    fake_tool = '<tool_call>{"name": "execute_arbitrary_shell", "arguments": {}}</tool_call>'
    assert agent.parse_tool_call(fake_tool) is None


# -----------------------------------------------------------------------------
# 7. Typo Resilience
# -----------------------------------------------------------------------------
def test_typo_resilience_in_queries():
    """Verify queries with common typos are correctly resolved and routed."""
    agent = InvestigationAgent()

    # 'Take me to the libary' -> routes to library
    res_lib = agent.run("Take me to the libary", agent_id="navigator", player_position=[0.0, -15.0])
    assert res_lib["tool_calls"][0]["tool"] == "navigate_to"
    assert "library" in res_lib["tool_calls"][0]["arguments"]["destination"].lower()

    # 'Find the nearst tabel' -> routes to table
    res_tab = agent.run("Find the nearst tabel", agent_id="navigator", player_position=[0.0, -15.0])
    assert res_tab["tool_calls"][0]["tool"] == "find_nearest_target"
    assert res_tab["tool_calls"][0]["arguments"]["target_class"] == "table"


# -----------------------------------------------------------------------------
# 8. State Propagation in API Routes
# -----------------------------------------------------------------------------
def test_api_route_synchronizes_state_before_agent_execution():
    """Verify POST /agent/query synchronizes request.player_position and request.active_room
    into global_game_session BEFORE executing the investigation agent."""
    client = TestClient(app)

    payload = {
        "query": "Where am I?",
        "agent_id": "navigator",
        "player_position": [6.2, 0.0, -4.8],
        "active_room": "Research Office",
    }
    response = client.post("/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check that game_state in response reflects the updated stance
    assert data["game_state"]["current_room"] == "Research Office"
    assert abs(data["game_state"]["player_position"][0] - 6.2) < 0.1
    assert abs(data["game_state"]["player_position"][2] - -4.8) < 0.1

    # Check global_game_session was authoritatively updated
    assert global_game_session.current_room == "Research Office"
    assert abs(global_game_session.player_position[0] - 6.2) < 0.1

