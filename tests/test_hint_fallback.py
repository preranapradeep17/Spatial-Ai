"""Tests for intent routing and prevention of generic request_hint fallback in Spatial Quest.

Ensures:
1. Non-hint queries (across object identification, scene understanding, location,
   navigation, spatial reasoning, investigation, and uncertainty) NEVER trigger request_hint.
2. Explicit hint queries trigger request_hint with a valid non-null topic.
3. Diverse 30+ query matrix covering all 8 categories passes with 100% correctness.
"""
import pytest
from backend.agent.investigation_agent import global_investigation_agent
from backend.game.missions import global_game_session
from backend.game.facility import seed_facility_scene
from backend.spatial.scene_graph import global_scene_graph


@pytest.fixture(autouse=True)
def setup_game_environment():
    global_game_session.return_to_hub()
    global_game_session.start_mission("mission_a")
    if not global_scene_graph.get_all_objects():
        seed_facility_scene()


def test_non_hint_queries_never_trigger_request_hint():
    """Verifies that queries outside explicit hint requests never invoke request_hint."""
    non_hint_queries = [
        # 1. Object Identification
        "Where is the chair?",
        "Find the table.",
        "Locate the bookshelf.",
        "Where is the cabinet?",
        # 2. Scene Understanding
        "What objects can you see?",
        "Describe this room.",
        "What is around me?",
        "List visible objects.",
        # 3. Location
        "Where am I?",
        "What room am I standing in?",
        "What room is this?",
        "Current location",
        # 4. Navigation
        "Take me to the Library.",
        "Guide me to the Research Office.",
        "Navigate to the Investigation Laboratory.",
        "Take me to the nearest exit.",
        # 5. Spatial Reasoning
        "What is closest to the table?",
        "Which object is beside the chair?",
        "How far is the chair from the table?",
        "What is near the desk?",
        # 6. Investigation
        "Find the missing key.",
        "Where was the key last reported?",
        "What does the incident report say?",
        "Inspect clue clue_safe_01.",
        "Show my inventory.",
        # 7. Uncertainty
        "Is the laptop visible?",
        "Why can't you find the key?",
        "Why is it unreachable?",
        "What is your confidence?",
        # 8. Mission Next Action (Guidance, NOT generic hint)
        "What should I do next?",
    ]

    for q in non_hint_queries:
        for mode in ["scout", "navigator", "archivist"]:
            res = global_investigation_agent.run(q, agent_id=mode)
            tools_called = [c["tool"] for c in res.get("tool_calls", [])]
            assert "request_hint" not in tools_called, (
                f"Query '{q}' in mode '{mode}' unexpectedly triggered 'request_hint'!"
            )


def test_explicit_hint_request_executes_safely_without_null_topic():
    """Verifies that explicit hint requests execute request_hint with a valid topic and without errors."""
    explicit_hint_queries = [
        "Give me a hint.",
        "I need a hint.",
        "Can you give me a clue hint?",
        "Give hint",
    ]

    for q in explicit_hint_queries:
        res = global_investigation_agent.run(q, agent_id="archivist")
        tools_called = [c["tool"] for c in res.get("tool_calls", [])]
        assert "request_hint" in tools_called, f"Explicit hint query '{q}' did not trigger request_hint"

        # Check tool call arguments
        hint_call = next(c for c in res["tool_calls"] if c["tool"] == "request_hint")
        topic = hint_call["arguments"].get("topic")
        assert topic is not None, f"Topic was None (null) for query '{q}'"
        assert topic != "null", f"Topic was 'null' string for query '{q}'"
        assert len(str(topic).strip()) > 0, "Topic was empty"

        # Check response content
        resp = res["final_response"]
        assert "Mission Hint: Examine nearby objects and documents to make progress." not in resp, (
            "Response used the generic fallback failure string!"
        )
        assert "Tactical" in resp or "Hint" in resp or "guidance" in resp.lower()


def test_all_eight_categories_routing_correctness():
    """Verifies specific expected tool targets across all 8 functional categories."""
    category_expectations = [
        # (Query, Mode, Expected Tool)
        ("Where is the chair?", "scout", "locate_object"),
        ("Find the table.", "scout", "locate_object"),
        ("Locate the bookshelf.", "scout", "locate_object"),
        ("What objects can you see?", "scout", "get_visible_objects"),
        ("Describe this room.", "scout", "get_visible_objects"),
        ("What is around me?", "scout", "get_visible_objects"),
        ("What is around me?", "navigator", "get_current_location"),
        ("What is around me?", "archivist", "search_documents"),
        ("Where am I?", "navigator", "get_current_location"),
        ("What room am I standing in?", "navigator", "get_current_location"),
        ("Take me to the Library.", "navigator", "navigate_to"),
        ("Guide me to the Research Office.", "navigator", "navigate_to"),
        ("What is closest to the table?", "navigator", "find_nearest_target"),
        ("Which object is beside the chair?", "navigator", "query_spatial_relation"),
        ("How far is the chair from the table?", "navigator", "calculate_distance"),
        ("Find the missing key.", "scout", "locate_object"),
        ("Where was the key last reported?", "archivist", "search_documents"),
        ("What does the incident report say?", "archivist", "search_documents"),
        ("Inspect clue clue_safe_01.", "archivist", "inspect_clue"),
        ("Show my inventory.", "archivist", "get_inventory"),
        ("Is the laptop visible?", "scout", "get_visible_objects"),
        ("Why is it unreachable?", "navigator", "explain_unreachable"),
        ("What should I do next?", "archivist", "get_mission_objectives"),
        ("Give me a hint.", "archivist", "request_hint"),
    ]

    for q, mode, expected_tool in category_expectations:
        res = global_investigation_agent.run(q, agent_id=mode)
        tools_called = [c["tool"] for c in res.get("tool_calls", [])]
        assert expected_tool in tools_called, (
            f"Query '{q}' in mode '{mode}' expected '{expected_tool}', but got {tools_called}"
        )

