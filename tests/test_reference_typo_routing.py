"""Comprehensive Regression Tests for Reference Resolution, Typographical Robustness,
Intent Routing Precedence, and Context Isolation in Spatial Quest.

Traceable to CS3235 (Working with LLMs - Multi-step Agent Loop & Dialogue State Tracking).
"""
import pytest
from backend.agent.investigation_agent import InvestigationAgent
from backend.agent.reference_resolver import ConversationReferenceState, ReferenceResolver
from backend.agent.typo_normalizer import global_typo_normalizer, levenshtein_distance
from backend.game.facility import seed_facility_scene
from backend.game.missions import global_game_session


@pytest.fixture(autouse=True)
def reset_session_and_scene():
    """Initializes standard facility layout and game state."""
    seed_facility_scene(source="facility_layout")
    global_game_session.return_to_hub()
    global_game_session.start_mission("mission_a")


class TestReferenceResolution:
    """Verifies typed anaphora resolution and multi-turn state tracking."""

    def test_multi_turn_anaphoric_chain(self):
        """Tests standard 4-turn reference chain: survey -> closest -> navigate there -> distance from here."""
        agent = InvestigationAgent(max_steps=2)
        pos = [0.0, 0.0, -15.0]
        room = "Reception / Central Hub"

        # Turn 1: Survey room
        r1 = agent.run("What objects can you see?", agent_id="scout", player_position=pos, active_room=room)
        assert r1["tool_calls"][0]["tool"] == "get_visible_objects"

        # Turn 2: Which one is closest to me?
        r2 = agent.run("Which one is closest to me?", agent_id="navigator", player_position=pos, active_room=room)
        assert r2["tool_calls"][0]["tool"] == "find_nearest_target"
        assert agent.reference_state.last_selected_object in ("table", "chair")

        # Turn 3: Take me there (resolves to selected object from Turn 2)
        r3 = agent.run("Take me there.", agent_id="navigator", player_position=pos, active_room=room)
        assert r3["tool_calls"][0]["tool"] == "navigate_to"
        dest = r3["tool_calls"][0]["arguments"]["destination"].lower()
        assert "table" in dest or "chair" in dest

        # Turn 4: How far is it from here? (resolves to target from Turn 3 and player position)
        r4 = agent.run("How far is it from here?", agent_id="navigator", player_position=pos, active_room=room)
        assert r4["tool_calls"][0]["tool"] == "find_nearest_target"
        target_cl = r4["tool_calls"][0]["arguments"]["target_class"].lower()
        assert "table" in target_cl or "chair" in target_cl

    def test_repetition_anaphora_same_object(self):
        """Tests 'Take me to the same object' resolution."""
        agent = InvestigationAgent(max_steps=2)
        agent.reference_state.last_selected_object = "bookshelf"
        resolved, was_mod, entity = agent.reference_resolver.resolve("Take me to the same object.")
        assert was_mod is True
        assert entity == "bookshelf"
        assert "take me to the bookshelf" in resolved.lower()

    def test_relative_spatial_anaphora(self):
        """Tests 'What is near that object?' resolution."""
        agent = InvestigationAgent(max_steps=2)
        agent.reference_state.last_selected_object = "table"
        resolved, was_mod, entity = agent.reference_resolver.resolve("What is near that object?")
        assert was_mod is True
        assert entity == "table"
        assert "what is near the table" in resolved.lower()

    def test_inspection_anaphora(self):
        """Tests 'Tell me more about it' resolving to inspected clue or object."""
        agent = InvestigationAgent(max_steps=2)
        agent.reference_state.last_inspected_clue = "clue_pda_02"
        resolved, was_mod, entity = agent.reference_resolver.resolve("Tell me more about it.")
        assert was_mod is True
        assert entity == "clue_pda_02"
        assert "inspect clue clue_pda_02" in resolved.lower()


class TestTypographicalRobustness:
    """Verifies controlled-vocabulary Levenshtein corrections and invariants."""

    def test_core_typo_queries(self):
        """Tests standard informal and misspelled queries."""
        cases = [
            ("wat objcts can u see?", "what objects can you see?"),
            ("fnd the nerest tabl", "find the nearest table"),
            ("tke me to libary", "take me to library"),
            ("giv me a hnt", "give me a hint"),
            ("shw my invntory", "show my inventory"),
        ]
        for query, expected_norm in cases:
            norm_q, _ = global_typo_normalizer.normalize(query)
            assert norm_q.lower().strip() == expected_norm.lower().strip()

    def test_domain_object_preservation(self):
        """Ensures valid objects are never mutated into other valid objects."""
        objects = ["table", "chair", "bookshelf", "cabinet", "sofa", "door", "desk"]
        for obj in objects:
            norm_q, corrs = global_typo_normalizer.normalize(f"where is the {obj}?")
            assert f"where is the {obj}?" == norm_q.lower()
            assert obj not in corrs

    def test_unknown_noun_preservation(self):
        """Ensures unknown nouns are preserved without aggressive mutation."""
        unknowns = ["quantum", "microscope", "teleporter", "plasma", "alien"]
        for word in unknowns:
            norm_w, was_changed = global_typo_normalizer.correct_word(word)
            assert was_changed is False
            assert norm_w == word

    def test_functional_words_never_mutated(self):
        """Ensures basic pronouns, prepositions, and stop words are never mutated."""
        words = ["that", "this", "it", "to", "in", "from", "here", "there"]
        for w in words:
            norm_w, was_changed = global_typo_normalizer.correct_word(w)
            assert was_changed is False
            assert norm_w == w


class TestIntentRoutingPrecedence:
    """Verifies that specific domain rules precede generic keyword matchers."""

    def test_why_uncertainty_vs_unreachable(self):
        """Differentiates sensor uncertainty from navigation unreachability."""
        agent = InvestigationAgent(max_steps=2)
        r_unc = agent.run("Why can't you detect the object?", agent_id="scout")
        assert r_unc["tool_calls"][0]["tool"] == "explain_uncertainty"

        r_unreach = agent.run("Explain why that destination is blocked.", agent_id="navigator")
        assert r_unreach["tool_calls"][0]["tool"] == "explain_unreachable"

    def test_exit_vs_navigate_door(self):
        """Differentiates nearest exit from navigation to a specific door."""
        agent = InvestigationAgent(max_steps=2)
        r_exit = agent.run("Find the nearest exit.", agent_id="navigator")
        assert r_exit["tool_calls"][0]["tool"] == "find_nearest_exit"

        r_section = agent.run("Which way leads to the next section?", agent_id="navigator")
        assert r_section["tool_calls"][0]["tool"] == "find_nearest_exit"

        r_door = agent.run("Take me to the door.", agent_id="navigator")
        assert r_door["tool_calls"][0]["tool"] == "navigate_to"
        assert r_door["tool_calls"][0]["arguments"]["destination"] == "door"

    def test_calculate_path_routing(self):
        """Ensures path calculation routes to find_path rather than find_nearest_exit."""
        agent = InvestigationAgent(max_steps=2)
        res = agent.run("Calculate a path to the door", agent_id="navigator")
        assert res["tool_calls"][0]["tool"] == "find_path"
        assert res["tool_calls"][0]["arguments"]["destination"] == "door"

    def test_tell_about_room_vs_visible_objects(self):
        """Ensures 'Tell me about this room' routes to get_current_location for Navigator."""
        agent = InvestigationAgent(max_steps=2)
        res = agent.run("Tell me about this room.", agent_id="navigator")
        assert res["tool_calls"][0]["tool"] == "get_current_location"

    def test_ambiguous_pointing_uncertainty(self):
        """Ensures blind pointing 'What is that over there?' without referent triggers explain_uncertainty."""
        agent = InvestigationAgent(max_steps=2)
        agent.reference_state.last_selected_object = None
        agent.reference_state.last_mentioned_object = None
        res = agent.run("What is that over there?", agent_id="scout")
        assert res["tool_calls"][0]["tool"] == "explain_uncertainty"


class TestContextIsolation:
    """Verifies that references in Room A do not leak into Room B."""

    def test_room_transition_clears_local_objects(self):
        """Switching rooms isolates object context."""
        state = ConversationReferenceState()
        state.last_selected_object = "table"
        state.last_mentioned_object = "table"
        state.current_room = "Reception / Central Hub"

        # Transition to Research Office
        state.reset_room_context("Research Office")
        assert state.last_selected_object is None
        assert state.last_mentioned_object is None
        assert state.current_room == "Research Office"
        assert state.last_mentioned_room == "Research Office"

