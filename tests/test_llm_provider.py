"""Unit tests for the Google Gemini SDK Provider, Session Memory, and Spatial Context Builder."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.game.facility import seed_facility_scene
from backend.language_model.llm_provider import (
    GeminiAPIClient,
    LLMProviderManager,
    SessionMemoryManager,
    build_spatial_context_snapshot,
)


class TestLLMProvider(unittest.TestCase):
    """Verifies official google-genai SDK integration, tool conversion, session memory, and error handling."""

    def setUp(self):
        self.client = GeminiAPIClient(api_key="AIzaSyDummyKeyForTestingPurposes12345")

    def test_client_configuration_check(self):
        self.assertTrue(GeminiAPIClient(api_key="AIzaSyDummyKeyForTestingPurposes12345").is_configured())
        self.assertFalse(GeminiAPIClient(api_key="").is_configured())
        self.assertFalse(GeminiAPIClient(api_key="your_gemini_api_key_here").is_configured())

    def test_schema_conversion_uses_genai_types(self):
        tools = self.client._convert_schemas_to_gemini_tools(["get_visible_objects", "calculate_distance", "get_inventory"])
        self.assertEqual(len(tools), 1)
        declarations = tools[0].function_declarations
        self.assertEqual(len(declarations), 3)

        names = [d.name for d in declarations]
        self.assertIn("get_visible_objects", names)
        self.assertIn("calculate_distance", names)
        self.assertIn("get_inventory", names)

        dist_decl = next(d for d in declarations if d.name == "calculate_distance")
        self.assertIn("object_a", dist_decl.parameters.properties)
        self.assertIn("object_b", dist_decl.parameters.properties)

        inv_decl = next(d for d in declarations if d.name == "get_inventory")
        self.assertIsNone(inv_decl.parameters)

    def test_client_uses_v1beta_api_version(self):
        with patch("backend.language_model.llm_provider.genai.Client") as mock_ctor:
            self.client._get_client()
            mock_ctor.assert_called_once()
            _, kwargs = mock_ctor.call_args
            self.assertEqual(kwargs["http_options"].api_version, "v1beta")

    def test_session_memory_isolation_and_clear(self):
        mem = SessionMemoryManager(max_turns=2)
        mem.append_turn("s1", "Where is the table?", "In the Research Office.")
        mem.append_turn("s1", "How far is it?", "4.2 meters.")
        mem.append_turn("s1", "Take me there", "Plotting route.")
        mem.append_turn("s2", "Hello", "Ready.")

        h1 = mem.get_history("s1")
        h2 = mem.get_history("s2")
        self.assertEqual(len(h1), 4)  # bounded to max_turns * 2 = 4
        self.assertEqual(h1[0]["text"], "How far is it?")
        self.assertEqual(len(h2), 2)

        removed = mem.clear_session("s1")
        self.assertEqual(removed, 4)
        self.assertEqual(mem.get_history("s1"), [])
        self.assertEqual(len(mem.get_history("s2")), 2)

    def test_spatial_context_snapshot_contains_scene_and_rag(self):
        seed_facility_scene()
        snapshot, rag_passages = build_spatial_context_snapshot(
            query="What is the Door 3 override code?",
            agent_id="archivist",
            active_room="Investigation Laboratory",
            player_position=[0.0, 0.0],
            selected_object_id="door_3",
        )
        self.assertIn("Investigation Laboratory", snapshot)
        self.assertIn("Selected Object in UI:", snapshot)
        self.assertIn("PRE-RETRIEVED FACILITY ARCHIVE PASSAGES", snapshot)
        self.assertTrue(len(rag_passages) >= 1)

    def test_manager_without_env_key_falls_back_safely(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "SPATIAL_LLM_MODE": "gemini_api"}, clear=False):
            with patch("backend.language_model.llm_provider.load_dotenv"):
                mgr = LLMProviderManager()
                mgr.gemini_client.api_key = ""
                status = mgr.get_status()
                self.assertEqual(status["active_mode"], LLMProviderManager.MODE_FROM_SCRATCH)
                self.assertFalse(status["is_api_available"])
                res = mgr.set_mode("gemini_api")
                self.assertFalse(res["success"])

    def test_manager_with_env_key_activates_gemini_and_never_exposes_key(self):
        fake_key = "AIzaSyServerOnlySecretKey987654321"
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": fake_key, "GEMINI_MODEL": "gemini-2.5-flash", "SPATIAL_LLM_MODE": "gemini_api"},
            clear=False,
        ):
            with patch("backend.language_model.llm_provider.load_dotenv"):
                mgr = LLMProviderManager()
                status = mgr.get_status()
                self.assertEqual(status["active_mode"], LLMProviderManager.MODE_GEMINI_API)
                self.assertTrue(status["is_api_available"])
                self.assertEqual(status["model_name"], "gemini-2.5-flash")
                self.assertNotIn(fake_key, str(status))

    def test_gemini_multi_step_tool_call_and_synthesis_mock(self):
        """Simulates step 1 returning function_call and step 2 returning final grounded text."""
        fc_part = SimpleNamespace(
            function_call=SimpleNamespace(name="locate_object", args={"object_name": "table"}),
            text=None,
        )
        text_part = SimpleNamespace(
            function_call=None,
            text="The central investigation table is located at [0.0, 0.75, 0.0] in the Investigation Laboratory.",
        )

        resp_step1 = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[fc_part]))])
        resp_step2 = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[text_part]))])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [resp_step1, resp_step2]

        with patch.object(self.client, "_get_client", return_value=mock_client):
            executor = MagicMock(return_value={"success": True, "tool": "locate_object", "result": {"found": True}})
            result = self.client.run_conversation_turn(
                query="Where is the table?",
                context_snapshot="FACILITY CONTEXT",
                history=[],
                tool_executor=executor,
                max_steps=4,
            )

        self.assertTrue(result["success"])
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["tool"], "locate_object")
        self.assertEqual(result["tool_calls"][0]["arguments"], {"object_name": "table"})
        self.assertIn("Investigation Laboratory", result["final_response"])
        executor.assert_called_once_with("locate_object", {"object_name": "table"})

    def test_gemini_error_classification(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("Read timed out")

        with patch.object(self.client, "_get_client", return_value=mock_client):
            result = self.client.run_conversation_turn(
                query="Check status",
                context_snapshot="CONTEXT",
                history=[],
                tool_executor=MagicMock(),
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "timeout")

    def test_gemini_empty_response_returns_clean_failure(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(candidates=[])
        with patch.object(self.client, "_get_client", return_value=mock_client):
            result = self.client.run_conversation_turn(
                query="Check status",
                context_snapshot="CONTEXT",
                history=[],
                tool_executor=MagicMock(),
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "empty_response")

    def test_spatial_rag_noise_suppressed_and_pair_relation_works(self):
        from backend.spatial.scene_graph import global_scene_graph
        seed_facility_scene()
        _, rag_spatial = build_spatial_context_snapshot(
            query="Where is the chair located?",
            agent_id="scout",
            active_room="Reception / Central Hub",
            player_position=[0.0, -17.2],
        )
        self.assertEqual(rag_spatial, [])
        rels = global_scene_graph.query_spatial_relation("table", "all", target_obj_name="chair")
        self.assertTrue(len(rels) > 0)
        self.assertTrue(any(r["relation"] in ("near", "in_front_of", "behind", "left_of", "right_of") for r in rels))

    def test_rooms_not_mutated_to_doors_and_memory_query_safe(self):
        from backend.agent.investigation_agent import InvestigationAgent
        from backend.agent.typo_normalizer import global_typo_normalizer
        norm_q, corrs = global_typo_normalizer.normalize("What rooms are available in this facility?")
        self.assertEqual(norm_q, "What rooms are available in this facility?")
        self.assertNotIn("rooms", corrs)
        agent = InvestigationAgent(max_steps=2)
        res_rooms = agent.run("What rooms are available in this facility?", agent_id="scout", session_id="t_rooms")
        self.assertTrue(any(tc["tool"] == "get_current_location" for tc in res_rooms["tool_calls"]))
        res_mem = agent.run(
            "In my previous messages, what did I ask you about Door or rooms, and which room did we just navigate to?",
            agent_id="scout",
            session_id="t_rooms",
        )
        self.assertFalse(any(tc["tool"] == "navigate_to" for tc in res_mem["tool_calls"]))

    def test_transient_503_retries_once_and_recovers(self):
        text_part = SimpleNamespace(function_call=None, text="Recovered cleanly after transient 503.")
        ok_resp = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[text_part]))])
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            RuntimeError("503 UNAVAILABLE"),
            ok_resp,
        ]
        with patch.object(self.client, "_get_client", return_value=mock_client), \
             patch("backend.language_model.llm_provider.time.sleep"):
            result = self.client.run_conversation_turn(
                query="Check status",
                context_snapshot="CONTEXT",
                history=[],
                tool_executor=MagicMock(),
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["final_response"], "Recovered cleanly after transient 503.")

    def test_pin_model_family_strict_and_status_tracks_actual_model(self):
        fake_key = "AIzaSyServerOnlySecretKey987654321"
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": fake_key, "GEMINI_MODEL": "gemini-2.5-flash", "SPATIAL_LLM_MODE": "gemini_api"},
            clear=False,
        ):
            with patch("backend.language_model.llm_provider.load_dotenv"):
                mgr = LLMProviderManager()
                mgr.record_turn_outcome(model_name="gemini-2.5-flash-lite", used_fallback=False, error=None)
                st = mgr.get_status()
                self.assertEqual(st["last_model_used"], "gemini-2.5-flash-lite")
                self.assertEqual(st["effective_model_name"], "gemini-2.5-flash-lite")
                self.assertFalse(st["last_used_fallback"])

        # Ensure pin_model_family=True never switches into another alias after a functionCall
        fc_part = SimpleNamespace(function_call=SimpleNamespace(name="get_inventory", args={}), text=None)
        resp_step1 = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[fc_part]))])
        mock_client = MagicMock()
        seen_models = []

        def fake_gen(model, contents, config):
            seen_models.append(model)
            if len(seen_models) == 1:
                return resp_step1
            raise RuntimeError("404 not found")

        mock_client.models.generate_content.side_effect = fake_gen
        with patch.object(self.client, "_get_client", return_value=mock_client):
            self.client.run_conversation_turn(
                query="Inventory?",
                context_snapshot="CTX",
                history=[],
                tool_executor=MagicMock(return_value={"success": True}),
            )
        self.assertEqual(seen_models, ["gemini-2.5-flash", "gemini-2.5-flash"])

    def test_reception_to_office_route_avoids_table_and_passes_doorway(self):
        from backend.game.facility import compute_turn_by_turn_route
        route = compute_turn_by_turn_route(
            start_pos=[0.0, -17.2],
            target_pos=[6.0, 0.75, -5.0],
            target_name="Research Office",
            target_room="Research Office",
            origin_room_override="Reception / Central Hub",
        )
        coords = [w["coord"] for w in route["waypoints"]]
        self.assertTrue(any(abs(c[0]) >= 1.8 and -17.0 <= c[2] <= -11.0 for c in coords))
        self.assertTrue(any(abs(c[0] - 2.8) < 0.1 and abs(c[2] - (-5.0)) < 0.1 for c in coords))





    def test_daily_quota_marks_sticky_and_cools_down_model(self):
        from backend.language_model.llm_provider import GeminiAPIClient
        text_part = SimpleNamespace(function_call=None, text="Served by fallback flash-lite.")
        ok_resp = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[text_part]))])
        mock_client = MagicMock()
        seen_models = []

        def fake_gen(model, contents, config):
            seen_models.append(model)
            if model == "gemini-2.5-flash":
                raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
            return ok_resp

        mock_client.models.generate_content.side_effect = fake_gen
        client = GeminiAPIClient(api_key="AIzaSyDummyKeyForTestingPurposes12345", model="gemini-2.5-flash")
        with patch.object(client, "_get_client", return_value=mock_client):
            res1 = client.run_conversation_turn("Where is the chair?", "CTX", [], MagicMock())
            res2 = client.run_conversation_turn("Where is the table?", "CTX", [], MagicMock())

        self.assertTrue(res1["success"])
        self.assertTrue(res2["success"])
        self.assertEqual(res1["model_used"], "gemini-2.5-flash-lite")
        self.assertEqual(res2["model_used"], "gemini-2.5-flash-lite")
        # Second turn should NOT retry daily-exhausted gemini-2.5-flash and should go straight to sticky model
        self.assertEqual(seen_models, ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash-lite"])

    def test_normalize_response_formatting_repairs_split_numbers(self):
        from backend.language_model.llm_provider import normalize_response_formatting
        raw = (
            "Security code is 7\n391.\n"
            "Estimated range: 1\n4.5 m to [0.0,\n 0.75,\n 0.0].\n"
            "1. First step.\n"
            "2. Second step."
        )
        fixed = normalize_response_formatting(raw)
        self.assertIn("7391", fixed)
        self.assertIn("14.5 m", fixed)
        self.assertIn("[0.0, 0.75, 0.0]", fixed)
        self.assertIn("1. First step.\n2. Second step.", fixed)


if __name__ == "__main__":
    unittest.main()
