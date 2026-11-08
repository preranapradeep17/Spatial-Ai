"""Unit tests for the Hybrid LLM Provider (From-Scratch vs. Gemini API)."""
import os
import unittest
from unittest.mock import MagicMock, patch

from backend.language_model.llm_provider import GeminiAPIClient, LLMProviderManager


class TestLLMProvider(unittest.TestCase):
    """Verifies schema conversion, fallback handling, and mode switching."""

    def setUp(self):
        self.client = GeminiAPIClient(api_key="AIzaSyDummyKeyForTestingPurposes12345")
        self.manager = LLMProviderManager()

    def test_client_configuration_check(self):
        configured_client = GeminiAPIClient(api_key="AIzaSyDummyKeyForTestingPurposes12345")
        self.assertTrue(configured_client.is_configured())

        empty_client = GeminiAPIClient(api_key="")
        self.assertFalse(empty_client.is_configured())

    def test_schema_conversion(self):
        """Verifies that registered Pydantic tools convert correctly to Gemini function_declarations."""
        tools = self.client._convert_schemas_to_gemini_tools(["get_visible_objects", "calculate_distance"])
        self.assertEqual(len(tools), 1)
        declarations = tools[0]["function_declarations"]
        self.assertEqual(len(declarations), 2)

        names = [d["name"] for d in declarations]
        self.assertIn("get_visible_objects", names)
        self.assertIn("calculate_distance", names)

        # Check parameter typing
        dist_decl = next(d for d in declarations if d["name"] == "calculate_distance")
        self.assertEqual(dist_decl["parameters"]["type"], "OBJECT")
        self.assertIn("object_a", dist_decl["parameters"]["properties"])
        self.assertIn("object_b", dist_decl["parameters"]["properties"])
        self.assertEqual(dist_decl["parameters"]["properties"]["object_a"]["type"], "STRING")

    def test_manager_default_mode(self):
        """Manager must default to from_scratch for academic compliance."""
        self.assertEqual(self.manager.mode, LLMProviderManager.MODE_FROM_SCRATCH)
        status = self.manager.get_status()
        self.assertTrue(status["academic_syllabus_mode"])
        self.assertEqual(status["active_mode"], "from_scratch")

    def test_manager_switch_mode_without_key_fails(self):
        """Switching to API mode without a valid key must safely fail and stay from_scratch."""
        mgr = LLMProviderManager()
        mgr.gemini_client.api_key = ""
        res = mgr.set_mode("gemini_api")
        self.assertFalse(res["success"])
        self.assertEqual(mgr.mode, LLMProviderManager.MODE_FROM_SCRATCH)

    def test_manager_switch_mode_with_key_succeeds(self):
        """Switching to API mode with an explicit key must succeed."""
        mgr = LLMProviderManager()
        res = mgr.set_mode("gemini_api", api_key="AIzaSyDummyValidFormatKey1234567890")
        self.assertTrue(res["success"])
        self.assertEqual(mgr.mode, LLMProviderManager.MODE_GEMINI_API)
        status = mgr.get_status()
        self.assertEqual(status["active_mode"], "gemini_api")
        self.assertFalse(status["academic_syllabus_mode"])

        # Switch back to from_scratch
        res_back = mgr.set_mode("from_scratch")
        self.assertTrue(res_back["success"])
        self.assertEqual(mgr.mode, LLMProviderManager.MODE_FROM_SCRATCH)

    @patch("requests.post")
    def test_gemini_tool_call_mock(self, mock_post):
        """Test mocked Gemini API response parsing for tool calling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "calculate_distance",
                                    "args": {"target_object": "table"}
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        tool_name, tool_args, reply = self.client.generate_tool_call(
            query="How far is the table?",
            system_prompt="You are a scout drone.",
            context_info={"player_position": [0, -10]}
        )
        self.assertEqual(tool_name, "calculate_distance")
        self.assertEqual(tool_args, {"target_object": "table"})
        self.assertIsNone(reply)

    @patch("requests.post")
    def test_gemini_direct_text_reply_mock(self, mock_post):
        """Test mocked Gemini API response parsing for direct conversational text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Hello investigator! All facility sensors are operating normally."
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        tool_name, tool_args, reply = self.client.generate_tool_call(
            query="Hello drone",
            system_prompt="You are a scout drone.",
            context_info={}
        )
        self.assertIsNone(tool_name)
        self.assertEqual(reply, "Hello investigator! All facility sensors are operating normally.")


if __name__ == "__main__":
    unittest.main()
