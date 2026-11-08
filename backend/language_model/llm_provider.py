"""Hybrid LLM Provider for SpatialAI / Spatial Quest.

Provides dual-mode architecture:
1. 'from_scratch' (Academic Default): Custom in-house 4-layer Transformer
   and BPE tokenizer trained from random weights, 100% offline, compliant with CS3235.
2. 'gemini_api' (Enhanced Mode): Google Gemini API (gemini-2.0-flash / gemini-1.5-flash)
   with native function calling for advanced zero-shot conversational reasoning,
   while executing against the exact same local 13-tool registry.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests

from backend.tools.schemas import TOOL_SCHEMA_MAP

logger = logging.getLogger("spatial_ai.llm_provider")


class GeminiAPIClient:
    """Lightweight REST client for Google Gemini API with native function calling."""

    DEFAULT_MODEL = "gemini-2.0-flash"
    FALLBACK_MODEL = "gemini-1.5-flash"
    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.timeout = 8.0

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def _convert_schemas_to_gemini_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Converts registered Pydantic tool schemas to Gemini function_declarations."""
        target_tools = tool_names or list(TOOL_SCHEMA_MAP.keys())
        declarations = []

        type_map = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT"
        }

        for name in target_tools:
            if name not in TOOL_SCHEMA_MAP:
                continue
            schema_cls = TOOL_SCHEMA_MAP[name]
            raw_schema = schema_cls.model_json_schema()
            doc = schema_cls.__doc__ or f"Execute {name} command"
            description = doc.split("\n")[0].strip()

            properties = {}
            for prop_name, prop_data in raw_schema.get("properties", {}).items():
                p_type = type_map.get(prop_data.get("type", "string"), "STRING")
                prop_entry = {
                    "type": p_type,
                    "description": prop_data.get("description", prop_name)
                }
                if p_type == "ARRAY":
                    item_type = type_map.get(prop_data.get("items", {}).get("type", "string"), "STRING")
                    prop_entry["items"] = {"type": item_type}
                properties[prop_name] = prop_entry

            declarations.append({
                "name": name,
                "description": description,
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": raw_schema.get("required", [])
                }
            })

        return [{"function_declarations": declarations}]

    def generate_tool_call(
        self,
        query: str,
        system_prompt: str,
        context_info: Dict[str, Any],
        allowed_tools: Optional[List[str]] = None
    ) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        """Calls Gemini with native tools to select the best action.
        
        Returns:
            (tool_name, tool_arguments, direct_text_reply)
        """
        if not self.is_configured():
            return None, {}, None

        url = f"{self.API_BASE}/{self.model}:generateContent?key={self.api_key}"
        tools_payload = self._convert_schemas_to_gemini_tools(allowed_tools)

        user_content = (
            f"Active Room: {context_info.get('active_room', 'Reception Hub')}\n"
            f"Player Position: {context_info.get('player_position', [0, -15])}\n"
            f"Companion Drone Persona: {context_info.get('agent_id', 'scout')}\n"
            f"Investigator Query: {query}"
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "tools": tools_payload,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 300
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 404 and self.model != self.FALLBACK_MODEL:
                # Fallback to gemini-1.5-flash if 2.0-flash is unavailable
                url = f"{self.API_BASE}/{self.FALLBACK_MODEL}:generateContent?key={self.api_key}"
                resp = requests.post(url, json=payload, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
                return None, {}, None

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, {}, None

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "functionCall" in part:
                    fn = part["functionCall"]
                    return fn.get("name"), fn.get("args", {}), None
                if "text" in part:
                    return None, {}, part.get("text", "").strip()

            return None, {}, None
        except Exception as exc:
            logger.warning(f"Gemini API request failed: {exc}")
            return None, {}, None

    def synthesize_response(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        observation: Any,
        companion_persona: str = "Scout Drone"
    ) -> Optional[str]:
        """Uses Gemini to synthesize an engaging in-character response grounded in the tool observation."""
        if not self.is_configured():
            return None

        url = f"{self.API_BASE}/{self.model}:generateContent?key={self.api_key}"

        prompt = (
            f"You are the {companion_persona} in Spatial Quest (Sector 4 Facility).\n"
            f"Investigator asked: \"{query}\"\n"
            f"Executed Action: {tool_name} with args {json.dumps(tool_args)}\n"
            f"Observed Metric Data: {json.dumps(observation) if isinstance(observation, (dict, list)) else str(observation)}\n\n"
            f"Provide a concise, immersion-friendly report (2-3 sentences max) conveying the exact findings to the player. "
            f"Do not hallucinate coordinates or objects not in the observed data."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200}
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            return None
        except Exception:
            return None


class LLMProviderManager:
    """Singleton managing active LLM mode (from-scratch vs. Gemini API) and configuration."""

    MODE_FROM_SCRATCH = "from_scratch"
    MODE_GEMINI_API = "gemini_api"

    def __init__(self):
        self._mode = self.MODE_FROM_SCRATCH
        self.gemini_client = GeminiAPIClient()

        # Check environment variable for initial preference
        env_mode = os.environ.get("SPATIAL_LLM_MODE", "").lower().strip()
        if env_mode in (self.MODE_GEMINI_API, "gemini", "api"):
            if self.gemini_client.is_configured():
                self._mode = self.MODE_GEMINI_API

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, new_mode: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Toggles or sets the active LLM mode."""
        if api_key:
            self.gemini_client.api_key = api_key.strip()
            os.environ["GEMINI_API_KEY"] = api_key.strip()

        clean_mode = new_mode.lower().strip()
        if clean_mode in (self.MODE_GEMINI_API, "gemini", "api"):
            if not self.gemini_client.is_configured():
                return {
                    "success": False,
                    "mode": self._mode,
                    "message": "Gemini API key is required to activate API mode. Set GEMINI_API_KEY in your environment or enter it in the dialog."
                }
            self._mode = self.MODE_GEMINI_API
            return {
                "success": True,
                "mode": self._mode,
                "model": self.gemini_client.model,
                "message": f"Switched to Enhanced LLM Mode using Google Gemini API ({self.gemini_client.model})."
            }
        else:
            self._mode = self.MODE_FROM_SCRATCH
            return {
                "success": True,
                "mode": self._mode,
                "message": "Switched to Academic From-Scratch Mode (In-house 4-layer Transformer, 100% offline, CS3235 syllabus compliant)."
            }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current LLM status for telemetry and frontend UI."""
        return {
            "active_mode": self._mode,
            "is_api_available": self.gemini_client.is_configured(),
            "model_name": self.gemini_client.model if self._mode == self.MODE_GEMINI_API else "custom_4layer_transformer",
            "provider_name": "Google Gemini API" if self._mode == self.MODE_GEMINI_API else "SpatialAI Custom Transformer (From-Scratch)",
            "has_api_key": bool(self.gemini_client.api_key),
            "academic_syllabus_mode": (self._mode == self.MODE_FROM_SCRATCH)
        }


# Global singleton instance
global_llm_provider = LLMProviderManager()

