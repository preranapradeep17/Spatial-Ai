"""Server-Side Google Gemini API Provider for Spatial Quest.

Uses the official `google-genai` Python SDK to drive conversational reasoning,
spatial scene-graph question answering, RAG synthesis, and Pydantic-validated
tool calling on behalf of the FastAPI backend.

Security & Configuration:
- Reads `GEMINI_API_KEY`, `GEMINI_MODEL`, `SPATIAL_LLM_MODE`, and `GEMINI_TIMEOUT_SECONDS`
  strictly from server environment / `.env` (loaded via python-dotenv).
- Never exposes API keys to frontend JavaScript or API responses.
- Executes only whitelisted, Pydantic-validated backend tools.
- Gracefully handles missing API keys, timeouts, quota limits, and upstream failures.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from backend.tools.schemas import TOOL_SCHEMA_MAP, validate_tool_arguments

_REPO_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_REPO_ENV_PATH):
    load_dotenv(dotenv_path=_REPO_ENV_PATH)
else:
    load_dotenv()

logger = logging.getLogger("spatial_ai.llm_provider")


class SessionMemoryManager:
    """Thread-safe bounded per-session conversational history store."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        sid = session_id or "default"
        with self._lock:
            return list(self._sessions.get(sid, []))

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        sid = session_id or "default"
        with self._lock:
            history = self._sessions.setdefault(sid, [])
            history.append({"role": "user", "text": user_text})
            history.append({"role": "model", "text": assistant_text})
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                self._sessions[sid] = history[-max_messages:]

    def clear_session(self, session_id: str) -> int:
        sid = session_id or "default"
        with self._lock:
            removed = len(self._sessions.get(sid, []))
            self._sessions.pop(sid, None)
            return removed


def build_spatial_context_snapshot(
    query: str,
    agent_id: str = "scout",
    active_room: Optional[str] = None,
    player_position: Optional[List[float]] = None,
    camera_heading: Optional[float] = None,
    selected_object_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Collects authoritative Scene Graph, Facility, Mission, and TF-IDF RAG context."""
    from backend.game.facility import (
        ROOM_DEFINITIONS,
        get_room_definition,
        get_room_for_position,
        parse_position_coords,
    )
    from backend.game.missions import global_game_session
    from backend.rag.retriever import global_retriever
    from backend.spatial.scene_graph import global_scene_graph

    px, py, pz = parse_position_coords(player_position or global_game_session.player_position)
    room_def = get_room_definition(active_room) or get_room_for_position(px, pz)
    room_name = room_def["name"]
    bounds = room_def["bounds"]

    all_objs = global_scene_graph.get_all_objects()
    live_ids = getattr(global_scene_graph, "live_perception_ids", set())

    in_room_objs = []
    other_objs = []
    for obj in all_objs:
        if not obj.estimated_3d_position:
            continue
        ox, oy, oz = obj.estimated_3d_position
        dist = round(math.dist([px, 0.0, pz], [ox, 0.0, oz]), 2)
        obj_room = get_room_for_position(ox, oz)["name"]
        provenance = "live_cnn_perception" if obj.object_id in live_ids else (
            getattr(obj, "source", None) or getattr(global_scene_graph, "source", "facility_layout")
        )
        entry = {
            "id": obj.object_id,
            "class_name": obj.class_name,
            "position": [round(ox, 2), round(oy, 2), round(oz, 2)],
            "distance_from_investigator_m": dist,
            "room": obj_room,
            "confidence": round(obj.confidence, 3),
            "depth_m": round(obj.depth_estimate, 2) if obj.depth_estimate is not None else None,
            "depth_source": getattr(obj, "depth_source", None) or ("estimated_bbox_height" if obj.object_id in live_ids else "seeded_layout"),
            "provenance": provenance,
        }
        if obj.object_id in live_ids or bounds["xmin"] <= ox <= bounds["xmax"] and bounds["zmin"] <= oz <= bounds["zmax"]:
            in_room_objs.append(entry)
        else:
            other_objs.append(entry)

    edges = global_scene_graph.compute_relations()
    room_obj_id_set = {o["id"] for o in in_room_objs}
    relevant_edges = []
    other_edges = []
    for e in edges:
        edge_txt = f"{e.source_id} is {e.relation} {e.target_id} ({e.distance_meters}m)"
        if e.source_id in room_obj_id_set or e.target_id in room_obj_id_set or (
            selected_object_id and (e.source_id == selected_object_id or e.target_id == selected_object_id)
        ):
            relevant_edges.append(edge_txt)
        else:
            other_edges.append(edge_txt)
    combined_edges = (relevant_edges + other_edges)[:28]

    # Pre-retrieve relevant facility archive passages via TF-IDF RAG (minimum relevance gate)
    if not global_retriever.is_indexed:
        doc_path = "data/documents/facility_documents.json"
        if os.path.exists(doc_path):
            global_retriever.load_documents_from_json(doc_path)
    rag_passages = global_retriever.retrieve(query, top_k=3, min_score=0.20)

    active_m = global_game_session.active_mission()
    objectives_summary = [
        {
            "id": o.get("objective_id"),
            "description": o.get("description"),
            "completed": bool(o.get("is_completed")),
            "points": o.get("points", 0),
        }
        for o in active_m.get("objectives", [])
    ]
    inventory_summary = [
        {"id": it.get("id"), "name": it.get("name"), "description": it.get("description")}
        for it in global_game_session.inventory
    ]

    selected_obj_info = None
    if selected_object_id:
        match = next((o for o in in_room_objs + other_objs if o["id"] == selected_object_id), None)
        selected_obj_info = match

    rag_block = "\n".join(
        f"- [{p.get('title', 'Document')} | score={p.get('score', 0)}]: {p.get('text', '')}"
        for p in rag_passages
    ) or "- None needed/matched above relevance threshold for this query (call `search_documents` or `search_safety_manual` if archival documents are requested)."

    zones_lines = "\n".join(
        f"  {i + 1}. {r['name']} — {r['description']} (bounds x:[{r['bounds']['xmin']},{r['bounds']['xmax']}], z:[{r['bounds']['zmin']},{r['bounds']['zmax']}])"
        for i, r in enumerate(ROOM_DEFINITIONS)
    )
    live_objs = [o for o in in_room_objs if o.get("provenance") == "live_cnn_perception"]
    seeded_in_room = [o for o in in_room_objs if o.get("provenance") != "live_cnn_perception"]
    live_block = "\n".join("  " + json.dumps(o) for o in live_objs) or "  (none — no live CNN camera scan detections currently in scene graph)"
    in_room_block = "\n".join("  " + json.dumps(o) for o in seeded_in_room) or "  (none)"
    other_room_block = "\n".join("  " + json.dumps(o) for o in other_objs) or "  (none)"

    snapshot = (
        f"### CURRENT FACILITY & INVESTIGATOR STATE\n"
        f"- Active Companion Focus Mode: {agent_id}\n"
        f"- Active Room: {room_name} ({room_def['description']})\n"
        f"- Investigator Position [X, Y, Z]: [{px:.2f}, {py:.2f}, {pz:.2f}] m | Camera Heading: {camera_heading if camera_heading is not None else 0.0} deg\n"
        f"- Active Mission: {active_m.get('title')} — {active_m.get('briefing')}\n"
        f"- Objectives: {json.dumps(objectives_summary)}\n"
        f"- Inventory: {json.dumps(inventory_summary)}\n"
        f"- Selected Object in UI: {json.dumps(selected_obj_info) if selected_obj_info else 'None'}\n\n"
        f"### ALL FACILITY ROOMS ({len(ROOM_DEFINITIONS)} ZONES)\n"
        f"{zones_lines}\n\n"
        f"### LIVE CNN PERCEPTION DETECTIONS (custom_detector) ({len(live_objs)})\n"
        f"{live_block}\n\n"
        f"### SEEDED FACILITY LAYOUT OBJECTS IN ACTIVE ROOM (facility_layout) ({len(seeded_in_room)})\n"
        f"{in_room_block}\n\n"
        f"### SEEDED FACILITY LAYOUT OBJECTS IN OTHER ROOMS (facility_layout) ({len(other_objs)})\n"
        f"{other_room_block}\n\n"
        f"### SCENE GRAPH: SPATIAL RELATIONSHIPS\n"
        f"{chr(10).join('- ' + r for r in combined_edges) if combined_edges else '- None computed'}\n\n"
        f"### PRE-RETRIEVED FACILITY ARCHIVE PASSAGES (TF-IDF RAG)\n"
        f"{rag_block}\n"
    )
    return snapshot, rag_passages


def normalize_response_formatting(text: str) -> str:
    """Repairs mid-number/mid-code line wraps and normalizes spacing without altering numeric values."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Join digits split across newlines inside security codes, metrics, or bold spans (e.g. "7\n391" -> "7391", "1\n4.5" -> "14.5")
    #    Protect actual numbered list items at line start (e.g., "\n2. Next step" or "\n2) Next step").
    cleaned = re.sub(r"(?<=\d)\n+(?=\d+(?!\s*[\.\)](?:\s|$)))", "", cleaned)
    cleaned = re.sub(r"(?<=\d\.)\n+(?=\d+(?!\s*[\.\)](?:\s|$)))", "", cleaned)
    cleaned = re.sub(r"(?<=\d)\n+(?=\.\d)", "", cleaned)

    # 2. Repair markdown bold/code wrappers containing broken whitespace around digits (e.g. "**7\n391**" -> "**7391**")
    cleaned = re.sub(r"\*\*(\d[\d\.\-]*)\s*\n+\s*([\d\.]+)\*\*", r"**\1\2**", cleaned)
    cleaned = re.sub(r"`(\d[\d\.\-]*)\s*\n+\s*([\d\.]+)`", r"`\1\2`", cleaned)

    # 3. Keep 3D coordinate tuples [X, Y, Z] on a single unbroken line
    cleaned = re.sub(
        r"\[([-\d.\s]+),\s*\n+\s*([-\d.\s]+),\s*\n+\s*([-\d.\s]+)\]",
        r"[\1, \2, \3]",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


SPATIAL_QUEST_SYSTEM_INSTRUCTION = """You are the Spatial Quest AI Investigation Partner assisting an investigator inside the Sector 4 3D Research Facility.

IDENTITY & ROLE:
- You act as the intelligent investigation assistant grounded in live Computer Vision (`SpatialDetector` / Scene Graph), 3D metric localization, TF-IDF RAG facility document search, and safe interactive navigation/tool execution.
- Sector 4 contains 5 connected rooms:
  1. Reception / Central Hub (Z < -10): visitor entry, security table, workstation terminal, lounge sofa.
  2. Research Office (Z in [-10, 0], X >= 0): office desk workstation, chair, filing cabinet.
  3. Library / Archives (Z in [-10, 0], X < 0): reference bookshelf, reading table, reading chair.
  4. Investigation Laboratory (Z in [0, 10]): central lab table/workbench, lab chair, lab cabinet, reference shelf, Security Door 3.
  5. Storage Room / Vault (Z > 10): archive bookshelves A & B, vault locker cabinet, emergency exit door.
- Companion focus modes adjust emphasis while keeping all tools available:
  * `scout`: Object detection, confidence, depth, FOV visibility, and live CNN (`custom_detector`) vs seeded `facility_layout` provenance.
  * `navigator`: 3D coordinates, metric distances, spatial relationships (`left_of`, `right_of`, `near`, `behind`, `in_front_of`, `above`, `on`), rooms, and turn-by-turn navigation.
  * `archivist`: Clues, TF-IDF facility documents, security override codes, safety protocols, and interactive unlocks.

STRICT GROUNDING & TOOL USE RULES:
1. NEVER invent, guess, or fabricate object locations, coordinates, passcodes, confidence scores, or facility lore.
2. Distinguish clearly between `facility_layout` (pre-seeded 3D architectural objects) and `live_cnn_perception` (`custom_detector` live camera detections). Never claim a seeded object came from live camera inference unless provenance is `custom_detector`.
3. Only state information verified in the Scene Graph snapshot, TF-IDF RAG passages, or executed tool outputs. If something is absent or unavailable, state that explicitly.
4. MANDATORY TOOL EXECUTION POLICY: Even when objects, coordinates, spatial edges, or pre-retrieved archive snippets appear in the Facility Context Snapshot, you MUST still emit the matching function tool call BEFORE answering so live metrics, 3D navigation, and mission objective progress verify and advance reliably — NEVER skip tool calls by reading from the snapshot alone:
   - Investigation role ("What is your role?"): explain your Spatial Quest investigation assistant role directly and concisely from IDENTITY & ROLE above.
   - Facility rooms overview ("What rooms are available in this facility?", "What are the rooms?", "List rooms"): ALWAYS call `get_current_location()` and list ALL 5 rooms from `available_rooms` with their names and descriptions.
   - Objects in a room ("What objects are in the investigation laboratory?"): ALWAYS call `get_visible_objects(room_name=...)`.
   - Live CNN detections ("What did the live camera scan detect?", "What live CV detections are visible?"): ALWAYS call `get_visible_objects(live_only=True)` and report ONLY `custom_detector` objects with their confidence, `[X, Y, Z]` position, depth, and `depth_source` (`measured_webgl_depth` vs `estimated_bbox_height`). If `live_only=True` returns empty, state clearly that no live CNN detections are currently recorded.
   - Finding an object ("Where is the chair?", "Where is the table located?", "Locate the table"): ALWAYS call `locate_object(object_name=...)` first and report ALL matching rooms/instances with exact `[X, Y, Z]` coordinates.
   - Nearest object ("Find the nearest bookshelf"): ALWAYS call `find_nearest_target(target_class=...)`.
   - Relationship between objects ("What is the relationship between the table and the chair?"): ALWAYS call `query_spatial_relation(object_name="table", target_object_name="chair", relation="all")` (and `calculate_distance` if needed), then report the exact spatial relationships and distances for the matching table–chair pairs across rooms (starting with the active room) rather than refusing or asking for clarification first.
   - Current investigation objective: ALWAYS call `get_mission_objectives()` rather than assuming objective status from prior turns.
   - Navigation ("Navigate me to the research office"): ALWAYS call `navigate_to(destination=...)` so the 3D avatar moves and turn-by-turn steps are returned.
   - Clues/codes/manuals ("Search documents", override codes, safety protocols): ALWAYS call `inspect_clue`, `search_documents`, `search_safety_manual`, or `interact_with_object`.
5. Keep responses concise, structured with clear markdown headings (`###`) and bullet points (`-`), keep numbers/coordinates/codes unbroken on a single line, and always ground spatial data in metric `[X, Y, Z]` values."""


class GeminiAPIClient:
    """Official google-genai SDK client with tool calling, memory, and streaming."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    FALLBACK_MODEL = "gemini-2.5-flash-lite"
    FALLBACK_MODELS = ("gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-3-flash-preview")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        env_model = os.environ.get("GEMINI_MODEL", "").strip()
        self.model = (model or env_model or self.DEFAULT_MODEL).strip()
        try:
            self.timeout = float(
                timeout_seconds if timeout_seconds is not None else os.environ.get("GEMINI_TIMEOUT_SECONDS", "15.0")
            )
        except ValueError:
            self.timeout = 15.0
        self._client: Optional[genai.Client] = None
        self._client_key: Optional[str] = None
        self._exhausted_until: Dict[str, float] = {}
        self._sticky_model: Optional[str] = None

    def is_configured(self) -> bool:
        key = self.api_key.strip()
        if not key or key in ("your_gemini_api_key_here", "YOUR_API_KEY"):
            return False
        return len(key) > 10

    def _get_client(self) -> genai.Client:
        if not self.is_configured():
            raise RuntimeError("GEMINI_API_KEY is not configured on the server.")
        if self._client is None or self._client_key != self.api_key:
            # Explicitly pin v1beta (required for Function Calling) and disable multi-minute SDK backoff hangs
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    api_version="v1beta",
                    timeout=int(self.timeout * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._client_key = self.api_key
        return self._client

    def _convert_schemas_to_gemini_tools(
        self,
        tool_names: Optional[List[str]] = None,
        tool_descriptions: Optional[Dict[str, str]] = None,
    ) -> List[types.Tool]:
        """Converts registered Pydantic schemas in TOOL_SCHEMA_MAP into Gemini Tool declarations."""
        target_tools = tool_names or list(TOOL_SCHEMA_MAP.keys())
        declarations: List[types.FunctionDeclaration] = []

        type_map = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        for name in target_tools:
            if name not in TOOL_SCHEMA_MAP:
                continue
            schema_cls = TOOL_SCHEMA_MAP[name]
            raw_schema = schema_cls.model_json_schema()
            desc = (tool_descriptions or {}).get(name) or (schema_cls.__doc__ or f"Execute {name} command").split("\n")[0].strip()

            properties: Dict[str, Any] = {}
            for prop_name, prop_data in raw_schema.get("properties", {}).items():
                raw_type = prop_data.get("type")
                if not raw_type and "anyOf" in prop_data:
                    for opt in prop_data["anyOf"]:
                        if opt.get("type") and opt.get("type") != "null":
                            raw_type = opt["type"]
                            if opt["type"] == "array" and "items" in opt:
                                prop_data = {**prop_data, "items": opt["items"]}
                            break
                p_type = type_map.get(raw_type or "string", "STRING")
                prop_entry: Dict[str, Any] = {
                    "type": p_type,
                    "description": prop_data.get("description", prop_name),
                }
                if p_type == "ARRAY":
                    item_type = type_map.get(prop_data.get("items", {}).get("type", "number"), "STRING")
                    prop_entry["items"] = {"type": item_type}
                properties[prop_name] = prop_entry

            if properties:
                param_schema: Dict[str, Any] = {
                    "type": "OBJECT",
                    "properties": properties,
                }
                req = raw_schema.get("required", [])
                if req:
                    param_schema["required"] = req
                declarations.append(
                    types.FunctionDeclaration(
                        name=name,
                        description=desc,
                        parameters=param_schema,
                    )
                )
            else:
                declarations.append(
                    types.FunctionDeclaration(
                        name=name,
                        description=desc,
                    )
                )

        return [types.Tool(function_declarations=declarations)]

    def _build_contents(
        self,
        query: str,
        context_snapshot: str,
        history: List[Dict[str, str]],
    ) -> List[types.Content]:
        contents: List[types.Content] = []
        for turn in history:
            role = "model" if turn.get("role") == "model" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=turn.get("text", ""))])
            )
        user_payload = f"{context_snapshot}\n### USER QUESTION / INSTRUCTION\n{query}"
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_payload)])
        )
        return contents

    @staticmethod
    def _classify_api_error(exc: Exception) -> Tuple[str, str]:
        msg = str(exc)
        low = msg.lower()
        if isinstance(exc, TimeoutError) or "timeout" in low or "timed out" in low or "504" in msg or "deadline_exceeded" in low:
            return "timeout", "Gemini API request timed out. Please retry in a moment."
        if "429" in msg or "resource_exhausted" in low or "quota" in low:
            return "quota_exceeded", "Gemini API quota or rate limit reached. Please wait briefly or check your billing/quota."
        if "503" in msg or "unavailable" in low or "high demand" in low:
            return "service_unavailable", "Gemini API temporarily overloaded (503). Please retry shortly."
        if "401" in msg or "403" in msg or "api key" in low or "permission_denied" in low:
            return "auth_error", "Gemini API key was rejected or lacks permission. Verify GEMINI_API_KEY in `.env`."
        if "404" in msg or "not found" in low:
            return "model_not_found", f"Configured Gemini model not found: {msg}"
        return "api_unavailable", f"Gemini API unavailable: {msg[:240]}"

    @staticmethod
    def _is_daily_quota_error(exc: Exception) -> bool:
        low = str(exc).lower()
        return any(tok in low for tok in ("perday", "per_day", "generaterequestsperday", "rpd"))

    def _generate_with_model_fallback(
        self,
        client: genai.Client,
        primary_model: str,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
        pin_model_family: bool = False,
    ) -> Tuple[Optional[Any], str, Optional[str], Optional[str]]:
        """Executes generate_content cleanly with sticky active model and quota cooldown."""
        now = time.time()
        if pin_model_family:
            # Never cross into another model alias once a functionCall has been emitted;
            # models such as gemini-flash-lite-latest or gemini-3 require matching thought_signatures.
            all_candidates = [primary_model]
            candidates_to_try = [primary_model]
        else:
            ordered: List[str] = []
            if self._sticky_model and self._exhausted_until.get(self._sticky_model, 0.0) <= now:
                ordered.append(self._sticky_model)
            if primary_model not in ordered:
                ordered.append(primary_model)
            for m in self.FALLBACK_MODELS:
                if m not in ordered:
                    ordered.append(m)
            all_candidates = ordered
            candidates_to_try = [m for m in all_candidates if self._exhausted_until.get(m, 0.0) <= now]
            if not candidates_to_try:
                candidates_to_try = all_candidates

        last_code = "api_unavailable"
        last_msg = "Gemini API unavailable."

        for m_name in candidates_to_try:
            response = None
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as exc:
                    code, friendly = self._classify_api_error(exc)
                    last_code, last_msg = code, friendly
                    if code == "service_unavailable" and attempt + 1 < max_attempts:
                        time.sleep(0.5)
                        continue
                    if code == "quota_exceeded":
                        if self._is_daily_quota_error(exc):
                            self._exhausted_until[m_name] = time.time() + 3600.0
                            if self._sticky_model == m_name:
                                self._sticky_model = None
                            response = None
                            break
                        if attempt + 1 < max_attempts:
                            time.sleep(0.6)
                            continue
                        self._exhausted_until[m_name] = time.time() + 90.0
                        if self._sticky_model == m_name:
                            self._sticky_model = None
                    if code in ("model_not_found", "quota_exceeded", "service_unavailable", "timeout"):
                        if code == "model_not_found":
                            self._exhausted_until[m_name] = time.time() + 3600.0
                        response = None
                        break
                    return None, m_name, code, friendly

            if response is None:
                continue

            cand = response.candidates[0] if response.candidates else None
            if not cand or not cand.content or not cand.content.parts:
                last_code, last_msg = "empty_response", "Gemini returned an empty candidate response."
                continue
            self._sticky_model = m_name
            return response, m_name, None, None

        return None, candidates_to_try[-1], last_code, last_msg

    def run_conversation_turn(
        self,
        query: str,
        context_snapshot: str,
        history: List[Dict[str, str]],
        tool_executor: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        tool_descriptions: Optional[Dict[str, str]] = None,
        max_steps: int = 4,
    ) -> Dict[str, Any]:
        """Executes a complete multi-step Gemini tool-calling turn cleanly."""
        if not self.is_configured():
            return {
                "success": False,
                "error_code": "missing_api_key",
                "error_message": "GEMINI_API_KEY is not configured in environment.",
                "tool_calls": [],
                "tool_results": [],
                "final_response": "",
                "model_used": self.model,
            }

        client = self._get_client()
        tools = self._convert_schemas_to_gemini_tools(tool_descriptions=tool_descriptions)
        contents = self._build_contents(query, context_snapshot, history)

        active_model = self.model
        tool_calls_log: List[Dict[str, Any]] = []
        tool_results_log: List[Dict[str, Any]] = []
        raw_parts_log: List[str] = []

        for _step in range(max_steps):
            config = types.GenerateContentConfig(
                system_instruction=SPATIAL_QUEST_SYSTEM_INSTRUCTION,
                tools=tools,
                temperature=0.2,
                max_output_tokens=2048,
            )
            response, active_model, err_code, err_msg = self._generate_with_model_fallback(
                client, active_model, contents, config, pin_model_family=(_step > 0)
            )
            if response is None:
                return {
                    "success": False,
                    "error_code": err_code or "api_unavailable",
                    "error_message": err_msg or "Gemini API unavailable.",
                    "tool_calls": tool_calls_log,
                    "tool_results": tool_results_log,
                    "final_response": "",
                    "model_used": active_model,
                }

            candidate = response.candidates[0]

            contents.append(candidate.content)
            fn_calls = [p.function_call for p in candidate.content.parts if p.function_call]

            if not fn_calls:
                text_chunks = [p.text for p in candidate.content.parts if p.text]
                final_text = normalize_response_formatting("\n".join(text_chunks))
                if not final_text:
                    return {
                        "success": False,
                        "error_code": "empty_response",
                        "error_message": "Gemini returned no text output.",
                        "tool_calls": tool_calls_log,
                        "tool_results": tool_results_log,
                        "final_response": "",
                        "model_used": active_model,
                    }
                raw_parts_log.append(final_text)
                return {
                    "success": True,
                    "tool_calls": tool_calls_log,
                    "tool_results": tool_results_log,
                    "final_response": final_text,
                    "model_raw_output": "\n".join(raw_parts_log),
                    "model_used": active_model,
                }

            response_parts: List[types.Part] = []
            for fc in fn_calls:
                t_name = str(fc.name or "").strip()
                raw_args = dict(fc.args or {})
                is_valid, validated_args, val_err = validate_tool_arguments(t_name, raw_args)
                if not is_valid:
                    exec_res = {"success": False, "tool": t_name, "error": f"Validation error: {val_err}"}
                else:
                    exec_res = tool_executor(t_name, validated_args)

                tool_calls_log.append({"tool": t_name, "arguments": validated_args if is_valid else raw_args})
                tool_results_log.append(exec_res)
                raw_parts_log.append(f"<tool_call>{json.dumps({'name': t_name, 'arguments': validated_args})}</tool_call>")

                response_parts.append(
                    types.Part.from_function_response(
                        name=t_name,
                        response={"result": exec_res},
                    )
                )

            # Official Gemini v1beta expects FunctionResponse parts under role="user"
            contents.append(types.Content(role="user", parts=response_parts))

        # Final synthesis turn without tools if max_steps completed right after tool calls
        synth_cfg = types.GenerateContentConfig(
            system_instruction=SPATIAL_QUEST_SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=2048,
        )
        synth_resp, active_model, s_code, s_msg = self._generate_with_model_fallback(
            client, active_model, contents, synth_cfg, pin_model_family=bool(tool_calls_log)
        )
        if synth_resp is None:
            return {
                "success": False,
                "error_code": s_code or "empty_response",
                "error_message": s_msg or "Gemini returned an empty synthesis response.",
                "tool_calls": tool_calls_log,
                "tool_results": tool_results_log,
                "final_response": "",
                "model_used": active_model,
            }
        s_cand = synth_resp.candidates[0]
        s_parts = s_cand.content.parts if s_cand and s_cand.content else []
        s_text = normalize_response_formatting("\n".join(p.text for p in (s_parts or []) if p.text))
        if s_text:
            raw_parts_log.append(s_text)
            return {
                "success": True,
                "tool_calls": tool_calls_log,
                "tool_results": tool_results_log,
                "final_response": s_text,
                "model_raw_output": "\n".join(raw_parts_log),
                "model_used": active_model,
            }

        return {
            "success": False,
            "error_code": "empty_response",
            "error_message": "Gemini did not produce a final text response after tool calls.",
            "tool_calls": tool_calls_log,
            "tool_results": tool_results_log,
            "final_response": "",
            "model_used": active_model,
        }

    def stream_final_text(
        self,
        contents: List[types.Content],
        tools: List[types.Tool],
        active_model: str,
    ) -> Generator[str, None, None]:
        """Streams final natural language response tokens using Gemini streaming API."""
        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=SPATIAL_QUEST_SYSTEM_INSTRUCTION,
            tools=tools,
            temperature=0.2,
            max_output_tokens=2048,
        )
        stream = client.models.generate_content_stream(
            model=active_model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            if chunk.candidates:
                parts = chunk.candidates[0].content.parts if chunk.candidates[0].content else []
                for p in parts or []:
                    if p.text:
                        yield p.text


class LLMProviderManager:
    """Singleton managing Gemini API provider, session memory, and graceful fallback."""

    MODE_GEMINI_API = "gemini_api"
    MODE_FROM_SCRATCH = "from_scratch"

    def __init__(self):
        self.gemini_client = GeminiAPIClient()
        self.memory = SessionMemoryManager(max_turns=10)
        self.last_error: Optional[str] = None
        self.last_model_used: Optional[str] = None
        self.last_used_fallback: bool = False
        self._manual_override: bool = False

        env_mode = os.environ.get("SPATIAL_LLM_MODE", "").lower().strip()
        if env_mode in (self.MODE_FROM_SCRATCH, "offline", "local"):
            self._mode = self.MODE_FROM_SCRATCH
        elif self.gemini_client.is_configured():
            self._mode = self.MODE_GEMINI_API
        elif env_mode in (self.MODE_GEMINI_API, "gemini", "api"):
            self._mode = self.MODE_GEMINI_API
        else:
            # No API key configured in offline/test environments -> safe offline fallback default
            self._mode = self.MODE_FROM_SCRATCH

    @property
    def mode(self) -> str:
        return self._mode

    def refresh_from_env(self) -> None:
        # Respect explicit pytest env patching when PYTEST_CURRENT_TEST is active
        should_override = not bool(os.environ.get("PYTEST_CURRENT_TEST"))
        if os.path.exists(_REPO_ENV_PATH):
            load_dotenv(dotenv_path=_REPO_ENV_PATH, override=should_override)
        else:
            load_dotenv(override=should_override)
        self.gemini_client.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        env_model = os.environ.get("GEMINI_MODEL", "").strip()
        if env_model and env_model != self.gemini_client.model:
            self.gemini_client.model = env_model
            self.gemini_client._sticky_model = None
            self.gemini_client._exhausted_until.clear()
        if not self._manual_override:
            env_mode = os.environ.get("SPATIAL_LLM_MODE", "").lower().strip()
            if env_mode in (self.MODE_FROM_SCRATCH, "offline", "local"):
                self._mode = self.MODE_FROM_SCRATCH
            elif self.gemini_client.is_configured():
                self._mode = self.MODE_GEMINI_API

    def sync_env_if_needed(self) -> None:
        if not self._manual_override and not os.environ.get("PYTEST_CURRENT_TEST"):
            self.refresh_from_env()

    def record_turn_outcome(
        self,
        model_name: Optional[str],
        used_fallback: bool,
        error: Optional[str] = None,
    ) -> None:
        """Tracks the actual model executed on the most recent turn for UI status accuracy."""
        self.last_used_fallback = bool(used_fallback)
        self.last_error = error
        if model_name:
            self.last_model_used = model_name

    def set_mode(self, new_mode: str) -> Dict[str, Any]:
        """Switches active mode on server (API keys are read strictly from server env)."""
        self.refresh_from_env()
        self._manual_override = True
        self.last_model_used = None
        self.last_used_fallback = False
        clean_mode = (new_mode or "").lower().strip()
        if clean_mode in (self.MODE_GEMINI_API, "gemini", "api"):
            if not self.gemini_client.is_configured():
                return {
                    "success": False,
                    "mode": self._mode,
                    "is_api_available": False,
                    "message": "GEMINI_API_KEY is not set in the server environment or `.env` file.",
                }
            self._mode = self.MODE_GEMINI_API
            return {
                "success": True,
                "mode": self._mode,
                "is_api_available": True,
                "model": self.gemini_client.model,
                "message": f"Switched to Google Gemini API ({self.gemini_client.model}).",
            }
        self._mode = self.MODE_FROM_SCRATCH
        return {
            "success": True,
            "mode": self._mode,
            "is_api_available": self.gemini_client.is_configured(),
            "message": "Switched to Offline Fallback Mode (From-Scratch Transformer + Deterministic Guardrails).",
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns safe status metadata without ever exposing the API key."""
        self.refresh_from_env()
        configured = self.gemini_client.is_configured()
        active = self._mode == self.MODE_GEMINI_API and configured
        actual_model = (
            self.last_model_used
            if (active and self.last_model_used and not self.last_used_fallback)
            else (self.gemini_client.model if active else "custom_4layer_transformer_fallback")
        )
        return {
            "active_mode": self.MODE_GEMINI_API if active else self.MODE_FROM_SCRATCH,
            "configured_mode": self._mode,
            "is_api_available": configured,
            "model_name": self.gemini_client.model if active else "custom_4layer_transformer_fallback",
            "effective_model_name": actual_model,
            "last_model_used": self.last_model_used,
            "last_used_fallback": self.last_used_fallback,
            "provider_name": f"Google Gemini ({actual_model})" if active else "Offline Fallback (Transformer + Scene Graph)",
            "has_api_key": configured,
            "last_error": self.last_error,
        }


global_llm_provider = LLMProviderManager()
