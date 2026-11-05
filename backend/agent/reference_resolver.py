"""Dedicated Conversational Reference Resolution Module for Spatial Quest.

Tracks structured, typed conversational references across turns:
- last_mentioned_object
- last_selected_object
- last_navigation_target
- last_mentioned_room
- last_inspected_clue
- last_active_mission_target

Resolves anaphoric expressions ("there", "it", "that", "the same object", "that object",
"from here", "near me") into explicit referents without indiscriminate noun replacement.
Guarantees clean context isolation across room transitions and mission switches.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List
import re

FACILITY_ROOMS = {
    "reception": "Reception / Central Hub",
    "hub": "Reception / Central Hub",
    "central hub": "Reception / Central Hub",
    "office": "Research Office",
    "research office": "Research Office",
    "library": "Library / Archives",
    "archives": "Library / Archives",
    "lab": "Investigation Lab",
    "laboratory": "Investigation Lab",
    "storage": "Storage Vault",
    "vault": "Storage Vault"
}

KNOWN_OBJECTS = {
    "table", "chair", "bookshelf", "cabinet", "sofa", "door", "desk",
    "pda", "safe", "datapad", "key", "master_key", "research_key",
    "laptop", "terminal", "exit"
}

@dataclass
class ConversationReferenceState:
    """Maintains strongly-typed contextual entity tracking across multi-turn dialogue."""
    last_mentioned_object: Optional[str] = None
    last_selected_object: Optional[str] = None
    last_navigation_target: Optional[str] = None
    last_mentioned_room: Optional[str] = None
    last_inspected_clue: Optional[str] = None
    last_active_mission_target: Optional[str] = None
    current_room: Optional[str] = None

    def update_from_query(self, query: str) -> None:
        """Extracts candidate entities explicitly mentioned in user query."""
        q_low = query.lower()

        # Check for room mentions
        for r_key, r_norm in FACILITY_ROOMS.items():
            if re.search(r"\b" + re.escape(r_key) + r"\b", q_low):
                self.last_mentioned_room = r_norm
                break

        # Check for object mentions (avoid generic pronouns like "it", "there", "any")
        for obj in KNOWN_OBJECTS:
            if re.search(r"\b" + re.escape(obj) + r"\b", q_low):
                self.last_mentioned_object = obj
                break

    def update_from_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        exec_res: Dict[str, Any]
    ) -> None:
        """Updates authoritative entity pointers from tool execution outcomes."""
        res_data = exec_res.get("result", {}) if isinstance(exec_res, dict) else {}
        if not isinstance(res_data, dict):
            res_data = {}

        # 1. find_nearest_target: outcome identifies the selected geometric target
        if tool_name == "find_nearest_target":
            found_obj = res_data.get("class_name") or res_data.get("nearest_object") or res_data.get("target_class")
            if found_obj and found_obj not in ("any", "object", "none"):
                self.last_selected_object = str(found_obj).lower().strip()
                self.last_mentioned_object = str(found_obj).lower().strip()
            elif tool_args.get("target_class") and tool_args["target_class"] != "any":
                self.last_mentioned_object = str(tool_args["target_class"]).lower().strip()

        # 2. locate_object: positive search sets mentioned & selected
        elif tool_name == "locate_object":
            obj_name = res_data.get("class_name") or tool_args.get("object_name")
            if obj_name:
                clean_name = str(obj_name).lower().strip()
                self.last_mentioned_object = clean_name
                self.last_selected_object = clean_name

        # 3. navigate_to / find_path: sets navigation target and object/room
        elif tool_name in ("navigate_to", "find_path"):
            dest = tool_args.get("destination")
            if dest:
                clean_dest = str(dest).lower().strip()
                self.last_navigation_target = clean_dest
                # Determine whether destination was a room or object
                matched_room = False
                for r_key, r_norm in FACILITY_ROOMS.items():
                    if r_key in clean_dest:
                        self.last_mentioned_room = r_norm
                        matched_room = True
                        break
                if not matched_room and clean_dest not in ("exit", "there"):
                    self.last_selected_object = clean_dest
                    self.last_mentioned_object = clean_dest

        # 4. inspect_clue: sets clue and selected object
        elif tool_name == "inspect_clue":
            clue = tool_args.get("clue_id") or tool_args.get("clue_name")
            if clue:
                clean_clue = str(clue).strip()
                self.last_inspected_clue = clean_clue
                self.last_selected_object = clean_clue

        # 5. inspect_object_closely:
        elif tool_name == "inspect_object_closely":
            obj = tool_args.get("object_name")
            if obj:
                clean_obj = str(obj).lower().strip()
                self.last_selected_object = clean_obj
                self.last_mentioned_object = clean_obj

        # 6. get_current_location: updates room reference
        elif tool_name == "get_current_location":
            rm = res_data.get("room_name") or tool_args.get("active_room")
            if rm:
                self.last_mentioned_room = str(rm).strip()
                self.current_room = str(rm).strip()

        # 7. get_mission_objectives / request_hint: tracks active mission target
        elif tool_name in ("get_mission_objectives", "request_hint"):
            target = res_data.get("active_target") or res_data.get("mission_target")
            if target:
                self.last_active_mission_target = str(target).strip()

    def reset_room_context(self, new_room: str) -> None:
        """Ensures context isolation when player transitions between rooms."""
        if self.current_room and self.current_room != new_room:
            # Clear room-local object pointers to avoid cross-room hallucinations
            self.last_selected_object = None
            self.last_mentioned_object = None
            self.last_navigation_target = None
        self.current_room = new_room
        self.last_mentioned_room = new_room

    def reset_mission_context(self, new_mission: str) -> None:
        """Clears mission-specific targets when mystery changes."""
        self.last_inspected_clue = None
        self.last_active_mission_target = None
        self.last_selected_object = None

    def to_dict(self) -> Dict[str, Any]:
        """Exports state for debugging and telemetry."""
        return asdict(self)


class ReferenceResolver:
    """Resolves anaphoric expressions in user utterances to typed entities."""

    def __init__(self, state: Optional[ConversationReferenceState] = None):
        self.state = state or ConversationReferenceState()

    def resolve(
        self,
        query: str,
        player_position: Optional[List[float]] = None,
        active_room: Optional[str] = None
    ) -> Tuple[str, bool, Optional[str]]:
        """Resolves pronouns and anaphoric targets in query.

        Returns:
            Tuple[resolved_query, was_modified, resolved_entity]
        """
        q_raw = query.strip()
        q_low = q_raw.lower()
        modified = False
        resolved_entity: Optional[str] = None

        # Synchronize room state for isolation
        if active_room:
            self.state.reset_room_context(active_room)

        # 1. Navigation Anaphora: "Take me there", "Go there", "Navigate there", "Guide me there"
        nav_there_match = re.search(r"\b(?:take me(?: to)?|guide me(?: to)?|navigate(?: to)?|lead me(?: to)?|head(?: to)?|go(?: to)?)\s+there\b", q_low)
        if nav_there_match:
            # Resolve target entity: preferred order: selected object -> navigation target -> mentioned room -> mentioned object
            target = (
                self.state.last_selected_object or
                self.state.last_navigation_target or
                self.state.last_mentioned_room or
                self.state.last_mentioned_object
            )
            if target and target not in ("there", "it"):
                # Ensure proper preposition "to the {target}"
                has_to = bool(re.search(r"\bto\s+there\b", q_low))
                replacement = f"the {target}" if has_to else f"to the {target}"
                resolved_query = re.sub(r"\bthere\b", replacement, q_raw, count=1, flags=re.IGNORECASE)
                return resolved_query, True, target

        # 2. Navigation Pronoun: "Take me to it", "Go to it", "Navigate to that", "Walk to it"
        nav_it_match = re.search(r"\b(?:take me(?: to)?|guide me(?: to)?|navigate(?: to)?|lead me(?: to)?|head(?: to)?|go(?: to)?)\s+(?:to\s+)?(?:it|that)\b", q_low)
        if nav_it_match:
            target = (
                self.state.last_selected_object or
                self.state.last_navigation_target or
                self.state.last_mentioned_object
            )
            if target and target not in ("it", "that", "any"):
                has_to = bool(re.search(r"\bto\s+(?:it|that)\b", q_low))
                replacement = f"the {target}" if has_to else f"to the {target}"
                resolved_query = re.sub(r"\b(?:it|that)\b", replacement, q_raw, count=1, flags=re.IGNORECASE)
                return resolved_query, True, target

        # 3. Distance Anaphora: "How far is it from here?", "How far is that from here?", "How far is it?"
        dist_it_here = re.search(r"\bhow far is (?:it|that)(?:\s+from (?:here|me))?\b", q_low)
        if dist_it_here:
            target = self.state.last_selected_object or self.state.last_mentioned_object
            if target and target not in ("it", "that", "any"):
                resolved_query = f"How far away is {target}"
                return resolved_query, True, target

        # 4. Proximity to Player: "Is it near me?", "Is that near me?", "Is it close to me?"
        near_me_match = re.search(r"\bis (?:it|that) (?:near|close to) me\b", q_low)
        if near_me_match:
            target = self.state.last_selected_object or self.state.last_mentioned_object
            if target and target not in ("it", "that", "any"):
                resolved_query = f"How far away is {target}"
                return resolved_query, True, target

        # 5. Spatial Relations Anaphora: "What is near that object?", "What is near it?", "What is next to it?"
        rel_it_match = re.search(r"\b(?:what is |which object is |is anything )?(near|next to|beside|left of|right of|above)\s+(?:that object|the same object|it|that)\b", q_low)
        if rel_it_match:
            rel_word = rel_it_match.group(1)
            target = self.state.last_selected_object or self.state.last_mentioned_object
            if target and target not in ("it", "that", "any"):
                resolved_query = f"What is {rel_word} the {target}?"
                return resolved_query, True, target

        # 6. Repetition Anaphora: "Navigate to the same object", "Take me to the same object"
        same_obj_match = re.search(r"\b(?:navigate to|take me to|go to|lead me to)\s+(?:the same object|that same object)\b", q_low)
        if same_obj_match:
            target = self.state.last_selected_object or self.state.last_mentioned_object
            if target:
                resolved_query = f"Take me to the {target}"
                return resolved_query, True, target

        # 7. Clue / Object Inspection Anaphora: "Tell me more about it", "Inspect it", "Examine it"
        inspect_it_match = re.search(r"\b(?:inspect|examine|tell me more about|read)\s+(?:it|that|this)\b", q_low)
        if inspect_it_match:
            if self.state.last_inspected_clue:
                resolved_query = f"Inspect clue {self.state.last_inspected_clue}"
                return resolved_query, True, self.state.last_inspected_clue
            elif self.state.last_selected_object:
                target = self.state.last_selected_object
                resolved_query = f"Inspect {target} closely"
                return resolved_query, True, target

        # 8. Ambiguous pointing with no active referent: "What is that over there?", "What is that?"
        if re.search(r"\bwhat is that(?:\s+over there)?\b", q_low):
            if not self.state.last_selected_object and not self.state.last_mentioned_object:
                # Return query preserved for explain_uncertainty to signal ambiguity
                return q_raw, False, None

        return q_raw, False, None
