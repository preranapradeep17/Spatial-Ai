"""Mission progression, scoring, inventory, and companion-agent personas for Spatial Quest.

Supports replayable mystery hunts (Mysteries A–C) with reproducible seed-based
randomization, inventory tracking, dynamic hints, and CS4231/CS3235 syllabus mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import random


COMPANION_AGENTS: Dict[str, Dict[str, Any]] = {
    "scout": {
        "id": "scout",
        "name": "Scout Drone",
        "specialty": "Computer vision survey and object localization",
        "preferred_tools": ["get_visible_objects", "locate_object"],
        "zone": "Zone 1: Reception / Central Hub",
    },
    "navigator": {
        "id": "navigator",
        "name": "Navigator Drone",
        "specialty": "3D path planning, distance calculation, and spatial relations",
        "preferred_tools": ["query_spatial_relation", "calculate_distance", "find_path", "navigate_to", "find_nearest_target"],
        "zone": "Zone 2: Investigation Laboratory",
    },
    "archivist": {
        "id": "archivist",
        "name": "Archivist Drone",
        "specialty": "Clue inspection, RAG document search, environment interaction",
        "preferred_tools": ["inspect_clue", "search_documents", "search_safety_manual", "interact_with_object", "get_inventory", "request_hint"],
        "zone": "Zone 3: Library / Archives",
    },
}


def _objective(objective_id: str, description: str, points: int, required_tools: List[str], required_item: Optional[str] = None) -> Dict[str, Any]:
    return {
        "objective_id": objective_id,
        "description": description,
        "points": points,
        "required_tools": required_tools,
        "required_item": required_item,
        "is_completed": False,
    }


def _mission_catalog(seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    # Controlled variation for replayability based on seed
    key_hiding_spot = "lab cabinet" if rng.random() > 0.5 else "office filing cabinet"
    archive_shelf = "Archive Bookshelf Alpha" if rng.random() > 0.5 else "Archive Bookshelf Beta"
    override_code = str(7000 + (rng.randint(100, 999))) if seed != 42 else "7391"

    return [
        {
            "id": "mission_a",
            "code": "A",
            "title": "Mystery A: Missing Research Key",
            "zone": "Location 1: Reception / Central Hub",
            "scene": "reception",
            "companion": "scout",
            "syllabus": "CS4231 object detection / CS3235 tool calling",
            "briefing": f"A master facility key is missing. Survey Reception and locate the security table to find where staff logged it (reported near {key_hiding_spot}). Retrieve the key into inventory to complete the mission.",
            "spawn_z": -15.0,
            "award": 200,
            "timed": False,
            "hint": "Ask the companion drone: 'What objects are visible?' then 'Where is the table?' and interact with the cabinet to collect the research key.",
            "required_inventory": ["research_key"],
            "objectives": [
                _objective("scout_01", "Ask: what objects are visible?", 100, ["get_visible_objects"]),
                _objective("scout_02", "Locate the table in the facility", 100, ["locate_object"]),
                _objective("scout_03", "Retrieve the missing Research Key into your inventory", 150, ["interact_with_object"], required_item="research_key"),
            ],
        },
        {
            "id": "mission_b",
            "code": "B",
            "title": "Mystery B: Missing Research File",
            "zone": "Location 2 & 3: Research Office & Archives",
            "scene": "office",
            "companion": "navigator",
            "syllabus": "CS4231 scene graph / 3D Euclidean geometry",
            "briefing": f"Incident Report 104-A is displaced between the Office and Library. Query spatial relations left of the table and compute distance to {archive_shelf}.",
            "spawn_z": -5.0,
            "award": 300,
            "timed": False,
            "hint": "Ask: 'What is left of the table?' and 'How far is the chair from the table?'",
            "objectives": [
                _objective("spatial_01", "Query what is left of the table", 150, ["query_spatial_relation"]),
                _objective("spatial_02", "Calculate distance between chair and table", 150, ["calculate_distance"]),
            ],
        },
        {
            "id": "mission_c",
            "code": "C",
            "title": "Mystery C: Laboratory Containment",
            "zone": "Location 4 → Location 5 Corridor",
            "scene": "laboratory",
            "companion": "navigator",
            "syllabus": "CS4231 nearest-neighbor localization / waypoint planning",
            "briefing": "Chemical anomaly in the Lab. Locate the nearest door or exit and command the companion drone to calculate safe navigation waypoints.",
            "spawn_z": 0.0,
            "award": 350,
            "timed": False,
            "hint": "Ask: 'Where is the nearest door?' then 'Calculate path to the door' or 'Take me to the nearest exit.'",
            "objectives": [
                _objective("nav_01", "Find the nearest door or exit", 150, ["find_nearest_target", "find_nearest_exit"]),
                _objective("nav_02", "Calculate a path or navigate to the door", 200, ["find_path", "navigate_to"]),
            ],
        },
        {
            "id": "mission_d",
            "code": "D",
            "title": "Clue Room & RAG",
            "zone": "Location 5: Storage Room & Vault",
            "scene": "archive",
            "companion": "archivist",
            "syllabus": "CS3235 TF-IDF RAG / validated tool execution",
            "briefing": f"Inspect the datapad clue, retrieve the Door 3 override code ({override_code}) from facility incident logs via RAG, and disengage the security lock.",
            "spawn_z": 6.0,
            "award": 650,
            "timed": False,
            "hint": "Inspect the clue datapad with 'inspect clue_pda_02', search documents for 'override code Door 3', then 'unlock door 3'.",
            "objectives": [
                _objective("clue_01", "Inspect a clue (datapad or safe)", 150, ["inspect_clue"]),
                _objective("clue_02", "Search documents for the Door 3 override", 250, ["search_documents", "search_safety_manual"]),
                _objective("clue_03", "Unlock Archive Door 3", 250, ["interact_with_object"]),
            ],
        },
        {
            "id": "mission_e",
            "code": "E",
            "title": "Timed Containment",
            "zone": "Whole facility (5 Rooms)",
            "scene": "containment",
            "companion": "navigator",
            "syllabus": "CS3235 multi-step agent loop under time constraint",
            "briefing": "180-second containment timer. Survey the lab, retrieve the override, unlock Door 3 before containment breach.",
            "spawn_z": 0.0,
            "award": 550,
            "timed": True,
            "timer_seconds": 180,
            "hint": "Run visible objects survey, quickly retrieve the override with RAG, and unlock Door 3.",
            "objectives": [
                _objective("timed_01", "Detect objects in the lab", 100, ["get_visible_objects"]),
                _objective("timed_02", "Retrieve override via RAG", 150, ["search_documents", "search_safety_manual"]),
                _objective("timed_03", "Unlock Door 3 before time expires", 300, ["interact_with_object"]),
            ],
        },
        {
            "id": "mission_f",
            "code": "F",
            "title": "Dataset Terminal",
            "zone": "Location 1: Reception Terminal",
            "scene": "reception",
            "companion": "scout",
            "syllabus": "CS4231 SUN RGB-D evaluation / viva station",
            "briefing": "Academic viva station. Inspect the SUN RGB-D terminal and verify detector classes against Princeton benchmark ground truth.",
            "spawn_z": -14.0,
            "award": 300,
            "timed": False,
            "hint": "Locate the reception terminal table and survey all SUN RGB-D objects.",
            "objectives": [
                _objective("term_01", "Locate the terminal table in Reception", 100, ["locate_object"]),
                _objective("term_02", "Survey visible SUN RGB-D classes in the facility", 200, ["get_visible_objects"]),
            ],
        },
    ]


@dataclass
class GameSession:
    """Tracks the player's current mystery mission, score, inventory, companion, and unlocks."""

    seed: int = 42
    missions: List[Dict[str, Any]] = field(default_factory=lambda: _mission_catalog(42))
    active_mission_id: str = "mission_a"
    active_agent_id: str = "scout"
    score: int = 0
    unlocked_through_index: int = 0
    started_at: float = field(default_factory=time.time)
    timer_started_at: Optional[float] = None
    timer_expired: bool = False
    inventory: List[Dict[str, Any]] = field(default_factory=list)
    collected_clues: List[str] = field(default_factory=list)
    player_position: List[float] = field(default_factory=lambda: [0.0, 0.0, -15.0])
    current_room: str = "Reception / Central Hub"

    def update_player_state(self, position: Optional[List[float]] = None, room: Optional[str] = None):
        """Updates and synchronizes real-time player coordinates and room."""
        from backend.game.facility import parse_position_coords, normalize_room_name, get_room_for_position
        if position is not None:
            px, py, pz = parse_position_coords(position)
            self.player_position = [round(px, 2), round(py, 2), round(pz, 2)]
            if not room:
                self.current_room = get_room_for_position(px, pz)["name"]
        if room is not None:
            self.current_room = normalize_room_name(room)

    def set_seed(self, seed: int):
        """Sets reproducible random seed and reconfigures mission catalog variants."""
        self.seed = seed
        self.missions = _mission_catalog(seed)

    def restart_current_mission(self) -> Dict[str, Any]:
        """Restarts the currently active mission, clearing its objectives and timer."""
        mission = self.active_mission()
        for obj in mission["objectives"]:
            obj["is_completed"] = False
        if mission.get("timed"):
            self.timer_started_at = time.time()
        self.timer_expired = False
        return self.to_dict()

    def return_to_hub(self) -> Dict[str, Any]:
        """Returns player and session focus to Reception / Central Hub."""
        self.player_position = [0.0, 0.0, -15.0]
        self.current_room = "Reception / Central Hub"
        return {
            "status": "hub_returned",
            "room": "Reception / Central Hub",
            "position": [0.0, 0.0, -15.0],
            "message": "Returned to Reception / Central Hub."
        }

    def select_mystery(self, mission_id: str) -> Dict[str, Any]:
        """Directly selects and unlocks any available mystery template."""
        idx = self._mission_index(mission_id)
        self.unlocked_through_index = max(self.unlocked_through_index, idx)
        return self.start_mission(mission_id, force=True)

    def _mission_index(self, mission_id: str) -> int:
        for i, mission in enumerate(self.missions):
            if mission["id"] == mission_id:
                return i
        return 0

    def active_mission(self) -> Dict[str, Any]:
        idx = self._mission_index(self.active_mission_id)
        return self.missions[idx]

    def remaining_timer(self) -> Optional[float]:
        mission = self.active_mission()
        if not mission.get("timed") or self.timer_started_at is None:
            return None
        remaining = float(mission.get("timer_seconds", 180)) - (time.time() - self.timer_started_at)
        if remaining <= 0:
            self.timer_expired = True
            return 0.0
        return round(remaining, 1)

    def select_agent(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in COMPANION_AGENTS:
            raise ValueError(f"Unknown companion '{agent_id}'. Allowed: {list(COMPANION_AGENTS.keys())}")
        self.active_agent_id = agent_id
        return COMPANION_AGENTS[agent_id]

    def start_mission(self, mission_id: str, force: bool = False) -> Dict[str, Any]:
        idx = self._mission_index(mission_id)
        if not force and idx > self.unlocked_through_index:
            raise ValueError("Mission is still locked. Complete the previous mission first.")
        self.active_mission_id = mission_id
        self.active_agent_id = self.missions[idx]["companion"]
        self.timer_expired = False
        if self.missions[idx].get("timed"):
            self.timer_started_at = time.time()
        else:
            self.timer_started_at = None
        return self.to_dict()

    def add_to_inventory(self, item_id: str, name: str, description: str):
        """Adds an item to the investigator's inventory if not already present."""
        if not any(item["id"] == item_id for item in self.inventory):
            self.inventory.append({
                "id": item_id,
                "name": name,
                "description": description,
                "collected_at": round(time.time() - self.started_at, 1)
            })

    def get_hint(self) -> str:
        """Returns a non-revealing procedural hint explaining exactly what remains incomplete."""
        mission = self.active_mission()
        for obj in mission["objectives"]:
            if not obj["is_completed"]:
                req_item = obj.get("required_item")
                if req_item and not any(it["id"] == req_item for it in self.inventory):
                    return f"Objective Incomplete: '{obj['description']}'. Item '{req_item}' is not yet in your inventory. Inspect facility cabinets or desks to retrieve it."
                return f"Hint for '{obj['description']}': Try using {', '.join(obj['required_tools'])}. {mission.get('hint', '')}"

        # If all objectives are completed but mission has remaining required inventory
        for req in mission.get("required_inventory", []):
            if not any(it["id"] == req for it in self.inventory):
                return f"Mission Incomplete: Missing required inventory item '{req}'. Retrieve it to complete {mission['title']}."

        return f"All objectives for {mission['title']} are completed! Proceed to the next location."

    def record_tools(self, tool_calls: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> List[str]:
        """Marks objectives complete when the matching investigation tools fire."""
        completed_now: List[str] = []
        if not tool_calls:
            return completed_now

        # 1. Auto-inventory on clue inspection, safe unlock, or container interactions first
        for call, res in zip(tool_calls, tool_results):
            t_name = call.get("tool") or call.get("name")
            if t_name == "inspect_clue" and res.get("success"):
                clue_data = res.get("result", {}).get("data", {})
                if clue_data:
                    c_id = call.get("arguments", {}).get("clue_id", "clue")
                    if c_id not in self.collected_clues:
                        self.collected_clues.append(c_id)
                    self.add_to_inventory(c_id, clue_data.get("name", "Clue Note"), clue_data.get("description", ""))
            elif t_name == "interact_with_object" and res.get("success"):
                obj_id = call.get("arguments", {}).get("object_id", "")
                if "safe" in obj_id.lower():
                    self.add_to_inventory("master_keycard", "Master Security Keycard", "Allows Level 3 access to Sector 4 Vault.")
                elif any(k in obj_id.lower() for k in ("cabinet", "key", "drawer", "box", "desk")):
                    self.add_to_inventory("research_key", "Master Research Key", "Physical backup key for restricted lab cabinets and security consoles.")

        used_successful = {
            (call.get("tool") or call.get("name"))
            for call, res in zip(tool_calls, tool_results)
            if res.get("success", True) is not False
        }
        mission = self.active_mission()
        timed_out = mission.get("timed") and self.remaining_timer() == 0.0

        # 2. Mark objectives complete only if both required tools and required items are satisfied
        for objective in mission["objectives"]:
            if objective["is_completed"]:
                continue
            if timed_out and objective["objective_id"] == "timed_03":
                continue
            if used_successful.intersection(objective["required_tools"]):
                req_item = objective.get("required_item")
                if req_item:
                    has_item = any(item["id"] == req_item for item in self.inventory)
                    if not has_item:
                        continue  # Cannot complete objective without required item in inventory

                objective["is_completed"] = True
                self.score += int(objective["points"])
                completed_now.append(objective["objective_id"])

        # 3. Check overall mission completion: all objectives done AND all required inventory collected
        req_inv = mission.get("required_inventory", [])
        has_all_req_inv = all(any(it["id"] == item_id for it in self.inventory) for item_id in req_inv)
        all_objs_done = bool(mission["objectives"]) and all(obj["is_completed"] for obj in mission["objectives"])

        if all_objs_done and has_all_req_inv:
            idx = self._mission_index(mission["id"])
            if idx + 1 < len(self.missions):
                self.unlocked_through_index = max(self.unlocked_through_index, idx + 1)

        return completed_now

    def to_dict(self) -> Dict[str, Any]:
        remaining = self.remaining_timer()
        mission = self.active_mission()
        idx = self._mission_index(mission["id"])
        catalog = []
        for i, item in enumerate(self.missions):
            item_req_inv = item.get("required_inventory", [])
            item_inv_done = all(any(it["id"] == req for it in self.inventory) for req in item_req_inv)
            item_objs_done = bool(item["objectives"]) and all(obj["is_completed"] for obj in item["objectives"])
            is_done = item_objs_done and item_inv_done
            catalog.append({
                "id": item["id"],
                "code": item["code"],
                "title": item["title"],
                "zone": item["zone"],
                "scene": item["scene"],
                "locked": i > self.unlocked_through_index,
                "active": item["id"] == self.active_mission_id,
                "completed": is_done,
            })
        active_req_inv = mission.get("required_inventory", [])
        active_inv_done = all(any(it["id"] == req for it in self.inventory) for req in active_req_inv)
        active_objs_done = bool(mission["objectives"]) and all(obj["is_completed"] for obj in mission["objectives"])
        active_completed = active_objs_done and active_inv_done

        return {
            "score": self.score,
            "seed": self.seed,
            "active_mission": {
                "id": mission["id"],
                "code": mission["code"],
                "title": mission["title"],
                "zone": mission["zone"],
                "scene": mission["scene"],
                "briefing": mission["briefing"],
                "syllabus": mission["syllabus"],
                "spawn_z": mission["spawn_z"],
                "timed": bool(mission.get("timed")),
                "timer_remaining": remaining,
                "timer_expired": self.timer_expired,
                "objectives": mission["objectives"],
                "completed": active_completed,
            },
            "missions": catalog,
            "companion": COMPANION_AGENTS[self.active_agent_id],
            "companions": list(COMPANION_AGENTS.values()),
            "inventory": self.inventory,
            "inventory_count": len(self.inventory),
            "player_position": self.player_position,
            "current_room": self.current_room,
            "unlocked_through_index": self.unlocked_through_index,
            "next_mission_id": self.missions[idx + 1]["id"] if idx + 1 < len(self.missions) and idx + 1 <= self.unlocked_through_index else None,
        }


global_game_session = GameSession()
