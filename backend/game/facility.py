"""Canonical 5-location facility layout shared by the 3D client and the scene graph.

Positions match `frontend/index.html` (Three.js) so language-agent answers
refer to the same metric coordinates the player sees.
Locations:
1. Reception / Central Hub (Z < -10)
2. Research Office (Z in [-10, 0], X >= 0)
3. Library / Archives (Z in [-10, 0], X < 0)
4. Investigation Laboratory (Z in [0, 10])
5. Storage Room (Z > 10)
"""
from typing import List, Dict, Any, Optional, Tuple
from backend.api.schemas import BoundingBox2D, DetectedObject
from backend.spatial.scene_graph import global_scene_graph

CLASS_IDS = {
    "table": 0,
    "chair": 1,
    "door": 2,
    "bookshelf": 3,
    "sofa": 4,
    "cabinet": 5,
}

ROOM_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "reception",
        "name": "Reception / Central Hub",
        "bounds": {"xmin": -12.0, "xmax": 12.0, "zmin": -24.0, "zmax": -10.0},
        "description": "Visitor entry, main security checkpoint, and security terminal.",
        "spawn_z": -15.0,
    },
    {
        "id": "office",
        "name": "Research Office",
        "bounds": {"xmin": 0.0, "xmax": 12.0, "zmin": -10.0, "zmax": 0.0},
        "description": "Administrative workstations and confidential filing cabinets.",
        "spawn_z": -5.0,
    },
    {
        "id": "library",
        "name": "Library / Archives",
        "bounds": {"xmin": -12.0, "xmax": 0.0, "zmin": -10.0, "zmax": 0.0},
        "description": "Scientific reference library, blueprint catalogs, and reading carrels.",
        "spawn_z": -5.0,
    },
    {
        "id": "laboratory",
        "name": "Investigation Laboratory",
        "bounds": {"xmin": -12.0, "xmax": 12.0, "zmin": 0.0, "zmax": 10.0},
        "description": "Central experimental facility with chemical workbench and laser security gate.",
        "spawn_z": 0.0,
    },
    {
        "id": "storage",
        "name": "Storage Room",
        "bounds": {"xmin": -12.0, "xmax": 12.0, "zmin": 10.0, "zmax": 24.0},
        "description": "High-security artifact vault, heavy storage lockers, and emergency exit.",
        "spawn_z": 14.0,
    },
]


def parse_position_coords(pos: Optional[List[float]]) -> Tuple[float, float, float]:
    """Extracts (x, y, z) float coordinates from 2D [x, z] or 3D [x, y, z] vectors."""
    if not pos:
        return 0.0, 0.0, -15.0
    px = float(pos[0])
    if len(pos) == 2:
        py = 0.0
        pz = float(pos[1])
    else:
        py = float(pos[1])
        pz = float(pos[2])
    return px, py, pz


def get_room_definition(room_identifier: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolves arbitrary room strings (e.g. 'Library', '3. Library / Archives', 'library',
    'Zone 3', 'office', 'laboratory') into canonical room definition."""
    import re
    if not room_identifier:
        return None
    s = str(room_identifier).lower().strip()

    # 1. Direct zone number / prefix matching first
    zm = re.search(r"\bzone\s*([1-5])\b", s)
    if zm:
        z_num = int(zm.group(1))
        return ROOM_DEFINITIONS[z_num - 1]

    # 2. Keyword matching
    if any(k in s for k in ["library", "archive", "reading", "study"]):
        return next(r for r in ROOM_DEFINITIONS if r["id"] == "library")
    if any(k in s for k in ["office", "workstation"]):
        return next(r for r in ROOM_DEFINITIONS if r["id"] == "office")
    if any(k in s for k in ["lab", "laboratory"]):
        return next(r for r in ROOM_DEFINITIONS if r["id"] == "laboratory")
    if any(k in s for k in ["storage", "vault"]):
        return next(r for r in ROOM_DEFINITIONS if r["id"] == "storage")
    if any(k in s for k in ["reception", "hub", "lobby", "entrance"]):
        return next(r for r in ROOM_DEFINITIONS if r["id"] == "reception")

    # 3. Normalized stripped string matching
    s_clean = re.sub(r"^(?:zone\s*\d+\s*[:.]?\s*|\d+\.\s*)", "", s).strip()
    if s_clean:
        for r in ROOM_DEFINITIONS:
            r_id = r["id"].lower()
            r_name = r["name"].lower()
            if s_clean == r_id or s_clean == r_name:
                return r
            if r_id in s_clean or s_clean in r_id or s_clean in r_name:
                return r

    # 4. Standalone zone digit fallback
    for digit, idx in [("3", 2), ("2", 1), ("4", 3), ("5", 4), ("1", 0)]:
        if re.search(rf"\b{digit}\b", s):
            return ROOM_DEFINITIONS[idx]

    return None


def normalize_room_name(room_identifier: Optional[str]) -> str:
    """Returns the canonical display name for any valid room identifier, or fallback."""
    r = get_room_definition(room_identifier)
    return r["name"] if r else (str(room_identifier) if room_identifier else "Reception / Central Hub")


def is_room_identifier(text: str) -> bool:
    """Determines whether a destination string refers to a room rather than an object."""
    return get_room_definition(text) is not None


def get_room_for_position(x: float, z: float) -> Dict[str, Any]:
    """Returns the room definition corresponding to world coordinates (x, z)."""
    if z < -10.0:
        return ROOM_DEFINITIONS[0]  # Reception
    elif z < 0.0:
        if x >= 0.0:
            return ROOM_DEFINITIONS[1]  # Research Office
        else:
            return ROOM_DEFINITIONS[2]  # Library / Archives
    elif z <= 10.0:
        return ROOM_DEFINITIONS[3]  # Investigation Laboratory
    else:
        return ROOM_DEFINITIONS[4]  # Storage Room


def _room_walkable_standpoint(room_name: str) -> Optional[List[float]]:
    """Returns an unobstructed floor stand-point inside each room for 3D avatar arrival."""
    standpoints = {
        "Reception / Central Hub": [0.0, 0.75, -17.0],
        "Research Office": [4.2, 0.75, -5.0],
        "Library / Archives": [-4.5, 0.75, -6.6],
        "Investigation Laboratory": [1.9, 0.75, 2.2],
        "Storage Room": [0.0, 0.75, 12.5],
    }
    return standpoints.get(room_name)


def compute_turn_by_turn_route(
    start_pos: List[float],
    target_pos: List[float],
    target_name: str,
    target_room: str = None,
    origin_room_override: Optional[str] = None
) -> Dict[str, Any]:
    """Generates Google Maps-style turn-by-turn navigation directions between two 3D points.

    Returns:
        dict containing:
        - origin_room: str
        - destination_room: str
        - distance_meters: float
        - turns: List[str]
        - waypoints: List[Dict[str, Any]]
    """
    import math

    px = float(start_pos[0])
    pz = float(start_pos[1] if len(start_pos) == 2 else start_pos[2])

    tx = float(target_pos[0])
    ty = float(target_pos[1]) if len(target_pos) > 2 else 0.75
    tz = float(target_pos[2] if len(target_pos) > 2 else target_pos[1])

    # Avoid placing final avatar waypoint directly inside center furniture colliders when routing to a room
    if origin_room_override is not None or target_name in (
        "Reception / Central Hub", "Research Office", "Library / Archives", "Investigation Laboratory", "Storage Room"
    ):
        stand = _room_walkable_standpoint(target_name)
        if stand is not None:
            tx, ty, tz = stand[0], stand[1], stand[2]

    dx = tx - px
    dz = tz - pz
    total_dist = round(math.sqrt(dx**2 + dz**2), 2)

    if origin_room_override and get_room_definition(origin_room_override):
        orig_room = get_room_definition(origin_room_override)["name"]
    else:
        orig_room = get_room_for_position(px, pz)["name"]

    if target_room and get_room_definition(target_room):
        dest_room = get_room_definition(target_room)["name"]
    else:
        dest_room = get_room_for_position(tx, tz)["name"]

    turns: List[str] = []
    waypoints: List[Dict[str, Any]] = []

    def _add_wp(coord: List[float], note: str):
        waypoints.append({"step": len(waypoints) + 1, "coord": coord, "note": note})

    # If already at destination
    if total_dist < 1.2:
        turns.append(f"🎯 You have reached the {target_name}. It is directly within reach at [{tx:.1f}, {ty:.1f}, {tz:.1f}].")
        _add_wp([tx, ty, tz], f"Arrival at {target_name}")
        return {
            "origin_room": orig_room,
            "destination_room": dest_room,
            "distance_meters": total_dist,
            "turns": turns,
            "waypoints": waypoints,
        }

    # Case 1: In the same room
    if orig_room == dest_room:
        _add_wp([round(px, 2), 0.0, round(pz, 2)], "Current stance")

        # Bypass central Reception table if walking across Z = -15 along center X
        if "Reception" in orig_room and abs(px) < 1.8 and abs(tx) < 1.8 and (pz - (-15.0)) * (tz - (-15.0)) < 0:
            _add_wp([1.9, 0.0, round(pz, 2)], "Sidestep Reception table")
            _add_wp([1.9, 0.0, round(tz, 2)], "Bypass Reception table")

        if abs(dz) >= 0.8:
            dir_str = "⬆️ Head straight forward (North)" if dz > 0 else "⬇️ Turn around and walk South"
            turns.append(f"{dir_str} for {abs(dz):.1f} meters.")
            _add_wp([round(tx if abs(px) < 1.8 and "Reception" in orig_room else px, 2), 0.0, round(tz, 2)], "Corridor transit")

        if abs(dx) >= 0.8:
            turn_str = "➡️ Turn 90° right (East)" if dx > 0 else "⬅️ Turn 90° left (West)"
            side_str = "right" if dx > 0 else "left"
            turns.append(f"{turn_str} and proceed for {abs(dx):.1f} meters.")
            turns.append(f"🎯 Destination reached: The {target_name} is directly on your {side_str} at coordinates [{tx:.1f}, {ty:.1f}, {tz:.1f}].")
        else:
            turns.append(f"🎯 Destination reached: The {target_name} is straight ahead at coordinates [{tx:.1f}, {ty:.1f}, {tz:.1f}].")

        _add_wp([round(tx, 2), round(ty, 2), round(tz, 2)], f"Arrival at {target_name}")
        return {
            "origin_room": orig_room,
            "destination_room": dest_room,
            "distance_meters": total_dist,
            "turns": turns,
            "waypoints": waypoints,
        }

    # Case 2: Across different rooms
    _add_wp([round(px, 2), 0.0, round(pz, 2)], "Departure stance")

    # Moving Northward from Reception (pz < -10)
    if pz < -10.0:
        dist_to_archway = abs(-10.0 - pz)
        turns.append(f"⬆️ Head straight North from {orig_room} toward the main corridor archway (proceed {dist_to_archway:.1f}m).")
        if abs(px) < 1.8 and pz < -13.5:
            lane_x = -1.9 if "Library" in dest_room else 1.9
            _add_wp([lane_x, 0.0, round(min(-16.2, pz), 2)], "Reception table bypass lane")
            _add_wp([lane_x, 0.0, -11.8], "Clear Reception furniture")
        _add_wp([0.0, 0.0, -10.0], "Reception exit archway")

        if "Office" in dest_room:
            turns.append("↗️ Turn right (East) through the archway into the Research Office.")
            turns.append(f"➡️ Walk {abs(tx):.1f}m East past the admin workstations, then bear North for {abs(tz - (-10.0)):.1f}m.")
            _add_wp([0.0, 0.0, -5.0], "Central corridor Office junction")
            _add_wp([2.8, 0.0, -5.0], "Research Office doorway threshold")
        elif "Library" in dest_room:
            turns.append("↖️ Turn left (West) through the archway into the Library / Archives.")
            turns.append(f"⬅️ Walk {abs(tx):.1f}m West past the reference bookshelves, then bear North for {abs(tz - (-10.0)):.1f}m.")
            _add_wp([0.0, 0.0, -5.0], "Central corridor Library junction")
            _add_wp([-2.8, 0.0, -5.0], "Library doorway threshold")
        elif "Laboratory" in dest_room:
            turns.append("⬆️ Continue straight North along the central corridor between Library and Office (proceed 10.0m).")
            turns.append("🚪 Pass through the Investigation Laboratory threshold at Z = 0.0m.")
            lab_x = -1.95 if tx < -2.0 else 1.95
            _add_wp([lab_x, 0.0, -1.6], "Lab workbench approach")
            _add_wp([lab_x, 0.0, 0.0], "Lab entry threshold (bypass workbench)")
            if abs(tx) > 1.0:
                t_dir = "➡️ Turn right" if tx > 0 else "⬅️ Turn left"
                turns.append(f"{t_dir} and proceed {abs(tx):.1f}m toward the bench.")
        elif "Storage" in dest_room:
            turns.append("⬆️ Continue straight North through the central corridor and through the Investigation Laboratory (proceed 20.0m).")
            turns.append("🚪 Approach Security Door 3 at [0.0, 2.0, 8.0] and pass through the vault doorway.")
            _add_wp([1.95, 0.0, -1.6], "Lab portal approach")
            _add_wp([1.95, 0.0, 1.8], "Bypass Lab workbench")
            _add_wp([0.0, 0.0, 6.2], "Door 3 approach")
            _add_wp([0.0, 0.0, 8.0], "Door 3 vault threshold")

    # Moving from Office/Library (-10 <= pz <= 0)
    elif -10.0 <= pz <= 0.0:
        if px >= 2.2:
            _add_wp([2.8, 0.0, -5.0], "Exit Research Office doorway")
        elif px <= -2.2:
            _add_wp([-2.8, 0.0, -5.0], "Exit Library doorway")
        if abs(px) > 1.0:
            turns.append(f"⬅️ Head toward the central hallway corridor (proceed {abs(px):.1f}m).")
            _add_wp([0.0, 0.0, -5.0], "Central corridor intersection")

        if "Laboratory" in dest_room:
            turns.append(f"⬆️ Head North through the Investigation Laboratory archway at Z = 0.0m (walk {abs(0.0 - pz):.1f}m).")
            lab_x = -1.95 if tx < -2.0 else 1.95
            _add_wp([lab_x, 0.0, -1.6], "Lab workbench approach")
            _add_wp([lab_x, 0.0, 0.0], "Lab entry threshold (bypass workbench)")
            if abs(tx) > 1.0:
                t_dir = "➡️ Turn right" if tx > 0 else "⬅️ Turn left"
                turns.append(f"{t_dir} and proceed {abs(tx):.1f}m toward the bench.")
        elif "Storage" in dest_room:
            turns.append("⬆️ Proceed North through the Lab toward Door 3 at [0.0, 2.0, 8.0].")
            turns.append("🚪 Pass through the vault threshold into the Storage Room.")
            _add_wp([1.95, 0.0, -1.6], "Lab portal approach")
            _add_wp([1.95, 0.0, 1.8], "Bypass Lab workbench")
            _add_wp([0.0, 0.0, 6.2], "Door 3 approach")
            _add_wp([0.0, 0.0, 8.0], "Door 3 vault threshold")
        elif "Reception" in dest_room:
            turns.append(f"⬇️ Turn South toward the Reception archway (proceed {abs(pz - (-10.0)):.1f}m).")
            _add_wp([0.0, 0.0, -10.0], "Reception archway")
            if tz < -14.0 and abs(tx) < 1.8:
                _add_wp([1.9, 0.0, -11.8], "Bypass Reception desk")
                _add_wp([1.9, 0.0, -16.2], "Reception South bypass")
        elif "Office" in dest_room and px < 0:
            turns.append("➡️ Cross East through the central corridor into the Research Office.")
            _add_wp([2.8, 0.0, -5.0], "Research Office doorway threshold")
        elif "Library" in dest_room and px >= 0:
            turns.append("⬅️ Cross West through the central corridor into the Library / Archives.")
            _add_wp([-2.8, 0.0, -5.0], "Library doorway threshold")

    # Moving from Laboratory (0 < pz <= 10)
    elif 0.0 < pz <= 10.0:
        if "Storage" in dest_room:
            turns.append(f"⬆️ Head North toward Security Door 3 at [0.0, 2.0, 8.0] (proceed {abs(8.0 - pz):.1f}m).")
            turns.append("🚪 Pass through the Door 3 vault archway into the Storage Room.")
            _add_wp([0.0, 0.0, 6.2], "Door 3 approach")
            _add_wp([0.0, 0.0, 8.0], "Door 3 threshold")
        else:
            turns.append(f"⬇️ Turn South through the Investigation Laboratory exit at Z = 0.0m (proceed {abs(pz - 0.0):.1f}m).")
            lab_x = -1.95 if px < 0 else 1.95
            _add_wp([lab_x, 0.0, 1.8], "Lab workbench bypass")
            _add_wp([lab_x, 0.0, 0.0], "Lab exit threshold")
            _add_wp([0.0, 0.0, -5.0], "Central corridor junction")
            if "Reception" in dest_room:
                turns.append("⬇️ Continue South along the central corridor past Library and Office into Reception.")
                _add_wp([0.0, 0.0, -10.0], "Reception archway")
                if tz < -14.0 and abs(tx) < 1.8:
                    _add_wp([1.9, 0.0, -11.8], "Bypass Reception desk")
                    _add_wp([1.9, 0.0, -16.2], "Reception South bypass")
            elif "Office" in dest_room:
                turns.append("↗️ Turn East into the Research Office corridor.")
                _add_wp([2.8, 0.0, -5.0], "Research Office doorway threshold")
            elif "Library" in dest_room:
                turns.append("↖️ Turn West into the Library / Archives corridor.")
                _add_wp([-2.8, 0.0, -5.0], "Library doorway threshold")

    # Moving from Storage Room (pz > 10)
    else:
        turns.append("⬇️ Turn South toward Security Door 3 at [0.0, 2.0, 8.0].")
        turns.append("🚪 Pass through Door 3 back into the Investigation Laboratory.")
        _add_wp([0.0, 0.0, 8.0], "Door 3 entry")
        if "Reception" in dest_room or "Office" in dest_room or "Library" in dest_room:
            turns.append("⬇️ Continue South across the Lab threshold into the central corridor.")
            _add_wp([1.95, 0.0, 1.8], "Bypass Lab workbench")
            _add_wp([1.95, 0.0, 0.0], "Lab exit threshold")
            _add_wp([0.0, 0.0, -5.0], "Central corridor junction")
            if "Office" in dest_room:
                _add_wp([2.8, 0.0, -5.0], "Research Office doorway threshold")
            elif "Library" in dest_room:
                _add_wp([-2.8, 0.0, -5.0], "Library doorway threshold")
            elif "Reception" in dest_room:
                _add_wp([0.0, 0.0, -10.0], "Reception archway")
                if tz < -14.0 and abs(tx) < 1.8:
                    _add_wp([1.9, 0.0, -11.8], "Bypass Reception desk")
                    _add_wp([1.9, 0.0, -16.2], "Reception South bypass")

    turns.append(f"🎯 Destination reached: The {target_name} is located at coordinates [{tx:.1f}, {ty:.1f}, {tz:.1f}].")
    _add_wp([round(tx, 2), round(ty, 2), round(tz, 2)], f"Arrival at {target_name}")

    return {
        "origin_room": orig_room,
        "destination_room": dest_room,
        "distance_meters": total_dist,
        "turns": turns,
        "waypoints": waypoints,
    }


def facility_detections() -> List[DetectedObject]:
    """Returns the static 5-location facility object set using canonical SUN RGB-D classes."""
    specs = [
        # Location 1: Reception / Central Hub
        ("table_reception", "table", [0.0, 0.75, -15.0], 15.0),
        ("chair_reception", "chair", [0.0, 0.45, -13.8], 13.8),
        ("sofa_visitor", "sofa", [-6.5, 0.3, -16.0], 17.3),
        ("table_terminal", "table", [6.5, 0.75, -14.0], 15.4),

        # Location 2: Research Office
        ("table_office_desk", "table", [6.0, 0.75, -5.0], 7.8),
        ("chair_office", "chair", [6.0, 0.45, -4.0], 7.2),
        ("cabinet_office_files", "cabinet", [9.0, 1.2, -5.0], 10.3),

        # Location 3: Library / Archives
        ("bookshelf_library_reference", "bookshelf", [-8.5, 1.6, -5.0], 9.9),
        ("table_library_reading", "table", [-4.5, 0.75, -5.0], 6.7),
        ("chair_library", "chair", [-4.5, 0.45, -4.0], 6.0),

        # Location 4: Investigation Laboratory
        ("table_lab", "table", [0.0, 0.75, 0.0], 2.5),
        ("chair_lab", "chair", [-1.4, 0.45, 0.6], 1.6),
        ("cabinet_lab", "cabinet", [8.5, 1.2, 2.0], 8.8),
        ("bookshelf_lab", "bookshelf", [-8.5, 1.6, 2.0], 8.8),
        ("door_3", "door", [0.0, 2.0, 8.0], 8.0),

        # Location 5: Storage Room
        ("bookshelf_archive_a", "bookshelf", [-8.5, 1.6, 14.0], 14.6),
        ("bookshelf_archive_b", "bookshelf", [-8.5, 1.6, 18.0], 18.1),
        ("cabinet_vault", "cabinet", [8.5, 1.2, 14.0], 16.4),
        ("door_emergency_exit", "door", [0.0, 2.0, 22.0], 22.0),
    ]

    detections: List[DetectedObject] = []
    for object_id, class_name, position, depth in specs:
        detections.append(
            DetectedObject(
                object_id=object_id,
                class_id=CLASS_IDS[class_name],
                class_name=class_name,
                confidence=0.91,
                bbox=BoundingBox2D(xmin=80.0, ymin=80.0, xmax=200.0, ymax=280.0),
                estimated_3d_position=position,
                depth_estimate=depth,
            )
        )
    return detections


def seed_facility_scene(source: str = "facility_layout") -> int:
    """Loads the 5-location facility object set into the global scene graph."""
    detections = facility_detections()
    global_scene_graph.update_objects(detections, source=source)
    return len(detections)
