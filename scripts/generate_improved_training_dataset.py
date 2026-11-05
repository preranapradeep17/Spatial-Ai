"""Improved Ground-Truth Synthetic Dataset Generator for Spatial Quest.

Generates 2,250 balanced, linguistically diverse Vision-Language dialogues
specifically addressing:
1. find_nearest_target (relative references & player proximity)
2. interact_with_object (doors, safes, terminals, cabinets)
3. search_safety_manual (fire safety, evacuation, chemical hazards)
4. get_visible_objects (casual phrasing, typos, sweeps, counts)
5. calculate_distance (entity preservation across diverse syntactic structures)
6. navigate_to / find_path (room and object destinations)

Enforces:
- Strict zero train/test leakage by checking against all normalized held-out test queries
- Schema validation via Pydantic schemas in backend/tools/schemas.py
- Authentic 5-room facility coordinates and room boundaries
- Preserved held-out test split (untouched data/corpora/corrected_test.jsonl)
"""
import os
import sys
import json
import random
import math
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

random.seed(1337)
sys.path.insert(0, ".")

from backend.game.facility import ROOM_DEFINITIONS, facility_detections, get_room_for_position
from backend.agent.prompt_builder import (
    build_canonical_prompt,
    format_visible_objects_block,
    format_spatial_relations_block,
    format_conversation_history_block,
)
from backend.tools.schemas import (
    GetVisibleObjectsArgs,
    LocateObjectArgs,
    QuerySpatialRelationArgs,
    CalculateDistanceArgs,
    FindNearestTargetArgs,
    FindPathArgs,
    NavigateToArgs,
    InspectClueArgs,
    SearchDocumentsArgs,
    InteractWithObjectArgs,
    SearchSafetyManualArgs,
    FindNearestExitArgs,
    InspectObjectCloselyArgs,
    RequestHintArgs,
    GetInventoryArgs,
    GetMissionObjectivesArgs,
    ExplainUncertaintyArgs,
    GetGameGuideArgs,
    ExplainUnreachableArgs,
    GetCurrentLocationArgs,
)

SCHEMA_MAP = {
    "get_visible_objects": GetVisibleObjectsArgs,
    "locate_object": LocateObjectArgs,
    "query_spatial_relation": QuerySpatialRelationArgs,
    "calculate_distance": CalculateDistanceArgs,
    "find_nearest_target": FindNearestTargetArgs,
    "find_path": FindPathArgs,
    "navigate_to": NavigateToArgs,
    "inspect_clue": InspectClueArgs,
    "search_documents": SearchDocumentsArgs,
    "interact_with_object": InteractWithObjectArgs,
    "search_safety_manual": SearchSafetyManualArgs,
    "find_nearest_exit": FindNearestExitArgs,
    "inspect_object_closely": InspectObjectCloselyArgs,
    "request_hint": RequestHintArgs,
    "get_inventory": GetInventoryArgs,
    "get_mission_objectives": GetMissionObjectivesArgs,
    "explain_uncertainty": ExplainUncertaintyArgs,
    "get_game_guide": GetGameGuideArgs,
    "explain_unreachable": ExplainUnreachableArgs,
    "get_current_location": GetCurrentLocationArgs,
}

# 1. Load exact held-out test queries to guarantee zero test leakage
TEST_QUERIES_NORMALIZED = set()
with open("data/corpora/corrected_test.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        m = re.search(r"USER:\s*(.*?)\nASSISTANT:", d["prompt"], re.DOTALL)
        if m:
            TEST_QUERIES_NORMALIZED.add(m.group(1).strip().lower())

print(f"Loaded {len(TEST_QUERIES_NORMALIZED)} normalized test queries for strict leakage guard.")

FACILITY_OBJECTS = facility_detections()
ROOM_OBJECT_MAP: Dict[str, List[Any]] = {r["id"]: [] for r in ROOM_DEFINITIONS}
for obj in FACILITY_OBJECTS:
    px, py, pz = obj.estimated_3d_position
    rm = get_room_for_position(px, pz)
    ROOM_OBJECT_MAP[rm["id"]].append(obj)

def get_realistic_room_position(room_id: str) -> List[float]:
    room_def = next(r for r in ROOM_DEFINITIONS if r["id"] == room_id)
    b = room_def["bounds"]
    min_x, max_x = b["xmin"] + 1.0, b["xmax"] - 1.0
    min_z, max_z = b["zmin"] + 1.0, b["zmax"] - 1.0
    if min_x > max_x: min_x = max_x = (b["xmin"] + b["xmax"]) / 2.0
    if min_z > max_z: min_z = max_z = (b["zmin"] + b["zmax"]) / 2.0
    px = round(random.uniform(min_x, max_x), 1)
    pz = round(random.uniform(min_z, max_z), 1)
    return [px, 0.0, pz]

def make_scene_prompt(query: str, room_id: str, agent_mode: str, history=None) -> str:
    room_def = next(r for r in ROOM_DEFINITIONS if r["id"] == room_id)
    active_objs = ROOM_OBJECT_MAP[room_id]
    player_pos = get_realistic_room_position(room_id)
    obj_str = format_visible_objects_block(active_objs, provenance="Scene Graph Registry", max_objects=5)

    rel_lines = []
    for i in range(len(active_objs)):
        for j in range(len(active_objs)):
            if i != j:
                a, b = active_objs[i], active_objs[j]
                pa, pb = a.estimated_3d_position, b.estimated_3d_position
                dist = math.dist(pa, pb)
                if dist <= 2.5:
                    rel_lines.append(f"- {a.class_name} is near {b.class_name}")
                elif pa[0] < pb[0] - 1.0:
                    rel_lines.append(f"- {a.class_name} is left_of {b.class_name}")
                elif pa[0] > pb[0] + 1.0:
                    rel_lines.append(f"- {a.class_name} is right_of {b.class_name}")
                if len(rel_lines) >= 3:
                    break
        if len(rel_lines) >= 3:
            break

    rel_str = format_spatial_relations_block(rel_lines, max_relations=3)
    hist_str = format_conversation_history_block(history, max_turns=2)

    return build_canonical_prompt(
        query=query,
        agent_mode=agent_mode,
        room_name=room_def["name"],
        player_position=player_pos,
        visible_objects_str=obj_str,
        spatial_relations_str=rel_str,
        active_mission_str="Mystery A: Missing Research Key (Objectives: 1/3)",
        inventory_str="None",
        history_str=hist_str
    )

def validate_and_create_sample(
    query: str,
    tool_name: str,
    args: Dict[str, Any],
    thought: str,
    room_id: str,
    agent_mode: str,
    grounded_ans: str,
    history=None
) -> Optional[Dict[str, Any]]:
    # 1. Check strict zero test leakage
    q_norm = query.strip().lower()
    if q_norm in TEST_QUERIES_NORMALIZED:
        return None

    # 2. Pydantic validation
    schema = SCHEMA_MAP.get(tool_name)
    if not schema:
        raise ValueError(f"Unknown tool name: {tool_name}")
    try:
        schema(**args)
    except Exception as e:
        raise ValueError(f"Schema validation failed for {tool_name} with args {args}: {e}")

    prompt_str = make_scene_prompt(query, room_id, agent_mode, history)
    tool_call_json = json.dumps({"name": tool_name, "arguments": args})
    completion_str = f" <thought>{thought}</thought><tool_call>{tool_call_json}</tool_call>"

    return {
        "prompt": prompt_str,
        "completion": completion_str,
        "grounded_response": grounded_ans
    }

def generate_improved_dataset() -> List[Dict[str, Any]]:
    samples = []
    seen_prompts = set()

    def add_sample(s: Optional[Dict[str, Any]]):
        if s is not None and s["prompt"] not in seen_prompts:
            seen_prompts.add(s["prompt"])
            samples.append(s)

    room_ids = list(ROOM_OBJECT_MAP.keys())
    all_obj_names = ["table", "chair", "bookshelf", "cabinet", "sofa", "door", "desk", "workbench"]

    print("Generating improved balanced samples across all 20 tools...")

    # -------------------------------------------------------------------------
    # 1. calculate_distance (~140 samples)
    # Focus: Entity preservation across varied syntactic structures & typos
    # -------------------------------------------------------------------------
    cd_templates = [
        "Calculate distance between {a} and {b}.",
        "How far is the {a} from the {b}?",
        "What is the distance between the {a} and the {b}?",
        "Measure the distance from {a} to {b}.",
        "Compute distance between {a} and {b}.",
        "Distance from {a} to {b} in meters.",
        "What is the metric separation between {a} and {b}?",
        "How many meters separate the {a} and the {b}?",
        "Tell me how far {a} is from {b}.",
        "distance {a} to {b}",
        "measure {a} and {b}",
        "distence from {a} to {b}",
        "how fr is {a} from {b}",
        "distance between {a} and {b}"
    ]
    for _ in range(180):
        room_id = random.choice(room_ids)
        a, b = random.sample(all_obj_names, 2)
        tmpl = random.choice(cd_templates)
        q = tmpl.format(a=a, b=b)
        thought = f"Compute 3D Euclidean distance between {a} and {b}."
        args = {"object_a": a, "object_b": b}
        ans = f"The Euclidean distance between {a} and {b} is calculated."
        add_sample(validate_and_create_sample(q, "calculate_distance", args, thought, room_id, "NAVIGATOR", ans))

    # -------------------------------------------------------------------------
    # 2. find_nearest_target (~140 samples)
    # Focus: Relative references (reference_object) & observer proximity
    # -------------------------------------------------------------------------
    fnt_ref_templates = [
        "Find the nearest object to the {ref}.",
        "What is closest to the {ref}?",
        "Which object is situated nearest to the {ref}?",
        "Locate whatever is closest to the {ref}.",
        "What item is nearest to the {ref}?",
        "Find nearest object to {ref}",
        "Which piece of furniture is closest to the {ref}?",
        "What is near the {ref}?",
        "nearest to {ref}",
        "closest to {ref}",
        "what is neerest to {ref}",
        "find clostest to {ref}"
    ]
    for _ in range(110):
        room_id = random.choice(room_ids)
        ref = random.choice(all_obj_names)
        tmpl = random.choice(fnt_ref_templates)
        q = tmpl.format(ref=ref)
        thought = f"Find the nearest registered object relative to reference object {ref}."
        args = {"target_class": "any", "reference_object": ref}
        ans = f"The nearest object to {ref} is identified."
        add_sample(validate_and_create_sample(q, "find_nearest_target", args, thought, room_id, "NAVIGATOR", ans))

    fnt_player_templates = [
        "What is closest to me?",
        "Which object is closest to my position?",
        "Find the nearest object",
        "What is near my current stance?",
        "Locate nearest object to player",
        "What is closest to here?",
        "nearest object",
        "closest thing",
        "what is neerest to me",
        "which object is clostest"
    ]
    for _ in range(70):
        room_id = random.choice(room_ids)
        q = random.choice(fnt_player_templates)
        thought = "Find the nearest registered object relative to live observer coordinates."
        args = {"target_class": "any"}
        ans = "The closest object to your current position is identified."
        add_sample(validate_and_create_sample(q, "find_nearest_target", args, thought, room_id, "NAVIGATOR", ans))

    # -------------------------------------------------------------------------
    # 3. interact_with_object (~130 samples)
    # Focus: Doors, Safes, Terminals, Cabinets
    # -------------------------------------------------------------------------
    iwo_scenarios = [
        # Door 3
        ("Unlock Door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        ("Disengage magnetic lock on Door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        ("Unlock the security barrier Door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        ("Open Archive Door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        ("unlock door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        ("access door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        ("unlok door 3", "door_3", "Execute physical interaction on Archive Door 3 security barrier.", "library"),
        # Other Doors
        ("Open the storage room door.", "door", "Execute environmental physical interaction with doorway threshold.", "storage"),
        ("Interact with the laboratory door.", "door", "Execute environmental physical interaction with doorway threshold.", "laboratory"),
        ("Unlock the office door.", "door", "Execute environmental physical interaction with doorway threshold.", "office"),
        ("open door", "door", "Execute environmental physical interaction with doorway threshold.", "reception"),
        # Safes
        ("Unlock the wall safe.", "clue_safe_01", "Execute environmental physical interaction to unlock laboratory safe.", "laboratory"),
        ("Open the electronic wall safe.", "clue_safe_01", "Execute environmental physical interaction to unlock laboratory safe.", "laboratory"),
        ("Access the laboratory safe.", "clue_safe_01", "Execute environmental physical interaction to unlock laboratory safe.", "laboratory"),
        ("Disengage lock on clue safe 01", "clue_safe_01", "Execute environmental physical interaction to unlock laboratory safe.", "laboratory"),
        ("unlock safe", "clue_safe_01", "Execute environmental physical interaction to unlock laboratory safe.", "laboratory"),
        # Terminals
        ("Log into the central terminal.", "clue_terminal_03", "Engage interactive environmental facility central terminal.", "reception"),
        ("Use the facility terminal console.", "clue_terminal_03", "Engage interactive environmental facility central terminal.", "reception"),
        ("Interact with the central terminal.", "clue_terminal_03", "Engage interactive environmental facility central terminal.", "reception"),
        ("Access terminal clue", "clue_terminal_03", "Engage interactive environmental facility central terminal.", "reception"),
        ("use terminal", "clue_terminal_03", "Engage interactive environmental facility central terminal.", "reception"),
        # Cabinets
        ("Interact with the security cabinet.", "cabinet", "Open research cabinet drawer to retrieve stored items.", "storage"),
        ("Open the steel cabinet drawer.", "cabinet", "Open research cabinet drawer to retrieve stored items.", "laboratory"),
        ("Unlock the research cabinet.", "cabinet", "Open research cabinet drawer to retrieve stored items.", "office"),
        ("Search the storage cabinet.", "cabinet", "Open research cabinet drawer to retrieve stored items.", "storage"),
        ("open cabinet", "cabinet", "Open research cabinet drawer to retrieve stored items.", "office")
    ]
    for _ in range(170):
        q, oid, thought, rm = random.choice(iwo_scenarios)
        args = {"object_id": oid}
        ans = f"Interacted with {oid} successfully."
        add_sample(validate_and_create_sample(q, "interact_with_object", args, thought, rm, "NAVIGATOR", ans))

    # -------------------------------------------------------------------------
    # 4. search_safety_manual (~120 samples)
    # Focus: Fire emergency, chemical spills, safety rules
    # -------------------------------------------------------------------------
    ssm_queries = [
        ("How do we handle electrical equipment fires?", "electrical fire carbon dioxide", "Retrieve relevant safety manual emergency procedures."),
        ("What is the fire emergency evacuation protocol?", "fire emergency evacuation", "Retrieve relevant safety manual emergency procedures."),
        ("Where are the fire safety rules?", "fire safety manual rules", "Retrieve relevant safety manual emergency procedures."),
        ("What do safety rules say about fire outbreaks?", "fire outbreak safety protocol", "Retrieve relevant safety manual emergency procedures."),
        ("Emergency fire evacuation procedures in safety manual", "emergency fire evacuation", "Retrieve relevant safety manual emergency procedures."),
        ("Chemical spill safety rules", "chemical spill safety rules", "Retrieve relevant safety manual emergency procedures."),
        ("What are the chemical hazard protocols?", "chemical hazard safety protocol", "Retrieve relevant safety manual emergency procedures."),
        ("Safety guidelines for volatile chemical storage", "volatile chemical safety guidelines", "Retrieve relevant safety manual emergency procedures."),
        ("Emergency containment protocols in safety manual", "emergency containment protocols", "Retrieve relevant safety manual emergency procedures."),
        ("What are the facility safety rules for power surges?", "power surge facility safety", "Retrieve relevant safety manual emergency procedures."),
        ("Safety instructions for emergency evacuation", "safety instructions evacuation", "Retrieve relevant safety manual emergency procedures."),
        ("Look up fire evacuation rules in the manual", "fire evacuation manual rules", "Retrieve relevant safety manual emergency procedures."),
        ("fire safety protocol", "fire safety protocol", "Retrieve relevant safety manual emergency procedures."),
        ("chemical spill rules", "chemical spill rules", "Retrieve relevant safety manual emergency procedures."),
        ("saftey manual fire", "fire safety", "Retrieve relevant safety manual emergency procedures."),
        ("evacuashun protocol", "evacuation protocol", "Retrieve relevant safety manual emergency procedures.")
    ]
    for _ in range(150):
        room_id = random.choice(room_ids)
        q, query_term, thought = random.choice(ssm_queries)
        args = {"query": query_term}
        ans = f"Safety manual search returned relevant instructions for '{query_term}'."
        add_sample(validate_and_create_sample(q, "search_safety_manual", args, thought, room_id, "ARCHIVIST", ans))

    # -------------------------------------------------------------------------
    # 5. get_visible_objects (~140 samples)
    # Focus: Casual phrasing, optical sweeps, item counts, typos
    # -------------------------------------------------------------------------
    gvo_queries = [
        "What objects can you see?",
        "Scan the room for objects.",
        "Look around and list what you see.",
        "What is currently visible in this room?",
        "Perform a visual survey of this room.",
        "Do an optical scan of the surroundings.",
        "What do your cameras detect here?",
        "List visible furniture and objects.",
        "Inspect room contents.",
        "Optical sweep of current room.",
        "look around pls",
        "scan this area",
        "wht can u see",
        "show visble objects",
        "what objcts are visible",
        "survey room",
        "Count the visible objects in this room.",
        "How many items are detected here?",
        "Count all visible furniture.",
        "Tally the objects in this sector.",
        "Perform optical sweep of this zone",
        "Sweep current room visually"
    ]
    for _ in range(170):
        room_id = random.choice(room_ids)
        q = random.choice(gvo_queries)
        rm_def = next(r for r in ROOM_DEFINITIONS if r["id"] == room_id)
        thought = f"Survey visual perception feed inside {rm_def['name']} to identify all detected objects."
        args = {"room_name": None, "player_position": None, "camera_heading": None, "filter_fov": False}
        ans = f"Visual survey of {rm_def['name']} complete. Detected registered objects."
        add_sample(validate_and_create_sample(q, "get_visible_objects", args, thought, room_id, "SCOUT", ans))

    # -------------------------------------------------------------------------
    # 6. navigate_to & find_path (~130 samples each)
    # Focus: Destination entities (rooms and objects)
    # -------------------------------------------------------------------------
    dests = ["office", "laboratory", "library", "reception", "storage", "table", "chair", "cabinet", "bookshelf", "door"]
    nav_templates = [
        "Guide me to the {d}",
        "Take me to the {d}",
        "Head toward the {d}",
        "Navigate to the {d}",
        "Lead the way to {d}",
        "Walk over to the {d}",
        "go to {d}",
        "walk to {d}",
        "tke me to {d}",
        "navgate to {d}"
    ]
    for _ in range(160):
        room_id = random.choice(room_ids)
        d = random.choice(dests)
        q = random.choice(nav_templates).format(d=d)
        thought = f"Plan collision-free turn-by-turn route and dispatch companion navigation to {d}."
        args = {"destination": d}
        ans = f"Navigating to {d}."
        add_sample(validate_and_create_sample(q, "navigate_to", args, thought, room_id, "NAVIGATOR", ans))

    path_templates = [
        "Calculate path to {d}",
        "Find route to {d}",
        "Compute waypoints to {d}",
        "What are the navigation waypoints to {d}?",
        "Route to {d}",
        "Calculate path to the {d}"
    ]
    for _ in range(150):
        room_id = random.choice(room_ids)
        d = random.choice(dests)
        q = random.choice(path_templates).format(d=d)
        thought = f"Compute obstacle-avoiding 3D geometric waypoints to {d}."
        args = {"destination": d}
        ans = f"Waypoints calculated to {d}."
        add_sample(validate_and_create_sample(q, "find_path", args, thought, room_id, "NAVIGATOR", ans))

    # -------------------------------------------------------------------------
    # 7. Remaining Tools (Balanced ~80-110 samples each)
    # -------------------------------------------------------------------------
    # locate_object
    for _ in range(120):
        room_id = random.choice(room_ids)
        obj = random.choice(all_obj_names)
        q = random.choice([f"Where is the {obj}?", f"Find 3D position of {obj}", f"Locate {obj} coordinates", f"where is {obj}"])
        thought = f"Retrieve the 3D metric coordinates of {obj}."
        args = {"object_name": obj}
        ans = f"The {obj} is located."
        add_sample(validate_and_create_sample(q, "locate_object", args, thought, room_id, "SCOUT", ans))

    # inspect_object_closely
    for _ in range(120):
        room_id = random.choice(room_ids)
        obj = random.choice(all_obj_names)
        q = random.choice([f"Inspect the {obj} closely.", f"Examine the {obj} details.", f"Close-up inspection of {obj}."])
        thought = f"Perform close-up visual crop of {obj} for detailed analysis."
        args = {"object_name": obj}
        ans = f"Close inspection of {obj} complete."
        add_sample(validate_and_create_sample(q, "inspect_object_closely", args, thought, room_id, "SCOUT", ans))

    # query_spatial_relation
    for _ in range(120):
        room_id = random.choice(room_ids)
        obj = random.choice(all_obj_names)
        rel = random.choice(["near", "left_of", "right_of"])
        q = random.choice([f"What is near the {obj}?", f"What is to the left of the {obj}?", f"Which object is beside the {obj}?"])
        thought = f"Query spatial relationship for {obj}."
        args = {"object_name": obj, "relation": rel}
        ans = f"Spatial query evaluated for {obj}."
        add_sample(validate_and_create_sample(q, "query_spatial_relation", args, thought, room_id, "NAVIGATOR", ans))

    # explain_unreachable
    for _ in range(110):
        room_id = random.choice(room_ids)
        q = random.choice(["Why is the destination unreachable?", "Why can't you navigate there?", "Explain why that route is inaccessible.", "Why is the path blocked?"])
        thought = "Explain target reachability constraints, camera FOV limits, or locked door partitions."
        args = {"query": "Destination unreachable"}
        ans = "The target is unreachable."
        add_sample(validate_and_create_sample(q, "explain_unreachable", args, thought, room_id, "NAVIGATOR", ans))

    # explain_uncertainty
    for _ in range(110):
        room_id = random.choice(room_ids)
        q = random.choice(["What are you uncertain about?", "Why is detection confidence low?", "Explain sensor uncertainty margins.", "What is your margin of error?"])
        thought = "Inspect visual perception confidence threshold and depth uncertainty margins."
        args = {"query": "detection confidence analysis"}
        ans = "Sensor detection confidence averages 91%."
        add_sample(validate_and_create_sample(q, "explain_uncertainty", args, thought, room_id, "SCOUT", ans))

    # get_inventory
    for _ in range(100):
        room_id = random.choice(room_ids)
        q = random.choice(["Show my inventory.", "What items do I have in my inventory?", "Check my collected items.", "inventory list", "show inventory"])
        thought = "Retrieve items and keys collected in investigator inventory."
        args = {}
        ans = "Investigator Inventory: collected items listed."
        add_sample(validate_and_create_sample(q, "get_inventory", args, thought, room_id, "ARCHIVIST", ans))

    # search_documents
    for _ in range(100):
        room_id = random.choice(room_ids)
        q = random.choice(["What happened during Incident 104?", "What is the emergency override code in the logs?", "Search documents for security clearance."])
        thought = "Retrieve relevant facility documentation."
        args = {"query": "Incident 104 power surge"}
        ans = "Document search complete."
        add_sample(validate_and_create_sample(q, "search_documents", args, thought, room_id, "ARCHIVIST", ans))

    # inspect_clue
    for _ in range(90):
        room_id = random.choice(room_ids)
        cid = random.choice(["clue_safe_01", "clue_pda_02", "clue_terminal_03"])
        q = random.choice([f"Inspect clue {cid}", f"Examine {cid} details", f"Read clue note on {cid}"])
        thought = f"Inspect active investigation clue {cid}."
        args = {"clue_id": cid}
        ans = f"Clue {cid} inspected."
        add_sample(validate_and_create_sample(q, "inspect_clue", args, thought, room_id, "ARCHIVIST", ans))

    # get_mission_objectives
    for _ in range(90):
        room_id = random.choice(room_ids)
        q = random.choice(["What are my current mission objectives?", "Have I completed my objectives?", "Check mission progress.", "mission status"])
        thought = "Query mission objective tracker and completion status."
        args = {}
        ans = "Current mission objectives evaluated."
        add_sample(validate_and_create_sample(q, "get_mission_objectives", args, thought, room_id, "ARCHIVIST", ans))

    # get_game_guide
    for _ in range(90):
        room_id = random.choice(room_ids)
        q = random.choice(["How to play this game?", "How do I control the drone?", "Show game guide", "controls guide"])
        thought = "Retrieve field guide instructions."
        args = {"topic": "all"}
        ans = "Spatial Quest Field Guide displayed."
        add_sample(validate_and_create_sample(q, "get_game_guide", args, thought, room_id, "ARCHIVIST", ans))

    # find_nearest_exit
    for _ in range(100):
        room_id = random.choice(room_ids)
        q = random.choice(["Where is the exit from this room?", "Which way leads to the next section?", "How do we leave this room?", "Find the nearest doorway threshold."])
        thought = "Locate nearest doorway or archway threshold to navigate egress."
        args = {}
        ans = "The nearest egress doorway is located along the primary corridor."
        add_sample(validate_and_create_sample(q, "find_nearest_exit", args, thought, room_id, "NAVIGATOR", ans))

    # get_current_location
    for _ in range(100):
        room_id = random.choice(room_ids)
        q = random.choice(["Where are we right now?", "What room is this?", "Where am I?", "What zone are we in?"])
        thought = "Determine player coordinates and active room boundaries in the facility."
        args = {}
        ans = "Current location identified."
        add_sample(validate_and_create_sample(q, "get_current_location", args, thought, room_id, "NAVIGATOR", ans))

    # request_hint
    for _ in range(80):
        room_id = random.choice(room_ids)
        q = random.choice(["Give me a hint.", "I am stuck, give me a clue hint.", "Tactical hint for this puzzle.", "hint please", "give hint"])
        thought = "Request tactical investigation hint."
        args = {"topic": "mission_progression"}
        ans = "Tactical Hint provided."
        add_sample(validate_and_create_sample(q, "request_hint", args, thought, room_id, "ARCHIVIST", ans))

    random.shuffle(samples)
    print(f"Generated {len(samples)} valid, leakage-free samples.")
    return samples

def main():
    samples = generate_improved_dataset()
    random.seed(42)
    random.shuffle(samples)

    total = len(samples)
    train_count = 2000
    val_count = min(250, total - train_count)

    train_data = samples[:train_count]
    val_data = samples[train_count:train_count + val_count]

    print(f"Split counts: Train={len(train_data)}, Val={len(val_data)}")

    # Audit tool distribution in new train split
    tool_counts = defaultdict(int)
    for s in train_data:
        m = re.search(r"<tool_call>(.*?)</tool_call>", s["completion"])
        if m:
            c = json.loads(m.group(1))
            tool_counts[c.get("name")] += 1

    print("\n--- Improved Train Tool Distribution ---")
    for t, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt} ({cnt/len(train_data)*100:.1f}%)")

    # Save to data/corpora/improved_train.jsonl and improved_val.jsonl
    out_dir = "data/corpora"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "improved_train.jsonl")
    with open(train_path, "w") as f:
        for s in train_data:
            f.write(json.dumps(s) + "\n")
    print(f"✓ Saved {len(train_data)} train samples to {train_path}")

    val_path = os.path.join(out_dir, "improved_val.jsonl")
    with open(val_path, "w") as f:
        for s in val_data:
            f.write(json.dumps(s) + "\n")
    print(f"✓ Saved {len(val_data)} val samples to {val_path}")

    print("\nHeld-out test set remains UNTOUCHED at data/corpora/corrected_test.jsonl")

if __name__ == "__main__":
    main()
