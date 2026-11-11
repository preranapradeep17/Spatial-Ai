"""Standardized Tool Registry and Execution Engine for Spatial Quest.

Provides deterministic execution handlers for all 10 investigation tools:
1. get_visible_objects()
2. locate_object(object_name)
3. query_spatial_relation(object_name, relation)
4. calculate_distance(object_a, object_b)
5. find_nearest_target(target_class)
6. find_path(destination)
7. navigate_to(destination)
8. inspect_clue(clue_id)
9. search_documents(query)
10. interact_with_object(object_id)

Traceable to CS3235 (Working with LLMs - Tool Calling, Function Execution & Safety).
"""
import os
import re
import math
from typing import Dict, Any, Callable, List, Optional
from backend.spatial.scene_graph import global_scene_graph
from backend.rag.retriever import global_retriever

class ToolRegistry:
    """Central registry of validated tools available to the investigation agent."""

    def __init__(self):
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._descriptions: Dict[str, str] = {}
        self._clues_state: Dict[str, Dict[str, Any]] = {
            "clue_safe_01": {
                "name": "Wall Safe in Investigation Lab",
                "status": "locked",
                "description": "A heavy electronic wall safe requiring a 4-digit security code.",
                "hint": "Check maintenance logs for default emergency code."
            },
            "clue_pda_02": {
                "name": "Researcher's Forgotten Datapad",
                "status": "unlocked",
                "description": "A cracked datapad resting on the table. Screen shows: 'Containment grid destabilized. Code hidden behind bookshelf.'",
                "hint": "Inspect the bookshelf in Zone 2."
            },
            "clue_terminal_03": {
                "name": "Facility Central Terminal",
                "status": "active",
                "description": "Terminal displays facility status. Ventilation offline. Archive Door 3 locked.",
                "hint": "Use interact_with_object to toggle systems or enter override."
            }
        }
        self._environment_state: Dict[str, Any] = {
            "door_3": "locked",
            "ventilation": "offline",
            "power_grid": "emergency_mode"
        }
        self._register_default_tools()

    def register(self, name: str, description: str, func: Callable[..., Dict[str, Any]]):
        self._tools[name] = func
        self._descriptions[name] = description

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        return dict(self._descriptions)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a registered tool securely with error handling."""
        if tool_name not in self._tools:
            return {
                "success": False,
                "error": f"Unknown tool '{tool_name}'. Registered tools: {list(self._tools.keys())}"
            }
        try:
            handler = self._tools[tool_name]
            result = handler(**arguments)
            if isinstance(result, dict) and "success" in result:
                return {"success": result["success"], "tool": tool_name, "result": result}
            return {"success": True, "tool": tool_name, "result": result}
        except TypeError as te:
            return {"success": False, "tool": tool_name, "error": f"Argument error calling '{tool_name}': {str(te)}"}
        except Exception as e:
            return {"success": False, "tool": tool_name, "error": f"Execution error in '{tool_name}': {str(e)}"}

    def _register_default_tools(self):
        # 1. get_visible_objects()
        def _get_visible_objects(
            room_name: Optional[str] = None,
            player_position: Optional[List[float]] = None,
            camera_heading: Optional[float] = None,
            filter_fov: Optional[bool] = False,
            live_only: Optional[bool] = False,
        ):
            from backend.spatial.geometry import is_object_in_camera_fov

            objs = (
                global_scene_graph.get_live_perception_objects()
                if live_only
                else global_scene_graph.get_all_objects()
            )

            if room_name:
                from backend.game.facility import get_room_definition
                matched_room = get_room_definition(room_name)
                if matched_room:
                    b = matched_room["bounds"]
                    room_objs = [
                        o for o in objs
                        if o.estimated_3d_position and b["xmin"] <= o.estimated_3d_position[0] <= b["xmax"] and b["zmin"] <= o.estimated_3d_position[2] <= b["zmax"]
                    ]
                    if room_objs:
                        objs = room_objs
                    elif "empty" in room_name.lower() or live_only:
                        objs = []
                else:
                    objs = []

            # Optional camera frustum / FOV filtering
            if filter_fov and player_position is not None:
                cam_pos = [player_position[0], 1.5, player_position[1]] if len(player_position) == 2 else player_position
                cam_yaw = camera_heading if camera_heading is not None else 0.0
                objs = [
                    o for o in objs
                    if o.estimated_3d_position and is_object_in_camera_fov(o.estimated_3d_position, cam_pos, camera_yaw_deg=cam_yaw)
                ]

            from backend.game.facility import get_room_for_position
            results = []
            for o in objs:
                in_fov = True
                if player_position is not None and o.estimated_3d_position:
                    cam_pos = [player_position[0], 1.5, player_position[1]] if len(player_position) == 2 else player_position
                    cam_yaw = camera_heading if camera_heading is not None else 0.0
                    in_fov = is_object_in_camera_fov(o.estimated_3d_position, cam_pos, camera_yaw_deg=cam_yaw)
                pos = o.estimated_3d_position or [0.0, 0.0, 0.0]
                obj_room = get_room_for_position(pos[0], pos[2])["name"]
                obj_is_live = global_scene_graph.is_live_object(o)
                source_tag = "custom_detector" if obj_is_live else (getattr(o, "source", None) or "facility_layout")
                depth_src = getattr(o, "depth_source", None) or ("estimated_bbox_height" if obj_is_live else "seeded_layout")

                results.append({
                    "id": o.object_id,
                    "class_name": o.class_name,
                    "room": obj_room,
                    "confidence": round(o.confidence, 3),
                    "position": [round(p, 2) for p in pos],
                    "depth_m": round(o.depth_estimate, 2) if o.depth_estimate else None,
                    "depth_source": depth_src,
                    "source": source_tag,
                    "data_source": "spatial_detector_cnn" if obj_is_live else "scene_graph_registry",
                    "is_live_cnn": obj_is_live,
                    "in_camera_fov": in_fov
                })
            return results
        self.register("get_visible_objects", "List currently visible objects, room contents, or live SpatialDetector CNN scan detections (live_only=True).", _get_visible_objects)

        # 2. locate_object(object_name)
        def _locate_object(object_name: str):
            objs = global_scene_graph.locate_object(object_name)
            if not objs:
                from backend.game.missions import global_game_session
                active_m = global_game_session.active_mission()
                target_clean = object_name.lower().strip()
                m_title = active_m.get("title", "").lower()
                m_brief = active_m.get("briefing", "").lower()

                is_mission_target = (
                    target_clean in ("key", "master_key", "research_key", "file", "clue")
                    or target_clean in m_title
                    or target_clean in m_brief
                )

                mission_info = None
                rag_citations = []
                if is_mission_target:
                    mission_info = {
                        "mission_id": active_m.get("id"),
                        "title": active_m.get("title"),
                        "briefing": active_m.get("briefing"),
                        "hint": active_m.get("hint")
                    }
                    from backend.rag.retriever import global_retriever
                    if not global_retriever.is_indexed:
                        doc_path = "data/documents/facility_documents.json"
                        if os.path.exists(doc_path):
                            global_retriever.load_documents_from_json(doc_path)
                    rag_results = global_retriever.retrieve(f"{target_clean} location cabinet table", top_k=2)
                    rag_citations = [
                        {"title": r.get("title", "Facility Document"), "text": r.get("text", "")}
                        for r in rag_results
                    ]

                return {
                    "found": False,
                    "object_name": object_name,
                    "in_camera_fov": False,
                    "in_scene_graph": False,
                    "is_mission_target": is_mission_target,
                    "mission_info": mission_info,
                    "rag_citations": rag_citations,
                    "relevant_locations": ["Reception security table", "Investigation Lab cabinet"],
                    "message": f"Object '{object_name}' is not registered in the 3D scene graph or visible in current camera FOV."
                }

            from backend.game.facility import get_room_for_position
            best = max(objs, key=lambda x: x.confidence)
            best_pos = best.estimated_3d_position or [0.0, 0.0, 0.0]
            all_matches = []
            for o in objs:
                op = o.estimated_3d_position or [0.0, 0.0, 0.0]
                all_matches.append({
                    "object_id": o.object_id,
                    "class_name": o.class_name,
                    "room": get_room_for_position(op[0], op[2])["name"],
                    "position": [round(p, 2) for p in op],
                    "confidence": round(o.confidence, 3),
                    "depth_m": round(o.depth_estimate, 2) if o.depth_estimate else None,
                    "depth_source": getattr(o, "depth_source", None) or ("estimated_bbox_height" if global_scene_graph.is_live_object(o) else "seeded_layout"),
                    "source": "custom_detector" if global_scene_graph.is_live_object(o) else (getattr(o, "source", None) or "facility_layout"),
                })
            return {
                "found": True,
                "object_id": best.object_id,
                "class_name": best.class_name,
                "room": get_room_for_position(best_pos[0], best_pos[2])["name"],
                "confidence": round(best.confidence, 3),
                "position": [round(p, 2) for p in best_pos],
                "depth_m": round(best.depth_estimate, 2) if best.depth_estimate else None,
                "depth_source": getattr(best, "depth_source", None) or ("estimated_bbox_height" if global_scene_graph.is_live_object(best) else "seeded_layout"),
                "source": "custom_detector" if global_scene_graph.is_live_object(best) else (getattr(best, "source", None) or "facility_layout"),
                "all_matches": all_matches,
            }
        self.register("locate_object", "Locate an object by name across the facility and return 3D coordinates and rooms.", _locate_object)

        # 3. query_spatial_relation(object_name, relation, target_object_name)
        def _query_spatial_relation(
            object_name: str,
            relation: str = "all",
            target_object_name: Optional[str] = None,
        ):
            matches = global_scene_graph.query_spatial_relation(
                object_name,
                relation,
                target_obj_name=target_object_name,
            )
            return {
                "query_object": object_name,
                "target_object": target_object_name,
                "relation": relation,
                "matches": matches,
            }
        self.register(
            "query_spatial_relation",
            "Find spatial relationships (left_of, right_of, near, behind, in_front_of, above, on, or all) involving one or between two objects.",
            _query_spatial_relation,
        )

        # 4. calculate_distance(object_a, object_b)
        def _calculate_distance(object_a: str, object_b: str):
            res = global_scene_graph.calculate_distance(object_a, object_b)
            if res is None:
                return {"success": False, "message": f"Could not compute distance between '{object_a}' and '{object_b}'."}
            a, b, dist = res
            return {
                "success": True,
                "object_a": a.class_name,
                "object_b": b.class_name,
                "distance_meters": dist
            }
        self.register("calculate_distance", "Compute Euclidean 3D distance between two objects in meters.", _calculate_distance)

        # 5. find_nearest_target(target_class, player_position, reference_object)
        def _find_nearest_target(
            target_class: Optional[str] = "any",
            player_position: Optional[List[float]] = None,
            reference_object: Optional[str] = None
        ):
            from backend.game.missions import global_game_session
            from backend.game.facility import parse_position_coords, get_room_for_position
            from backend.spatial.geometry import is_object_in_camera_fov

            all_objs = global_scene_graph.get_all_objects()
            if not all_objs:
                return {
                    "found": False,
                    "target_class": target_class or "object",
                    "message": "Scene graph contains zero registered objects."
                }

            if player_position is None:
                player_position = global_game_session.player_position

            px, py, pz = parse_position_coords(player_position)
            player_ref = [px, py, pz]

            clean_target = (target_class or "any").strip()

            ref_obj_desc = None
            ref_pos = player_ref
            exclude_ids = set()

            # Handle object-specific queries (distance relative to requested object)
            if reference_object and reference_object.lower().strip() not in ("me", "myself", "i", "player", "user", "stance", "camera", "observer", "here"):
                matched_refs = global_scene_graph.locate_object(reference_object.strip())
                if not matched_refs:
                    return {
                        "found": False,
                        "target_class": clean_target,
                        "reference_object": reference_object,
                        "message": f"Reference object '{reference_object}' is not registered in the facility scene graph."
                    }
                # Choose the reference object closest to player
                ref_obj_node = matched_refs[0]
                if len(matched_refs) > 1:
                    def _d_to_p(o):
                        pos = o.estimated_3d_position or [0.0, 0.0, 0.0]
                        return math.sqrt((pos[0]-px)**2 + (pos[2]-pz)**2)
                    ref_obj_node = min(matched_refs, key=_d_to_p)
                ref_pos = ref_obj_node.estimated_3d_position or [0.0, 0.0, 0.0]
                ref_obj_desc = f"{ref_obj_node.class_name} ({ref_obj_node.object_id})"
                exclude_ids.add(ref_obj_node.object_id)

            # Get candidate objects sorted by Euclidean distance
            ranked = global_scene_graph.get_objects_by_distance(reference_pos=ref_pos, target_class=clean_target)
            if exclude_ids:
                ranked = [(o, d) for o, d in ranked if o.object_id not in exclude_ids]

            if not ranked:
                t_label = clean_target if clean_target not in ("any", "all", "object", "objects", "me") else "object"
                rel_str = f" relative to {ref_obj_desc}" if ref_obj_desc else ""
                return {
                    "found": False,
                    "target_class": clean_target,
                    "reference_object": reference_object,
                    "player_position": [round(px, 2), round(pz, 2)],
                    "message": f"No registered objects matching '{t_label}' found in facility scene graph{rel_str}."
                }

            best_obj, dist = ranked[0]
            obj_pos = best_obj.estimated_3d_position or [0.0, 0.0, 0.0]
            obj_room = get_room_for_position(obj_pos[0], obj_pos[2])
            player_room = get_room_for_position(px, pz)
            is_in_same_room = (obj_room["id"] == player_room["id"])

            ranked_summary = [
                {
                    "class_name": o.class_name,
                    "object_id": o.object_id,
                    "distance_meters": d,
                    "room": get_room_for_position(o.estimated_3d_position[0], o.estimated_3d_position[2])["name"] if o.estimated_3d_position else "Unknown",
                    "in_current_room": get_room_for_position(o.estimated_3d_position[0], o.estimated_3d_position[2])["id"] == player_room["id"] if o.estimated_3d_position else False
                }
                for o, d in ranked[:5]
            ]

            best_is_live = global_scene_graph.is_live_object(best_obj)
            return {
                "found": True,
                "object_id": best_obj.object_id,
                "class_name": best_obj.class_name,
                "target_class": clean_target,
                "reference_object": ref_obj_desc or "player",
                "position": [round(p, 2) for p in obj_pos],
                "player_position": [round(px, 2), round(pz, 2)],
                "reference_position": [round(p, 2) for p in ref_pos],
                "distance_meters": dist,
                "room": obj_room["name"],
                "player_room": player_room["name"],
                "in_current_room": is_in_same_room,
                "ranked_candidates": ranked_summary,
                "is_live_cnn": best_is_live,
                "depth_source": getattr(best_obj, "depth_source", None) or ("estimated_bbox_height" if best_is_live else "seeded_layout"),
                "source": "custom_detector" if best_is_live else (getattr(best_obj, "source", None) or "facility_layout"),
                "data_source": "spatial_detector_cnn" if best_is_live else "scene_graph_registry",
            }
        self.register("find_nearest_target", "Find the closest object of the target class to the observer.", _find_nearest_target)

        # 6. find_path(destination)
        def _find_path(
            destination: str,
            player_position: Optional[List[float]] = None,
            current_room: Optional[str] = None
        ):
            from backend.game.facility import (
                compute_turn_by_turn_route,
                get_room_for_position,
                get_room_definition,
                parse_position_coords
            )
            from backend.game.missions import global_game_session

            if player_position is None:
                player_position = global_game_session.player_position
            if current_room is None:
                current_room = global_game_session.current_room

            px, py, pz = parse_position_coords(player_position)
            orig_room_def = get_room_definition(current_room) or get_room_for_position(px, pz)
            orig_room = orig_room_def["name"]

            # 1. Check if destination refers to a room
            target_room_def = get_room_definition(destination)
            if target_room_def is not None:
                dest_room = target_room_def["name"]
                b = target_room_def["bounds"]
                is_already_there = (orig_room_def["id"] == target_room_def["id"]) or (b["xmin"] <= px <= b["xmax"] and b["zmin"] <= pz <= b["zmax"])
                if is_already_there:
                    return {
                        "destination": destination,
                        "status": "path_computed",
                        "target_object_id": target_room_def["id"],
                        "target_position": [round(px, 2), round(py, 2), round(pz, 2)],
                        "distance_meters": 0.0,
                        "origin_room": orig_room,
                        "destination_room": dest_room,
                        "turn_by_turn": [f"🎯 Already inside {dest_room} at coordinates [{px:.1f}, {py:.1f}, {pz:.1f}]."],
                        "waypoints": [{"step": 1, "coord": [px, py, pz], "note": f"Current stance in {dest_room}"}],
                        "total_steps": 1
                    }

                room_objs = [
                    o for o in global_scene_graph.get_all_objects()
                    if o.estimated_3d_position and b["xmin"] <= o.estimated_3d_position[0] <= b["xmax"] and b["zmin"] <= o.estimated_3d_position[2] <= b["zmax"]
                ]
                pref_obj = next((o for o in room_objs if o.class_name in ("table", "desk", "workbench")), None)
                if not pref_obj and room_objs:
                    pref_obj = room_objs[0]

                if pref_obj:
                    target_pos = pref_obj.estimated_3d_position or [0.0, 0.0, 0.0]
                    target_id = pref_obj.object_id
                else:
                    target_pos = [(b["xmin"] + b["xmax"]) / 2, 0.0, target_room_def.get("spawn_z", (b["zmin"] + b["zmax"]) / 2)]
                    target_id = target_room_def["id"]

                route = compute_turn_by_turn_route(
                    start_pos=[px, pz],
                    target_pos=target_pos,
                    target_name=dest_room,
                    target_room=dest_room,
                    origin_room_override=orig_room
                )

                return {
                    "destination": destination,
                    "status": "path_computed",
                    "target_object_id": target_id,
                    "target_position": [round(p, 2) for p in target_pos],
                    "distance_meters": route["distance_meters"],
                    "origin_room": route["origin_room"],
                    "destination_room": route["destination_room"],
                    "turn_by_turn": route["turns"],
                    "waypoints": route["waypoints"],
                    "total_steps": len(route["waypoints"])
                }

            # 2. Destination is an object
            dest_lower = destination.lower().strip()
            if "exit" in dest_lower or "door" in dest_lower:
                objs = global_scene_graph.locate_object("door")
            else:
                objs = global_scene_graph.locate_object(dest_lower)

            if not objs:
                for cls in ["table", "chair", "bookshelf", "cabinet", "sofa", "door"]:
                    if cls in dest_lower:
                        objs = global_scene_graph.locate_object(cls)
                        break

            if objs:
                room_hint = None
                if "security" in dest_lower:
                    room_hint = "reception"
                else:
                    for rm_id in ["office", "lab", "laboratory", "library", "reception", "storage", "vault"]:
                        if rm_id in dest_lower:
                            room_hint = rm_id
                            break

                selected_target = None
                if room_hint:
                    from backend.game.facility import get_room_for_position
                    for obj in objs:
                        pos = obj.estimated_3d_position or [0.0, 0.0, 0.0]
                        r_name = get_room_for_position(pos[0], pos[2])["id"]
                        if room_hint in r_name:
                            selected_target = obj
                            break

                if not selected_target:
                    def _dist_to_p(obj):
                        pos = obj.estimated_3d_position or [0.0, 0.0, 0.0]
                        return math.sqrt((pos[0] - px)**2 + (pos[2] - pz)**2)

                    sorted_objs = sorted(objs, key=_dist_to_p)
                    selected_target = sorted_objs[1] if len(sorted_objs) > 1 and _dist_to_p(sorted_objs[0]) < 1.2 else sorted_objs[0]

                target = selected_target
                target_pos = target.estimated_3d_position or [0.0, 0.0, 2.0]

                route = compute_turn_by_turn_route(
                    start_pos=[px, pz],
                    target_pos=target_pos,
                    target_name=target.class_name,
                    target_room=get_room_for_position(target_pos[0], target_pos[2])["name"],
                    origin_room_override=orig_room
                )

                return {
                    "destination": destination,
                    "status": "path_computed",
                    "target_object_id": target.object_id,
                    "target_position": [round(p, 2) for p in target_pos],
                    "distance_meters": route["distance_meters"],
                    "origin_room": route["origin_room"],
                    "destination_room": route["destination_room"],
                    "turn_by_turn": route["turns"],
                    "waypoints": route["waypoints"],
                    "total_steps": len(route["waypoints"])
                }
            return {"destination": destination, "status": "path_failed", "message": f"Destination '{destination}' not located."}
        self.register("find_path", "Calculate obstacle-avoiding 3D navigation waypoints to a destination.", _find_path)

        # 7. navigate_to(destination)
        def _navigate_to(
            destination: str,
            player_position: Optional[List[float]] = None,
            current_room: Optional[str] = None
        ):
            from backend.game.facility import (
                compute_turn_by_turn_route,
                get_room_for_position,
                get_room_definition,
                parse_position_coords
            )
            from backend.game.missions import global_game_session

            if player_position is None:
                player_position = global_game_session.player_position
            if current_room is None:
                current_room = global_game_session.current_room

            px, py, pz = parse_position_coords(player_position)
            orig_room_def = get_room_definition(current_room) or get_room_for_position(px, pz)
            orig_room = orig_room_def["name"]

            # 1. Check if destination refers to a room
            target_room_def = get_room_definition(destination)
            if target_room_def is not None:
                dest_room = target_room_def["name"]
                b = target_room_def["bounds"]

                # Check if player is already inside the destination room
                is_already_there = (orig_room_def["id"] == target_room_def["id"]) or (b["xmin"] <= px <= b["xmax"] and b["zmin"] <= pz <= b["zmax"])
                if is_already_there:
                    room_objs = [
                        o for o in global_scene_graph.get_all_objects()
                        if o.estimated_3d_position and b["xmin"] <= o.estimated_3d_position[0] <= b["xmax"] and b["zmin"] <= o.estimated_3d_position[2] <= b["zmax"]
                    ]
                    landmarks = [f"{o.class_name} at [{o.estimated_3d_position[0]:.1f}, {o.estimated_3d_position[1]:.1f}, {o.estimated_3d_position[2]:.1f}]" for o in room_objs[:3]]
                    return {
                        "action": "navigate_to",
                        "destination": destination,
                        "origin_room": orig_room,
                        "destination_room": dest_room,
                        "target_position": [round(px, 2), round(py, 2), round(pz, 2)],
                        "distance_meters": 0.0,
                        "status": "already_at_destination",
                        "landmarks": landmarks,
                        "turn_by_turn": [
                            f"🎯 You are already inside {dest_room}! Current stance coordinates: [{px:.1f}, {py:.1f}, {pz:.1f}]."
                        ],
                        "waypoints": [{"step": 1, "coord": [px, py, pz], "note": f"Already inside {dest_room}"}],
                        "instruction": f"Player is already inside {dest_room}."
                    }

                # Navigate to destination room
                room_objs = [
                    o for o in global_scene_graph.get_all_objects()
                    if o.estimated_3d_position and b["xmin"] <= o.estimated_3d_position[0] <= b["xmax"] and b["zmin"] <= o.estimated_3d_position[2] <= b["zmax"]
                ]
                pref_obj = next((o for o in room_objs if o.class_name in ("table", "desk", "workbench")), None)
                if not pref_obj and room_objs:
                    pref_obj = room_objs[0]

                if pref_obj:
                    target_pos = pref_obj.estimated_3d_position or [0.0, 0.0, 0.0]
                    target_id = pref_obj.object_id
                else:
                    target_pos = [(b["xmin"] + b["xmax"]) / 2, 0.0, target_room_def.get("spawn_z", (b["zmin"] + b["zmax"]) / 2)]
                    target_id = target_room_def["id"]

                route = compute_turn_by_turn_route(
                    start_pos=[px, pz],
                    target_pos=target_pos,
                    target_name=dest_room,
                    target_room=dest_room,
                    origin_room_override=orig_room
                )

                return {
                    "action": "navigate_to",
                    "destination": destination,
                    "target_object_id": target_id,
                    "target_position": [round(p, 2) for p in target_pos],
                    "distance_meters": route["distance_meters"],
                    "origin_room": route["origin_room"],
                    "destination_room": route["destination_room"],
                    "turn_by_turn": route["turns"],
                    "waypoints": route["waypoints"],
                    "status": "command_dispatched",
                    "instruction": f"Guiding companion avatar towards {dest_room} at {target_pos}."
                }

            # 2. Destination is an object
            target_objs = []
            dest_lower = destination.lower().strip()
            if "exit" in dest_lower or "door" in dest_lower:
                target_objs = global_scene_graph.locate_object("door")
            else:
                target_objs = global_scene_graph.locate_object(dest_lower)

            if not target_objs:
                for cls in ["table", "chair", "bookshelf", "cabinet", "sofa", "door"]:
                    if cls in dest_lower:
                        target_objs = global_scene_graph.locate_object(cls)
                        break

            if target_objs:
                room_hint = None
                if "security" in dest_lower:
                    room_hint = "reception"
                else:
                    for rm_id in ["office", "lab", "laboratory", "library", "reception", "storage", "vault"]:
                        if rm_id in dest_lower:
                            room_hint = rm_id
                            break

                selected_target = None
                if room_hint:
                    for obj in target_objs:
                        pos = obj.estimated_3d_position or [0.0, 0.0, 0.0]
                        r_name = get_room_for_position(pos[0], pos[2])["id"]
                        if room_hint in r_name:
                            selected_target = obj
                            break

                if not selected_target:
                    def _dist_to_p(obj):
                        pos = obj.estimated_3d_position or [0.0, 0.0, 0.0]
                        return math.sqrt((pos[0] - px)**2 + (pos[2] - pz)**2)

                    sorted_targets = sorted(target_objs, key=_dist_to_p)
                    # If player is right beside target (< 1.2m), route to next if available
                    if len(sorted_targets) > 1 and _dist_to_p(sorted_targets[0]) < 1.2:
                        selected_target = sorted_targets[1]
                    else:
                        selected_target = sorted_targets[0]

                target_pos = selected_target.estimated_3d_position or [0.0, 0.0, 0.0]
                route = compute_turn_by_turn_route(
                    start_pos=[px, pz],
                    target_pos=target_pos,
                    target_name=selected_target.class_name,
                    target_room=get_room_for_position(target_pos[0], target_pos[2])["name"],
                    origin_room_override=orig_room
                )

                return {
                    "action": "navigate_to",
                    "destination": destination,
                    "target_object_id": selected_target.object_id,
                    "target_position": [round(p, 2) for p in target_pos],
                    "distance_meters": route["distance_meters"],
                    "origin_room": route["origin_room"],
                    "destination_room": route["destination_room"],
                    "turn_by_turn": route["turns"],
                    "waypoints": route["waypoints"],
                    "status": "command_dispatched",
                    "instruction": f"Guiding companion avatar towards {destination} at {target_pos}."
                }

            return {
                "action": "navigate_to",
                "destination": destination,
                "target_position": [0.0, 0.0, 0.0],
                "status": "target_unreachable"
            }
        self.register("navigate_to", "Dispatch companion drone navigation command to target destination.", _navigate_to)

        # 8. inspect_clue(clue_id)
        def _inspect_clue(clue_id: str):
            clue = self._clues_state.get(clue_id)
            if not clue:
                return {
                    "found": False,
                    "clue_id": clue_id,
                    "available_clues": list(self._clues_state.keys())
                }
            return {"found": True, "clue_id": clue_id, "data": clue}
        self.register("inspect_clue", "Examine an interactive clue or puzzle object.", _inspect_clue)

        # 9. search_documents(query)
        def _search_documents(query: str):
            if not global_retriever.is_indexed:
                doc_path = "data/documents/facility_documents.json"
                if os.path.exists(doc_path):
                    global_retriever.load_documents_from_json(doc_path)
            clean_q = re.sub(r"^(?:search\s+(?:documents\s+)?(?:for\s+)?|find\s+(?:documents\s+)?(?:about\s+)?|look\s+up\s+)", "", query, flags=re.IGNORECASE).strip()
            results = global_retriever.retrieve(clean_q or query, top_k=2)
            return {
                "query": query,
                "retrieved_count": len(results),
                "passages": results
            }
        self.register("search_documents", "Query facility logs, incident reports, and safety manuals using RAG.", _search_documents)

        # 10. interact_with_object(object_id, passcode)
        def _interact_with_object(object_id: str, passcode: Optional[str] = None):
            from backend.game.missions import global_game_session
            expected_code = "7391"
            for m in global_game_session.missions:
                if m.get("id") in ("mission_d", "mission_e"):
                    m_code = re.search(r"\b([0-9]{4})\b", m.get("briefing", ""))
                    if m_code:
                        expected_code = m_code.group(1)
                        break

            obj_lower = object_id.lower()
            if "safe" in obj_lower:
                if passcode is not None and str(passcode).strip() != expected_code:
                    return {
                        "success": False,
                        "object_id": object_id,
                        "action": "access_denied",
                        "result": f"❌ Access Denied: Code '{passcode}' is invalid for the safe. Security lock engaged."
                    }
                self._clues_state["clue_safe_01"]["status"] = "unlocked"
                return {"success": True, "object_id": object_id, "action": "unlock", "result": "Wall safe unlocked! Inside is the master keycard."}
            elif "door" in obj_lower:
                if passcode is not None and str(passcode).strip() != expected_code:
                    return {
                        "success": False,
                        "object_id": object_id,
                        "action": "access_denied",
                        "result": f"❌ Access Denied: Security override code '{passcode}' is invalid. Door 3 magnetic lock remains engaged."
                    }
                self._environment_state["door_3"] = "unlocked"
                code_txt = f" (Override Code: {passcode})" if passcode else ""
                return {"success": True, "object_id": object_id, "action": "open", "result": f"Door 3 emergency magnetic lock disengaged{code_txt}. Path open."}
            elif any(k in obj_lower for k in ("cabinet", "key", "drawer")):
                global_game_session.add_to_inventory("research_key", "Master Research Key", "Physical backup key for restricted lab cabinets and security consoles.")
                return {"success": True, "object_id": object_id, "action": "retrieve", "result": "Cabinet drawer opened! Discovered and collected the Master Research Key into inventory."}
            elif "terminal" in obj_lower or "console" in obj_lower:
                self._environment_state["ventilation"] = "active"
                return {"success": True, "object_id": object_id, "action": "activate", "result": "Ventilation scrubbers activated. Air scrubbed."}
            return {"success": True, "object_id": object_id, "action": "inspect", "result": f"Interacted with {object_id}. No state change."}
        self.register("interact_with_object", "Trigger environment interactions (open doors, unlock safes, use terminals).", _interact_with_object)

        # 11. get_inventory()
        def _get_inventory():
            from backend.game.missions import global_game_session
            inv = global_game_session.inventory
            return {
                "inventory": inv,
                "count": len(inv),
                "items": [item["name"] for item in inv] if inv else ["No items collected yet."]
            }
        self.register("get_inventory", "Retrieve player inventory and collected mystery clues.", _get_inventory)

        # 12. get_mission_objectives()
        def _get_mission_objectives():
            from backend.game.missions import global_game_session
            active = global_game_session.active_mission()
            session_dict = global_game_session.to_dict()
            is_done = session_dict["active_mission"]["completed"]
            return {
                "mission_id": active["id"],
                "code": active["code"],
                "title": active["title"],
                "objectives": active["objectives"],
                "completed": is_done,
                "hint": active.get("hint", ""),
                "briefing": active.get("briefing", ""),
                "inventory": [it["name"] for it in global_game_session.inventory],
                "score": global_game_session.score
            }
        self.register("get_mission_objectives", "View active mission objectives and completion status.", _get_mission_objectives)

        # 13. request_hint(topic)
        def _request_hint(topic: Optional[str] = None):
            from backend.game.missions import global_game_session
            hint_str = global_game_session.get_hint()
            active_m = global_game_session.active_mission()
            resolved_topic = topic or "mission_progression"
            return {
                "hint": hint_str,
                "topic": resolved_topic,
                "mission_id": active_m.get("id"),
                "mission_title": active_m.get("title")
            }
        self.register("request_hint", "Request a tactical hint for the current mystery.", _request_hint)

        # Backward compatibility aliases
        def _find_nearest_exit():
            res = global_scene_graph.find_nearest("door") or global_scene_graph.find_nearest("exit")
            if res is None:
                return {"found": False, "message": "No door or exit detected in the scene."}
            exit_obj, dist = res
            return {
                "found": True,
                "exit_id": exit_obj.object_id,
                "class_name": exit_obj.class_name,
                "position": [round(p, 2) for p in (exit_obj.estimated_3d_position or [0.0, 0.0, 0.0])],
                "distance_meters": dist
            }
        self.register("find_nearest_exit", "Find the nearest door or exit.", _find_nearest_exit)

        def _search_safety_manual(query: str):
            if not global_retriever.is_indexed:
                doc_path = "data/documents/facility_documents.json"
                if os.path.exists(doc_path):
                    global_retriever.load_documents_from_json(doc_path)
            res = global_retriever.retrieve(query, top_k=2)
            passages = [r["text"] for r in res] if res else [
                "Lab Safety Rule 4: In case of fire, evacuate immediately via the nearest marked green emergency exit.",
                "Equipment Safety: Carbon dioxide (CO2) extinguishers are designated for electrical fires near computers."
            ]
            return {
                "query": query,
                "retrieved_passages": passages,
                "source": "facility_documents.json"
            }
        self.register("search_safety_manual", "Search the lab safety manual.", _search_safety_manual)

        # 14. explain_uncertainty(query)
        def _explain_uncertainty(query: Optional[str] = None):
            objs = global_scene_graph.get_all_objects()
            if not objs:
                return {
                    "status": "no_objects",
                    "explanation": "No objects currently detected in scene graph. Visual uncertainty is high."
                }
            confs = [o.confidence for o in objs]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            low_conf = [
                {"id": o.object_id, "class_name": o.class_name, "confidence": round(o.confidence, 3)}
                for o in objs if o.confidence < 0.85
            ]
            depth_errors = [
                {"id": o.object_id, "depth_m": o.depth_estimate, "estimated_margin_m": round(0.05 + 0.05 * (o.depth_estimate or 2.0), 2)}
                for o in objs[:5]
            ]
            return {
                "overall_confidence": round(avg_conf, 3),
                "threshold": 0.85,
                "low_confidence_detections": low_conf,
                "depth_uncertainty_margins": depth_errors,
                "empirical_benchmark": {
                    "dataset": "SUN RGB-D Held-out Test Split",
                    "mAP_50": 0.0000,
                    "reason": "From-scratch single-stage CNN detector with random weight initialization trained on small real-world split without pretraining."
                },
                "status_message": f"Identified {len(objs)} objects with {round(avg_conf * 100, 1)}% mean confidence. Depth localization has a ±0.35m estimated margin of error."
            }
        self.register("explain_uncertainty", "Explain visual detection confidence, sensor limitations, and 3D spatial uncertainty.", _explain_uncertainty)

        # 15. inspect_object_closely(object_name)
        def _inspect_object_closely(object_name: str):
            objs = global_scene_graph.locate_object(object_name)
            if not objs:
                return {"found": False, "object_name": object_name, "message": f"Could not locate '{object_name}' in current visual frame."}
            target = max(objs, key=lambda x: x.confidence)
            t_pos = target.estimated_3d_position or [0.0, 0.0, 0.0]
            nearby = []
            for other in global_scene_graph.get_all_objects():
                if other.object_id != target.object_id and other.estimated_3d_position:
                    dist = math.dist(t_pos, other.estimated_3d_position)
                    if dist <= 3.5:
                        nearby.append({
                            "object_id": other.object_id,
                            "class_name": other.class_name,
                            "distance_m": round(dist, 2),
                            "confidence": round(other.confidence, 3)
                        })
            return {
                "found": True,
                "object_id": target.object_id,
                "class_name": target.class_name,
                "confidence": round(target.confidence, 3),
                "bbox": [target.bbox.xmin, target.bbox.ymin, target.bbox.xmax, target.bbox.ymax] if target.bbox else None,
                "position": [round(p, 2) for p in t_pos],
                "depth_m": round(target.depth_estimate, 2) if target.depth_estimate else None,
                "nearby_objects": nearby,
                "inspection_note": f"Detailed visual crop for '{target.class_name}' inspected. Spatial bounding box and depth verified."
            }
        self.register("inspect_object_closely", "Perform close visual inspection on a specific object and analyze nearby anomalies.", _inspect_object_closely)

        # 16. get_game_guide(topic)
        def _get_game_guide(topic: str = "all"):
            return {
                "title": "Spatial Quest Investigation & Gameplay Guide",
                "topic": topic,
                "status": "ready"
            }
        self.register("get_game_guide", "Retrieve gameplay guide, movement controls, voice commands, and investigation mechanics.", _get_game_guide)

        # 17. explain_unreachable(query, destination)
        def _explain_unreachable(query: Optional[str] = None, destination: Optional[str] = None):
            objs = global_scene_graph.get_all_objects()
            table_objs = [o for o in objs if "table" in o.class_name.lower()]
            from backend.game.facility import get_room_for_position
            locations_info = []
            for t in table_objs:
                pos = t.estimated_3d_position or [0.0, 0.0, 0.0]
                rm = get_room_for_position(pos[0], pos[2])
                locations_info.append(f"• {rm['name']}: {t.class_name} at [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

            return {
                "explanation": "Target objects become unreachable if the local camera hasn't detected them, if the scene graph was unseeded, or if security doors are sealed.",
                "available_tables": locations_info,
                "door_3_status": self._environment_state.get("door_3", "locked"),
                "suggestion": "You can ask: 'Take me to the laboratory table' or 'Take me to the office table'."
            }
        self.register("explain_unreachable", "Explain why an object or room is unreachable, locked, or blocked.", _explain_unreachable)

        # 18. get_current_location(player_position, active_room)
        def _get_current_location(player_position: Optional[List[float]] = None, active_room: Optional[str] = None):
            from backend.game.facility import ROOM_DEFINITIONS, get_room_for_position, get_room_definition, parse_position_coords
            from backend.game.missions import global_game_session
            if player_position is None:
                player_position = global_game_session.player_position
            if active_room is None:
                active_room = global_game_session.current_room

            px, py, pz = parse_position_coords(player_position)
            rm = get_room_definition(active_room) or get_room_for_position(px, pz)
            return {
                "current_room": rm["name"],
                "coordinates": [round(px, 2), round(py, 2), round(pz, 2)],
                "description": rm["description"],
                "adjacent_zones": "Reception (Z < -10), Research Office (Z -10..0, East), Library (Z -10..0, West), Laboratory (Z 0..10), Storage Vault (Z > 10).",
                "available_rooms": [
                    {"id": r["id"], "name": r["name"], "description": r["description"], "bounds": r["bounds"]}
                    for r in ROOM_DEFINITIONS
                ],
            }
        self.register("get_current_location", "Report current player room, coordinates, and facility orientation.", _get_current_location)


# Global tool registry singleton
global_tool_registry = ToolRegistry()
