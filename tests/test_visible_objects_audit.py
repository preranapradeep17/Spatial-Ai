"""Tests for get_visible_objects() audit in Spatial Quest.

Verifies:
1. Room Changes: Detected object list updates dynamically based on the player's active room.
2. Camera Visibility: Frustum / FOV filtering excludes objects outside the camera's viewing angle.
3. Confidence Scores: Confirms 91% confidence originates from static scene graph seeding and is marked as scene graph data rather than live CNN inference.
4. Empty Detection Results: Handles empty rooms and unobserved zones cleanly without fabricating detections.
"""
import pytest
from backend.tools.registry import global_tool_registry
from backend.spatial.scene_graph import global_scene_graph
from backend.spatial.geometry import is_object_in_camera_fov
from backend.game.facility import seed_facility_scene
from backend.agent.investigation_agent import global_investigation_agent
from backend.game.missions import global_game_session
from backend.api.schemas import DetectedObject, BoundingBox2D


@pytest.fixture(autouse=True)
def setup_environment():
    global_game_session.return_to_hub()
    global_game_session.start_mission("mission_a")
    seed_facility_scene(source="facility_layout")


def test_room_changes_return_distinct_object_sets():
    """Verifies that get_visible_objects returns distinct objects when moving between rooms."""
    reception_res = global_tool_registry.execute("get_visible_objects", {"room_name": "Reception / Central Hub"})
    assert reception_res["success"] is True
    reception_objs = reception_res["result"]
    reception_classes = {o["class_name"] for o in reception_objs}
    assert "sofa" in reception_classes
    assert "table" in reception_classes

    office_res = global_tool_registry.execute("get_visible_objects", {"room_name": "Research Office"})
    assert office_res["success"] is True
    office_objs = office_res["result"]
    office_ids = {o["id"] for o in office_objs}
    assert "cabinet_office_files" in office_ids
    assert "sofa_visitor" not in office_ids

    library_res = global_tool_registry.execute("get_visible_objects", {"room_name": "Library / Archives"})
    assert library_res["success"] is True
    library_objs = library_res["result"]
    library_ids = {o["id"] for o in library_objs}
    assert "bookshelf_library_reference" in library_ids
    assert "cabinet_office_files" not in library_ids

    lab_res = global_tool_registry.execute("get_visible_objects", {"room_name": "Investigation Laboratory"})
    assert lab_res["success"] is True
    lab_objs = lab_res["result"]
    lab_ids = {o["id"] for o in lab_objs}
    assert "door_3" in lab_ids
    assert "cabinet_lab" in lab_ids


def test_confidence_scores_and_data_source_provenance():
    """Audits confidence scores (0.91) and confirms data source is explicitly scene_graph_registry."""
    res = global_tool_registry.execute("get_visible_objects", {"room_name": "Reception / Central Hub"})
    assert res["success"] is True
    objs = res["result"]

    assert len(objs) > 0
    for obj in objs:
        # Verify 91% confidence originates from static facility seeding
        assert obj["confidence"] == 0.91
        # Data source must clearly indicate scene_graph_registry (not live CNN)
        assert obj["data_source"] == "scene_graph_registry"
        assert obj["is_live_cnn"] is False


def test_camera_fov_visibility_filtering():
    """Verifies camera frustum / FOV calculation correctly filters objects outside view cone."""
    # Player at [0, -15] looking North (heading 0 degrees)
    # table_reception is at [0.0, 0.75, -15.0] (dist 0 -> in FOV)
    # sofa_visitor is at [-6.5, 0.3, -16.0] (behind / far left -> outside 60-degree North cone)
    res_filtered = global_tool_registry.execute("get_visible_objects", {
        "room_name": "Reception / Central Hub",
        "player_position": [0.0, -15.0],
        "camera_heading": 0.0,
        "filter_fov": True
    })
    assert res_filtered["success"] is True
    filtered_ids = {o["id"] for o in res_filtered["result"]}
    assert "sofa_visitor" not in filtered_ids

    # Rotate camera to face West (-90 or 270 deg) towards sofa at [-6.5, -16.0]
    res_west = global_tool_registry.execute("get_visible_objects", {
        "room_name": "Reception / Central Hub",
        "player_position": [0.0, -15.0],
        "camera_heading": 270.0,
        "filter_fov": True
    })
    assert res_west["success"] is True
    west_ids = {o["id"] for o in res_west["result"]}
    assert "sofa_visitor" in west_ids


def test_empty_detection_results_zero_fabrication():
    """Verifies that an empty room or unobserved zone returns zero objects without fabricating detections."""
    # Query with a non-existent or empty room name
    empty_res = global_tool_registry.execute("get_visible_objects", {"room_name": "Empty Quarantine Chamber"})
    assert empty_res["success"] is True
    assert empty_res["result"] == []

    # Agent response for empty detection must honestly state no objects detected
    agent_res = global_investigation_agent.run("What objects can you see?", active_room="Empty Chamber")
    assert "No objects" in agent_res["final_response"] or "0 objects" in agent_res["final_response"]
    assert "Zero-Fabrication Policy" in agent_res["final_response"]


def test_live_cnn_source_distinction_when_cnn_inference_runs():
    """Verifies that if live SpatialDetector CNN updates the scene graph, source transitions to spatial_detector_cnn."""
    cnn_detections = [
        DetectedObject(
            object_id="live_table_01",
            class_id=0,
            class_name="table",
            confidence=0.874,
            bbox=BoundingBox2D(xmin=100.0, ymin=150.0, xmax=400.0, ymax=350.0),
            estimated_3d_position=[0.0, 0.75, 2.5],
            depth_estimate=2.5
        )
    ]
    global_scene_graph.update_objects(cnn_detections, source="custom_detector")

    res = global_tool_registry.execute("get_visible_objects", {})
    assert res["success"] is True
    objs = res["result"]
    assert len(objs) == 1
    assert objs[0]["id"] == "live_table_01"
    assert objs[0]["confidence"] == 0.874
    assert objs[0]["data_source"] == "spatial_detector_cnn"
    assert objs[0]["is_live_cnn"] is True
