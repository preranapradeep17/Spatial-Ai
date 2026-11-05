"""Unit tests for spatial scene graph and geometric relation inference."""
import pytest
import math
from backend.spatial.scene_graph import SceneGraphManager
from backend.api.schemas import DetectedObject, BoundingBox2D

def test_scene_graph_distance_and_nearest():
    sg = SceneGraphManager()
    table = DetectedObject(
        object_id="table_01",
        class_id=0,
        class_name="table",
        confidence=0.9,
        bbox=BoundingBox2D(xmin=100, ymin=200, xmax=400, ymax=400),
        estimated_3d_position=[0.0, 0.75, 2.0],
        depth_estimate=2.0
    )
    door = DetectedObject(
        object_id="door_01",
        class_id=4,
        class_name="door",
        confidence=0.95,
        bbox=BoundingBox2D(xmin=450, ymin=100, xmax=600, ymax=450),
        estimated_3d_position=[3.0, 1.2, 6.0],
        depth_estimate=6.0
    )
    sg.update_objects([table, door])

    # Distance calculation: dx = 3, dy = 0.45, dz = 4 -> sqrt(9 + 0.2025 + 16) = sqrt(25.2025) ~ 5.02
    res = sg.calculate_distance("table", "door")
    assert res is not None
    _, _, dist = res
    expected_dist = round(math.sqrt(3.0**2 + 0.45**2 + 4.0**2), 2)
    assert dist == expected_dist

    # Nearest exit from origin [0, 0, 0]
    nearest_res = sg.find_nearest("door")
    assert nearest_res is not None
    nearest_obj, n_dist = nearest_res
    assert nearest_obj.object_id == "door_01"

def test_spatial_relations_laptop_on_table():
    sg = SceneGraphManager()
    table = DetectedObject(
        object_id="table_01",
        class_id=0,
        class_name="table",
        confidence=0.9,
        bbox=BoundingBox2D(xmin=100, ymin=200, xmax=400, ymax=400),
        estimated_3d_position=[0.0, 0.75, 2.0],
        depth_estimate=2.0
    )
    laptop = DetectedObject(
        object_id="laptop_01",
        class_id=2,
        class_name="laptop",
        confidence=0.88,
        bbox=BoundingBox2D(xmin=200, ymin=220, xmax=300, ymax=300),
        estimated_3d_position=[0.05, 0.95, 2.05],  # 0.2m higher, minimal horizontal offset
        depth_estimate=2.05
    )
    chair = DetectedObject(
        object_id="chair_01",
        class_id=1,
        class_name="chair",
        confidence=0.85,
        bbox=BoundingBox2D(xmin=20, ymin=250, xmax=100, ymax=400),
        estimated_3d_position=[-0.8, 0.5, 2.0],  # Lateral left
        depth_estimate=2.0
    )
    sg.update_objects([table, laptop, chair])

    relations = sg.query_spatial_relation("table", "on")
    assert any(r["subject"] == "laptop" for r in relations), "Expected laptop to be on table"

    left_relations = sg.query_spatial_relation("table", "left_of")
    assert any(r["subject"] == "chair" for r in left_relations), "Expected chair to be left of table"
