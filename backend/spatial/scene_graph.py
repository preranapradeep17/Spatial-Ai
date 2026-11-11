"""Spatial scene graph management and relation inference."""
from typing import List, Dict, Optional, Tuple, Any
import math
import time
from backend.api.schemas import DetectedObject, SceneGraphNode, SceneGraphEdge, SceneGraphResponse

class SceneGraphManager:
    """Maintains 3D object positions and extracts spatial relationships."""

    def __init__(self, proximity_threshold: float = 2.0, on_vertical_threshold: float = 0.8):
        self.proximity_threshold = proximity_threshold
        self.on_vertical_threshold = on_vertical_threshold
        self.objects: Dict[str, DetectedObject] = {}
        self.live_perception_ids: set = set()
        self.last_updated: float = time.time()
        self.source: str = "initialized"

    def clear(self):
        self.objects.clear()
        self.live_perception_ids.clear()
        self.last_updated = time.time()

    def update_objects(self, detections: List[DetectedObject], source: str = "custom_detector"):
        """Updates internal state with new detections, annotating node provenance."""
        self.objects = {}
        self.live_perception_ids.clear()
        for d in detections:
            d_copy = d.model_copy(update={"source": d.source or source})
            self.objects[d_copy.object_id] = d_copy
            if source == "custom_detector":
                self.live_perception_ids.add(d_copy.object_id)
        self.last_updated = time.time()
        self.source = source

    def merge_live_detections(self, detections: List[DetectedObject], source: str = "custom_detector"):
        """Records live CNN detections alongside existing seeded facility layout objects
        without wiping out static room landmarks, keeping provenance strictly distinct."""
        # Remove previous live perception overlay nodes before adding latest live frame
        for oid in list(self.live_perception_ids):
            self.objects.pop(oid, None)
        self.live_perception_ids.clear()

        for idx, d in enumerate(detections):
            live_id = d.object_id if d.object_id.startswith("cnn_") else f"cnn_{d.object_id}"
            d_copy = d.model_copy(update={"object_id": live_id, "source": source})
            self.objects[live_id] = d_copy
            self.live_perception_ids.add(live_id)

        self.last_updated = time.time()
        if detections:
            self.source = f"hybrid_layout+{source}"

    def get_all_objects(self) -> List[DetectedObject]:
        return list(self.objects.values())

    def get_seeded_objects(self) -> List[DetectedObject]:
        return [o for o in self.objects.values() if o.object_id not in self.live_perception_ids]

    def get_live_perception_objects(self) -> List[DetectedObject]:
        return [o for o in self.objects.values() if o.object_id in self.live_perception_ids]

    def is_live_object(self, obj: DetectedObject) -> bool:
        return (obj.object_id in self.live_perception_ids) or (getattr(obj, "source", None) == "custom_detector")

    def locate_object(self, class_name: str) -> List[DetectedObject]:
        """Finds all detected objects of a given class or synonym (case-insensitive)."""
        target = class_name.lower().strip()
        synonyms = {
            "desk": ["desk", "table"],
            "table": ["table", "desk"],
            "couch": ["couch", "sofa"],
            "sofa": ["sofa", "couch"],
            "bookcase": ["bookcase", "bookshelf"],
            "bookshelf": ["bookshelf", "bookcase"],
            "exit": ["exit", "door"],
            "door": ["door", "exit"],
        }
        target_set = set(synonyms.get(target, [target]))
        target_set.add(target)

        matches = []
        for obj in self.objects.values():
            c_low = obj.class_name.lower()
            id_low = obj.object_id.lower()
            if any(t in c_low or c_low in t or t in id_low for t in target_set):
                matches.append(obj)
        return matches

    def calculate_distance(self, obj_a_query: str, obj_b_query: str) -> Optional[Tuple[DetectedObject, DetectedObject, float]]:
        """Calculates 3D Euclidean distance between two objects."""
        matches_a = self.locate_object(obj_a_query)
        matches_b = self.locate_object(obj_b_query)

        if not matches_a or not matches_b:
            return None

        # Choose highest confidence instances
        a = max(matches_a, key=lambda x: x.confidence)
        b = max(matches_b, key=lambda x: x.confidence)

        if not a.estimated_3d_position or not b.estimated_3d_position:
            return None

        pa = a.estimated_3d_position
        pb = b.estimated_3d_position
        dist = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2 + (pa[2] - pb[2])**2)
        return a, b, round(dist, 2)

    def get_objects_by_distance(
        self,
        reference_pos: Optional[List[float]] = None,
        target_class: Optional[str] = "any"
    ) -> List[Tuple[DetectedObject, float]]:
        """Calculates Euclidean distance to each candidate object and returns them sorted ascending.

        If target_class is None/any/all/object/me, searches all registered scene objects.
        """
        if reference_pos is None:
            reference_pos = [0.0, 0.0, 0.0]

        ref_clean = reference_pos if len(reference_pos) == 3 else [reference_pos[0], 0.0, reference_pos[1]]

        t_clean = (target_class or "").lower().strip()
        if not t_clean or t_clean in ("any", "all", "object", "objects", "me", "anything", "target"):
            candidates = list(self.objects.values())
        else:
            candidates = self.locate_object(t_clean)

        ranked: List[Tuple[DetectedObject, float]] = []
        for obj in candidates:
            if not obj.estimated_3d_position:
                continue
            p = obj.estimated_3d_position
            dist = math.sqrt((p[0] - ref_clean[0])**2 + (p[1] - ref_clean[1])**2 + (p[2] - ref_clean[2])**2)
            ranked.append((obj, round(dist, 2)))

        ranked.sort(key=lambda x: x[1])
        return ranked

    def find_nearest(
        self,
        target_class: Optional[str] = None,
        reference_pos: Optional[List[float]] = None
    ) -> Optional[Tuple[DetectedObject, float]]:
        """Finds nearest object of target_class to reference_pos (default [0,0,0]).
        If target_class is None/any/object/me, searches all registered scene objects.
        """
        ranked = self.get_objects_by_distance(reference_pos=reference_pos, target_class=target_class)
        if not ranked:
            return None
        return ranked[0]

    def compute_relations(self) -> List[SceneGraphEdge]:
        """Computes geometric directional and proximity relations between all pairs."""
        edges: List[SceneGraphEdge] = []
        obj_list = list(self.objects.values())

        for i in range(len(obj_list)):
            for j in range(len(obj_list)):
                if i == j:
                    continue
                a = obj_list[i]
                b = obj_list[j]

                if not a.estimated_3d_position or not b.estimated_3d_position:
                    continue

                pa = a.estimated_3d_position
                pb = b.estimated_3d_position

                dx = pa[0] - pb[0]
                dy = pa[1] - pb[1]
                dz = pa[2] - pb[2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                # Check "on" / "above" relationship
                horiz_dist = math.sqrt(dx**2 + dz**2)
                if 0.1 <= dy <= self.on_vertical_threshold and horiz_dist < 0.8:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="on", distance_meters=round(dist, 2)))
                if dy > 0.3 and horiz_dist < 1.5:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="above", distance_meters=round(dist, 2)))

                # Directional relations (lateral: left_of / right_of, longitudinal: in_front_of / behind)
                if dx < -0.3 and abs(dz) < 1.5:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="left_of", distance_meters=round(dist, 2)))
                elif dx > 0.3 and abs(dz) < 1.5:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="right_of", distance_meters=round(dist, 2)))

                if dz > 0.3 and abs(dx) < 1.8 and abs(dz) <= 3.5:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="in_front_of", distance_meters=round(dist, 2)))
                elif dz < -0.3 and abs(dx) < 1.8 and abs(dz) <= 3.5:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="behind", distance_meters=round(dist, 2)))

                # Proximity relation
                if dist <= self.proximity_threshold:
                    edges.append(SceneGraphEdge(source_id=a.object_id, target_id=b.object_id, relation="near", distance_meters=round(dist, 2)))

        return edges

    def query_spatial_relation(
        self,
        obj_name: str,
        relation: str = "all",
        target_obj_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Finds spatial relations involving obj_name (and optionally between target_obj_name)."""
        target_objs = self.locate_object(obj_name)
        if not target_objs:
            return []

        second_objs = self.locate_object(target_obj_name) if target_obj_name else []
        second_ids = {o.object_id for o in second_objs}
        if target_obj_name and not second_ids:
            return []

        rel_norm = (relation or "all").lower().strip()
        match_all_rels = rel_norm in ("all", "any", "between", "")
        all_edges = self.compute_relations()
        results = []

        for edge in all_edges:
            if not match_all_rels and edge.relation != rel_norm:
                continue
            for t in target_objs:
                if target_obj_name:
                    # Pair query: match edges in either direction between obj_name and target_obj_name
                    if edge.target_id == t.object_id and edge.source_id in second_ids:
                        other_obj = self.objects.get(edge.source_id)
                        if other_obj:
                            results.append({
                                "subject": other_obj.class_name,
                                "subject_id": other_obj.object_id,
                                "relation": edge.relation,
                                "object": t.class_name,
                                "object_id": t.object_id,
                                "distance_meters": edge.distance_meters,
                            })
                    elif edge.source_id == t.object_id and edge.target_id in second_ids:
                        other_obj = self.objects.get(edge.target_id)
                        if other_obj:
                            results.append({
                                "subject": t.class_name,
                                "subject_id": t.object_id,
                                "relation": edge.relation,
                                "object": other_obj.class_name,
                                "object_id": other_obj.object_id,
                                "distance_meters": edge.distance_meters,
                            })
                else:
                    if edge.target_id == t.object_id:
                        source_obj = self.objects.get(edge.source_id)
                        if source_obj:
                            results.append({
                                "subject": source_obj.class_name,
                                "subject_id": source_obj.object_id,
                                "relation": edge.relation,
                                "object": t.class_name,
                                "object_id": t.object_id,
                                "distance_meters": edge.distance_meters,
                            })

        return results

    def to_response(self) -> SceneGraphResponse:
        nodes = [
            SceneGraphNode(
                id=o.object_id,
                class_name=o.class_name,
                position=o.estimated_3d_position or [0.0, 0.0, 0.0],
                confidence=o.confidence,
                depth=o.depth_estimate,
                depth_source=getattr(o, "depth_source", None),
                source=getattr(o, "source", None) or ("custom_detector" if o.object_id in self.live_perception_ids else self.source)
            )
            for o in self.objects.values()
        ]
        edges = self.compute_relations()
        return SceneGraphResponse(
            nodes=nodes,
            edges=edges,
            last_updated=self.last_updated,
            source=self.source
        )

# Global singleton scene graph instance
global_scene_graph = SceneGraphManager()
