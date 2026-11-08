"""Integration tests for FastAPI REST API endpoints."""
import pytest
from starlette.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_serves_web_terminal():
    """Verify GET / returns the interactive Spatial Quest Web Terminal."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Spatial" in response.text
    assert "Command" in response.text or "Terminal" in response.text

def test_health_check_endpoint():
    """Verify GET /health returns valid status, device info, and uptime."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "device_info" in data
    assert "selected_device" in data["device_info"]
    assert data["uptime_seconds"] >= 0


def test_perception_endpoint_mock_mode():
    """Verify POST /perception updates scene graph and returns detections."""
    payload = {"mock_mode": True}
    response = client.post("/perception", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "detections" in data
    assert len(data["detections"]) == 5
    assert data["source"] == "test_mock_pipeline"
    assert data["inference_time_ms"] >= 0


def test_perception_endpoint_real_mode():
    """Verify POST /perception executes from-scratch SpatialDetector when mock_mode is False."""
    import base64
    import io
    from PIL import Image

    # Create a small synthetic test image to test real forward pass
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {"mock_mode": False, "image_base64": b64}
    response = client.post("/perception", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "custom_detector"
    assert "detections" in data
    assert data["inference_time_ms"] >= 0


def test_scene_graph_endpoint():
    """Verify GET /scene returns updated nodes and computed edges."""
    # First populate via mock perception
    client.post("/perception", json={"mock_mode": True})

    response = client.get("/scene")
    assert response.status_code == 200
    data = response.json()

    assert "nodes" in data
    assert len(data["nodes"]) >= 5
    assert "edges" in data
    assert isinstance(data["edges"], list)

def test_agent_query_nearest_exit():
    """Verify POST /agent/query triggers find_nearest_exit and navigation action."""
    client.post("/perception", json={"mock_mode": True})

    payload = {"query": "Take me to the nearest exit."}
    response = client.post("/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == payload["query"]
    assert len(data["tool_calls"]) >= 1
    assert data["tool_calls"][0]["tool"] == "find_nearest_exit"
    assert "nearest exit" in data["final_response"].lower()
    assert data["unity_action"] is not None
    assert data["unity_action"]["action"] == "navigate_to"

def test_agent_query_fire_safety():
    """Verify POST /agent/query retrieves safety manual guidelines."""
    payload = {"query": "What should I do during a fire?"}
    response = client.post("/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert any(tc["tool"] == "search_safety_manual" for tc in data["tool_calls"])
    assert "safety" in data["final_response"].lower()

def test_game_state_and_companion_switch():
    """Verify mission catalog and companion whitelist are exposed to the game client."""
    response = client.get("/game/state")
    assert response.status_code == 200
    data = response.json()
    assert data["active_mission"]["code"] == "A"
    assert len(data["missions"]) == 6
    assert data["companion"]["id"] in {"scout", "navigator", "archivist"}

    switched = client.post("/game/agent", json={"agent_id": "navigator"})
    assert switched.status_code == 200
    assert switched.json()["companion"]["id"] == "navigator"

    locked = client.post("/game/start", json={"mission_id": "mission_f"})
    assert locked.status_code == 400


def test_unity_action_callback():
    """Verify POST /unity/action-result successfully logs avatar arrival."""
    payload = {
        "action_id": "test_act_123",
        "action_type": "navigate_to",
        "success": True,
        "message": "Avatar arrived at door.",
        "avatar_position": [2.5, 0.0, 4.5]
    }
    response = client.post("/unity/action-result", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "acknowledged"
    assert data["action_id"] == "test_act_123"
    assert data["success"] is True

def test_telemetry_latest_endpoint():
    """Verify GET /telemetry/latest returns correlated CV, CG, and LLM telemetry."""
    # Run an agent query to trigger telemetry recording
    q_resp = client.post("/agent/query", json={
        "query": "Where is the table?",
        "player_position": [0.0, -15.0],
        "active_room": "Reception / Central Hub"
    })
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    corr_id = q_data.get("correlation_id")
    assert corr_id is not None

    t_resp = client.get("/telemetry/latest")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert t_data["correlation_id"] == corr_id
    assert "cv" in t_data
    assert "cg" in t_data
    assert "llm" in t_data
    assert t_data["cv"]["model_name"].startswith("SpatialDetector")
    assert t_data["cg"]["active_room"] == "Reception / Central Hub"
    assert t_data["llm"]["model_name"].startswith("DecoderTransformer")

def test_telemetry_by_id_endpoint():
    """Verify GET /telemetry/{correlation_id} retrieves matching record and 404s on unknown."""
    t_resp = client.get("/telemetry/latest")
    assert t_resp.status_code == 200
    corr_id = t_resp.json()["correlation_id"]

    id_resp = client.get(f"/telemetry/{corr_id}")
    assert id_resp.status_code == 200
    assert id_resp.json()["correlation_id"] == corr_id

    missing_resp = client.get("/telemetry/corr-nonexistent-999")
    assert missing_resp.status_code == 404

def test_game_reset_endpoint():
    """Verify POST /game/reset reconfigures session seed and resets progression."""
    reset_resp = client.post("/game/reset", json={"seed": 101})
    assert reset_resp.status_code == 200
    data = reset_resp.json()
    assert data["seed"] == 101
    assert data["score"] == 0
    assert data["active_mission"]["code"] == "A"
    assert len(data["inventory"]) == 0

def test_inventory_and_hint_query_tools():
    """Verify get_inventory and request_hint tools execute cleanly via agent."""
    inv_resp = client.post("/agent/query", json={"query": "Show me my inventory items"})
    assert inv_resp.status_code == 200
    inv_data = inv_resp.json()
    assert any(tc["tool"] == "get_inventory" for tc in inv_data["tool_calls"])
    assert "inventory" in inv_data["final_response"].lower()

    hint_resp = client.post("/agent/query", json={"query": "Give me a hint for this mission"})
    assert hint_resp.status_code == 200
    hint_data = hint_resp.json()
    assert any(tc["tool"] == "request_hint" for tc in hint_data["tool_calls"])
    assert "hint" in hint_data["final_response"].lower()

def test_llm_mode_endpoints():
    """Verify GET /llm/status and POST /llm/mode toggling."""
    st_resp = client.get("/llm/status")
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert "active_mode" in st_data
    assert "provider_name" in st_data

    # Toggle to from_scratch (guaranteed valid without key)
    mode_resp = client.post("/llm/mode", json={"mode": "from_scratch"})
    assert mode_resp.status_code == 200
    assert mode_resp.json()["mode"] == "from_scratch"

