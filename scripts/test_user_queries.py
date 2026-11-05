"""Test agent responses to user's queries."""
import sys
sys.path.insert(0, ".")
from backend.agent.investigation_agent import InvestigationAgent

agent = InvestigationAgent(max_steps=2)

queries = [
    ("Where is the research key?", "scout"),
    ("How do I get the key?", "archivist"),
    ("Open the cabinet", "navigator"),
    ("Enter code 7391", "archivist"),
    ("Use code 7391 to unlock Door 3", "archivist"),
    ("Have I completed my mission?", "archivist"),
    ("How do I complete this mission?", "archivist")
]

for q, mode in queries:
    res = agent.run(q, agent_id=mode, player_position=[0.0, 0.0, -15.0], active_room="Reception / Central Hub")
    tool = res["tool_calls"][0]["tool"] if res["tool_calls"] else "NONE"
    args = res["tool_calls"][0].get("arguments") or res["tool_calls"][0].get("args", {}) if res["tool_calls"] else {}
    resp = res.get("final_response", "")
    print(f"Q: '{q}' [{mode}]")
    print(f"   Tool: {tool} -> {args}")
    print(f"   Resp: {resp[:130]}...")
    print("-" * 60)
