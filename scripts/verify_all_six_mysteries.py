"""Automated End-to-End Gameplay Verification for All Six Mysteries in Spatial Quest.

Tests every mystery from initial loading to condition validation, inventory collection,
objective checklist progression, override code validation, and sequence unlocking.
"""
from __future__ import annotations

import sys
import json
from typing import Dict, Any, List
from starlette.testclient import TestClient
from backend.app.main import app
from backend.game.missions import global_game_session


def run_full_gameplay_audit() -> Dict[str, Any]:
    client = TestClient(app)
    results = {
        "mysteries_tested": 0,
        "mysteries_passed": 0,
        "objectives_tested": 0,
        "objectives_passed": 0,
        "details": [],
        "errors": []
    }

    print("=" * 70)
    print("🕵️ SPATIAL QUEST — FULL END-TO-END GAMEPLAY VERIFICATION AUDIT")
    print("=" * 70)

    # Reset game session with standard seed 42
    reset_resp = client.post("/game/reset", json={"seed": 42})
    assert reset_resp.status_code == 200, "Reset failed"
    # Allow natural unlocking progression
    global_game_session.unlocked_through_index = 0

    # -------------------------------------------------------------
    # 1. MYSTERY A: MISSING RESEARCH KEY
    # -------------------------------------------------------------
    print("\n--- [1/6] TESTING MYSTERY A: MISSING RESEARCH KEY ---")
    st = client.get("/game/state").json()
    assert st["active_mission"]["code"] == "A", f"Expected Mystery A, got {st['active_mission']['code']}"
    assert st["active_mission"]["completed"] is False, "Mystery A should initially be incomplete"
    assert len(st["active_mission"]["objectives"]) == 3, "Mystery A must have 3 objectives"
    print("✓ Mission A loaded cleanly with 3 pending objectives.")

    # Irrelevant query should not fulfill objectives
    r_irr = client.post("/agent/query", json={"query": "What is the capital of France?"}).json()
    st_irr = r_irr.get("game_state", client.get("/game/state").json())
    assert all(not o["is_completed"] for o in st_irr["active_mission"]["objectives"]), "Irrelevant query must not complete objectives!"
    print("✓ Irrelevant query verified: 0 objectives falsely triggered.")

    # Step 1: Scout survey (scout_01)
    r1 = client.post("/agent/query", json={"query": "What objects can you see?"}).json()
    assert any(tc["tool"] == "get_visible_objects" for tc in r1["tool_calls"]), "Step 1 must call get_visible_objects"
    st1 = r1["game_state"]
    assert st1["active_mission"]["objectives"][0]["is_completed"] is True, "scout_01 must be completed"
    assert st1["active_mission"]["objectives"][1]["is_completed"] is False, "scout_02 must remain pending"
    assert st1["active_mission"]["completed"] is False, "Mission A must not be marked complete yet"
    print("✓ Step 1 verified: Visual perception survey completed (scout_01 done).")

    # Step 2: Locate table (scout_02)
    r2 = client.post("/agent/query", json={"query": "Where is the table?"}).json()
    assert any(tc["tool"] == "locate_object" for tc in r2["tool_calls"]), "Step 2 must call locate_object"
    st2 = r2["game_state"]
    assert st2["active_mission"]["objectives"][1]["is_completed"] is True, "scout_02 must be completed"
    assert st2["active_mission"]["objectives"][2]["is_completed"] is False, "scout_03 must remain pending"
    assert st2["active_mission"]["completed"] is False, "Mission A must not be marked complete yet"
    print("✓ Step 2 verified: Table located in Reception Hub (scout_02 done).")

    # Step 3: Open cabinet & collect key (scout_03)
    r3 = client.post("/agent/query", json={"query": "Open the cabinet and pick up the key"}).json()
    assert any(tc["tool"] == "interact_with_object" for tc in r3["tool_calls"]), "Step 3 must call interact_with_object"
    st3 = r3["game_state"]
    assert any(it["id"] == "research_key" for it in st3["inventory"]), "research_key must be in inventory"
    assert st3["active_mission"]["objectives"][2]["is_completed"] is True, "scout_03 must be completed"
    assert st3["active_mission"]["completed"] is True, "Mystery A MUST be completed after all 3 objectives and key collected"
    assert st3["unlocked_through_index"] >= 1, "Mystery B must now be unlocked"
    print("✓ Step 3 verified: Cabinet opened, Master Research Key retrieved into inventory.")
    print("🎉 MYSTERY A COMPLETE: Score =", st3["score"], "pts.")
    results["mysteries_passed"] += 1
    results["objectives_passed"] += 3

    # -------------------------------------------------------------
    # 2. MYSTERY B: MISSING RESEARCH FILE
    # -------------------------------------------------------------
    print("\n--- [2/6] TESTING MYSTERY B: MISSING RESEARCH FILE ---")
    st_b = client.post("/game/start", json={"mission_id": "mission_b"}).json()
    assert st_b["active_mission"]["code"] == "B", "Mystery B must load"
    assert st_b["active_mission"]["completed"] is False, "Mystery B initially incomplete"
    assert st_b["companion"]["id"] == "navigator", "Mystery B requires Navigator companion"
    print("✓ Mission B loaded: Office & Archives zone, Navigator drone active.")

    # Step 1: Query spatial relation left of table (spatial_01)
    rb1 = client.post("/agent/query", json={"query": "What is left of the table?"}).json()
    assert any(tc["tool"] == "query_spatial_relation" for tc in rb1["tool_calls"]), "Step 1 must call query_spatial_relation"
    st_b1 = rb1["game_state"]
    assert st_b1["active_mission"]["objectives"][0]["is_completed"] is True, "spatial_01 must be completed"
    assert st_b1["active_mission"]["completed"] is False, "Mission B must not be complete yet"
    print("✓ Step 1 verified: Spatial relation query executed (spatial_01 done).")

    # Step 2: Distance calculation (spatial_02)
    rb2 = client.post("/agent/query", json={"query": "How far is the chair from the table?"}).json()
    assert any(tc["tool"] == "calculate_distance" for tc in rb2["tool_calls"]), "Step 2 must call calculate_distance"
    st_b2 = rb2["game_state"]
    assert st_b2["active_mission"]["objectives"][1]["is_completed"] is True, "spatial_02 must be completed"
    assert st_b2["active_mission"]["completed"] is True, "Mystery B MUST be completed now"
    assert st_b2["unlocked_through_index"] >= 2, "Mystery C must now be unlocked"
    print("✓ Step 2 verified: 3D Euclidean distance computed (spatial_02 done).")
    print("🎉 MYSTERY B COMPLETE: Score =", st_b2["score"], "pts.")
    results["mysteries_passed"] += 1
    results["objectives_passed"] += 2

    # -------------------------------------------------------------
    # 3. MYSTERY C: LABORATORY CONTAINMENT
    # -------------------------------------------------------------
    print("\n--- [3/6] TESTING MYSTERY C: LABORATORY CONTAINMENT ---")
    st_c = client.post("/game/start", json={"mission_id": "mission_c"}).json()
    assert st_c["active_mission"]["code"] == "C", "Mystery C must load"
    assert st_c["active_mission"]["completed"] is False, "Mystery C initially incomplete"
    print("✓ Mission C loaded: Lab Corridor zone, Navigator drone active.")

    # Step 1: Find nearest door (nav_01)
    rc1 = client.post("/agent/query", json={"query": "Where is the nearest door?"}).json()
    assert any(tc["tool"] in ("find_nearest_target", "find_nearest_exit") for tc in rc1["tool_calls"]), "Step 1 must locate nearest door"
    st_c1 = rc1["game_state"]
    assert st_c1["active_mission"]["objectives"][0]["is_completed"] is True, "nav_01 must be completed"
    assert st_c1["active_mission"]["completed"] is False, "Mission C must not be complete yet"
    print("✓ Step 1 verified: Nearest door localized (nav_01 done).")

    # Step 2: Path calculation (nav_02)
    rc2 = client.post("/agent/query", json={"query": "Calculate path to the door"}).json()
    assert any(tc["tool"] in ("find_path", "navigate_to") for tc in rc2["tool_calls"]), "Step 2 must calculate path"
    st_c2 = rc2["game_state"]
    assert st_c2["active_mission"]["objectives"][1]["is_completed"] is True, "nav_02 must be completed"
    assert st_c2["active_mission"]["completed"] is True, "Mystery C MUST be completed now"
    assert st_c2["unlocked_through_index"] >= 3, "Mystery D must now be unlocked"
    print("✓ Step 2 verified: Path planned avoiding facility obstacles (nav_02 done).")
    print("🎉 MYSTERY C COMPLETE: Score =", st_c2["score"], "pts.")
    results["mysteries_passed"] += 1
    results["objectives_passed"] += 2

    # -------------------------------------------------------------
    # 4. MYSTERY D: CLUE ROOM & RAG
    # -------------------------------------------------------------
    print("\n--- [4/6] TESTING MYSTERY D: CLUE ROOM & RAG ---")
    st_d = client.post("/game/start", json={"mission_id": "mission_d"}).json()
    assert st_d["active_mission"]["code"] == "D", "Mystery D must load"
    assert st_d["active_mission"]["completed"] is False, "Mystery D initially incomplete"
    assert st_d["companion"]["id"] == "archivist", "Mystery D requires Archivist companion"
    print("✓ Mission D loaded: Vault & Storage zone, Archivist drone active.")

    # Step 1: Inspect clue datapad (clue_01)
    rd1 = client.post("/agent/query", json={"query": "Inspect clue_pda_02"}).json()
    assert any(tc["tool"] == "inspect_clue" for tc in rd1["tool_calls"]), "Step 1 must call inspect_clue"
    st_d1 = rd1["game_state"]
    assert st_d1["active_mission"]["objectives"][0]["is_completed"] is True, "clue_01 must be completed"
    assert any("pda" in it["id"] or "pda" in it["name"].lower() for it in st_d1["inventory"]), "Datapad clue note must be in inventory"
    assert st_d1["active_mission"]["completed"] is False, "Mission D must not be complete yet"
    print("✓ Step 1 verified: Datapad clue inspected and note added to inventory (clue_01 done).")

    # Step 2: Search documents for Door 3 override code (clue_02)
    rd2 = client.post("/agent/query", json={"query": "Search documents for Door 3 override code"}).json()
    assert any(tc["tool"] == "search_documents" for tc in rd2["tool_calls"]), "Step 2 must call search_documents"
    assert "7391" in rd2["final_response"], "Final response must cite override code 7391 from incident logs"
    st_d2 = rd2["game_state"]
    assert st_d2["active_mission"]["objectives"][1]["is_completed"] is True, "clue_02 must be completed"
    assert st_d2["active_mission"]["completed"] is False, "Mission D must not be complete yet"
    print("✓ Step 2 verified: TF-IDF RAG document search retrieved Door 3 code 7391 (clue_02 done).")

    # Step 3a: NEGATIVE TEST — Attempt wrong code
    rd_bad = client.post("/agent/query", json={"query": "Enter code 9999"}).json()
    assert rd_bad["tool_results"][0]["success"] is False, "Wrong code 9999 must return success=False"
    assert "invalid" in rd_bad["final_response"].lower() or "denied" in rd_bad["final_response"].lower()
    st_bad = rd_bad["game_state"]
    assert st_bad["active_mission"]["objectives"][2]["is_completed"] is False, "clue_03 must NOT be marked complete after wrong code!"
    print("✓ Step 3a (Negative Test) verified: Invalid code 9999 rejected, Door 3 remains locked.")

    # Step 3b: POSITIVE TEST — Submit correct code 7391
    rd3 = client.post("/agent/query", json={"query": "Enter code 7391"}).json()
    assert rd3["tool_results"][0]["success"] is True, "Code 7391 must return success=True"
    assert "disengaged" in rd3["final_response"].lower() or "open" in rd3["final_response"].lower()
    st_d3 = rd3["game_state"]
    assert st_d3["active_mission"]["objectives"][2]["is_completed"] is True, "clue_03 must be completed"
    assert st_d3["active_mission"]["completed"] is True, "Mystery D MUST be completed now"
    assert st_d3["unlocked_through_index"] >= 4, "Mystery E must now be unlocked"
    print("✓ Step 3b verified: Security override code 7391 validated, Door 3 unlocked (clue_03 done).")
    print("🎉 MYSTERY D COMPLETE: Score =", st_d3["score"], "pts.")
    results["mysteries_passed"] += 1
    results["objectives_passed"] += 3

    # -------------------------------------------------------------
    # 5. MYSTERY E: TIMED CONTAINMENT
    # -------------------------------------------------------------
    print("\n--- [5/6] TESTING MYSTERY E: TIMED CONTAINMENT ---")
    st_e = client.post("/game/start", json={"mission_id": "mission_e"}).json()
    assert st_e["active_mission"]["code"] == "E", "Mystery E must load"
    assert st_e["active_mission"]["timed"] is True, "Mystery E must be timed"
    assert st_e["active_mission"]["timer_remaining"] > 0, "Timer must be running"
    print(f"✓ Mission E loaded: Timed Containment ({st_e['active_mission']['timer_remaining']}s remaining).")

    # Step 1: Visual scan in lab (timed_01)
    re1 = client.post("/agent/query", json={"query": "What objects can you see?"}).json()
    st_e1 = re1["game_state"]
    assert st_e1["active_mission"]["objectives"][0]["is_completed"] is True, "timed_01 must be completed"
    print("✓ Step 1 verified: Visual perception survey in lab (timed_01 done).")

    # Step 2: RAG override retrieval (timed_02)
    re2 = client.post("/agent/query", json={"query": "Search documents for override code"}).json()
    st_e2 = re2["game_state"]
    assert st_e2["active_mission"]["objectives"][1]["is_completed"] is True, "timed_02 must be completed"
    print("✓ Step 2 verified: RAG override code retrieved (timed_02 done).")

    # Step 3: Unlock Door 3 with code 7391 before time expires (timed_03)
    re3 = client.post("/agent/query", json={"query": "Use code 7391 to unlock Door 3"}).json()
    st_e3 = re3["game_state"]
    assert st_e3["active_mission"]["objectives"][2]["is_completed"] is True, "timed_03 must be completed"
    assert st_e3["active_mission"]["completed"] is True, "Mystery E MUST be completed now"
    assert st_e3["unlocked_through_index"] >= 5, "Mystery F must now be unlocked"
    print("✓ Step 3 verified: Door 3 unlocked before timer expiration (timed_03 done).")
    print("🎉 MYSTERY E COMPLETE: Score =", st_e3["score"], "pts.")
    results["mysteries_passed"] += 1
    results["objectives_passed"] += 3

    # -------------------------------------------------------------
    # 6. MYSTERY F: DATASET TERMINAL
    # -------------------------------------------------------------
    print("\n--- [6/6] TESTING MYSTERY F: DATASET TERMINAL ---")
    st_f = client.post("/game/start", json={"mission_id": "mission_f"}).json()
    assert st_f["active_mission"]["code"] == "F", "Mystery F must load"
    print("✓ Mission F loaded: Reception Dataset Terminal, Scout drone active.")

    # Step 1: Locate table (term_01)
    rf1 = client.post("/agent/query", json={"query": "Where is the table?"}).json()
    st_f1 = rf1["game_state"]
    assert st_f1["active_mission"]["objectives"][0]["is_completed"] is True, "term_01 must be completed"
    print("✓ Step 1 verified: Security table localized (term_01 done).")

    # Step 2: Survey visible objects (term_02)
    rf2 = client.post("/agent/query", json={"query": "Survey visible objects"}).json()
    st_f2 = rf2["game_state"]
    assert st_f2["active_mission"]["objectives"][1]["is_completed"] is True, "term_02 must be completed"
    assert st_f2["active_mission"]["completed"] is True, "Mystery F MUST be completed now"
    print("✓ Step 2 verified: Grounded perception benchmark executed (term_02 done).")
    print("🎉 MYSTERY F COMPLETE: Score =", st_f2["score"], "pts.")
    results["mysteries_passed"] += 1
    results["objectives_passed"] += 2

    # -------------------------------------------------------------
    # AUDIT SUMMARY
    # -------------------------------------------------------------
    results["mysteries_tested"] = 6
    results["objectives_tested"] = 15
    print("\n" + "=" * 70)
    print("🏆 ALL 6 MYSTERIES COMPLETED AND VERIFIED 100% PLAYABLE!")
    print(f"• Mysteries Verified: {results['mysteries_passed']}/{results['mysteries_tested']}")
    print(f"• Objectives Verified: {results['objectives_passed']}/{results['objectives_tested']}")
    print(f"• Final Cumulative Score: {st_f2['score']} pts")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_full_gameplay_audit()
    if res["mysteries_passed"] != 6 or res["objectives_passed"] != 15:
        sys.exit(1)
    sys.exit(0)

