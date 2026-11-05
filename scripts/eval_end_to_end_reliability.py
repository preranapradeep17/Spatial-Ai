"""End-to-end LLM and Agent Reliability Evaluation Script for Spatial Quest.

Runs live evaluation of:
1. Natural language query suite (13 core queries + variations, typos, ambiguous)
2. Tool selection and argument validation audit
3. Multi-turn anaphoric conversation
4. Negative scenarios (missing objects, empty rooms, invalid targets)
5. Pure custom Transformer evaluation (raw model generation isolated from guardrails)

Produces empirical data for the evaluation report.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from backend.agent.investigation_agent import InvestigationAgent
from backend.game.facility import seed_facility_scene
from backend.game.missions import GameSession, global_game_session
from backend.tools.registry import global_tool_registry

def run_evaluation():
    seed_facility_scene(source="facility_layout")
    global_game_session.return_to_hub()
    global_game_session.start_mission("mission_a")

    agent = InvestigationAgent(max_steps=2)

    # -------------------------------------------------------------
    # 1. Core and Variational Natural Language Queries
    # -------------------------------------------------------------
    queries = [
        # Core 13
        ("What objects can you see?", "get_visible_objects", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Find the nearest table.", "find_nearest_target", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("What is closest to me?", "find_nearest_target", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("What is near the desk?", "query_spatial_relation", "navigator", [6.0, 0.0, -5.0], "Research Office"),
        ("Which object is next to the bookshelf?", "query_spatial_relation", "navigator", [-5.0, 0.0, -5.0], "Library / Archives"),
        ("Take me to the library.", "navigate_to", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Where is the research key?", "locate_object", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("What should I do next?", "get_mission_objectives", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Have I completed my mission?", "get_mission_objectives", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Show my inventory.", "get_inventory", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Read the note.", "inspect_clue", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Give me a hint.", "request_hint", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Why can't you detect the object?", "explain_uncertainty", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),

        # Variations
        ("List all visible objects in this sector.", "get_visible_objects", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Can you guide me towards the office?", "navigate_to", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("How far is the chair from the table?", "calculate_distance", "navigator", [6.0, 0.0, -5.0], "Research Office"),
        ("What items are currently in my bag?", "get_inventory", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Tell me what my goals are.", "get_mission_objectives", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Explain why that destination is blocked.", "explain_unreachable", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),

        # Spelling mistakes & informal phrasing
        ("wat objcts can u see?", "get_visible_objects", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("fnd the nerest tabl", "find_nearest_target", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("tke me to libary", "navigate_to", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("giv me a hnt", "request_hint", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("shw my invntory", "get_inventory", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),

        # Ambiguous / General Queries
        ("What is that over there?", "explain_uncertainty", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("How do I play?", "get_game_guide", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Tell me about this room.", "get_current_location", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Who are you?", "get_game_guide", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub")
    ]

    print("=" * 75)
    print("PHASE A: EVALUATING FULL AGENT PIPELINE (NATURAL LANGUAGE & ROUTING)")
    print("=" * 75)

    agent_results = []
    for q, exp_tool, companion, pos, room in queries:
        t0 = time.time()
        res = agent.run(
            q,
            agent_id=companion,
            player_position=pos,
            active_room=room
        )
        dur = round(time.time() - t0, 3)

        tool_calls = res.get("tool_calls", [])
        act_tool = tool_calls[0]["tool"] if tool_calls else "none"
        act_args = tool_calls[0]["arguments"] if tool_calls else {}
        is_correct = (act_tool == exp_tool)
        error_msg = "" if is_correct else f"Expected {exp_tool}, got {act_tool}"

        # Argument check
        arg_error = ""
        if act_tool == "navigate_to" and ("destination" not in act_args or not act_args.get("destination")):
            arg_error = "Missing required destination argument"
        elif act_tool == "locate_object" and ("object_name" not in act_args or not act_args.get("object_name")):
            arg_error = "Missing required object_name argument"
        elif act_tool == "calculate_distance" and ("object_a" not in act_args or "object_b" not in act_args):
            arg_error = "Missing object_a or object_b"

        agent_results.append({
            "query": q,
            "expected_tool": exp_tool,
            "actual_tool": act_tool,
            "actual_args": act_args,
            "validation_status": res.get("validation_status", "UNKNOWN"),
            "correct": is_correct,
            "arg_error": arg_error,
            "error": error_msg,
            "final_response": res.get("final_response", ""),
            "duration": dur
        })

    # -------------------------------------------------------------
    # 2. Multi-turn Conversational Context Test
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE B: MULTI-TURN CONVERSATION EVALUATION")
    print("=" * 75)

    multi_turn_convo = [
        ("What objects can you see?", "get_visible_objects", "scout"),
        ("Which one is closest to me?", "find_nearest_target", "navigator"),
        ("Take me there.", "navigate_to", "navigator"),
        ("How far is it from here?", "find_nearest_target", "navigator")
    ]
    convo_agent = InvestigationAgent(max_steps=2)
    current_pos = [0.0, 0.0, -15.0]
    current_rm = "Reception / Central Hub"
    multi_turn_results = []

    for turn_idx, (turn_q, exp_t, comp) in enumerate(multi_turn_convo):
        res = convo_agent.run(turn_q, agent_id=comp, player_position=current_pos, active_room=current_rm)
        t_calls = res.get("tool_calls", [])
        act_t = t_calls[0]["tool"] if t_calls else "none"
        act_a = t_calls[0]["arguments"] if t_calls else {}

        # If navigate_to was executed, update player position to simulated target position
        if act_t == "navigate_to" and res.get("tool_results"):
            target_pos = res["tool_results"][0].get("result", {}).get("target_position")
            if target_pos:
                current_pos = target_pos

        multi_turn_results.append({
            "turn": turn_idx + 1,
            "query": turn_q,
            "expected_tool": exp_t,
            "actual_tool": act_t,
            "actual_args": act_a,
            "final_response": res.get("final_response", "")[:120] + "...",
            "last_mentioned": convo_agent.last_mentioned_object,
            "player_stance": current_pos
        })

    # -------------------------------------------------------------
    # 3. Negative Scenarios Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE C: NEGATIVE SCENARIOS EVALUATION")
    print("=" * 75)

    neg_cases = [
        ("Where is the laptop?", "scout", "explain_uncertainty", "Non-existent object in facility"),
        ("Where is the quantum microscope?", "scout", "explain_uncertainty", "Non-existent object in facility"),
        ("What objects can you see?", "scout", "get_visible_objects", "Empty Chamber test", "Empty Quarantine Chamber"),
        ("Take me to the moon base.", "navigator", "navigate_to", "Invalid destination"),
        ("Show my inventory.", "archivist", "get_inventory", "Empty inventory check"),
        ("Search documents for secret alien blueprints.", "archivist", "search_documents", "Unavailable documentation query")
    ]
    negative_results = []
    for item in neg_cases:
        q = item[0]
        comp = item[1]
        exp_t = item[2]
        desc = item[3]
        test_room = item[4] if len(item) > 4 else "Reception / Central Hub"

        res = agent.run(q, agent_id=comp, player_position=[0.0, 0.0, -15.0], active_room=test_room)
        t_calls = res.get("tool_calls", [])
        act_t = t_calls[0]["tool"] if t_calls else "none"
        resp_text = res.get("final_response", "")

        negative_results.append({
            "query": q,
            "scenario": desc,
            "expected_tool": exp_t,
            "actual_tool": act_t,
            "response_sample": resp_text[:140] + "...",
            "zero_fabrication_maintained": "Zero-Fabrication" in resp_text or "not detected" in resp_text.lower() or "0 items" in resp_text.lower() or "no matching" in resp_text.lower() or "not found" in resp_text.lower() or "unreachable" in resp_text.lower()
        })

    # -------------------------------------------------------------
    # 4. Pure Custom Transformer Evaluation (Isolated from Guardrails)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE D: EVALUATING PURE CUSTOM TRANSFORMER (NO GUARDRAILS)")
    print("=" * 75)

    test_split_path = "data/corpora/lm_test_dataset.jsonl"
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()][:50]  # Sample 50 for deep audit

    raw_model_results = []
    valid_json_count = 0
    correct_tool_count = 0
    valid_args_count = 0

    for sample in test_samples:
        prompt_text = sample["prompt"]
        expected_raw = sample["completion"]
        exp_tool = None
        if "<tool_call>" in expected_raw:
            try:
                exp_json = json.loads(expected_raw.split("<tool_call>")[1].split("</tool_call>")[0].strip())
                exp_tool = exp_json.get("name")
            except Exception:
                pass

        # Raw model generation
        input_ids = agent.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=agent.device)
        with torch.no_grad():
            gen_ids = agent.model.generate(
                input_tensor,
                max_new_tokens=45,
                temperature=0.1,
                stop_token_ids=[agent.tokenizer.token_to_id.get("</tool_call>", agent.tokenizer.eos_token_id)]
            )
        new_tokens = gen_ids[0][len(input_ids):].tolist()
        gen_text = agent.tokenizer.decode(new_tokens, skip_special_tokens=False)

        parsed = agent.parse_tool_call(gen_text)
        is_valid_json = (parsed is not None)
        pred_tool = parsed.get("name") if parsed else None
        pred_args = parsed.get("arguments") if parsed else {}

        if is_valid_json:
            valid_json_count += 1
            if pred_tool == exp_tool:
                correct_tool_count += 1
            if isinstance(pred_args, dict) and len(pred_args) > 0:
                valid_args_count += 1

        raw_model_results.append({
            "user_prompt_snippet": prompt_text.split("USER:")[-1].split("ASSISTANT:")[0].strip(),
            "expected_tool": exp_tool,
            "raw_generated_text": gen_text.strip(),
            "parsed_tool": pred_tool,
            "parsed_args": pred_args,
            "is_valid_json": is_valid_json,
            "tool_correct": (pred_tool == exp_tool)
        })

    out_payload = {
        "agent_eval": agent_results,
        "multi_turn": multi_turn_results,
        "negative_cases": negative_results,
        "raw_transformer": {
            "total_tested": len(test_samples),
            "valid_json_count": valid_json_count,
            "valid_json_pct": round((valid_json_count / len(test_samples)) * 100, 2),
            "correct_tool_count": correct_tool_count,
            "correct_tool_pct": round((correct_tool_count / len(test_samples)) * 100, 2),
            "valid_args_count": valid_args_count,
            "valid_args_pct": round((valid_args_count / len(test_samples)) * 100, 2),
            "samples": raw_model_results[:15]
        }
    }

    os.makedirs("results/metrics", exist_ok=True)
    out_file = "results/metrics/end_to_end_reliability_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n✓ Saved complete evaluation payload to {out_file}")

if __name__ == "__main__":
    run_evaluation()

