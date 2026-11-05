"""Controlled Prompt Experiment for SpatialAI Custom Transformer.

Compares raw Transformer checkpoint inference across three prompt representations:
- Format A: Original Training Prompt (from scripts/generate_lm_dataset.py)
- Format B: Legacy Runtime Prompt (from backend/agent/investigation_agent.py with long personas)
- Format C: New Canonical Shared Prompt (from backend/agent/prompt_builder.py)

Evaluates on identical game-state and query pairs without retraining.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from backend.agent.investigation_agent import InvestigationAgent
from backend.agent.prompt_builder import build_canonical_prompt

TEST_BENCHMARK = [
    {
        "query": "Where is the table?",
        "expected_tool": "locate_object",
        "mode": "scout",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]\n- chair: [0.0, 0.5, -13.8]",
        "relations": "- chair is near table"
    },
    {
        "query": "What objects can you see?",
        "expected_tool": "get_visible_objects",
        "mode": "scout",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]\n- chair: [0.0, 0.5, -13.8]\n- sofa: [-6.5, 0.3, -16.0]",
        "relations": "- chair is near table"
    },
    {
        "query": "What is closest to me?",
        "expected_tool": "find_nearest_target",
        "mode": "navigator",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]\n- chair: [0.0, 0.5, -13.8]",
        "relations": "- chair is near table"
    },
    {
        "query": "How far is the chair from the table?",
        "expected_tool": "calculate_distance",
        "mode": "navigator",
        "room": "Research Office",
        "pos": [6.0, 0.0, -5.0],
        "visible": "- table: [6.0, 0.8, -5.0]\n- chair: [6.0, 0.5, -4.0]",
        "relations": "- chair is near table"
    },
    {
        "query": "What is to the left of the table?",
        "expected_tool": "query_spatial_relation",
        "mode": "navigator",
        "room": "Investigation Laboratory",
        "pos": [0.0, 0.0, 0.0],
        "visible": "- table: [0.0, 0.8, 0.0]\n- cabinet: [8.5, 1.2, 2.0]",
        "relations": "- table is left_of cabinet"
    },
    {
        "query": "Take me to the library.",
        "expected_tool": "navigate_to",
        "mode": "navigator",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]\n- door: [0.0, 2.0, -10.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Where is the exit?",
        "expected_tool": "find_nearest_exit",
        "mode": "navigator",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]\n- door: [0.0, 2.0, -10.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Disengage the door_3.",
        "expected_tool": "interact_with_object",
        "mode": "archivist",
        "room": "Investigation Laboratory",
        "pos": [0.0, 0.0, 5.0],
        "visible": "- door: [0.0, 2.0, 8.0]\n- cabinet: [8.5, 1.2, 2.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Can you find the laptop?",
        "expected_tool": "explain_uncertainty",
        "mode": "scout",
        "room": "Research Office",
        "pos": [6.0, 0.0, -5.0],
        "visible": "- table: [6.0, 0.8, -5.0]\n- chair: [6.0, 0.5, -4.0]",
        "relations": "- chair is near table"
    },
    {
        "query": "Why is the target unreachable?",
        "expected_tool": "explain_unreachable",
        "mode": "navigator",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "What are my current mission objectives?",
        "expected_tool": "get_mission_objectives",
        "mode": "archivist",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Show me what items I have collected.",
        "expected_tool": "get_inventory",
        "mode": "archivist",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Give me a hint.",
        "expected_tool": "request_hint",
        "mode": "archivist",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Inspect the table closely.",
        "expected_tool": "inspect_object_closely",
        "mode": "scout",
        "room": "Research Office",
        "pos": [6.0, 0.0, -5.0],
        "visible": "- table: [6.0, 0.8, -5.0]\n- chair: [6.0, 0.5, -4.0]",
        "relations": "- chair is near table"
    },
    {
        "query": "What should I do during a fire?",
        "expected_tool": "search_safety_manual",
        "mode": "archivist",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "What is the emergency override code for Door 3?",
        "expected_tool": "search_documents",
        "mode": "archivist",
        "room": "Library / Archives",
        "pos": [-5.0, 0.0, -5.0],
        "visible": "- bookshelf: [-5.0, 1.6, -5.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "Where are we right now?",
        "expected_tool": "get_current_location",
        "mode": "navigator",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "What is near the cabinet?",
        "expected_tool": "query_spatial_relation",
        "mode": "navigator",
        "room": "Investigation Laboratory",
        "pos": [0.0, 0.0, 0.0],
        "visible": "- table: [0.0, 0.8, 0.0]\n- cabinet: [8.5, 1.2, 2.0]",
        "relations": "- table is left_of cabinet"
    },
    {
        "query": "Calculate a path to the laboratory.",
        "expected_tool": "find_path",
        "mode": "navigator",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    },
    {
        "query": "How to play this game?",
        "expected_tool": "get_game_guide",
        "mode": "scout",
        "room": "Reception / Central Hub",
        "pos": [0.0, 0.0, -15.0],
        "visible": "- table: [0.0, 0.8, -15.0]",
        "relations": "- No immediate adjacency"
    }
]

def run_experiment():
    print("=" * 75)
    print("SpatialAI: Running Controlled Prompt Format Experiment")
    print("Testing 20 benchmark queries across 3 prompt representation formats")
    print("=" * 75)

    agent = InvestigationAgent(max_steps=1)
    if not agent.model or not agent.tokenizer:
        raise RuntimeError("Trained Transformer checkpoint or tokenizer not loaded.")

    formats = [
        ("Format A (Original Training)", "original_training"),
        ("Format B (Legacy Runtime with Long Personas)", "legacy_runtime"),
        ("Format C (New Canonical Shared)", "canonical")
    ]

    results_by_format = {}

    for label, fmt in formats:
        print(f"\n--- Testing {label} ---")
        total = len(TEST_BENCHMARK)
        valid_json = 0
        correct_tools = 0
        valid_args = 0
        samples_log = []

        for item in TEST_BENCHMARK:
            prompt = build_canonical_prompt(
                query=item["query"],
                agent_mode=item["mode"],
                room_name=item["room"],
                player_position=item["pos"],
                visible_objects_str=item["visible"],
                spatial_relations_str=item["relations"],
                prompt_format=fmt
            )

            input_ids = agent.tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=agent.device)
            with torch.no_grad():
                gen_ids = agent.model.generate(
                    input_tensor,
                    max_new_tokens=45,
                    temperature=0.0,  # Deterministic greedy evaluation
                    stop_token_ids=[agent.tokenizer.token_to_id.get("</tool_call>", agent.tokenizer.eos_token_id)]
                )
            new_tokens = gen_ids[0][len(input_ids):].tolist()
            gen_text = agent.tokenizer.decode(new_tokens, skip_special_tokens=False)

            parsed = agent.parse_tool_call(gen_text)
            has_valid_json = (parsed is not None)
            pred_tool = parsed.get("name") if parsed else None
            pred_args = parsed.get("arguments") if parsed else {}
            is_tool_correct = (pred_tool == item["expected_tool"])
            has_valid_args = bool(pred_args and isinstance(pred_args, dict))

            if has_valid_json:
                valid_json += 1
            if is_tool_correct:
                correct_tools += 1
            if has_valid_args:
                valid_args += 1

            samples_log.append({
                "query": item["query"],
                "expected_tool": item["expected_tool"],
                "predicted_tool": pred_tool,
                "predicted_args": pred_args,
                "tool_correct": is_tool_correct,
                "valid_json": has_valid_json,
                "raw_output": gen_text.strip()
            })

        json_rate = round((valid_json / total) * 100, 1)
        tool_rate = round((correct_tools / total) * 100, 1)
        args_rate = round((valid_args / total) * 100, 1)

        print(f"  Valid JSON Rate:    {json_rate}% ({valid_json}/{total})")
        print(f"  Correct Tool Rate:  {tool_rate}% ({correct_tools}/{total})")
        print(f"  Valid Args Rate:    {args_rate}% ({valid_args}/{total})")

        results_by_format[fmt] = {
            "label": label,
            "total_tested": total,
            "valid_json_count": valid_json,
            "valid_json_pct": json_rate,
            "correct_tool_count": correct_tools,
            "correct_tool_pct": tool_rate,
            "valid_args_count": valid_args,
            "valid_args_pct": args_rate,
            "samples": samples_log
        }

    out_file = "results/metrics/prompt_experiment_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_by_format, f, indent=2)

    print(f"\n✓ Saved prompt experiment metrics to {out_file}")

if __name__ == "__main__":
    run_experiment()

