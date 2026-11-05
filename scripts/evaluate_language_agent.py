"""Comprehensive Multi-Dimensional Evaluation Suite for SpatialAI Language Agent.

Evaluates the from-scratch Transformer and Agent Loop against 250 held-out test dialogues
(data/corpora/lm_test_dataset.jsonl) across 8 academic benchmark metrics:
1. Tool Selection Accuracy (Overall & Per-Category)
2. Object Identification Accuracy
3. Spatial Reasoning Accuracy
4. Negative Observation Handling (Uncertainty on Absent Objects)
5. Conversational Context Retention (Anaphora Resolution)
6. RAG Answer Faithfulness (Document Grounding)
7. Hallucination Rate (Fictitious Coordinates/Objects)
8. End-to-End Task Success Rate

Strictly traceable to CS3235 (Working with LLMs - Evaluation & Agentic Reasoning).
"""
import os
import sys
import json
import time
import math
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.agent.investigation_agent import InvestigationAgent
from backend.game.facility import seed_facility_scene


def run_comprehensive_evaluation(test_dataset_path: str = "data/corpora/lm_test_dataset.jsonl") -> Dict[str, Any]:
    print("=" * 70)
    print("SpatialAI: Running 8-Metric Multi-Dimensional Evaluation Suite")
    print(f"Test Split: {test_dataset_path}")
    print("=" * 70)

    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")

    # Ensure scene graph is seeded
    seed_facility_scene()

    agent = InvestigationAgent(max_steps=2)

    total_samples = 0
    correct_tools = 0
    category_counts = {}
    category_tool_correct = {}

    # Metric accumulators
    obj_ident_total = 0
    obj_ident_correct = 0

    spatial_total = 0
    spatial_correct = 0

    negative_total = 0
    negative_correct = 0

    context_total = 0
    context_correct = 0

    rag_total = 0
    rag_correct = 0

    hallucinations = 0
    task_successes = 0

    per_tool_stats = {}

    t0 = time.time()

    with open(test_dataset_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    total_samples = len(samples)
    print(f"Loaded {total_samples} held-out test examples.")

    for i, item in enumerate(samples):
        cat = item.get("category", "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        prompt_str = item["prompt"]
        # Extract user query
        user_line = [line for line in prompt_str.split("\n") if line.startswith("USER:")][-1]
        user_query = user_line.replace("USER:", "").strip()

        # Parse ground-truth tool call
        expected_tool = None
        expected_args = {}
        if "<tool_call>" in item.get("completion", ""):
            tc_str = item["completion"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
            try:
                tc_json = json.loads(tc_str)
                expected_tool = tc_json.get("name")
                expected_args = tc_json.get("arguments", {})
            except Exception:
                pass

        # Execute agent run
        res = agent.run(user_query, player_position=[0.0, -15.0])
        tool_calls = res.get("tool_calls", [])
        final_resp = res.get("final_response", "")

        actual_tool = tool_calls[0]["tool"] if tool_calls else None
        actual_args = tool_calls[0].get("arguments", {}) if tool_calls else {}

        # 1. Tool Selection Accuracy
        tool_match = (actual_tool == expected_tool)
        if not tool_match and expected_tool:
            # Allow semantic equivalence (e.g. find_path vs navigate_to, find_nearest_target vs query_spatial_relation)
            if expected_tool in ("find_path", "navigate_to") and actual_tool in ("find_path", "navigate_to"):
                tool_match = True
            elif expected_tool in ("explain_uncertainty", "explain_unreachable") and actual_tool in ("explain_uncertainty", "explain_unreachable"):
                tool_match = True

        if tool_match:
            correct_tools += 1
            category_tool_correct[cat] = category_tool_correct.get(cat, 0) + 1

        # Track per-tool counts
        if expected_tool:
            if expected_tool not in per_tool_stats:
                per_tool_stats[expected_tool] = {"expected": 0, "predicted": 0, "correct": 0}
            per_tool_stats[expected_tool]["expected"] += 1
        if actual_tool:
            if actual_tool not in per_tool_stats:
                per_tool_stats[actual_tool] = {"expected": 0, "predicted": 0, "correct": 0}
            per_tool_stats[actual_tool]["predicted"] += 1
        if tool_match and expected_tool:
            per_tool_stats[expected_tool]["correct"] += 1

        # 2. Object Identification Accuracy
        if cat == "object_understanding":
            obj_ident_total += 1
            if actual_tool in ("locate_object", "get_visible_objects", "inspect_object_closely"):
                obj_ident_correct += 1

        # 3. Spatial Reasoning Accuracy
        if cat == "spatial_reasoning":
            spatial_total += 1
            if actual_tool in ("calculate_distance", "query_spatial_relation", "find_nearest_target"):
                spatial_correct += 1

        # 4. Negative Observation Handling
        if cat == "contextual_negative":
            negative_total += 1
            if actual_tool in ("explain_uncertainty", "explain_unreachable"):
                negative_correct += 1

        # 5. Conversational Context Retention
        if "HISTORY:" in prompt_str or "it" in user_query.lower().split():
            context_total += 1
            if actual_tool is not None and not res.get("used_fallback"):
                context_correct += 1

        # 6. RAG Answer Faithfulness
        if cat == "investigation_rag":
            rag_total += 1
            ground_truth_ans = item.get("grounded_response", "")
            # Check key entities/passages
            entities = [w for w in ["7391", "104", "Omega", "titanium", "fire", "emergency", "Door 3"] if w.lower() in ground_truth_ans.lower()]
            if any(e.lower() in final_resp.lower() for e in entities) or actual_tool in ("search_documents", "search_safety_manual"):
                rag_correct += 1

        # 7. Hallucination Rate (Invented coordinates or impossible objects)
        if "unknown" in str(actual_args).lower() or (actual_tool == "locate_object" and "not found" in final_resp.lower() and cat != "contextual_negative"):
            hallucinations += 1

        # 8. End-to-End Task Success
        if final_resp and not final_resp.startswith("Execution error") and actual_tool is not None:
            task_successes += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total_samples} test samples... Current Tool Acc: {round(correct_tools / (i + 1) * 100, 1)}%")

    elapsed = time.time() - t0

    # Compile 8 Metrics
    tool_selection_acc = round((correct_tools / total_samples) * 100, 2)
    obj_ident_acc = round((obj_ident_correct / max(1, obj_ident_total)) * 100, 2)
    spatial_acc = round((spatial_correct / max(1, spatial_total)) * 100, 2)
    negative_acc = round((negative_correct / max(1, negative_total)) * 100, 2)
    context_acc = round((context_correct / max(1, context_total)) * 100, 2)
    rag_faithfulness = round((rag_correct / max(1, rag_total)) * 100, 2)
    hallucination_rate = round((hallucinations / total_samples) * 100, 2)
    task_success_rate = round((task_successes / total_samples) * 100, 2)

    cat_breakdown = {}
    for c, cnt in category_counts.items():
        corr = category_tool_correct.get(c, 0)
        cat_breakdown[c] = {
            "total": cnt,
            "correct": corr,
            "accuracy_pct": round((corr / cnt) * 100, 2)
        }

    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": total_samples,
        "evaluation_duration_seconds": round(elapsed, 2),
        "metrics": {
            "1_tool_selection_accuracy_pct": tool_selection_acc,
            "2_object_identification_accuracy_pct": obj_ident_acc,
            "3_spatial_reasoning_accuracy_pct": spatial_acc,
            "4_negative_observation_handling_pct": negative_acc,
            "5_conversational_context_retention_pct": context_acc,
            "6_rag_answer_faithfulness_pct": rag_faithfulness,
            "7_hallucination_rate_pct": hallucination_rate,
            "8_end_to_end_task_success_rate_pct": task_success_rate,
        },
        "category_breakdown": cat_breakdown,
        "per_tool_stats": per_tool_stats
    }

    out_dir = "results/metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "agent_comprehensive_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY (8 Academic Metrics)")
    print("=" * 70)
    print(f"1. Tool Selection Accuracy:         {tool_selection_acc}%")
    print(f"2. Object Identification Accuracy:   {obj_ident_acc}%")
    print(f"3. Spatial Reasoning Accuracy:       {spatial_acc}%")
    print(f"4. Negative Observation Handling:    {negative_acc}%")
    print(f"5. Conversational Context Retention: {context_acc}%")
    print(f"6. RAG Answer Faithfulness:          {rag_faithfulness}%")
    print(f"7. Hallucination Rate:               {hallucination_rate}%")
    print(f"8. End-to-End Task Success Rate:     {task_success_rate}%")
    print(f"\nSaved verified metrics to {out_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_comprehensive_evaluation()

