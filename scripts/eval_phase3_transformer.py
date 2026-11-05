"""Phase 3 Comprehensive Evaluation Script for Spatial Quest Transformer.

Evaluates the new checkpoint on the untouched held-out test split (corrected_test.jsonl):
1. Cross-entropy loss & perplexity on held-out test data
2. Valid JSON generation rate
3. Correct tool selection rate
4. Valid argument rate (Pydantic schema validation)
5. Exact tool-call match accuracy (tool name + all arguments)
6. Per-tool accuracy breakdown
7. Confusion matrix across all predicted vs expected tools
8. Detailed list of failed examples with expected and predicted outputs
9. Raw Transformer evaluation isolated from guardrails, regex, typo normalization, and fallback
10. Combined agent evaluation
11. Direct comparative evaluation between Legacy Checkpoint vs Phase 3 Checkpoint

Saves complete empirical evaluation metrics to results/metrics/phase3_evaluation_report.json.
"""
import os
import sys
import json
import re
import math
import time
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict

# Workspace path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F

from backend.language_model.tokenizer import SpatialTokenizer
from backend.language_model.transformer import DecoderTransformer
from backend.perception.device import get_optimal_device
from backend.tools.schemas import validate_tool_arguments, TOOL_SCHEMA_MAP
from backend.agent.investigation_agent import InvestigationAgent
from backend.game.facility import seed_facility_scene
from backend.game.missions import global_game_session


def evaluate_raw_model_on_test_split(
    model: DecoderTransformer,
    tokenizer: SpatialTokenizer,
    test_samples: List[Dict[str, Any]],
    device: torch.device,
    sample_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluates raw Transformer output independently of any guardrails, regex, or fallbacks."""
    model.eval()

    samples_to_eval = test_samples if sample_limit is None else test_samples[:sample_limit]
    total_eval = len(samples_to_eval)

    total_loss = 0.0
    loss_batches = 0

    valid_json_count = 0
    correct_tool_count = 0
    valid_args_count = 0
    exact_match_count = 0

    per_tool_stats = defaultdict(lambda: {"expected": 0, "correct": 0, "exact_match": 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    failed_examples = []
    sample_results = []

    all_registered_tools = list(TOOL_SCHEMA_MAP.keys())

    for idx, item in enumerate(samples_to_eval):
        prompt_text = item["prompt"]
        completion_text = item["completion"]

        # Parse ground truth completion
        exp_match = re.search(r"<tool_call>(.*?)</tool_call>", completion_text, re.DOTALL)
        exp_tool = None
        exp_args = {}
        if exp_match:
            try:
                exp_json = json.loads(exp_match.group(1).strip())
                exp_tool = exp_json.get("name")
                exp_args = exp_json.get("arguments", {})
            except Exception:
                pass

        if exp_tool:
            per_tool_stats[exp_tool]["expected"] += 1

        # 1. Compute Cross-Entropy Loss on Completion Tokens
        p_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        if p_ids and p_ids[-1] == tokenizer.eos_token_id:
            p_ids = p_ids[:-1]
        c_ids = tokenizer.encode(completion_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        full_ids = p_ids + c_ids

        if len(full_ids) > model.max_seq_len:
            full_ids = full_ids[:model.max_seq_len]

        input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
        
        # Mask prompt tokens for loss calculation
        p_len = len(p_ids)
        target_masked = target_tensor.clone()
        if p_len - 1 < target_masked.shape[1]:
            target_masked[:, :p_len - 1] = -100

        with torch.no_grad():
            _, loss = model(input_tensor, targets=target_masked)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                loss_batches += 1

        # 2. Autoregressive Generation from Prompt
        gen_input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        gen_inp_tensor = torch.tensor([gen_input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = model.generate(
                gen_inp_tensor,
                max_new_tokens=100,
                temperature=0.0,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.token_to_id.get("</tool_call>", -1)]
            )

        new_ids = gen_ids[0][len(gen_input_ids):].tolist()
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        # 3. Pure Parsing (No regex or guardrails)
        pred_tool = None
        pred_args = {}
        is_valid_json = False

        call_m = re.search(r"<tool_call>(.*?)</tool_call>", gen_text, re.DOTALL)
        if call_m:
            try:
                parsed_c = json.loads(call_m.group(1).strip())
                pred_tool = parsed_c.get("name")
                pred_args = parsed_c.get("arguments", {})
                is_valid_json = True
            except Exception:
                is_valid_json = False
        else:
            # Check if valid JSON was generated without tags
            try:
                parsed_c = json.loads(gen_text.strip())
                pred_tool = parsed_c.get("name")
                pred_args = parsed_c.get("arguments", {})
                is_valid_json = True
            except Exception:
                is_valid_json = False

        # Evaluation metrics
        tool_correct = (pred_tool == exp_tool) if (exp_tool and pred_tool) else False
        is_valid_args = False
        exact_match = False

        if is_valid_json:
            valid_json_count += 1
            if pred_tool in TOOL_SCHEMA_MAP and isinstance(pred_args, dict):
                is_v_arg, val_args, _ = validate_tool_arguments(pred_tool, pred_args)
                is_valid_args = is_v_arg
            if is_valid_args:
                valid_args_count += 1

        if tool_correct:
            correct_tool_count += 1
            if exp_tool:
                per_tool_stats[exp_tool]["correct"] += 1
            if is_valid_args and pred_args == exp_args:
                exact_match = True
                exact_match_count += 1
                if exp_tool:
                    per_tool_stats[exp_tool]["exact_match"] += 1

        # Record confusion matrix
        pred_label = pred_tool if pred_tool else "INVALID_OUTPUT"
        exp_label = exp_tool if exp_tool else "UNKNOWN"
        confusion_matrix[exp_label][pred_label] += 1

        # Record failure if incorrect
        if not tool_correct or not exact_match:
            if len(failed_examples) < 25:
                user_q_m = re.search(r"USER:\s*(.*?)\nASSISTANT:", prompt_text, re.DOTALL)
                q_text = user_q_m.group(1).strip() if user_q_m else ""
                failed_examples.append({
                    "idx": idx,
                    "query": q_text,
                    "expected_tool": exp_tool,
                    "expected_args": exp_args,
                    "predicted_tool": pred_tool,
                    "predicted_args": pred_args,
                    "generated_text": gen_text.strip(),
                    "reason": "TOOL_MISMATCH" if not tool_correct else ("ARG_MISMATCH" if not exact_match else "JSON_ERROR")
                })

        if len(sample_results) < 15:
            user_q_m = re.search(r"USER:\s*(.*?)\nASSISTANT:", prompt_text, re.DOTALL)
            sample_results.append({
                "query": user_q_m.group(1).strip() if user_q_m else "",
                "expected_tool": exp_tool,
                "predicted_tool": pred_tool,
                "generated_text": gen_text.strip(),
                "tool_correct": tool_correct,
                "exact_match": exact_match
            })

    avg_loss = round(total_loss / max(1, loss_batches), 4)
    perplexity = round(math.exp(min(avg_loss, 20.0)), 2)

    per_tool_accuracy = {}
    for t, stat in sorted(per_tool_stats.items()):
        exp_n = stat["expected"]
        corr_n = stat["correct"]
        exact_n = stat["exact_match"]
        per_tool_accuracy[t] = {
            "expected_count": exp_n,
            "correct_count": corr_n,
            "tool_accuracy_pct": round((corr_n / max(1, exp_n)) * 100, 2),
            "exact_match_count": exact_n,
            "exact_match_pct": round((exact_n / max(1, exp_n)) * 100, 2)
        }

    return {
        "total_evaluated": total_eval,
        "cross_entropy_loss": avg_loss,
        "perplexity": perplexity,
        "valid_json_count": valid_json_count,
        "valid_json_pct": round((valid_json_count / total_eval) * 100, 2),
        "correct_tool_count": correct_tool_count,
        "correct_tool_pct": round((correct_tool_count / total_eval) * 100, 2),
        "valid_args_count": valid_args_count,
        "valid_args_pct": round((valid_args_count / total_eval) * 100, 2),
        "exact_match_count": exact_match_count,
        "exact_match_pct": round((exact_match_count / total_eval) * 100, 2),
        "per_tool_accuracy": per_tool_accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "failed_examples": failed_examples,
        "sample_predictions": sample_results
    }


def run_phase3_evaluation():
    print("=" * 75)
    print("Spatial Quest: Comprehensive Phase 3 Evaluation")
    print("=" * 75)

    device = get_optimal_device()
    print(f"Device: {device}")

    # Load test split (UNTOUCHED)
    test_path = "data/corpora/corrected_test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(test_samples)} held-out test samples from {test_path}")

    # Load Tokenizer
    tokenizer = SpatialTokenizer.load("models/language_model/tokenizer.json")

    # 1. Load Legacy Checkpoint
    legacy_ckpt_path = "models/language_model/transformer_legacy_checkpoint.pt"
    print(f"\n[Evaluating Legacy Pre-Training Checkpoint] ({legacy_ckpt_path})")
    legacy_ckpt = torch.load(legacy_ckpt_path, map_location="cpu", weights_only=False)
    legacy_model = DecoderTransformer(
        vocab_size=legacy_ckpt["vocab_size"],
        max_seq_len=legacy_ckpt["max_seq_len"],
        d_model=legacy_ckpt["d_model"],
        n_heads=legacy_ckpt["n_heads"],
        n_layers=legacy_ckpt["n_layers"],
        d_ff=legacy_ckpt["d_ff"]
    ).to(device)
    legacy_model.load_state_dict(legacy_ckpt["model_state_dict"])

    legacy_results = evaluate_raw_model_on_test_split(
        legacy_model, tokenizer, test_samples, device, sample_limit=50
    )
    print(f"  Legacy Checkpoint: Loss={legacy_results['cross_entropy_loss']} | Valid JSON={legacy_results['valid_json_pct']}% | Tool Acc={legacy_results['correct_tool_pct']}% | Exact Match={legacy_results['exact_match_pct']}%")

    # 2. Load Phase 3 Trained Checkpoint
    phase3_ckpt_path = "models/language_model/transformer_phase3_best.pt"
    if not os.path.exists(phase3_ckpt_path):
        phase3_ckpt_path = "models/language_model/transformer_best.pt"

    print(f"\n[Evaluating Phase 3 Retrained Checkpoint] ({phase3_ckpt_path})")
    phase3_ckpt = torch.load(phase3_ckpt_path, map_location="cpu", weights_only=False)
    phase3_model = DecoderTransformer(
        vocab_size=phase3_ckpt["vocab_size"],
        max_seq_len=phase3_ckpt["max_seq_len"],
        d_model=phase3_ckpt["d_model"],
        n_heads=phase3_ckpt["n_heads"],
        n_layers=phase3_ckpt["n_layers"],
        d_ff=phase3_ckpt["d_ff"]
    ).to(device)
    phase3_model.load_state_dict(phase3_ckpt["model_state_dict"])

    # Evaluate on all 251 held-out test samples (raw model)
    phase3_raw_results = evaluate_raw_model_on_test_split(
        phase3_model, tokenizer, test_samples, device, sample_limit=None
    )
    print(f"  Phase 3 Raw Model (N={phase3_raw_results['total_evaluated']}):")
    print(f"    • Cross-Entropy Loss:          {phase3_raw_results['cross_entropy_loss']} | Perplexity: {phase3_raw_results['perplexity']}")
    print(f"    • Valid JSON Generation Rate:  {phase3_raw_results['valid_json_count']}/{phase3_raw_results['total_evaluated']} ({phase3_raw_results['valid_json_pct']}%)")
    print(f"    • Correct Tool Selection Rate: {phase3_raw_results['correct_tool_count']}/{phase3_raw_results['total_evaluated']} ({phase3_raw_results['correct_tool_pct']}%)")
    print(f"    • Valid Argument Rate:         {phase3_raw_results['valid_args_count']}/{phase3_raw_results['total_evaluated']} ({phase3_raw_results['valid_args_pct']}%)")
    print(f"    • Exact Tool-Call Match Rate:  {phase3_raw_results['exact_match_count']}/{phase3_raw_results['total_evaluated']} ({phase3_raw_results['exact_match_pct']}%)")

    # 3. Evaluate Combined Agent on the 28-Query Reliability Suite
    print(f"\n[Evaluating Combined Agent Pipeline]")
    seed_facility_scene(source="facility_layout")
    global_game_session.return_to_hub()
    global_game_session.start_mission("mission_a")

    agent = InvestigationAgent(max_steps=2, prompt_format="canonical")

    core_queries = [
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
        ("List all visible objects in this sector.", "get_visible_objects", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Can you guide me towards the office?", "navigate_to", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("How far is the chair from the table?", "calculate_distance", "navigator", [6.0, 0.0, -5.0], "Research Office"),
        ("What items are currently in my bag?", "get_inventory", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Tell me what my goals are.", "get_mission_objectives", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Explain why that destination is blocked.", "explain_unreachable", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("wat objcts can u see?", "get_visible_objects", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("fnd the nerest tabl", "find_nearest_target", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("tke me to libary", "navigate_to", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("giv me a hnt", "request_hint", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("shw my invntory", "get_inventory", "archivist", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("What is that over there?", "explain_uncertainty", "scout", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("How do I play?", "get_game_guide", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Tell me about this room.", "get_current_location", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub"),
        ("Calculate a path to the door.", "find_path", "navigator", [0.0, 0.0, -15.0], "Reception / Central Hub")
    ]

    agent_correct = 0
    agent_results = []
    for q, exp_t, mode, pos, room in core_queries:
        res = agent.run(q, agent_id=mode, player_position=pos, active_room=room)
        act_t = res["tool_calls"][0]["tool"] if res["tool_calls"] else "none"
        is_c = (act_t == exp_t)
        if is_c:
            agent_correct += 1
        agent_results.append({
            "query": q,
            "expected_tool": exp_t,
            "actual_tool": act_t,
            "correct": is_c,
            "guardrail_status": res.get("validation_status")
        })

    agent_accuracy_pct = round((agent_correct / len(core_queries)) * 100, 2)
    print(f"  Combined Agent Accuracy: {agent_correct}/{len(core_queries)} ({agent_accuracy_pct}%)")

    # 4. Save Comprehensive Evaluation Report
    out_payload = {
        "evaluation_metadata": {
            "test_split_path": test_path,
            "total_held_out_samples": len(test_samples),
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "checkpoint_comparison": {
            "legacy_checkpoint": {
                "checkpoint_path": legacy_ckpt_path,
                "samples_tested": legacy_results["total_evaluated"],
                "cross_entropy_loss": legacy_results["cross_entropy_loss"],
                "perplexity": legacy_results["perplexity"],
                "valid_json_pct": legacy_results["valid_json_pct"],
                "correct_tool_pct": legacy_results["correct_tool_pct"],
                "valid_args_pct": legacy_results["valid_args_pct"],
                "exact_match_pct": legacy_results["exact_match_pct"]
            },
            "phase3_checkpoint": {
                "checkpoint_path": phase3_ckpt_path,
                "samples_tested": phase3_raw_results["total_evaluated"],
                "cross_entropy_loss": phase3_raw_results["cross_entropy_loss"],
                "perplexity": phase3_raw_results["perplexity"],
                "valid_json_pct": phase3_raw_results["valid_json_pct"],
                "correct_tool_pct": phase3_raw_results["correct_tool_pct"],
                "valid_args_pct": phase3_raw_results["valid_args_pct"],
                "exact_match_pct": phase3_raw_results["exact_match_pct"]
            }
        },
        "raw_transformer_metrics": phase3_raw_results,
        "combined_agent_metrics": {
            "queries_evaluated": len(core_queries),
            "correct_queries": agent_correct,
            "accuracy_pct": agent_accuracy_pct,
            "details": agent_results
        }
    }

    os.makedirs("results/metrics", exist_ok=True)
    out_file = "results/metrics/phase3_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n✓ Saved complete Phase 3 evaluation report -> {out_file}")


if __name__ == "__main__":
    run_phase3_evaluation()

