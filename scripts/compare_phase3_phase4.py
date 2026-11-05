"""Compare Phase 3 and Phase 4 evaluation results."""
import json

with open("results/metrics/phase4_evaluation_report.json") as f:
    rep = json.load(f)

p3 = rep["phase3_baseline"]
p4 = rep["phase4_improved"]

print("=" * 80)
print(f"{'Tool':<25} | {'P3 Tool Acc':<16} | {'P4 Tool Acc':<16} | {'P3 Exact':<10} | {'P4 Exact':<10}")
print("-" * 80)

for tool in sorted(p3["per_tool_accuracy"].keys()):
    p3_stat = p3["per_tool_accuracy"][tool]
    p4_stat = p4["per_tool_accuracy"][tool]
    exp = p3_stat["expected_count"]
    
    p3_tc = f"{p3_stat['tool_accuracy_pct']:.1f}% ({p3_stat['correct_count']}/{exp})"
    p4_tc = f"{p4_stat['tool_accuracy_pct']:.1f}% ({p4_stat['correct_count']}/{exp})"
    p3_em = f"{p3_stat['exact_match_pct']:.1f}%"
    p4_em = f"{p4_stat['exact_match_pct']:.1f}%"
    
    print(f"{tool:<25} | {p3_tc:<16} | {p4_tc:<16} | {p3_em:<10} | {p4_em:<10}")

print("=" * 80)
print(f"{'OVERALL':<25} | {p3['correct_tool_pct']:.1f}% ({p3['correct_tool_count']}/{p3['total_evaluated']})     | {p4['correct_tool_pct']:.1f}% ({p4['correct_tool_count']}/{p4['total_evaluated']})     | {p3['exact_match_pct']:.1f}%     | {p4['exact_match_pct']:.1f}%")
print(f"Valid JSON Rate:        | {p3['valid_json_pct']:.1f}% ({p3['valid_json_count']}/{p3['total_evaluated']})     | {p4['valid_json_pct']:.1f}% ({p4['valid_json_count']}/{p4['total_evaluated']})     | -          | -")
print(f"Cross-Entropy Loss:     | {p3['cross_entropy_loss']}          | {p4['cross_entropy_loss']}          | -          | -")
print(f"Perplexity:             | {p3['perplexity']}            | {p4['perplexity']}            | -          | -")
print(f"Entity Accuracy:        | {p3['entity_accuracy_pct']:.1f}% ({p3['entity_match_count']}/{p3['entity_eval_count']})     | {p4['entity_accuracy_pct']:.1f}% ({p4['entity_match_count']}/{p4['entity_eval_count']})     | -          | -")

print("\n--- Phase 4 Confusion Matrix ---")
cm4 = p4["confusion_matrix"]
for exp_t, preds in sorted(cm4.items()):
    preds_str = ", ".join(f"{k}: {v}" for k, v in sorted(preds.items(), key=lambda x: -x[1]))
    print(f"{exp_t:<25} -> {preds_str}")

print("\n--- Sample Phase 4 Failures (First 10) ---")
for fail in p4["failed_examples"][:10]:
    print(f"Q: '{fail['query']}'")
    print(f"   Exp: {fail['expected_tool']} {fail['expected_args']}")
    print(f"   Pred: {fail['predicted_tool']} {fail['predicted_args']}")
    print(f"   Reason: {fail['reason']} | Raw: {fail['generated_text'][:100]}")

