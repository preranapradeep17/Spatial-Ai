"""Categorize all failures into the 4 requested categories."""
import json

with open("scratch/phase3_all_failures.json") as f:
    failures = json.load(f)

categories = {
    "tool_selection_error": [],
    "argument_generation_error": [],
    "malformed_output": [],
    "incomplete_generation": []
}

for item in failures:
    gt = item["gen_text"]
    is_valid = item["is_valid_json"]
    tc = item["tool_correct"]

    if not is_valid:
        if "</tool_call>" not in gt:
            categories["incomplete_generation"].append(item)
        else:
            categories["malformed_output"].append(item)
    elif not tc:
        categories["tool_selection_error"].append(item)
    else:
        categories["argument_generation_error"].append(item)

print(f"Total evaluated failures: {len(failures)} / 251 held-out test items")
for k, v in categories.items():
    print(f"  {k}: {len(v)}")

print("\n--- Tool Selection Error Breakdown (Confusion pairs) ---")
ts_by_tool = {}
for item in categories["tool_selection_error"]:
    pair = f"{item['expected_tool']} -> {item['predicted_tool']}"
    ts_by_tool[pair] = ts_by_tool.get(pair, 0) + 1
for p, c in sorted(ts_by_tool.items(), key=lambda x: -x[1]):
    print(f"  {p}: {c}")

print("\n--- Argument Generation Error Breakdown ---")
arg_by_tool = {}
for item in categories["argument_generation_error"]:
    et = item["expected_tool"]
    arg_by_tool[et] = arg_by_tool.get(et, 0) + 1
for t, c in sorted(arg_by_tool.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

with open("scratch/categorized_failures.json", "w") as f:
    json.dump({k: len(v) for k, v in categories.items()}, f, indent=2)

