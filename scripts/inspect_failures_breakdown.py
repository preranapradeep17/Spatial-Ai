"""Inspect failures breakdown."""
import json

with open("scratch/phase3_all_failures.json") as f:
    failures = json.load(f)

print(f"Total failures loaded: {len(failures)}")

by_tool = {}
for item in failures:
    et = item["expected_tool"]
    by_tool.setdefault(et, []).append(item)

tools_to_check = [
    "find_nearest_target",
    "interact_with_object",
    "search_safety_manual",
    "get_visible_objects",
    "calculate_distance",
    "navigate_to"
]

for tool in tools_to_check:
    items = by_tool.get(tool, [])
    print(f"\n==================== {tool} ({len(items)} failures) ====================")
    for i, it in enumerate(items[:5]):
        q = it["query"]
        et = it["expected_tool"]
        ea = it["expected_args"]
        pt = it["predicted_tool"]
        pa = it["predicted_args"]
        gt = it["gen_text"]
        print(f"[{i+1}] Query: '{q}'")
        print(f"    Expected: {et} -> {ea}")
        print(f"    Predicted: {pt} -> {pa}")
        print(f"    Gen Text: {gt[:140]}")

