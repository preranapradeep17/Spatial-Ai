"""Analyze linguistic patterns causing confusion."""
import json
from collections import Counter

with open("scratch/phase3_all_failures.json") as f:
    failures = json.load(f)

print("=== 1. get_visible_objects Confusions ===")
gvo_uncertainty_queries = []
gvo_inspect_queries = []
for it in failures:
    if it["expected_tool"] == "get_visible_objects":
        if it["predicted_tool"] == "explain_uncertainty":
            gvo_uncertainty_queries.append(it["query"])
        elif it["predicted_tool"] == "inspect_object_closely":
            gvo_inspect_queries.append(it["query"])

print(f"-> explain_uncertainty (N={len(gvo_uncertainty_queries)}):")
for q, c in Counter(gvo_uncertainty_queries).items():
    print(f"   [{c}x] '{q}'")

print(f"\n-> inspect_object_closely (N={len(gvo_inspect_queries)}):")
for q, c in Counter(gvo_inspect_queries).items():
    print(f"   [{c}x] '{q}'")

print("\n=== 2. find_nearest_target Confusions ===")
fnt_queries = []
for it in failures:
    if it["expected_tool"] == "find_nearest_target":
        fnt_queries.append((it["query"], it["predicted_tool"]))
fnt_grouped = {}
for q, pt in fnt_queries:
    fnt_grouped.setdefault(pt, []).append(q)
for pt, qlist in fnt_grouped.items():
    print(f"-> {pt} (N={len(qlist)}):")
    for q, c in Counter(qlist).items():
        print(f"   [{c}x] '{q}'")

print("\n=== 3. interact_with_object Confusions ===")
iwo_queries = []
for it in failures:
    if it["expected_tool"] == "interact_with_object":
        iwo_queries.append((it["query"], it["predicted_tool"]))
iwo_grouped = {}
for q, pt in iwo_queries:
    iwo_grouped.setdefault(pt, []).append(q)
for pt, qlist in iwo_grouped.items():
    print(f"-> {pt} (N={len(qlist)}):")
    for q, c in Counter(qlist).items():
        print(f"   [{c}x] '{q}'")

print("\n=== 4. search_safety_manual Confusions ===")
ssm_queries = []
for it in failures:
    if it["expected_tool"] == "search_safety_manual":
        ssm_queries.append((it["query"], it["predicted_tool"]))
ssm_grouped = {}
for q, pt in ssm_queries:
    ssm_grouped.setdefault(pt, []).append(q)
for pt, qlist in ssm_grouped.items():
    print(f"-> {pt} (N={len(qlist)}):")
    for q, c in Counter(qlist).items():
        print(f"   [{c}x] '{q}'")

print("\n=== 5. calculate_distance Argument Mismatches ===")
cd_queries = []
for it in failures:
    if it["expected_tool"] == "calculate_distance":
        cd_queries.append((it["query"], it["expected_args"], it["predicted_args"]))
print(f"Total calculate_distance argument errors: {len(cd_queries)}")
for q, ea, pa in cd_queries[:5]:
    print(f"   Q: '{q}' | Exp: {ea} | Pred: {pa}")

