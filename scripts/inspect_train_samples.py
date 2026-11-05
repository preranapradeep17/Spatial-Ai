"""Inspect training samples for failing tools."""
import json
import re

with open("data/corpora/corrected_train.jsonl") as f:
    train_data = [json.loads(line) for line in f]

def inspect_tool(tname, n=5):
    print(f"\n==================== Training: {tname} ====================")
    count = 0
    for d in train_data:
        m = re.search(r"<tool_call>(.*?)</tool_call>", d["completion"], re.DOTALL)
        if m:
            c = json.loads(m.group(1).strip())
            if c.get("name") == tname:
                count += 1
                if count <= n:
                    qm = re.search(r"USER:\s*(.*?)\nASSISTANT:", d["prompt"], re.DOTALL)
                    q = qm.group(1).strip() if qm else ""
                    am = re.search(r"AGENT_MODE:\s*(.*?)\n", d["prompt"])
                    mode = am.group(1).strip() if am else ""
                    comp = d["completion"]
                    print(f"[{count}] Mode: {mode} | Query: '{q}'")
                    print(f"    Completion: {comp}")
    print(f"Total count in training: {count}")

for t in ["calculate_distance", "navigate_to", "interact_with_object", "find_nearest_target", "search_safety_manual", "get_visible_objects"]:
    inspect_tool(t, 4)

