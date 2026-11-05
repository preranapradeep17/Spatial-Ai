"""Analyze all failures of Phase 3 Transformer on held-out test set."""

import json
import re
import os
import sys

sys.path.insert(0, ".")
import torch
from backend.language_model.transformer import DecoderTransformer
from backend.language_model.tokenizer import SpatialTokenizer

os.makedirs("scratch", exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = SpatialTokenizer.load("models/language_model/tokenizer.json")

ckpt = torch.load("models/language_model/transformer_phase3_best.pt", map_location=device)
cfg = ckpt.get("config", {})
model = DecoderTransformer(
    vocab_size=cfg.get("vocab_size", 1024),
    max_seq_len=cfg.get("max_seq_len", 416),
    d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4),
    n_layers=cfg.get("n_layers", 4),
    d_ff=cfg.get("d_ff", 512),
    dropout=cfg.get("dropout", 0.1)
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

failures = []
tool_stats = {}

with open("data/corpora/corrected_test.jsonl") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        prompt = data["prompt"]
        comp = data["completion"]

        # Parse expected tool & args
        exp_match = re.search(r"<tool_call>(.*?)</tool_call>", comp, re.DOTALL)
        exp_tool = None
        exp_args = {}
        if exp_match:
            try:
                ec = json.loads(exp_match.group(1).strip())
                exp_tool = ec.get("name")
                exp_args = ec.get("arguments", {})
            except Exception:
                pass

        # Generate with model
        gen_in = tokenizer.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([gen_in], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                inp,
                max_new_tokens=100,
                temperature=0.0,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.token_to_id.get("</tool_call>", -1)]
            )
        new_ids = out[0][len(gen_in):].tolist()
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        pred_tool = None
        pred_args = {}
        is_valid_json = False
        call_m = re.search(r"<tool_call>(.*?)</tool_call>", gen_text, re.DOTALL)
        if call_m:
            try:
                pc = json.loads(call_m.group(1).strip())
                pred_tool = pc.get("name")
                pred_args = pc.get("arguments", {})
                is_valid_json = True
            except Exception:
                pass

        tool_correct = (pred_tool == exp_tool)
        exact_match = tool_correct and (pred_args == exp_args)

        # Extract user query
        qm = re.search(r"USER:\s*(.*?)\nASSISTANT:", prompt, re.DOTALL)
        query = qm.group(1).strip() if qm else ""

        if exp_tool not in tool_stats:
            tool_stats[exp_tool] = {"total": 0, "tool_correct": 0, "exact_match": 0}
        tool_stats[exp_tool]["total"] += 1
        if tool_correct:
            tool_stats[exp_tool]["tool_correct"] += 1
        if exact_match:
            tool_stats[exp_tool]["exact_match"] += 1

        if not exact_match:
            failures.append({
                "idx": idx,
                "query": query,
                "expected_tool": exp_tool,
                "expected_args": exp_args,
                "predicted_tool": pred_tool,
                "predicted_args": pred_args,
                "tool_correct": tool_correct,
                "is_valid_json": is_valid_json,
                "gen_text": gen_text.strip()
            })

print(f"Total failures: {len(failures)} / 251")
print("\nTool stats:")
for t, st in sorted(tool_stats.items()):
    tc = st["tool_correct"]
    em = st["exact_match"]
    tot = st["total"]
    print(f"  {t}: {tc}/{tot} tool correct ({em}/{tot} exact match)")

with open("scratch/phase3_all_failures.json", "w") as f:
    json.dump(failures, f, indent=2)
print("Saved all failures to scratch/phase3_all_failures.json")
