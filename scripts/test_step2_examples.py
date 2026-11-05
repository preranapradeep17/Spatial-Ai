"""Test the 5 specific Step 2 audit examples on Phase 3 checkpoint."""
import json
import re
import sys

sys.path.insert(0, ".")
import torch
from backend.language_model.transformer import DecoderTransformer
from backend.language_model.tokenizer import SpatialTokenizer
from backend.agent.prompt_builder import build_canonical_prompt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = SpatialTokenizer.load("models/language_model/tokenizer.json")

ckpt = torch.load("models/language_model/transformer_phase4_best.pt", map_location=device)
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

step2_queries = [
    ("Calculate distance between bookshelf and laboratory door.", "NAVIGATOR", "laboratory"),
    ("How far is the cabinet from the chair?", "NAVIGATOR", "office"),
    ("Find the nearest object to the research desk.", "NAVIGATOR", "laboratory"),
    ("Interact with the security cabinet.", "NAVIGATOR", "laboratory"),
    ("Open the storage room door.", "NAVIGATOR", "storage")
]

print("=" * 70)
print("STEP 2 — Argument Generation Audit on Baseline Phase 3 Checkpoint")
print("=" * 70)

for q, mode, room in step2_queries:
    prompt = build_canonical_prompt(
        query=q,
        agent_mode=mode,
        room_name=room,
        player_position=[0.0, 0.0, 0.0],
        visible_objects_str="- bookshelf: [-2.0, 0.0, 3.0]\n- cabinet: [2.0, 0.0, 1.0]\n- chair: [1.0, 0.0, -1.0]\n- door: [0.0, 0.0, 5.0]",
        spatial_relations_str="- bookshelf is near door"
    )

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

    call_m = re.search(r"<tool_call>(.*?)</tool_call>", gen_text, re.DOTALL)
    parsed = {}
    if call_m:
        try:
            parsed = json.loads(call_m.group(1).strip())
        except Exception as e:
            parsed = {"error": str(e)}

    print(f"\nQuery: '{q}'")
    print(f"  Raw Generated: {gen_text.strip()}")
    print(f"  Parsed Tool:   {parsed.get('name')}")
    print(f"  Parsed Args:   {parsed.get('arguments')}")
