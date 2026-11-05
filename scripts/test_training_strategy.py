"""Experimental comparison between Full Causal LM vs SFT Target Masking for SpatialAI Transformer."""
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.language_model.tokenizer import SpatialTokenizer
from backend.language_model.transformer import DecoderTransformer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
tokenizer = SpatialTokenizer.load("models/language_model/tokenizer.json")
pad_id = tokenizer.pad_token_id

class DialogueDataset(Dataset):
    def __init__(self, jsonl_path: str, max_seq_len: int = 416, mask_prompt: bool = True):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_items = [json.loads(line) for line in f if line.strip()]

        for item in raw_items:
            p = item["prompt"]
            c = item["completion"]
            p_ids = tokenizer.encode(p, add_special_tokens=True)[:-1]
            c_ids = tokenizer.encode(c, add_special_tokens=False) + [tokenizer.eos_token_id]
            full_ids = p_ids + c_ids
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
                
            input_ids = full_ids[:-1]
            targets = full_ids[1:]
            
            p_len = len(p_ids)
            if mask_prompt:
                targets = [-100 if i < p_len - 1 else targets[i] for i in range(len(targets))]
                
            # Pad to max_seq_len - 1
            pad_len = (max_seq_len - 1) - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_id] * pad_len
                targets = targets + [-100] * pad_len
                
            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

print("Preparing datasets for 3-epoch comparison...")
train_sft = DialogueDataset("data/corpora/corrected_train.jsonl", mask_prompt=True)
val_sft = DialogueDataset("data/corpora/corrected_val.jsonl", mask_prompt=True)

train_loader = DataLoader(train_sft, batch_size=16, shuffle=True)
val_loader = DataLoader(val_sft, batch_size=16, shuffle=False)

torch.manual_seed(42)
model = DecoderTransformer(vocab_size=1024, max_seq_len=416, d_model=128, n_heads=4, n_layers=4, d_ff=512, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

print("\n--- Training SFT Model for 3 Epochs ---")
for epoch in range(1, 4):
    model.train()
    total_loss, batches = 0.0, 0
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    train_loss = total_loss / batches
    
    # Val loss
    model.eval()
    val_loss, val_b = 0.0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            _, vloss = model(vx, targets=vy)
            val_loss += vloss.item()
            val_b += 1
    val_loss /= val_b
    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-t0:.1f}s")

# Test generation on a validation prompt
print("\n--- Test Generation ---")
with open("data/corpora/corrected_val.jsonl") as f:
    sample_val = json.loads(f.readline())
p_val = sample_val["prompt"]
p_toks = tokenizer.encode(p_val, add_special_tokens=False)
inp = torch.tensor([p_toks], dtype=torch.long, device=device)
with torch.no_grad():
    gen_ids = model.generate(inp, max_new_tokens=40, temperature=0.1, stop_token_ids=[tokenizer.eos_token_id])
gen_str = tokenizer.decode(gen_ids[0][len(p_toks):].tolist(), skip_special_tokens=False)
print("Expected Completion:\n", sample_val["completion"])
print("Generated Completion:\n", gen_str)

