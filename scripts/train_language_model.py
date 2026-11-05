"""Training script for from-scratch Spatial Quest Decoder-Only Transformer.

Trains custom tokenizer and decoder Transformer on the facility domain corpus
and instruction tuning dialogues from random weights.
Traceable to CS3235 (Working with LLMs - Pretraining & Instruction Tuning).
Strictly adheres to: NO pretrained black-box LLM weights, NO commercial API calls.
"""
import os
import sys
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Workspace path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.language_model.tokenizer import SpatialTokenizer
from backend.language_model.transformer import DecoderTransformer
from backend.perception.device import get_optimal_device

class TextDataset(Dataset):
    """Chunks tokenized text into fixed-length sequence blocks for causal LM."""

    def __init__(self, token_ids: list, seq_len: int = 128):
        self.seq_len = seq_len
        self.samples = []
        # Sliding window with 50% overlap for comprehensive coverage
        step = max(1, seq_len // 2)
        for i in range(0, len(token_ids) - seq_len, step):
            chunk = token_ids[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_language_model():
    print("=" * 65)
    print("Spatial Quest: Training Custom Transformer LM from Random Weights")
    print("=" * 65)

    device = get_optimal_device()
    print(f"Device: {device}")

    corpus_path = "data/corpora/lm_training_corpus.txt"
    if not os.path.exists(corpus_path):
        corpus_path = "data/corpora/investigation_corpus.txt"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Include facility documents in text if using base corpus
    if "lm_training_corpus" not in corpus_path:
        doc_path = "data/documents/facility_documents.json"
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
                for d in docs:
                    raw_text += f"\n\nTitle: {d.get('title')}\n{d.get('content')}"

    print(f"Corpus source: {corpus_path}")
    print(f"Corpus size: {len(raw_text):,} characters, {len(raw_text.split()):,} words")

    # 1. Train Tokenizer
    tokenizer_dir = "models/language_model"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")

    print("\n--- Training BPE Subword Tokenizer (Vocab Target: 1,024) ---")
    tokenizer = SpatialTokenizer(vocab_size=1024)
    tokenizer.train_from_text(raw_text)
    tokenizer.save(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Learned vocabulary size: {vocab_size} tokens")
    print(f"Saved tokenizer to {tokenizer_path}")

    # 2. Tokenize corpus
    token_ids = tokenizer.encode(raw_text, add_special_tokens=True)
    print(f"Total token sequence length: {len(token_ids):,} tokens")

    seq_len = 256
    dataset = TextDataset(token_ids, seq_len=seq_len)
    print(f"Prepared {len(dataset)} causal LM training blocks (seq_len={seq_len})")

    # Prepare Validation Split
    val_dataset = None
    val_path = "data/corpora/lm_val_dataset.jsonl"
    if os.path.exists(val_path):
        val_text = ""
        with open(val_path, "r", encoding="utf-8") as vf:
            for line in vf:
                item = json.loads(line)
                val_text += item["prompt"] + item["completion"] + "\n\n"
        val_tokens = tokenizer.encode(val_text, add_special_tokens=True)
        if len(val_tokens) > seq_len:
            val_dataset = TextDataset(val_tokens, seq_len=seq_len)
            print(f"Prepared {len(val_dataset)} validation blocks (seq_len={seq_len})")

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    # 3. Initialize Transformer Architecture
    d_model = 128
    n_heads = 4
    n_layers = 4
    d_ff = 512

    model = DecoderTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized with {total_params:,} trainable parameters.")

    # 4. Optimizer and Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    epochs = 20
    print(f"\n--- Training for {epochs} Epochs ---")

    history = {"epoch": [], "loss": [], "val_loss": [], "perplexity": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20.0))

        # Evaluate validation loss
        val_loss_str = ""
        val_loss = None
        if val_loader:
            model.eval()
            val_tot = 0.0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, v_loss = model(vx, targets=vy)
                    val_tot += v_loss.item()
                    val_batches += 1
            if val_batches > 0:
                val_loss = round(val_tot / val_batches, 4)
                val_loss_str = f" | Val Loss: {val_loss:.4f}"

        history["epoch"].append(epoch)
        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(val_loss)
        history["perplexity"].append(round(perplexity, 4))

        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f}{val_loss_str} | Perplexity: {perplexity:.2f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    # 5. Save model checkpoint
    model_save_path = os.path.join(tokenizer_dir, "transformer_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "max_seq_len": seq_len,
        "final_loss": history["loss"][-1],
        "final_perplexity": history["perplexity"][-1],
    }, model_save_path)
    print(f"Saved trained Transformer weights to {model_save_path}")

    # 6. Save metrics and loss curve
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    metrics_path = "results/metrics/transformer_training_log.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training log to {metrics_path}")

    plot_path = "results/plots/transformer_loss_curve.png"
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["loss"], "b-o", label="Cross-Entropy Loss")
    plt.title("Spatial Quest Transformer LM: Training Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved loss curve plot to {plot_path}")

    # 7. Verification Test Generation
    print("\n--- Verifying Autoregressive Generation ---")
    model.eval()
    test_prompt = "User: Where is the closest door?\nAssistant:"
    prompt_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(input_tensor, max_new_tokens=30, temperature=0.7, stop_token_ids=[tokenizer.eos_token_id])
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
    print("Prompt:\n", test_prompt)
    print("Generated Continuation:\n", output_text[len(test_prompt):].strip())
    print("\n[DONE] Custom Transformer successfully trained and verified.")

if __name__ == "__main__":
    train_language_model()
