"""Phase 3 Custom Transformer Training Script for Spatial Quest.

Trains the from-scratch DecoderTransformer architecture on the corrected canonical dataset:
- Architecture: 4 layers, 4 heads, d_model=128, d_ff=512, max_seq_len=416, vocab_size=1024
- Dataset: corrected_train.jsonl (2,000 samples) and corrected_val.jsonl (249 samples)
- Held-out test set (corrected_test.jsonl) is strictly preserved and UNTOUCHED
- Evaluates validation loss and perplexity at every epoch
- Implements early stopping based on validation loss
- Saves the best checkpoint to models/language_model/transformer_phase3_best.pt and models/language_model/transformer_best.pt
- Preserves the legacy checkpoint at models/language_model/transformer_legacy_checkpoint.pt
- Saves training log and metrics to results/metrics/transformer_phase3_training_log.json
"""
import os
import sys
import json
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Workspace path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.language_model.tokenizer import SpatialTokenizer
from backend.language_model.transformer import DecoderTransformer
from backend.perception.device import get_optimal_device


class DialogueSFTDataset(Dataset):
    """Prepares canonical dialogue samples with SFT target masking."""

    def __init__(self, jsonl_path: str, tokenizer: SpatialTokenizer, max_seq_len: int = 416):
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_items = [json.loads(line) for line in f if line.strip()]

        for item in raw_items:
            p = item["prompt"]
            c = item["completion"]

            # Tokenize prompt without EOS
            p_ids = tokenizer.encode(p, add_special_tokens=True)
            if p_ids and p_ids[-1] == tokenizer.eos_token_id:
                p_ids = p_ids[:-1]

            # Tokenize completion with EOS
            c_ids = tokenizer.encode(c, add_special_tokens=False) + [tokenizer.eos_token_id]

            full_ids = p_ids + c_ids
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            input_ids = full_ids[:-1]
            targets = full_ids[1:]

            # Mask prompt tokens in target with -100 so loss is computed on completion tokens
            p_len = len(p_ids)
            targets = [-100 if i < p_len - 1 else targets[i] for i in range(len(targets))]

            # Pad to max_seq_len - 1
            pad_len = (max_seq_len - 1) - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_id] * pad_len
                targets = targets + [-100] * pad_len

            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_phase3_transformer():
    print("=" * 75)
    print("Spatial Quest: Phase 3 From-Scratch Transformer Training")
    print("=" * 75)

    device = get_optimal_device()
    print(f"Device: {device}")

    # 1. Verify file paths
    train_path = "data/corpora/corrected_train.jsonl"
    val_path = "data/corpora/corrected_val.jsonl"
    test_path = "data/corpora/corrected_test.jsonl"
    tokenizer_path = "models/language_model/tokenizer.json"

    assert os.path.exists(train_path), f"Train dataset missing: {train_path}"
    assert os.path.exists(val_path), f"Val dataset missing: {val_path}"
    assert os.path.exists(test_path), f"Test dataset missing: {test_path}"
    assert os.path.exists(tokenizer_path), f"Tokenizer missing: {tokenizer_path}"

    # Load tokenizer
    tokenizer = SpatialTokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded with vocabulary size: {vocab_size} tokens")

    # 2. Prepare Datasets (Held-out test set is explicitly NOT loaded)
    max_seq_len = 416
    print(f"\nPreparing datasets with max_seq_len={max_seq_len}...")
    t0_prep = time.time()
    train_dataset = DialogueSFTDataset(train_path, tokenizer, max_seq_len=max_seq_len)
    val_dataset = DialogueSFTDataset(val_path, tokenizer, max_seq_len=max_seq_len)
    print(f"Prepared {len(train_dataset)} training samples and {len(val_dataset)} validation samples in {time.time()-t0_prep:.1f}s.")
    print(f"Note: Held-out test set ({test_path}) is strictly untouched during training.")

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Transformer Architecture from Scratch
    d_model = 128
    n_heads = 4
    n_layers = 4
    d_ff = 512
    dropout = 0.1

    torch.manual_seed(42)
    model = DecoderTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture initialized from random weights:")
    print(f"  • Layers: {n_layers} | Heads: {n_heads} | d_model: {d_model} | d_ff: {d_ff}")
    print(f"  • Max Seq Len: {max_seq_len} | Vocab Size: {vocab_size}")
    print(f"  • Trainable Parameters: {total_params:,}")

    # 4. Optimizer, Scheduler, and Early Stopping Configuration
    epochs = 15
    lr = 6e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    patience = 4
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    best_epoch = 0

    history = {
        "config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "vocab_size": vocab_size,
            "total_params": total_params,
            "batch_size": batch_size,
            "initial_lr": lr,
            "epochs_scheduled": epochs,
            "patience": patience,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        },
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "lr": [],
        "epoch_time_s": []
    }

    print(f"\n--- Launching Phase 3 Training (Max {epochs} Epochs, Patience={patience}) ---")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(1, train_batches)

        # Validation evaluation
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                _, v_loss = model(vx, targets=vy)
                total_val_loss += v_loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_perplexity"].append(round(val_ppl, 2))
        history["lr"].append(round(current_lr, 6))
        history["epoch_time_s"].append(round(epoch_duration, 2))

        # Check for best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_marker = "★ BEST"
        else:
            patience_counter += 1
            best_marker = f"  (patience {patience_counter}/{patience})"

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Time: {epoch_duration:.1f}s {best_marker}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] Validation loss did not improve for {patience} consecutive epochs.")
            break

    total_training_duration = time.time() - start_total_time
    print(f"\nTraining completed in {total_training_duration:.2f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Overfitting analysis
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    loss_gap = round(final_val_loss - final_train_loss, 4)
    has_overfitting = (loss_gap > 0.5) and (patience_counter > 0)
    overfitting_assessment = (
        f"MILD OVERFITTING DETECTED (Train/Val gap: {loss_gap:.4f}). Early stopping restored optimal weights from Epoch {best_epoch}."
        if has_overfitting else
        f"NO SEVERE OVERFITTING (Train/Val gap: {loss_gap:.4f}). Well-regularized convergence across splits."
    )
    print(f"Overfitting Assessment: {overfitting_assessment}")

    history["summary"] = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_perplexity": round(math.exp(min(best_val_loss, 20.0)), 2),
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "train_val_gap": loss_gap,
        "overfitting_assessment": overfitting_assessment,
        "total_training_time_s": round(total_training_duration, 2)
    }

    # 5. Save Model Checkpoints
    # Restore best weights
    model.load_state_dict(best_weights)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "max_seq_len": max_seq_len,
        "final_loss": best_val_loss,
        "final_perplexity": math.exp(min(best_val_loss, 20.0)),
        "training_phase": "Phase 3 Canonical Aligned SFT",
        "best_epoch": best_epoch
    }

    # Save to dedicated Phase 3 best checkpoint
    phase3_ckpt_path = "models/language_model/transformer_phase3_best.pt"
    torch.save(checkpoint_payload, phase3_ckpt_path)
    print(f"✓ Saved Phase 3 best checkpoint -> {phase3_ckpt_path}")

    # Also update transformer_best.pt so runtime agent and evaluation automatically use it
    best_ckpt_path = "models/language_model/transformer_best.pt"
    torch.save(checkpoint_payload, best_ckpt_path)
    print(f"✓ Updated runtime checkpoint -> {best_ckpt_path}")
    print(f"✓ Legacy checkpoint preserved at -> models/language_model/transformer_legacy_checkpoint.pt")

    # 6. Save Training History Log and Loss Curve Plot
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    log_path = "results/metrics/transformer_phase3_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history log -> {log_path}")

    plot_path = "results/plots/transformer_phase3_loss_curve.png"
    plt.figure(figsize=(9, 4.5))
    plt.plot(history["epochs"], history["train_loss"], "b-o", label="Train Loss (SFT)")
    plt.plot(history["epochs"], history["val_loss"], "r--s", label="Val Loss (Held-Out Val)")
    plt.axvline(x=best_epoch, color="g", linestyle=":", label=f"Best Checkpoint (Epoch {best_epoch})")
    plt.title("Spatial Quest Phase 3 Transformer: Convergence & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved loss curve visualization -> {plot_path}")

    print("\n" + "=" * 75)
    print("Phase 3 Transformer Training Completed Successfully.")
    print("=" * 75)
    return history


if __name__ == "__main__":
    train_phase3_transformer()

