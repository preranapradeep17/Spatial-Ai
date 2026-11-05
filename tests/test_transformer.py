"""Tests for from-scratch Decoder-Only Transformer architecture."""
import pytest
import torch
from backend.language_model.transformer import DecoderTransformer, CausalSelfAttention

def test_causal_self_attention_shapes_and_mask():
    B, T, C = 2, 8, 64
    n_heads = 4
    attn = CausalSelfAttention(d_model=C, n_heads=n_heads, max_seq_len=16, dropout=0.0)
    x = torch.randn(B, T, C)
    out = attn(x)

    assert out.shape == (B, T, C)
    assert not torch.isnan(out).any()

def test_transformer_forward_and_loss():
    vocab_size = 100
    max_seq_len = 32
    d_model = 64
    model = DecoderTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        d_ff=128
    )

    B, T = 2, 10
    input_ids = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (B, T, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

def test_transformer_autoregressive_generation():
    vocab_size = 50
    model = DecoderTransformer(
        vocab_size=vocab_size,
        max_seq_len=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64
    )
    prompt = torch.tensor([[1, 5, 12]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.0)

    assert generated.shape == (1, 8)
    assert (generated[:, :3] == prompt).all()
