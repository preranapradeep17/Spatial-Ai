"""From-scratch Decoder-Only Transformer Architecture for Spatial Quest.

Implemented directly using PyTorch tensor math and nn.Module primitives:
- Scaled Dot-Product Attention: Softmax(Q @ K.T / sqrt(d_k)) with causal lower-triangular mask
- Multi-Head Attention projection and tensor slicing
- Position-wise Feed-Forward Network with GELU activation
- Pre-LayerNorm residual connections
- Autoregressive token generation loop with temperature and top-k filtering

Traceable to CS3235 (Working with LLMs - Transformer Architecture & Attention Math).
No pretrained HuggingFace or commercial LLM models used.
"""
import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism computed from tensor math."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, Query, Value projections combined into a single linear map for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix (1 where i >= j, 0 where i < j)
        # Register as buffer so it moves with the module across devices
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        B, T, C = x.shape

        # 1. Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape into (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention: (Q @ K.T) / sqrt(d_k)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # 4. Apply causal mask: mask upper triangular positions with -inf
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # 5. Softmax and attention dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # 6. Compute context output: weights @ V
        out = torch.matmul(weights, v)  # (B, n_heads, T, head_dim)

        # 7. Concatenate heads back to (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection and residual dropout
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU non-linearity."""

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard pre-LayerNorm Transformer decoder block."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture with residual skip connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderTransformer(nn.Module):
    """Complete Decoder-Only Autoregressive Transformer implemented from scratch."""

    def __init__(
        self,
        vocab_size: int = 512,
        max_seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Stack of Transformer decoder layers
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # Output LM head projecting from d_model to vocab logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head
        self.token_emb.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for causal language modeling.
        
        Args:
            input_ids: LongTensor of shape (B, T)
            targets: Optional LongTensor of shape (B, T) for loss calculation
        Returns:
            logits: FloatTensor of shape (B, T, vocab_size)
            loss: Optional cross entropy loss scalar
        """
        device = input_ids.device
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Position indices: [0, 1, ..., T-1]
        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Combine token embedding and position embedding
        tok_embeddings = self.token_emb(input_ids)       # (B, T, d_model)
        pos_embeddings = self.pos_emb(positions)         # (1, T, d_model)
        x = self.emb_dropout(tok_embeddings + pos_embeddings)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization and logits
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to compute CrossEntropyLoss across all batch and sequence positions
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        stop_token_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Autoregressive text generation loop with causal autoregression.
        
        Args:
            input_ids: Tensor of shape (1, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy argmax)
            top_k: Top-k filtering threshold
            stop_token_ids: List of token IDs that immediately terminate generation
        Returns:
            Tensor of generated token IDs (1, T + generated)
        """
        self.eval()
        stop_token_ids = stop_token_ids or []

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            logits, _ = self(cond)
            # Take logits for the very last sequence step
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            if temperature == 0.0:
                # Greedy decoding (argmax)
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(temperature, 1e-5)
                # Optional top-k filtering
                if top_k > 0:
                    k = min(top_k, next_logits.size(-1))
                    topk_vals, _ = torch.topk(next_logits, k)
                    min_val = topk_vals[:, -1].unsqueeze(-1)
                    next_logits = torch.where(next_logits < min_val, torch.tensor(float("-inf"), device=next_logits.device), next_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() in stop_token_ids:
                break

        return input_ids
