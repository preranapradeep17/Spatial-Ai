"""From-scratch Subword/BPE Tokenizer for Spatial Quest.

Implements Byte-Pair Encoding (BPE) merge learning, vocabulary generation,
encoding, decoding, and special token handling without any external NLP libraries.
Traceable to CS3235 (Working with LLMs - Tokenization & Vocabulary Construction).
"""
import json
import os
import re
from typing import List, Dict, Tuple, Optional, Set

DEFAULT_SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<tool_call>",
    "</tool_call>",
    "<thought>",
    "</thought>",
    "<observation>",
    "</observation>",
]

class SpatialTokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer implemented from scratch."""

    def __init__(self, vocab_size: int = 512, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []  # Ordered list of learned merges

        # Pre-assign special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<s>"]
        self.eos_token_id = self.token_to_id["</s>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def train_from_text(self, text: str, verbose: bool = False):
        """Learns subword vocabulary using BPE merge iterations."""
        # 1. Split text into pre-tokens (whitespace + punctuation splitting)
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        # 2. Represent each word as a tuple of single characters with end-of-word marker </w>
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for w in words:
            if not w:
                continue
            chars = tuple(list(w) + ["</w>"])
            word_freqs[chars] = word_freqs.get(chars, 0) + 1

        # 3. Add all unique base characters to vocabulary
        unique_chars: Set[str] = set()
        for chars in word_freqs.keys():
            unique_chars.update(chars)

        for c in sorted(unique_chars):
            if c not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[c] = new_id
                self.id_to_token[new_id] = c

        # 4. Iteratively find most frequent adjacent pair and merge
        self.merges = []
        target_merges = self.vocab_size - len(self.token_to_id)

        for merge_round in range(max(0, target_merges)):
            pair_counts: Dict[Tuple[str, str], int] = {}
            for token_tuple, freq in word_freqs.items():
                for i in range(len(token_tuple) - 1):
                    pair = (token_tuple[i], token_tuple[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            best_pair = max(pair_counts.keys(), key=lambda p: pair_counts[p])
            if pair_counts[best_pair] < 1:
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            new_id = len(self.token_to_id)
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            # Update word frequencies with merged pair
            new_word_freqs: Dict[Tuple[str, ...], int] = {}
            for token_tuple, freq in word_freqs.items():
                new_tuple: List[str] = []
                i = 0
                while i < len(token_tuple):
                    if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == best_pair:
                        new_tuple.append(merged_token)
                        i += 2
                    else:
                        new_tuple.append(token_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_tuple)] = freq
            word_freqs = new_word_freqs

            if verbose and (merge_round + 1) % 50 == 0:
                print(f"BPE merge round {merge_round + 1}: merged {best_pair} -> '{merged_token}'")

    def _tokenize_word(self, word: str) -> List[str]:
        """Splits a single word into learned BPE subwords."""
        tokens = list(word) + ["</w>"]
        for pair in self.merges:
            if len(tokens) < 2:
                break
            merged = pair[0] + pair[1]
            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text to a list of integer token IDs."""
        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        # Protect special tokens during tokenization
        # Form pattern for all special tokens
        pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + r"|\w+|[^\w\s]|\s+)"
        matches = re.finditer(pattern, text)

        for match in matches:
            piece = match.group(0)
            if piece in self.token_to_id:
                token_ids.append(self.token_to_id[piece])
            elif piece.isspace():
                continue
            else:
                subwords = self._tokenize_word(piece)
                for sw in subwords:
                    token_ids.append(self.token_to_id.get(sw, self.unk_token_id))

        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back to a readable string."""
        tokens: List[str] = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "<unk>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        text = "".join(tokens)
        # Replace end-of-word marker with a single space
        # Strip </w> right before punctuation/closing quotes to avoid JSON space inflation (e.g. "name " : "table ")
        text = re.sub(r"</w>(?=[^a-zA-Z0-9\s])", "", text)
        text = text.replace("</w>", " ")
        # Clean up stray whitespace
        return re.sub(r"\s+", " ", text).strip()

    def save(self, filepath: str):
        """Serializes tokenizer vocabulary and merges to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SpatialTokenizer":
        """Loads tokenizer vocabulary and merges from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], special_tokens=data["special_tokens"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok
