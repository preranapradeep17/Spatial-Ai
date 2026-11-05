"""Tests for from-scratch Subword/BPE Tokenizer."""
import os
import pytest
from backend.language_model.tokenizer import SpatialTokenizer

def test_tokenizer_train_and_encode_decode(tmp_path):
    corpus = (
        "The drone is navigating towards the table. "
        "The chair is located 1.5 meters away from the door. "
        "<tool_call>{\"name\": \"get_visible_objects\"}</tool_call> "
        "Locate object cabinet in the facility."
    )
    tok = SpatialTokenizer(vocab_size=120)
    tok.train_from_text(corpus)

    assert len(tok) <= 120
    assert "<tool_call>" in tok.token_to_id
    assert "</tool_call>" in tok.token_to_id

    # Test encoding
    ids = tok.encode("The drone is navigating", add_special_tokens=True)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id

    # Test decoding
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "drone" in decoded
    assert "navigating" in decoded

    # Test save and load
    save_file = str(tmp_path / "test_tok.json")
    tok.save(save_file)
    assert os.path.exists(save_file)

    loaded = SpatialTokenizer.load(save_file)
    assert len(loaded) == len(tok)
    assert loaded.bos_token_id == tok.bos_token_id
    assert loaded.eos_token_id == tok.eos_token_id
