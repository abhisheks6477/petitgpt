"""Checks for the canonical released tokenizer bytes and seven-special-token contract."""

import hashlib
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from src.special_tokens import SPECIAL_TOKEN_IDS, assert_special_token_ids

TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "tokenizer"
    / "releases"
    / "tokenizer_v1"
    / "tokenizer.json"
)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    assert TOKENIZER_PATH.is_file(), f"canonical tokenizer missing: {TOKENIZER_PATH}"
    assert hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest() == (
        "d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce"
    )
    assert_special_token_ids(str(TOKENIZER_PATH))
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_vocab_fits_uint16(tok):
    # packed pretrain shards default to uint16
    assert tok.get_vocab_size() < 65536


def test_special_token_ids_are_stable(tok):
    for token, tid in SPECIAL_TOKEN_IDS.items():
        assert tok.id_to_token(tid) == token


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog.",
        "def add(a, b):\n    return a + b\n",
        "Numbers: 3.14159 and 42 — and unicode: café, naïve.",
    ],
)
def test_roundtrip_preserves_exact_text(tok, text):
    ids = tok.encode(text).ids
    decoded = tok.decode(ids)
    assert decoded == text


def test_encode_is_deterministic(tok):
    text = "deterministic encoding check"
    assert tok.encode(text).ids == tok.encode(text).ids
