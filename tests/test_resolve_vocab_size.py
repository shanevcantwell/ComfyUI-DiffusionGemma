"""`dgemma.loop.resolve_vocab_size` unit tests (issue #64 §3.4) — the
`vocab_size` resolver `run_diffusion` feeds to `validate_constraints`'s C3
check. Pure function of a processor/tokenizer duck-type; no fake pipeline
needed.
"""
from __future__ import annotations

from dgemma.loop import resolve_vocab_size


class _TokenizerWithLen:
    def __len__(self):
        return 32_000


class _TokenizerWithVocabSizeOnly:
    vocab_size = 50_000


class _TokenizerWithNeither:
    pass


class _ProcessorWrapping:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_prefers_len_tokenizer_when_available():
    assert resolve_vocab_size(_ProcessorWrapping(_TokenizerWithLen())) == 32_000


def test_falls_back_to_vocab_size_attribute_when_len_unavailable():
    assert resolve_vocab_size(_ProcessorWrapping(_TokenizerWithVocabSizeOnly())) == 50_000


def test_returns_none_when_neither_is_available():
    assert resolve_vocab_size(_ProcessorWrapping(_TokenizerWithNeither())) is None


def test_bare_tokenizer_without_dot_tokenizer_attribute():
    """Mirrors `resolve_thought_channel_ids`'s own fallback: a bare tokenizer
    handed in directly (no `.tokenizer` wrapper) still works."""
    assert resolve_vocab_size(_TokenizerWithLen()) == 32_000


def test_len_taking_priority_even_when_vocab_size_also_present():
    class _Both:
        vocab_size = 999

        def __len__(self):
            return 111

    assert resolve_vocab_size(_ProcessorWrapping(_Both())) == 111
