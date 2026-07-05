"""Pin the thinking-toggle mechanism against the REAL chat template (finding
from P2 review, 2026-07-05).

`run_diffusion(thinking=True)` injects `<|think|>` as a system-message
content because the pipeline's `_prepare_inputs` never forwards
`enable_thinking` to `apply_chat_template` — message injection is the only
viable path through `pipeline.__call__`. But the template renders system
content through `| trim`, which eats the newline the native
`enable_thinking=True` path places after `<|think|>` — so the two renders
are NOT token-identical, and this test pins the exact delta instead of
letting prose claim identity.

Tokenizer-only: loads `AutoProcessor` from the local HF cache
(`local_files_only=True` — never the network, never the 26B weights).
Skip-gated when the cached checkpoint is unreachable.
"""
from __future__ import annotations

import pytest

from dgemma.loop import THINK_TOKEN
from dgemma.model import DEFAULT_REPO_ID

NEWLINE_ID = 107  # ordinary vocab id for "\n" (issue #8 grounding)


def _load_processor():
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(DEFAULT_REPO_ID, local_files_only=True)
    except Exception:
        return None


processor = _load_processor()

pytestmark = pytest.mark.skipif(
    processor is None,
    reason=f"{DEFAULT_REPO_ID} tokenizer/processor not in the local HF cache.",
)


def _render(messages, **kwargs) -> list[int]:
    ids = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    # apply_chat_template(tokenize=True) returns a batch; flatten to one row.
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def test_injected_think_message_differs_from_native_by_exactly_one_newline():
    """The load-bearing pin: the injected-message render equals the native
    `enable_thinking=True` render with exactly ONE token removed — the
    newline (id 107) immediately after `<|think|>`, eaten by the template's
    `| trim` on system content. If a template update ever changes this
    delta (or closes it), this test is the tripwire."""
    native = _render([{"role": "user", "content": "hi"}], enable_thinking=True)
    injected = _render(
        [{"role": "system", "content": THINK_TOKEN}, {"role": "user", "content": "hi"}]
    )

    think_id = processor.tokenizer.convert_tokens_to_ids(THINK_TOKEN)
    assert think_id in native and think_id in injected  # the toggle lands on both paths

    i = native.index(think_id)
    assert native[i + 1] == NEWLINE_ID  # the native path's newline after <|think|>
    # Removing exactly that one token from the native render yields the
    # injected render — one-token delta, nothing else differs.
    assert native[: i + 1] + native[i + 2 :] == injected


def test_trailing_newline_in_injected_content_is_trimmed_so_parity_is_unreachable():
    """Negative control for the 'can we reach parity?' question: appending
    the missing newline to the injected content does NOT close the delta —
    the template's `| trim` eats it, so `"<|think|>\\n"` renders identically
    to `"<|think|>"`. Documents that the one-token gap is structural, not a
    content-choice mistake."""
    injected_plain = _render(
        [{"role": "system", "content": THINK_TOKEN}, {"role": "user", "content": "hi"}]
    )
    injected_with_newline = _render(
        [{"role": "system", "content": THINK_TOKEN + "\n"}, {"role": "user", "content": "hi"}]
    )

    assert injected_with_newline == injected_plain
