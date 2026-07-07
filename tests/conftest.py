"""Shared fixtures for the `live` half of the suite (see `tests/README.md`
for the `pytest` vs `pytest -m live` convention).

One gate, not two: `test_integration.py` and `test_live_seams.py` both need
"real weights cached + a CUDA device, else skip gracefully" — kept here so
that check exists exactly once instead of being re-derived per file.
`require_live_weights` SKIPS (never errors) when either precondition is
missing, so `pytest -m live` on a box without the checkpoint/GPU reports
skips, not failures.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.model import DEFAULT_REPO_ID


def weights_cached(repo_id: str = DEFAULT_REPO_ID) -> bool:
    """True iff `repo_id` is present in the local Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False
    return any(repo.repo_id == repo_id for repo in cache_info.repos)


@pytest.fixture(scope="session")
def require_live_weights():
    """Depend on this fixture from any `@pytest.mark.live` test that needs
    the real checkpoint + a CUDA device. Session-scoped: the two checks are
    cheap and their result can't change mid-run, so there is no reason to
    repeat them per test or per module.
    """
    if not weights_cached():
        pytest.skip(
            f"{DEFAULT_REPO_ID} not present in the local HF cache (~53.6GB) — "
            "skipping live test."
        )
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available — skipping live test.")
