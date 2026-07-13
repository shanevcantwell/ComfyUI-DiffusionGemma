"""Session-scoped headless-ComfyUI lifecycle for the black-box E2E battery
(ADR-CDG-013, issue #59 phase E0).

Independence discipline (the architectural commitment this whole tier
exists to honor): this module imports **only** stdlib +
`requests`/`websocket-client` — never `dgemma`/`surfaces`/`consumers`. The
enforcement surface for that invariant is `test_e2e_import_guard.py`, not
this docstring; keep it that way when editing this file.

Three operator-scheduled preconditions are named in issue #59 §5 / ADR-CDG-013
and are NOT satisfied as of this phase:
1. The `custom_nodes` symlink at `/srv/dev/ComfyUI/custom_nodes/
   ComfyUI-DiffusionGemma` is dead (points at a repo that no longer exists) —
   repointing it touches the shared ComfyUI install and is operator-gated.
2. The GPU must be free of llauncher's resident model-server tenants
   (`localhost:8081/8082`) — operator-coordinated infra.
3. The real weights (`google/diffusiongemma-26B-A4B-it`, ~53.6GB) must be
   HF-cached.

Every fixture here therefore SKIPs (never errors) when any precondition is
unmet, so `pytest -m e2e` is mergeable and green today with the live tier
fully skip-gated — the same discipline `tests/conftest.py`'s
`require_live_weights` already established for the `live` tier.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

# --- Precondition gates (SKIP, never error) ---------------------------------

# Overridable so this isn't hardcoded to one box's layout (the E2E-run box
# happens to be this one, per ADR-CDG-013/issue #59's grounding pass).
COMFYUI_ROOT = Path(os.environ.get("DGEMMA_E2E_COMFYUI_ROOT", "/srv/dev/ComfyUI"))
COMFYUI_VENV_PYTHON = COMFYUI_ROOT / ".venv" / "bin" / "python"
PACK_ROOT = Path(__file__).resolve().parent.parent.parent
CUSTOM_NODES_LINK = COMFYUI_ROOT / "custom_nodes" / "ComfyUI-DiffusionGemma"

E2E_HOST = "127.0.0.1"
E2E_PORT = int(os.environ.get("DGEMMA_E2E_PORT", "8199"))  # isolated from the operator's interactive 8188
READINESS_TIMEOUT_S = 120.0
READINESS_POLL_INTERVAL_S = 0.5


def _weights_cached(repo_id: str = "google/diffusiongemma-26B-A4B-it") -> bool:
    """Same check as `tests/conftest.py:weights_cached`, reimplemented here
    rather than imported — the E2E tier's independence invariant forbids
    importing anything from this repo's own packages, and `tests/conftest.py`
    lives outside `tests/e2e/`'s own collection root but importing *test*
    helpers is still a coupling the import-guard test would have to special-
    case; a few duplicated lines are cheaper than a carve-out in the guard."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False
    return any(repo.repo_id == repo_id for repo in cache_info.repos)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _comfyui_installed() -> bool:
    return COMFYUI_ROOT.is_dir() and (COMFYUI_ROOT / "main.py").is_file() and COMFYUI_VENV_PYTHON.is_file()


def _pack_loadable() -> bool:
    """The custom_nodes symlink must resolve to *this* pack root — the named
    operator-gated precondition (issue #59 §5 precondition 1). A dead or
    missing symlink means ComfyUI will not load the pack at all."""
    if not CUSTOM_NODES_LINK.is_symlink() and not CUSTOM_NODES_LINK.exists():
        return False
    try:
        return CUSTOM_NODES_LINK.resolve() == PACK_ROOT.resolve() and CUSTOM_NODES_LINK.exists()
    except OSError:
        return False


def _free_port(port: int) -> bool:
    """Best-effort check that nothing is already listening on the battery's
    isolated port — a stale prior run's process would otherwise make the
    readiness poll pass against the WRONG server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((E2E_HOST, port)) != 0


def _skip_reason() -> str | None:
    """Returns a human-readable skip reason, or None if every precondition
    holds. Centralized so every fixture that depends on a live server skips
    with the same message instead of re-deriving the check."""
    if not _comfyui_installed():
        return (
            f"ComfyUI install not found/usable at {COMFYUI_ROOT} "
            "(expected main.py + .venv/bin/python) — skipping e2e battery."
        )
    if not _pack_loadable():
        return (
            f"custom_nodes symlink {CUSTOM_NODES_LINK} does not resolve to this pack "
            f"({PACK_ROOT}) — operator-gated precondition (issue #59 §5.2) unmet; "
            "skipping e2e battery."
        )
    if not _weights_cached():
        return (
            "google/diffusiongemma-26B-A4B-it not present in the local HF cache "
            "(~53.6GB) — skipping e2e battery."
        )
    if not _cuda_available():
        return "No CUDA device available — skipping e2e battery."
    return None


class ComfyUIServer(NamedTuple):
    base_url: str
    client_id: str
    ws_url: str


@pytest.fixture(scope="session")
def e2e_preconditions() -> None:
    """Depend on this from any `e2e` test that does NOT need the live
    server itself (e.g. a pure workflow-JSON shape check) but still needs
    the three named preconditions honored before running at all."""
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def comfyui_server(e2e_preconditions: None) -> Iterator[ComfyUIServer]:
    """Launch ComfyUI headless as a subprocess for the whole battery
    (ADR-CDG-013 Decision 2/3): one process, one model load, amortized
    across every scenario. Polls `/object_info` for readiness (bounded),
    yields the base URL + a websocket client_id, and SIGTERMs + reaps on
    teardown — so a battery run cannot corrupt or be corrupted by the
    operator's interactive instance (port isolation, never 8188)."""
    if not _free_port(E2E_PORT):
        pytest.skip(
            f"Port {E2E_PORT} is already in use — a stale e2e server or the operator's "
            "own instance may be running there; skipping e2e battery rather than "
            "risk asserting against the wrong process."
        )

    import requests

    output_dir = Path(os.environ.get("DGEMMA_E2E_OUTPUT_DIR", "/tmp/dgemma-e2e-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # ADR-CDG-013 §3 / #59-§3: subprocess-merged coverage. Set unconditionally —
    # harmless (a no-op) if `coverage`/`sitecustomize.py` aren't resolvable, and
    # this is the one place the server subprocess's env is constructed.
    coverage_rc = PACK_ROOT / "pyproject.toml"
    env.setdefault("COVERAGE_PROCESS_START", str(coverage_rc))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACK_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    proc = subprocess.Popen(
        [
            str(COMFYUI_VENV_PYTHON),
            str(COMFYUI_ROOT / "main.py"),
            "--listen", E2E_HOST,
            "--port", str(E2E_PORT),
            "--output-directory", str(output_dir),
            "--disable-auto-launch",
        ],
        cwd=str(COMFYUI_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://{E2E_HOST}:{E2E_PORT}"
    deadline = time.monotonic() + READINESS_TIMEOUT_S
    ready = False
    last_error: Exception | None = None
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                pytest.fail(
                    f"ComfyUI subprocess exited early (code {proc.returncode}) "
                    f"before becoming ready:\n{output}"
                )
            try:
                resp = requests.get(f"{base_url}/object_info", timeout=5)
                if resp.status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException as exc:
                last_error = exc
            time.sleep(READINESS_POLL_INTERVAL_S)

        if not ready:
            pytest.fail(
                f"ComfyUI did not become ready at {base_url}/object_info within "
                f"{READINESS_TIMEOUT_S}s (last error: {last_error})"
            )

        client_id = "dgemma-e2e-battery"
        ws_url = f"ws://{E2E_HOST}:{E2E_PORT}/ws?clientId={client_id}"
        yield ComfyUIServer(base_url=base_url, client_id=client_id, ws_url=ws_url)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
