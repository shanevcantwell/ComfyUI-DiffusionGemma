"""Belt-and-braces post-install script (issue #147).

ComfyUI's built-in Extensions flow (ComfyUI-Manager merged into core) runs a
registered pack's `requirements.txt` per-line via `sys.executable -m pip`,
then — if present — executes this file as `[sys.executable, "install.py"]`
(mechanism traced in issue #147's Extensions-flow research comment). Manager
itself already ran the requirements step by the time this file executes; this
script is a second, self-contained pass over the SAME requirements.txt that:

1. Logs enough about the target interpreter and already-installed versions
   to diagnose a broken install after the fact (issue #147's operator field
   report: a fresh install arrived with `diffusers` absent and `transformers`
   at ComfyUI-core's bundled version, not this pack's pin — evidence the
   requirements step never touched the right interpreter).
2. Re-checks each requirement against what's actually importable in THIS
   interpreter and installs anything still missing/mismatched — a second bite
   if Manager's own pass silently failed, was skipped, or landed in the wrong
   environment.

This is diagnostic insurance, not a claimed root-cause fix — see issue #147
for the parked root-cause question (mechanism on the affected box is still
unpinned). It converts the next broken install into a diagnosis.

MECHANISM PINNED (issue #147, operator field test, StabilityMatrix-packaged
ComfyUI, Python 3.12): two independent Manager/environment-side defects were
observed together. (1) Manager's own dependency step blacklist-skips
`transformers` as a protected package — `[ComfyUI-Manager] skip black listed
pip installation: 'transformers==5.13.0'` — so Manager's pass never applies
our pin; THIS script installs it anyway, deliberately, because the pack
requires the pin — that divergence from Manager's own choice is intentional,
and it is loudly logged (`is_satisfied` catches the mismatch, `run()` prints
`MISSING/MISMATCHED` and installs). (2) On a `uv`-managed venv (observed:
`.../venv/bin/python3 -m uv pip install diffusers>=0.39.0`), `uv pip install`
demanded a `venv/uv-build-constraints.txt` that does not exist in a
StabilityMatrix-provisioned venv, and Manager's uv-driven step failed outright
(`error: File not found`). This script never invokes `uv` with a constraints
file — see `resolve_installer()` / `uv_install()` below — specifically to
avoid inheriting that failure mode.

Import-safety: this module does no work at import time — everything lives
under `if __name__ == "__main__":` / functions called from `main()` — so it
is always safe for the test suite (or anything else) to import without
triggering pip subprocesses or network access.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HEADER = "[ComfyUI-DiffusionGemma install]"

REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# Packages worth logging even when they're not in requirements.txt (core
# ComfyUI may already provide a version; auto-round is optional/INT4-only).
# Kept in one place so the "what versions are present" log and the
# requirements-parse below can't drift on naming.
DIAGNOSTIC_PACKAGES = ("transformers", "diffusers", "accelerate", "auto-round")

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _bare_name(spec: str) -> str:
    """Strip a PEP 508-ish dependency spec down to its bare distribution
    name — handles version specifiers (==, >=, ...), extras, and markers.
    Mirrors `tests/test_requirements_sync.py`'s `_bare_name` (kept
    independent on purpose: install.py must stay a standalone, dependency-free
    script runnable before anything else in the pack is importable)."""
    spec = spec.split(";", 1)[0].strip()
    match = _NAME_RE.match(spec)
    if not match:
        raise ValueError(f"Could not parse a distribution name from spec: {spec!r}")
    return match.group(1)


def parse_requirements(text: str) -> list[str]:
    """Parse requirements.txt content into a list of verbatim spec lines,
    dropping blank lines and full-line '#' comments. requirements.txt is
    already filtered to exclude core-provided packages (see its own header
    comment and tests/test_requirements_sync.py) — this function does no
    filtering of its own, it only consumes what's there."""
    specs = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        specs.append(stripped)
    return specs


def installed_version(dist_name: str) -> str | None:
    """Return the installed version of `dist_name` via importlib.metadata,
    or None if it isn't installed. Deliberately metadata-only — never imports
    the package itself (importing torch/transformers at install time is slow
    and can fail for reasons unrelated to whether the version is right)."""
    from importlib import metadata

    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def is_satisfied(spec: str) -> tuple[bool, str | None]:
    """Check whether `spec` (a requirements.txt line, e.g. 'transformers==5.13.0'
    or 'accelerate') is already satisfied in this interpreter.

    Returns (satisfied, installed_version_or_none). Uses importlib.metadata
    for presence/version and packaging.specifiers for the comparison; if
    `packaging` isn't importable (unexpected — it's a pip/transformers dep,
    but this script must not hard-crash on that), falls back to "attempt
    install" (i.e. reports unsatisfied) rather than guessing.
    """
    name = _bare_name(spec)
    version = installed_version(name)
    if version is None:
        return False, None

    # Bare name, no version specifier (e.g. "accelerate") — presence is enough.
    specifier_part = spec.split(";", 1)[0].strip()[len(name):].strip()
    if not specifier_part:
        return True, version

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        satisfied = Version(version) in SpecifierSet(specifier_part)
        return satisfied, version
    except Exception:
        # packaging unavailable or spec unparseable: don't guess — treat as
        # unsatisfied so the install step gets a chance to fix it.
        return False, version


def log(msg: str = "") -> None:
    print(f"{HEADER} {msg}" if msg else "")


def print_environment_report() -> None:
    log(f"python executable: {sys.executable}")
    log(f"python version: {sys.version.split()[0]}")
    for name in DIAGNOSTIC_PACKAGES:
        version = installed_version(name)
        log(f"  {name}: {version if version is not None else '(not installed)'}")


def _module_available(module: str) -> bool:
    """Probe whether `sys.executable -m <module> --version` resolves.

    Quick subprocess check, not an import in THIS process — some venvs
    (StabilityMatrix's `uv`-provisioned ones, per issue #147's field report)
    omit the `pip` module entirely from the venv's site-packages, so
    `import pip` from inside this process is not a reliable proxy for what a
    freshly spawned `sys.executable -m pip` subprocess would see."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", module, "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except OSError:
        return False


def _pip_available() -> bool:
    return _module_available("pip")


def _uv_available() -> bool:
    return _module_available("uv")


def resolve_installer() -> str | None:
    """Pick the installer this run will use: 'pip' if `sys.executable -m pip`
    resolves; else 'uv' if `sys.executable -m uv` resolves (issue #147's
    pip-absent uv-venv fallback); else None if neither is usable. Cheap and
    called once per run — logged in the environment-report header so a
    diagnosing reader knows immediately which path was taken."""
    if _pip_available():
        return "pip"
    if _uv_available():
        return "uv"
    return None


def pip_install(spec: str) -> subprocess.CompletedProcess:
    """Install a single requirement spec into THIS interpreter, mirroring
    Manager's own per-line (not batched `-r`) install choice."""
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", spec],
        capture_output=True,
        text=True,
    )


def uv_install(spec: str) -> subprocess.CompletedProcess:
    """Install a single requirement spec via `uv pip install`, for venvs
    where `sys.executable -m pip` is unavailable (issue #147: StabilityMatrix
    `uv`-provisioned venv, no `pip` module present).

    Deliberately passes NO constraints-file argument. The pinned failure on
    the affected box was Manager's own uv-driven step demanding
    `venv/uv-build-constraints.txt` — a file that does not exist in that
    venv — and aborting with `error: File not found`. This invocation must
    not reproduce that: no `--build-constraints` / `-c` / config lookup that
    would pull one in, just `uv pip install <spec>`."""
    return subprocess.run(
        [sys.executable, "-m", "uv", "pip", "install", spec],
        capture_output=True,
        text=True,
    )


def install_spec(spec: str, installer: str) -> subprocess.CompletedProcess:
    """Dispatch a single spec to the resolved installer."""
    if installer == "uv":
        return uv_install(spec)
    return pip_install(spec)


def _manual_fallback_command(installer: str = "pip") -> str:
    if installer == "uv":
        return f'"{sys.executable}" -m uv pip install -r requirements.txt'
    return f'"{sys.executable}" -m pip install -r requirements.txt'


def _both_manual_fallback_commands() -> str:
    """Both manual remedies, for the case where neither installer resolved —
    the operator's environment is unknown at that point, so name both."""
    return (
        f"  pip:  {_manual_fallback_command('pip')}\n"
        f"  uv:   {_manual_fallback_command('uv')}"
    )


def print_failure_block(failed_specs: list[str], installer: str = "pip") -> None:
    log()
    log("!" * 70)
    log("INSTALL FAILED for the following requirement(s):")
    for spec in failed_specs:
        log(f"  - {spec}")
    log()
    log("Manual fix — run this in ComfyUI's own Python environment:")
    log(f"  {_manual_fallback_command(installer)}")
    log()
    log(
        "Until this is resolved, ComfyUI-DiffusionGemma will fail at import "
        "with a version-guard message (see dgemma/model.py's "
        "_check_transformers_version / dgemma/loop.py's "
        "_check_diffusers_version) rather than silently misbehaving."
    )
    log("!" * 70)


def print_no_installer_block() -> None:
    """Neither `pip` nor `uv` resolved in this interpreter — nothing this
    script can do automatically. Name both attempts and both manual
    remedies so a reader diagnosing the log has everything in one place."""
    log()
    log("!" * 70)
    log("INSTALL FAILED: no usable installer found in this interpreter.")
    log(f"  tried: {sys.executable} -m pip --version — not available")
    log(f"  tried: {sys.executable} -m uv --version — not available")
    log()
    log("Manual fix — run ONE of these in ComfyUI's own Python environment:")
    log(_both_manual_fallback_commands())
    log("!" * 70)


def run() -> int:
    """Do the install-check-and-fix pass. Returns a process exit code
    (0 = success/no-op, nonzero = failure)."""
    print_environment_report()

    if not REQUIREMENTS_PATH.exists():
        log(f"no requirements.txt found at {REQUIREMENTS_PATH} — nothing to do.")
        return 0

    specs = parse_requirements(REQUIREMENTS_PATH.read_text())
    if not specs:
        log("requirements.txt has no requirement lines — nothing to do.")
        return 0

    installer = resolve_installer()
    if installer == "pip":
        log("installer: pip")
    elif installer == "uv":
        log("installer: uv (pip module absent)")
    else:
        log("installer: none (neither pip nor uv module available)")
        print_no_installer_block()
        return 1

    failed_specs: list[str] = []
    for spec in specs:
        satisfied, version = is_satisfied(spec)
        if satisfied:
            log(f"OK  {spec} (installed: {version})")
            continue

        log(f"MISSING/MISMATCHED  {spec} (found: {version if version is not None else 'not installed'}) — installing...")
        result = install_spec(spec, installer)
        if result.returncode == 0:
            log(f"OK  {spec} (installed by this script)")
        else:
            log(f"FAILED  {spec} — {installer} exited {result.returncode}")
            if result.stderr:
                log(f"  stderr (tail): {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ''}")
            failed_specs.append(spec)

    if failed_specs:
        print_failure_block(failed_specs, installer)
        return 1

    log("all requirements satisfied.")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # last-resort: never let this script itself crash uninformatively
        log(f"install.py raised an unexpected error: {exc!r}")
        log("Manual fix — run ONE of these in ComfyUI's own Python environment:")
        log(_both_manual_fallback_commands())
        return 1


if __name__ == "__main__":
    sys.exit(main())
