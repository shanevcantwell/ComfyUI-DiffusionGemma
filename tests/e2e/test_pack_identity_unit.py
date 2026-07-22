"""Unit tests for `tests/e2e/conftest.py`'s pack-identity gate
(`_pack_identity()` / `_git_head_sha()`) — issue #122.

These run in the **default fast suite** (no `e2e`/`live` marker), same
discipline as `test_driver_unit.py`: the gate LOGIC is exercisable with
nothing but `tmp_path`-built fake layouts and two tiny real git repos —
none of the three operator-scheduled preconditions (issue #59 §5) are
needed to prove the gate itself is correct.

Coverage, per issue #122's acceptance criteria:
- symlink case (resolves into the source root — accept)
- symlink case (resolves elsewhere — reject, reason names both paths)
- matching-SHA real-checkout case (two independent git repos, same HEAD — accept)
- mismatched-SHA real-checkout case (two independent git repos, different
  HEAD — reject, reason names both SHAs)
- no-`.git` real-directory case (reject, reason names the missing checkout)
- missing deployed-path case (reject)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.conftest import _git_head_sha, _pack_identity, _pack_loadable


def _init_git_repo(repo_dir: Path) -> str:
    """Creates a real, minimal git repo at `repo_dir` with one commit and
    returns its HEAD SHA. Uses subprocess `git` directly (not a mock) —
    the gate's own implementation shells out to `git`, so a real repo is
    the only fixture that actually proves the subprocess plumbing works."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }

    def _run(*args: str) -> subprocess.CompletedProcess:
        import os

        env = os.environ.copy()
        env.update(env_overrides)
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    _run("init", "--initial-branch=main")
    (repo_dir / "marker.txt").write_text("content\n", encoding="utf-8")
    _run("add", "marker.txt")
    _run("commit", "-m", "initial commit")
    sha = _run("rev-parse", "HEAD").stdout.strip()
    return sha


# --- _git_head_sha() ---------------------------------------------------


def test_git_head_sha_returns_sha_for_real_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    expected_sha = _init_git_repo(repo)
    assert _git_head_sha(repo) == expected_sha


def test_git_head_sha_returns_none_when_no_dot_git(tmp_path: Path) -> None:
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    assert _git_head_sha(plain_dir) is None


def test_git_head_sha_returns_none_for_nonexistent_dir(tmp_path: Path) -> None:
    assert _git_head_sha(tmp_path / "does-not-exist") is None


def test_git_head_sha_returns_none_when_git_command_fails(tmp_path: Path) -> None:
    """A `.git` entry exists but `git rev-parse HEAD` fails (e.g. a corrupt
    or incomplete repo, or `.git` is a stray file/directory git doesn't
    recognize) — must degrade to None, not raise."""
    repo_dir = tmp_path / "corrupt_repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()  # present but not a real git repo — git will error

    assert _git_head_sha(repo_dir) is None


def test_git_head_sha_returns_none_when_git_binary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If `git` itself isn't invocable (not on PATH, permission error, etc.)
    `subprocess.run` raises OSError — must degrade to None, not propagate."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    def _raise_oserror(*args: object, **kwargs: object) -> None:
        raise OSError("git: command not found")

    monkeypatch.setattr(subprocess, "run", _raise_oserror)
    assert _git_head_sha(repo) is None


# --- _pack_identity(): symlink cases -------------------------------------


def test_pack_identity_accepts_symlink_resolving_to_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    deployed_path = tmp_path / "deployed_link"
    deployed_path.symlink_to(source_root, target_is_directory=True)

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is True
    assert reason is None


def test_pack_identity_rejects_symlink_resolving_elsewhere(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    deployed_path = tmp_path / "deployed_link"
    deployed_path.symlink_to(other, target_is_directory=True)

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is False
    assert reason is not None
    assert str(other.resolve()) in reason
    assert str(source_root.resolve()) in reason


def test_pack_identity_rejects_dangling_symlink(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    deployed_path = tmp_path / "dangling_link"
    deployed_path.symlink_to(tmp_path / "does-not-exist")

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is False
    assert reason is not None


# --- _pack_identity(): real-checkout cases -------------------------------


def test_pack_identity_accepts_real_checkout_with_matching_sha(tmp_path: Path) -> None:
    """The issue #122 fix: a real, independent git checkout (not a symlink)
    whose HEAD matches the source clone's HEAD must be accepted — this is
    the exact deployed-topology shape the operator's infra runs today."""
    source_root = tmp_path / "source"
    _init_git_repo(source_root)

    deployed_path = tmp_path / "deployed_checkout"
    subprocess.run(
        ["git", "clone", str(source_root), str(deployed_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is True
    assert reason is None


def test_pack_identity_rejects_real_checkout_with_mismatched_sha(tmp_path: Path) -> None:
    """A stale independent checkout (different HEAD) must SKIP-worthy-reject
    (return False with a reason), not silently pass — and the reason must
    name both SHAs so the skip message is actionable."""
    source_root = tmp_path / "source"
    source_sha = _init_git_repo(source_root)

    deployed_path = tmp_path / "deployed_checkout"
    subprocess.run(
        ["git", "clone", str(source_root), str(deployed_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    # Advance the source clone by one more commit so the two diverge.
    import os

    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
    )
    (source_root / "marker2.txt").write_text("more content\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker2.txt"], cwd=str(source_root), env=env, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "second commit"],
        cwd=str(source_root),
        env=env,
        check=True,
        capture_output=True,
    )
    new_source_sha = _git_head_sha(source_root)
    assert new_source_sha is not None
    assert new_source_sha != source_sha

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is False
    assert reason is not None
    assert source_sha in reason  # deployed checkout's (stale) SHA
    assert new_source_sha in reason  # current source clone's SHA
    assert "stale deploy" in reason


def test_pack_identity_rejects_real_directory_without_git(tmp_path: Path) -> None:
    """A real directory (not a symlink) with no `.git` at all — e.g. a bare
    copy or an unrelated directory — cannot be verified and must be
    rejected with a reason naming the missing checkout, never raise."""
    source_root = tmp_path / "source"
    _init_git_repo(source_root)

    deployed_path = tmp_path / "deployed_no_git"
    deployed_path.mkdir()
    (deployed_path / "some_file.py").write_text("# not a git repo\n", encoding="utf-8")

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is False
    assert reason is not None
    assert "no usable" in reason


def test_pack_identity_rejects_missing_deployed_path(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _init_git_repo(source_root)
    deployed_path = tmp_path / "does-not-exist-at-all"

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is False
    assert reason is not None
    assert str(deployed_path) in reason


def test_pack_identity_rejects_when_source_root_has_no_git(tmp_path: Path) -> None:
    """Degenerate case: the deployed path is a real checkout with a valid
    HEAD, but the source clone itself has no usable `.git` to compare
    against — should reject gracefully, never raise."""
    source_root = tmp_path / "source_no_git"
    source_root.mkdir()

    deployed_path = tmp_path / "deployed_checkout"
    _init_git_repo(deployed_path)

    ok, reason = _pack_identity(deployed_path=deployed_path, source_root=source_root)
    assert ok is False
    assert reason is not None
    assert "source clone" in reason


# --- _pack_loadable(): boolean wrapper ------------------------------------


def test_pack_loadable_wraps_pack_identity_boolean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_pack_loadable()` takes no args (it's the module-level convenience
    wrapper used by `_skip_reason()`), so exercise it via monkeypatched
    module globals rather than passing overrides directly."""
    import tests.e2e.conftest as conftest_module

    source_root = tmp_path / "source"
    source_root.mkdir()
    deployed_path = tmp_path / "deployed_link"
    deployed_path.symlink_to(source_root, target_is_directory=True)

    monkeypatch.setattr(conftest_module, "CUSTOM_NODES_LINK", deployed_path)
    monkeypatch.setattr(conftest_module, "PACK_ROOT", source_root)

    assert _pack_loadable() is True
