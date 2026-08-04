"""`tools/leak_guard.py` — generic-pattern topology leak guard (ADR-CDG-022
Decision 4b, issue #237).

Same import pattern as `test_q2_preflight.py`: the tool lives outside
`dgemma`/`surfaces`/`consumers` (a standalone dev tool), imported directly
off `tools/` via `sys.path.insert`.

All positive-detection cases run against **tmp fixture files in a throwaway
git repo** — never the live tree — so the suite is independent of whatever
the live tree currently contains. The synthetic example strings that seed
those tmp fixtures (e.g. `192.168.1.1`, `/home/alice`, `/srv/dev/somebox`)
are still Python literals inside *this* tracked file, though, and are
structurally indistinguishable from a real leak to the generic-pattern
scan — so this file's path is itself allowlisted in `.leak-guard-allow`
(`path:tests/test_leak_guard.py`), not because it's exempt from scrutiny
but because everything in it is a known-synthetic pattern-match example.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
import leak_guard  # noqa: E402  (path insert must precede this)


def _init_fixture_repo(tmp_path: Path) -> Path:
    """A throwaway git repo so `leak_guard.tracked_files()` (which shells
    out to `git ls-files`) has something real to walk, isolated from the
    actual ComfyUI-DiffusionGemma tree."""
    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    return repo


def _write_and_track(repo: Path, rel_path: str, content: str) -> None:
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    subprocess.run(["git", "add", rel_path], cwd=repo, check=True)


def _run_guard(repo: Path) -> tuple[int, list[leak_guard.Hit]]:
    """Mirrors `leak_guard.main()`'s scan-and-decide logic, but returns the
    hit list too so tests can assert on pattern_class specifically."""
    allowlist = leak_guard.load_allowlist(repo / ".leak-guard-allow")
    files = leak_guard.tracked_files(repo)
    hits: list[leak_guard.Hit] = []
    for rel_path in files:
        hits.extend(leak_guard.scan_file(repo, rel_path))
    unallowlisted = [h for h in hits if not allowlist.covers(h)]
    return (1 if unallowlisted else 0, hits)


# ---------------------------------------------------------------------------
# Positive detection — one test per pattern class
# ---------------------------------------------------------------------------


class TestPatternClasses:
    def test_detects_192_168_private_ip(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "notes.md", "the host is at 192.168.1.42\n")

        code, hits = _run_guard(repo)

        assert code == 1
        assert any(h.pattern_class == "private-ip-192.168" for h in hits)
        assert any(h.matched_text == "192.168.1.42" for h in hits)

    def test_detects_10_x_private_ip(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "notes.md", "internal box: 10.0.5.200\n")

        code, hits = _run_guard(repo)

        assert code == 1
        assert any(h.pattern_class == "private-ip-10" for h in hits)
        assert any(h.matched_text == "10.0.5.200" for h in hits)

    def test_detects_172_16_31_private_ip(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "notes.md", "vpn range 172.20.3.4 in use\n")

        code, hits = _run_guard(repo)

        assert code == 1
        assert any(h.pattern_class == "private-ip-172.16-31" for h in hits)
        assert any(h.matched_text == "172.20.3.4" for h in hits)

    def test_172_99_is_not_a_false_positive(self, tmp_path):
        """172.16-31 is the RFC 1918 block; 172.99.x.x is public-range-shaped
        and must NOT be flagged as a private IP (CIDR-aware, not a naive
        string-prefix match on '172.')."""
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "notes.md", "public endpoint 172.99.1.1\n")

        code, hits = _run_guard(repo)

        assert code == 0
        assert not any(h.pattern_class.startswith("private-ip") for h in hits)

    def test_172_15_and_172_32_are_not_false_positives(self, tmp_path):
        """Boundary check just outside the 16-31 range on both sides."""
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(
            repo, "notes.md", "edges: 172.15.0.1 and 172.32.0.1\n"
        )

        code, hits = _run_guard(repo)

        assert code == 0
        assert not any(h.pattern_class.startswith("private-ip") for h in hits)

    def test_detects_home_path(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "script.sh", "BIN=/home/alice/.local/bin/tool\n")

        code, hits = _run_guard(repo)

        assert code == 1
        assert any(h.pattern_class == "home-path" for h in hits)
        assert any(h.matched_text == "/home/alice" for h in hits)

    def test_detects_srv_dev_path(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "script.sh", "ROOT=/srv/dev/somebox/repo\n")

        code, hits = _run_guard(repo)

        assert code == 1
        assert any(h.pattern_class == "srv-dev-path" for h in hits)
        assert any(h.matched_text == "/srv/dev/somebox" for h in hits)


# ---------------------------------------------------------------------------
# Allowlist mechanism
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_path_prefix_allowlist_suppresses_hit(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(
            repo,
            "docs/experiments/deferred/notes.md",
            "vpn range 192.168.5.5 in use\n",
        )
        _write_and_track(
            repo,
            ".leak-guard-allow",
            "path:docs/experiments/deferred/\n",
        )

        code, _hits = _run_guard(repo)

        assert code == 0

    def test_placeholder_allowlist_suppresses_hit(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(
            repo, "tests/fixture.py", 'BIN = "/home/user/.local/bin/uv"\n'
        )
        _write_and_track(repo, ".leak-guard-allow", "placeholder:/home/user\n")

        code, _hits = _run_guard(repo)

        assert code == 0

    def test_allowlist_does_not_suppress_unrelated_hits(self, tmp_path):
        """A path-prefix allowlist entry only covers files under that
        prefix — a hit elsewhere in the tree still fails the run."""
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(
            repo,
            "docs/experiments/deferred/notes.md",
            "vpn range 192.168.5.5 in use\n",
        )
        _write_and_track(
            repo, "README.md", "internal host 192.168.9.9\n"
        )
        _write_and_track(
            repo,
            ".leak-guard-allow",
            "path:docs/experiments/deferred/\n",
        )

        code, hits = _run_guard(repo)

        assert code == 1
        assert any(h.file == "README.md" for h in hits)


# ---------------------------------------------------------------------------
# Clean tree
# ---------------------------------------------------------------------------


class TestCleanTree:
    def test_clean_tree_passes(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(
            repo,
            "README.md",
            "This project has no topology-adjacent residue in it.\n"
            "Public URLs like https://github.com/example/repo are fine.\n",
        )

        code, hits = _run_guard(repo)

        assert code == 0
        assert hits == []

    def test_clean_single_file_passes(self, tmp_path):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "notes.md", "nothing sensitive here\n")

        allowlist = leak_guard.load_allowlist(repo / ".leak-guard-allow")
        hits = leak_guard.scan_file(repo, "notes.md")
        unallowlisted = [h for h in hits if not allowlist.covers(h)]

        assert unallowlisted == []


# ---------------------------------------------------------------------------
# main() end-to-end (exit code via subprocess, no monkeypatching of git)
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    def test_main_exits_zero_on_clean_repo(self, tmp_path, monkeypatch):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "README.md", "clean content\n")

        monkeypatch.setattr(leak_guard, "REPO_ROOT", repo)
        monkeypatch.setattr(leak_guard, "ALLOWLIST_FILE", repo / ".leak-guard-allow")

        assert leak_guard.main() == 0

    def test_main_exits_nonzero_on_leaky_repo(self, tmp_path, monkeypatch, capsys):
        repo = _init_fixture_repo(tmp_path)
        _write_and_track(repo, "README.md", "leaked ip 10.1.2.3\n")

        monkeypatch.setattr(leak_guard, "REPO_ROOT", repo)
        monkeypatch.setattr(leak_guard, "ALLOWLIST_FILE", repo / ".leak-guard-allow")

        assert leak_guard.main() == 1
        captured = capsys.readouterr()
        assert "private-ip-10" in captured.err
