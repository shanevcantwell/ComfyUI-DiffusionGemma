#!/usr/bin/env python3
"""leak_guard.py — generic-pattern topology leak guard (ADR-CDG-022 Decision 4b).

Standalone dev tool. Does NOT import into, or get imported by, the
ComfyUI-DiffusionGemma node pack (same discipline as `tools/flipbook/` and
`tools/q2_preflight.py`).

Scans every `git ls-files`-tracked file's content for GENERIC, structural
patterns only:

  - private-IP ranges: 192.168.x.x, 10.x.x.x, 172.16.x.x-172.31.x.x
    (CIDR-aware — does not false-positive on e.g. 172.99.x.x)
  - `/home/<user>` absolute paths
  - `/srv/dev/<name>` absolute paths

CRITICAL CONSTRAINT (ADR-CDG-022 Decision 4b, Alternative D rejected): this
script NEVER encodes a literal hostname/IP denylist. A tracked denylist of
the actual sensitive values would itself re-leak them the moment this file
is public — the guard would become the leak. Patterns here are generic and
structural only; nothing in this file names an actual host, IP, or username
observed in the repo's history.

Allowlist: `.leak-guard-allow` at repo root, one entry per line, two forms:
  path:<prefix>          — skip any match whose file path starts with <prefix>
  placeholder:<string>   — skip any match whose matched text equals <string>
Blank lines and lines starting with `#` are ignored.

Exit code: 0 if no unallowlisted hits, non-zero otherwise.

Usage:
    python tools/leak_guard.py            # run from repo root
    python3 tools/leak_guard.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_FILE = REPO_ROOT / ".leak-guard-allow"

# --- Generic, structural patterns only (never a literal value denylist) ----

# RFC 1918 private-IP ranges, CIDR-aware (not a naive string-prefix match —
# 172.16-31 only, so 172.99.x.x does not false-positive).
_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
PRIVATE_IP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-ip-192.168", re.compile(rf"\b192\.168\.{_OCTET}\.{_OCTET}\b")),
    ("private-ip-10", re.compile(rf"\b10\.{_OCTET}\.{_OCTET}\.{_OCTET}\b")),
    (
        "private-ip-172.16-31",
        re.compile(rf"\b172\.(?:1[6-9]|2[0-9]|3[0-1])\.{_OCTET}\.{_OCTET}\b"),
    ),
]

PATH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("home-path", re.compile(r"/home/\w+")),
    ("srv-dev-path", re.compile(r"/srv/dev/\w+")),
]

ALL_PATTERNS = PRIVATE_IP_PATTERNS + PATH_PATTERNS

# This script's own source is exempt from its path-pattern scan: the
# patterns above are regex *literals* (e.g. `r"/home/\w+"`) — matching them
# against this file's own text would flag the guard's own pattern
# definitions as if they were leaked paths. This is a structural exemption
# (this file's path only), not a value denylist.
SELF_PATH = Path(__file__).resolve()


class Hit(NamedTuple):
    file: str
    line_no: int
    pattern_class: str
    matched_text: str
    line_text: str


class Allowlist(NamedTuple):
    path_prefixes: list[str]
    placeholders: list[str]

    def covers(self, hit: Hit) -> bool:
        for prefix in self.path_prefixes:
            if hit.file.startswith(prefix):
                return True
        for placeholder in self.placeholders:
            if hit.matched_text == placeholder:
                return True
        return False


def load_allowlist(path: Path = ALLOWLIST_FILE) -> Allowlist:
    path_prefixes: list[str] = []
    placeholders: list[str] = []
    if not path.exists():
        return Allowlist(path_prefixes, placeholders)
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            path_prefixes.append(line[len("path:") :])
        elif line.startswith("placeholder:"):
            placeholders.append(line[len("placeholder:") :])
    return Allowlist(path_prefixes, placeholders)


def tracked_files(repo_root: Path = REPO_ROOT) -> list[str]:
    """`git ls-files`, respecting being run from repo root (or anywhere
    inside the repo — `-C` pins the invocation to repo_root explicitly)."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def scan_file(repo_root: Path, rel_path: str) -> list[Hit]:
    abs_path = repo_root / rel_path
    if abs_path.resolve() == SELF_PATH:
        return []
    try:
        text = abs_path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        # Binary or unreadable file — nothing to scan as text.
        return []

    hits: list[Hit] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern_class, pattern in ALL_PATTERNS:
            for match in pattern.finditer(line):
                hits.append(
                    Hit(
                        file=rel_path,
                        line_no=line_no,
                        pattern_class=pattern_class,
                        matched_text=match.group(0),
                        line_text=line.strip(),
                    )
                )
    return hits


def main() -> int:
    allowlist = load_allowlist()
    files = tracked_files()

    all_hits: list[Hit] = []
    for rel_path in files:
        all_hits.extend(scan_file(REPO_ROOT, rel_path))

    unallowlisted = [hit for hit in all_hits if not allowlist.covers(hit)]

    if unallowlisted:
        for hit in unallowlisted:
            print(
                f"{hit.file}:{hit.line_no}: {hit.pattern_class}: {hit.matched_text}",
                file=sys.stderr,
            )
        print(
            f"\nleak_guard: {len(unallowlisted)} unallowlisted hit(s) found "
            f"across {len(files)} tracked files.",
            file=sys.stderr,
        )
        return 1

    print(
        f"leak_guard: clean — 0 unallowlisted hits across {len(files)} tracked files "
        f"({len(all_hits)} allowlisted hit(s) suppressed)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
