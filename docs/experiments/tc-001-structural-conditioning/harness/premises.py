"""premises.py — TC-001's 10 pre-registered premises, the two condition
templates, and the paired seeds, all taken VERBATIM from
`docs/experiments/tc-001-structural-conditioning/protocol.md` (issue #286).

Nothing here is inferred or paraphrased: the premise list, the skeleton
block (including its own heading line), and the instruction suffix are
copy-pasted out of the registered protocol so the harness runs exactly the
pre-registered design, not a redescription of it. If the protocol changes,
this module must be re-synced by hand — there is no single-source import
across the docs/code boundary (the protocol is prose, not data).
"""
from __future__ import annotations

# Protocol §Design, "Premises (reskinned from distinct human openings; use
# as-is)" — order is significant: PREMISES[i] pairs with SEEDS[i] across
# both conditions (protocol §Design: "Paired ... same seed per pair").
PREMISES: list[str] = [
    "A young woman sits alone on a New York subway; 365 days without a date.",
    "A night-shift radiologist finds his own name on a scan dated next month.",
    "Two brothers split their late father's beehives; one hive won't accept either.",
    "A ferry cook in the Aegean discovers the ship has been sailing in circles for a day.",
    "A retired forger is asked to authenticate a painting she faked forty years ago.",
    "A rural lineman keeps restoring power to a house the county says is vacant.",
    "A translator realizes the dictator's speech she's rendering live contains a message for her.",
    "A minor-league umpire calls his last game the day his eyesight officially fails.",
    "A landlady inherits a tenant's ashes and his unpaid rent in the same envelope.",
    "A glacier guide finds last season's missing client's camera, still recording.",
]

# Protocol §Design: "Pre-registered seeds: [101,102,...,110], seed i pairs
# premise i across both conditions." len(SEEDS) == len(PREMISES) == 10.
SEEDS: list[int] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

# Protocol §Design, Condition A: "premise + 'Write a short story, about 800
# words.'" — the exact instruction suffix, verbatim.
CONDITION_A_INSTRUCTION = "Write a short story, about 800 words."

# Protocol §Design → "Skeleton block (Condition B)" — copied verbatim,
# INCLUDING its own heading line ("STRUCTURE — commit to these before
# writing; enact them, never mention them:"), which is part of the prefix
# text the model sees, not a doc-only label.
STRUCTURE_SKELETON = """STRUCTURE — commit to these before writing; enact them, never mention them:
1. The theme is never stated. No narrator moral, no character voicing the
   lesson. If a sentence explains what the story means, delete it.
2. Open mid-crisis. Exactly one flashback, entered without announcement.
3. The ending is caused by an external event. The protagonist's plan fails.
4. Name at least three real things: a real book, a real place, a real brand."""


def condition_a_prompt(premise: str) -> str:
    """Protocol §Design, Condition A (control): premise + "\\n\\n" +
    the instruction suffix, verbatim."""
    return f"{premise}\n\n{CONDITION_A_INSTRUCTION}"


def condition_b_prompt(premise: str) -> str:
    """Protocol §Design, Condition B (conditioned): "identical [to A],
    plus the skeleton block below, verbatim, before the premise." — i.e.
    the skeleton block, then "\\n\\n", then the identical Condition A text
    (premise + instruction suffix)."""
    return f"{STRUCTURE_SKELETON}\n\n{condition_a_prompt(premise)}"


CONDITIONS = ("A", "B")

assert len(PREMISES) == 10, f"expected 10 pre-registered premises, got {len(PREMISES)}"
assert len(SEEDS) == 10, f"expected 10 pre-registered seeds, got {len(SEEDS)}"
