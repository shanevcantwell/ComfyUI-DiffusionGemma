# ADR-CDG-023 — mcp SDK 2.x port strategy: hold the `<2.0.0` cap as standing posture, with a named revisit trigger

**Status**: `proposed` — ratification authority: the OPERATOR, per this repo's
process convention (independent Opus design-gate review; the operator sets
intent, reads conclusions, holds the standing veto). This ADR does not open
implementation — it is the decision record the issue's own bounce clause
required before any port work could begin.
**Date**: 2026-08-04
**Related**:
- [Issue #151](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/151) — this ADR's tracking issue (mcp SDK 2.x breakage; the immediate-fix/follow-up-port split; the 2026-08-03 bounce this ADR resolves)
- [Ledger #247](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/247) — the autonomous-run batch this work executes under
- [PR #157](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/157) (`af0d7a1`) — landed the `mcp>=1.0.0,<2.0.0` cap; reconciled onto `main` via the `pre-0.5.0-release` merge train (confirmed present at `origin/main@6557b21`, this ADR's research pass)
- ADR-CDG-008 (MCP-center multi-surface topology) — the "MCP is the canonical surface" rule this ADR's Option analysis is weighed against
- ADR-CDG-019 (MCP-as-contract topology remediation) — the in-flight `dgemma_mcp/` rename bracket (#138); this ADR's chosen posture does not touch that bracket's scope

---

## Context

`surfaces/mcp/server.py` is a **thin adapter** (ARCHITECTURE.md: "unpack args →
call one `dgemma.*` function → wrap the result") built against the `mcp` SDK's
**1.x low-level `Server`** API: `@server.list_tools()` / `@server.call_tool()`
decorator methods, module-scope `_HANDLERS` dispatch table, and a bare
`list[TextContent]` / `{"error": ...}`-in-`TextContent` error convention —
transcribed near-verbatim from the `semantic-kinematics-mcp` sibling project's
shape (`surfaces/mcp/server.py` docstring, throughout).

`pyproject.toml`'s `[mcp]` extra originally read `mcp>=1.0.0` (open-ended). The
`mcp` SDK's 2.0.0 release (2026-07, per issue #151's discovery via CI's #146
honest-state dry-run) is a **breaking major rework**, not a version bump: the
SDK's own PyPI metadata calls it "a major rework... to fix long-standing
architectural issues," with a dedicated migration guide. Concretely
(confirmed 2026-08-03 against the installed 2.0.0 wheel in a scratch venv,
recorded verbatim on issue #151):

- `mcp.server.lowlevel.server.Server` **no longer has** `list_tools`/
  `call_tool` as decorator methods — `AttributeError` on the exact pattern
  this codebase uses. The replacement is constructor-only:
  `Server(..., on_list_tools=, on_call_tool=)`, whose callbacks take a
  different signature (`(ServerRequestContext, PaginatedRequestParams |
  CallToolRequestParams)`) and must return `ListToolsResult` /
  `CallToolResult` dataclasses — `CallToolResult` carrying `content:
  list[ContentBlock]` **and an `is_error: bool` flag** — instead of this
  codebase's current bare `list[Tool]` / `list[TextContent]` return and its
  `{"error": ...}`-embedded-in-`TextContent` convention.
- `Tool`/`TextContent` themselves still exist in `mcp.types`, so tool
  *identity* (names, JSON schemas) is preservable across the port. What
  changes is the handler contract and the error-surfacing convention.
- 2.0.0 also ships a new high-level `mcp.server.MCPServer` (the FastMCP-lineage
  `@mcp.tool` decorator API, schema inferred from type hints), now the SDK
  README's *featured* pattern — the low-level manual-dispatch-table style this
  codebase deliberately chose (to mirror `semantic-kinematics-mcp` and keep
  JSON-schema authorship explicit/declarative rather than inferred) is no
  longer the showcased example.

Issue #151 explicitly separated an **immediate fix** (cap the extra) from a
**follow-up port** ("separate, not this issue"). The immediate fix landed
(PR #157 / `af0d7a1`) and — per this ADR's own research pass, superseding
some stale mid-thread comments on #151 that reported it missing — **is
confirmed present on `origin/main`** at commit `6557b21`
(`pyproject.toml:59`, `mcp = ["mcp>=1.0.0,<2.0.0"]`). No mechanical PR is
needed for the cap; see this ADR's companion ledger comment.

A 2026-08-03 autonomous-run attempt at the follow-up port hit the issue's own
bounce clause ("If the 2.x API forces a user-visible surface change ... that
is a design fork: STOP and bounce") and correctly stopped rather than picking
between the two non-equivalent 2.x-recommended shapes. This ADR is that
stopped decision, now made explicitly with the trade-offs named.

## Decision

**Adopt Option A: hold the `mcp>=1.0.0,<2.0.0` cap as standing posture, with a
named revisit trigger — do not port to 2.x now.**

1. **The cap stays.** `pyproject.toml`'s `[mcp]` extra remains
   `mcp>=1.0.0,<2.0.0`. No code in `surfaces/mcp/` changes as a result of this
   ADR.
2. **Issue #151 stays open**, rescoped from "port pending" to "parked behind
   this ADR's revisit trigger" (Decision-3).
3. **Revisit trigger (named, not open-ended):** re-open the port question when
   **any** of the following becomes true:
   - The `mcp` 1.x line stops receiving security/compat fixes from upstream
     (i.e., the cap starts trading a live vulnerability for API stability), or
   - A concrete consumer needs a 2.x-only capability (e.g., an MCP client the
     project must interoperate with drops 1.x support), or
   - The operator schedules the port as deliberate roadmap work (this ADR does
     not preclude that choice — it only avoids defaulting into the port under
     autonomous-run pressure without the design fork resolved).
4. **When the trigger fires, the design fork (Option B vs. Option C, below)
   is decided at that time** — not pre-committed here. This ADR's job is to
   stop the drift (bounce → cap → parked, with the reasoning recorded), not to
   pre-select the eventual port shape.

## Rationale

### Why hold rather than port now

The two 2.x-recommended shapes are **not equivalent** under this repo's own
doctrine, and picking one is a genuine design fork the issue never settled
(this is exactly the condition that triggered the 2026-08-03 bounce):

- **ADR-CDG-008's "MCP is the canonical surface; thin adapter" rule**
  (ARCHITECTURE.md rule 2; `surfaces/mcp/` description: "unpack args → call
  one `dgemma.*` function → wrap the result") is satisfied by the **current
  low-level manual-dispatch-table shape** — JSON schemas are authored
  explicitly in each command module's `get_tools()`, dispatch is one
  `_HANDLERS` dict, nothing is inferred. The high-level `MCPServer`
  decorator API (Option C) infers schema from Python type hints, which
  moves schema authorship from an explicit, reviewable JSON-Schema literal
  into implicit type-hint-driven generation — a real shift in *how* the thin
  adapter stays thin and auditable, not a mechanical swap.
- **The `is_error` flag is a user-visible protocol change**, not an internal
  refactor, under **either** 2.x shape (Option B or C): today, a handler
  failure surfaces as `{"error": str(exc)}` embedded in a `TextContent`
  payload the caller must inspect; 2.x's `CallToolResult.is_error` is a
  first-class protocol field every MCP client speaking 2.x will expect to be
  populated correctly. Getting this wrong silently degrades error handling
  for every tool call — an entropy-budget-disguised-as-a-tensor-shaped risk
  in the sense ADR-CDG-001 names for payload trust, now at the protocol
  layer instead of the socket-type layer.
- Holding the cap costs nothing today: `mcp` is an **optional extra**
  (`pyproject.toml` comment, `tests/test_seam.py`), so a ComfyUI-only install
  never resolves it, and no other part of this pack imports
  `surfaces.mcp.server` (`tests/test_mcp_surface_seam.py`,
  `tests/test_mcp_import_guard.py`). The cap only affects contributors who
  explicitly install `.[mcp]` — a small, already-warned surface (PR #157's
  own commit message: "2.x API moved under `surfaces/mcp/server.py`").

### Positive Consequences
- Stops the drift the 2026-08-03 bounce correctly refused to absorb: no design
  fork gets picked under autonomous-run time pressure without an independent
  gate review.
- Zero implementation risk introduced by this ADR — it is a decision record
  only; `surfaces/mcp/server.py` is untouched.
- The revisit trigger is concrete and falsifiable (security-fix cutoff /
  named consumer need / operator schedule), so "hold" does not silently
  become "never revisit."

### Negative Consequences
- **Invariant this cap protects, named per the greenfield rule:** *the `[mcp]`
  extra's resolved SDK version MUST match the API `surfaces/mcp/server.py`
  is written against; an unbounded `mcp` constraint MUST NOT be reintroduced
  without the port landing first.* **Anticipated failure prevented:** the
  exact #151 regression recurring — a future contributor loosens or drops the
  cap (e.g. during a routine dependency bump) believing it is stale, and the
  extra breaks again on a fresh install, silently, until CI's honest-state job
  (#146) or a user hits the `AttributeError`. **Enforcement surface:** today,
  this is **prose-only** — the cap is a hand-maintained pin with no test
  asserting *why* it's capped. `tests/test_requirements_sync.py` covers the
  base `[project].dependencies` list, not extras; no equivalent sync/pin-
  rationale test exists for `[mcp]`. Naming this gap rather than closing it:
  closing it (e.g. a test asserting the cap's upper bound matches a comment
  citing this ADR) is in-scope follow-up work, not performed here.
- The project forgoes 2.x's SDK-team-recommended improvements (whatever
  "long-standing architectural issues" motivated the rework) for as long as
  the cap holds — an opportunity cost, not a defect, but real.
- `mcp` 1.x is a third-party dependency outside this project's control; if
  upstream stops shipping 1.x fixes before the operator schedules the port,
  the revisit trigger fires reactively rather than on a planned cadence. No
  automated staleness check exists for this (see Open Questions).

## Alternatives Considered

### Option B: port to the mcp 2.x low-level callback protocol

Rewrite `surfaces/mcp/server.py`'s `@server.list_tools()`/`@server.call_tool()`
decorators as `Server(..., on_list_tools=, on_call_tool=)` constructor
callbacks, adapting to the new `ServerRequestContext`/`*Params` request shape
and `ListToolsResult`/`CallToolResult` response shape. Tool identity (names,
JSON schemas authored in `commands/*.get_tools()`) is preserved unchanged;
only the transport-facing envelope and error convention (`is_error`) change.

**Pros:**
- Closest structural match to the current dispatch-table shape — smallest
  diff, easiest to review against ADR-CDG-008's thin-adapter rule.
- Preserves explicit JSON-Schema authorship in `commands/*.py` — no inferred
  schemas to audit.

**Cons:**
- Still requires deciding and documenting the `is_error` mapping (does every
  handler exception become `is_error=True`? does the existing
  `{"error": ...}` JSON payload also get preserved inside `content` for
  backward-compatible parsing by any existing client code, or is it dropped
  in favor of the flag alone?) — a real design question, not mechanical.
- The low-level callback API is no longer the SDK's featured/primary pattern,
  so this path carries more risk of being the *less* maintained of the two
  2.x-supported shapes going forward (unconfirmed — would need checking at
  revisit time, not now).

### Option C: migrate to the high-level `MCPServer` decorator API

Rewrite as `@mcp.tool`-decorated functions with schemas inferred from Python
type hints (the FastMCP-lineage pattern the 2.x SDK now showcases).

**Pros:**
- Matches the SDK's own recommended/primary pattern going forward — likely
  the best-maintained path, and less boilerplate per tool.
- Type-hint-inferred schemas reduce the chance of the JSON-Schema literal and
  the actual handler signature drifting apart (a class of bug explicit
  authorship doesn't fully prevent either, but the failure mode differs).

**Cons:**
- **Direct tension with ADR-CDG-008's "thin adapter, explicit schema
  authorship" pattern** — this repo deliberately chose the low-level style
  over the (already-then-available) high-level decorator style to mirror
  `semantic-kinematics-mcp` and keep schemas declarative rather than
  inferred (`surfaces/mcp/server.py` docstring). Adopting Option C reopens
  that prior decision, which this ADR does not have grounds to relitigate
  under autonomous-run scope.
- Larger structural rewrite — bigger diff, bigger review surface, on a
  surface ADR-CDG-008/019 are actively also restructuring (`dgemma_mcp/`
  rename bracket, #138) — higher chance of merge/rebase friction against
  concurrent topology work.

### Option D (implicitly rejected by silence — named for completeness): drop the `mcp` extra / stop shipping the MCP surface

Not seriously considered — MCP is this pack's canonical surface per
ADR-CDG-008 rule 2, not an optional feature; dropping it contradicts the
repo's core topology decision. Not analyzed further; recorded only so a
future reader does not wonder whether it was overlooked.

## Open Questions

- [ ] No automated staleness check exists for "has upstream `mcp` 1.x stopped
      receiving fixes" (Decision-3's first trigger condition). **Resolution:**
      out of scope for this ADR (would require an external dependency-
      advisory feed this project doesn't currently consume); revisit if/when
      a dependency-monitoring mechanism is adopted project-wide.
- [ ] Whether the eventual port (Option B or C) also needs to preserve the
      current `{"error": str(exc)}`-in-payload convention *alongside*
      `is_error` for any external client already parsing today's shape, or
      whether a clean break is acceptable. **Resolution:** deferred to the
      port ADR/plan written when the revisit trigger fires — no known
      external client exists today to constrain the answer (unconfirmed,
      not verified against usage telemetry this project doesn't collect).
- [ ] Whether the cap's enforcement surface (currently prose-only, per
      Negative Consequences) should gain a mechanical test asserting the
      upper-bound pin cites this ADR. **Resolution:** named as in-scope
      follow-up, not performed by this ADR; candidate for a small dedicated
      issue if the operator wants the enforcement surface closed before the
      revisit trigger fires.

## Supersession Relationships

**Supersedes:** none — this is the first decision record for the `mcp` SDK
version posture; the cap itself (PR #157) predates this ADR and was a
mechanical fix without its own decision record.
**Superseded by:** TBD — the eventual port ADR, written when Decision-3's
revisit trigger fires, supersedes this ADR's "hold" posture with whichever of
Option B/C (or a then-current Option E) is chosen.

## Implementation Notes

This ADR performs **no implementation**. It is a decision record only,
per this repo's waterfall process convention (design completes before
implementation opens) and per the operator-set instruction this ADR was
drafted under (STOP at ratification; no 2.x port performed).

| File | Change Type | Description |
|------|-------------|--------------|
| `decisions/adr-cdg-023-mcp-2x-port-strategy.md` | Created | This ADR |
| `pyproject.toml` | Unchanged | Cap already present at `origin/main@6557b21` (`mcp>=1.0.0,<2.0.0`); no PR needed — verified during this ADR's research pass, reported on the tracking issue |
| `surfaces/mcp/server.py` | Unchanged | No port performed; Decision-3's trigger gates future work |

## References

- [Issue #151](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/151) — full investigation trail, including the 2026-08-03 bounce comment this ADR resolves (verbatim API findings against the installed mcp 2.0.0 wheel).
- `surfaces/mcp/server.py` — current low-level `Server` dispatch implementation and its `semantic-kinematics-mcp`-mirroring docstring.
- `ARCHITECTURE.md` rule 2 ("MCP is the canonical surface") and `surfaces/mcp/` description ("Thin adapter: unpack args → call one `dgemma.*` function → wrap the result").
- ADR-CDG-001 — `EMIT-CANONICAL / PARSE-AT-THE-DOOR`; the payload-trust framing this ADR applies to the `is_error` protocol change.
- ADR-CDG-008, ADR-CDG-019 — MCP topology decisions this ADR's Option analysis is weighed against and does not alter.
