# Decompose dgemma/loop.py into responsibility modules

**Status**: accepted
**Date**: 2026-07-23
**Related**: ADR-CDG-004 (drive seam — `run_diffusion` is the single entry this ADR preserves), ARCHITECTURE.md rule 1 ("one core, one contract")

---

## Context

`dgemma/loop.py` is 1,631 lines carrying six distinct responsibilities that accumulated across phases: P1 (vertical slice), P2 (knobs as widgets), P3 (instrumentation + trace), and the crystalline proxy framework. The file grew by accretion — each phase added its own section without reshaping what came before — because the collection seam was designed to iterate every step regardless of phase, so new code could be appended without touching existing sections.

That strategy worked for getting phases shipped. It left a single module that is 25,000 tokens (well above the 5,000-token comfort threshold) with no clear decomposition surface visible from its public API: `run_diffusion()` is the one entry point, but it imports helpers scattered across 1,631 lines of mixed concerns.

The responsibilities currently bundled in `loop.py`:

| Responsibility | Lines | Size | Coupling to other responsibilities |
|---|---|---|---|
| Diffusers version guard + structural probe | 79–262 | ~184 | None — pure import-time validation, no loop state |
| Constants (DEFAULT_*, KNOB_DOCS) | 263–395 | ~133 | Referenced by `run_diffusion` defaults and MCP schema |
| DGemmaPipeline class | 399–420 | ~22 | Pipeline wrapper, used by `run_diffusion` |
| Temperature annealing + pinned mask | 421–472 | ~52 | Utility functions called per-step inside the loop |
| `_FrameCollector` (telemetry capture) | 473–814 | **~342** | Captures per-step frames; independent of denoising logic |
| Canvas state derivation | 815–876 | ~62 | Post-run state computation from collected frames |
| Vocab + thought channel resolution | 877–929 | ~53 | Utility helpers, no loop state |
| ThoughtChannelExcision class | 930–954 | ~25 | Post-processing, no loop state |
| Thought excision + decoding utilities | 955–1097 | **~143** | Frame decode, thought text extraction — post-run only |
| `run_diffusion()` (the drive seam) | 1098–1545 | ~448 | The single entry point; orchestrates all above |
| `_build_result()` | 1546–end | ~85 | Helper for `run_diffusion` return construction |

**The problem:** a cold reader cannot locate any one responsibility without scanning the full file. A change to frame capture (telemetry) requires understanding diffusers version guards and thought channel excision because they share the same module namespace. The MCP surface's schema (`surfaces/mcp/commands/generate.py`) imports constants from `loop.py` — pulling in the entire 1,631-line module for what should be a config lookup.

## Decision

Decompose `dgemma/loop.py` into five modules by responsibility, preserving `run_diffusion()` as the single drive entry (ADR-CDG-004) and keeping all public API surface unchanged:

```
dgemma/
  compat.py          # diffusers version guard + structural probe (~184 lines)
  config.py          # DEFAULT_*, KNOB_DOCS, THINK_TOKEN constants (~200 lines)
  capture.py         # _FrameCollector telemetry capture (~342 lines)
  excision.py        # thought channel excision + frame decoding (~200 lines)
  loop.py            # run_diffusion() + helpers (~550 lines — the drive seam)
```

**Target: `loop.py` drops from 1,631 to ~550 lines.** The extracted modules are self-contained — each can be read independently without carrying context from the others.

## Rationale

### Positive Consequences
- **Each module is a single responsibility.** A change to frame capture touches only `capture.py`; a diffusers version bump touches only `compat.py`. No cross-concern blast radius.
- **MCP surface imports are cheap.** `surfaces/mcp/commands/generate.py` currently imports from `loop.py`, pulling in 1,631 lines for constant lookups. After decomposition it imports from `config.py` (~200 lines) — a real reduction in import cost and cognitive load.
- **Test isolation improves.** Each extracted module has its own test surface: `test_compat.py`, `test_capture.py`, `test_excision.py`. Currently all tests route through `loop.py`'s namespace, making it hard to isolate which responsibility failed.
- **The drive seam stays clean.** `run_diffusion()` in the decomposed `loop.py` imports from sibling modules — same dependency graph, just expressed across files instead of one monolith.

### Negative Consequences
- **Import path churn for existing callers.** Any code importing `_FrameCollector`, `THINK_TOKEN`, or `DEFAULT_ENTROPY_BOUND` from `dgemma.loop` must update to the new module path. Internal-only imports (no external API surface) — but still a mechanical change across tests and surfaces.
- **`__init__.py` re-export surface grows.** The public face (`dgemma/__init__.py`) currently re-exports from `.loop`. After decomposition it re-exports from four modules instead of one. This is mechanical, not conceptual — the exported symbols don't change, only their source module.
- **A refactor where behavior does not change.** No surface behavior changes; this is a structural move. The seam test (`tests/test_seam.py`) must still pass (core imports no surface), and all existing tests must continue to exercise the same paths — just from different import locations.

## Alternatives Considered

### Option A: Keep loop.py as-is, add section comments
**Why rejected:** Section comments don't reduce cognitive load for a cold reader scanning 1,631 lines. The file is already well-commented (each section has a docstring), but the problem isn't documentation — it's that six responsibilities share one namespace and one import path. Comments cannot make `import dgemma.loop` cheaper when you only need constants.

### Option B: Extract only _FrameCollector (the largest block at 342 lines)
**Why rejected:** Partial extraction leaves the same structural problem for the remaining five responsibilities. The diffusers guard alone is 184 lines of import-time validation that has nothing to do with denoising — it belongs in its own module regardless of what else happens. Extracting one block without decomposing the rest is a band-aid on a structural issue.

### Option C: Split by phase (P1, P2, P3 modules)
**Why rejected:** Phases are historical artifacts, not responsibility boundaries. Frame capture was added in P3 but is conceptually telemetry — it belongs with other capture logic regardless of when it shipped. Decomposing by phase would create modules named after implementation history rather than function, which is worse for cold readers.

## Supersession Relationships

**Supersedes:** none (structural refactor, not a design decision replacing another)
**Superseded by:** TBD — future work may further decompose `run_diffusion()` itself if it grows beyond the 5,000-token threshold again

---

## Implementation plan (staged, each stage independently verifiable)

### Stage 1: Extract constants → `dgemma/config.py`
**Touching:** `loop.py` (remove DEFAULT_*, KNOB_DOCS, THINK_TOKEN, THOUGHT_CHANNEL_*), new `config.py`, all import sites (`surfaces/mcp/commands/generate.py`, tests)

**Verifiable when it lands:** `from dgemma.config import DEFAULT_ENTROPY_BOUND` works; `loop.py` shrinks by ~133 lines; no test regression.

### Stage 2: Extract diffusers guard → `dgemma/compat.py`
**Touching:** `loop.py` (remove `_check_diffusers_version`, `_tuple_version`, `_check_diffusers_structure`, `REQUIRED_DIFFUSERS_MINIMUM`), new `compat.py`, `__init__.py` import

**Verifiable when it lands:** `from dgemma.compat import check_diffusers` works; `tests/test_diffusers_version_guard.py` still passes (import path updated); `loop.py` shrinks by ~184 lines.

### Stage 3: Extract frame capture → `dgemma/capture.py`
**Touching:** `loop.py` (remove `_FrameCollector` class), new `capture.py`, `run_diffusion()` import update

**Verifiable when it lands:** `_FrameCollector` imported from `dgemma.capture`; all trace/instrumentation tests pass; `loop.py` shrinks by ~342 lines.

### Stage 4: Extract thought excision → `dgemma/excision.py`
**Touching:** `loop.py` (remove `ThoughtChannelExcision`, `excise_thought_channel`, `_decode_ids`, `_extract_thought_text`, `decode_frames`, `resolve_vocab_size`, `resolve_thought_channel_ids`), new `excision.py`, import sites

**Verifiable when it lands:** Thought excision functions imported from `dgemma.excision`; all thought-channel tests pass; `loop.py` shrinks by ~200 lines.

### Stage 5: Final cleanup — verify loop.py is the drive seam
**Touching:** `loop.py` (should now be ~550 lines: `DGemmaPipeline`, `anneal_temperature`, `_build_pinned_mask`, `derive_canvas_state`, `run_diffusion()`, `_build_result()`), `__init__.py` re-export audit

**Verifiable when it lands:** `loop.py` is under 600 lines; `run_diffusion()` is the single public entry; all tests pass including seam test (`test_seam.py`).

---

## References

- ADR-CDG-004 (`decisions/adr-cdg-004-diffusers-pipeline-drive-seam.md`) — `run_diffusion` single-entry drive contract preserved by this refactor
- ARCHITECTURE.md rule 1 ("one core, one contract") — the decomposition keeps the contract surface unchanged
- `tests/test_seam.py` — subprocess import-leak assertion that must pass after every stage
