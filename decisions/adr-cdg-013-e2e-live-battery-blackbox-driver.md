# ADR-CDG-013 — The E2E live-model battery is a black-box ComfyUI-API driver with subprocess-merged coverage

**Status**: proposed
**Date**: 2026-07-13
**Related**: ADR-CDG-003 (node/engine seam — this battery drives the node layer's *external* face, the complement to CDG-003's headless-engine testability), ADR-CDG-001 (native socket types — the wire-level honesty this battery confirms at runtime), ADR-CDG-005 (`CANVAS_STATE` resumable save-state — scenario S6 is its live round-trip), ADR-CDG-008 (MCP-center topology — the ComfyUI surface this battery drives is one peer surface), ADR-CDG-012 (Delivery & verification contract — this battery is the **live-execution tier above** CDG-012 DV.2's *static* workflow-conformance; they compose), Issue #59 (the banked plan this ADR records the architecture for), Issues #9 / #36 / #38 (the live bugs the battery's sharp scenarios are designed to catch), Issue #50 (the pytest-cov + torch C-tracer flake this ADR's coverage mechanism is designed to sidestep)

---

## Context

The pack has two test halves today (`tests/README.md`): a fast mocked suite (fakes at every real boundary) and an in-process `live` suite (`test_live_seams.py`, `test_integration.py`) that loads the real 53GB model **but imports the implementation directly** — `from dgemma.loop import run_diffusion`, `from surfaces.comfyui.sampler import DGemmaSampler`. Both halves reach *into* the pack.

The operator's requirement for the final battery is stronger and explicitly architectural: *"ComfyUI provides the tools and workflow schema to be able to produce this E2E live integration test suite completely independently of the implementation."* The second sentence is a commitment to honor, not a suggestion — the battery must drive ComfyUI as a **black box**, queuing workflow-schema JSONs through ComfyUI's own API and asserting only on API responses, importing nothing from the code under test. Independence is what makes it a true integration proof rather than a mock at larger scale.

Two forces make the naive path (reuse the in-process `live` marker) wrong:

1. **The process boundary.** The node-pack code executes inside the **ComfyUI server process**, not the pytest process. An in-process test that imports the pack is not driving the wiring ComfyUI actually constructs (node cache, `/prompt` scheduling, websocket push, interrupt propagation). Exactly the live bugs the battery must catch live in that wiring: #36 (ComfyUI node cache not invalidated on a knob change in a loop), #38 (interrupt not reaching the sampling loop), #9 (thinking consumes the whole canvas — a lying validity readout on an empty answer). None of these is reachable by importing `run_diffusion`; they are properties of the *loaded graph running under ComfyUI*.
2. **The coverage flake (#50).** In-process `pytest --cov` intermittently `SystemError`s importing torch under the C-tracer; the known workaround is `python -m coverage run`. A coverage mechanism for this tier cannot depend on the in-process pytest-cov C-tracer being healthy.

## Decision

**The E2E live-model battery is a standalone black-box driver that speaks only ComfyUI's HTTP + websocket API, and its coverage is measured inside the ComfyUI subprocess via `COVERAGE_PROCESS_START` and merged with the unit suite's data — deliberately NOT in-process `pytest-cov`.**

1. **Black-box driver, new `e2e` marker.** A pytest module marked `e2e` (distinct from the existing in-process `live` marker) whose import graph touches **only** `requests`/`websocket-client`/`json`/`pathlib` + the shipped `examples/*.api.json` files — zero imports from `dgemma`/`surfaces`/`consumers`. Enforced by an import-guard test (same shape as `tests/test_seam.py`'s boundary guard): the independence is an invariant with an enforcement surface, not a convention.
2. **Headless server lifecycle, per battery run.** A session-scoped fixture launches ComfyUI headless (`python main.py --listen 127.0.0.1 --port 8199 --output-directory <tmp> --disable-auto-launch`, `/srv/dev/ComfyUI/.venv` interpreter, port isolated from the operator's interactive 8188), polls `/object_info` for readiness, yields base URL + websocket `client_id`, and on teardown `SIGTERM`s and reaps.
3. **Model-load amortized once per battery.** One server process + a cache-stable `DGemmaLoader` config means the ~53GB bf16 load is paid once; every scenario runs warm off ComfyUI's model cache. The load is never a per-test cost.
4. **Subprocess-merged coverage.** `[tool.coverage.run]` in `pyproject.toml` (`parallel = true`, `concurrency = ["thread"]`, `sigterm = true`, `source = ["dgemma","surfaces","consumers"]`); the server-launch fixture sets `COVERAGE_PROCESS_START` and a `sitecustomize.py` calling `coverage.process_startup()` on the subprocess `PYTHONPATH`, so measurement starts before ComfyUI imports the pack; after the run, `coverage combine` merges the server-process `.coverage.*` with the unit suite's data into one report.
5. **Assertions on honesty, not just success.** Scenarios read ComfyUI `/history/{id}` outputs and `/ws` events and assert on *contradiction-freedom* (#9: empty STRING must not co-exist with `converged=True committed_fraction=1.0`), *cancellation actually stopping* (#38: partial `steps_used` after `/interrupt`), and *re-execution* (#36: distinct executions across a loop sweep) — not merely `status_str == "success"`.

Full scenario inventory, phasing, runtime/VRAM budget, and operator-gated preconditions are banked in **issue #59**; this ADR records the *architecture* those scenarios sit on.

## Rationale

### Positive Consequences
- **A true integration proof.** The battery exercises the exact wiring ComfyUI constructs — node cache, `/prompt` scheduling, websocket push, `/interrupt` propagation — which is where #9/#36/#38 live and where an in-process test is structurally blind.
- **Independence is enforceable, not aspirational.** The import-guard test makes "imports nothing from the implementation" a tripwire, so the battery cannot silently degrade into a mock-at-larger-scale.
- **Sidesteps #50.** Measuring coverage via `coverage.process_startup()` in the *server* process (where torch is already imported by ComfyUI's boot) avoids the pytest-cov + C-tracer interaction #50 flags, and reuses the same `coverage` machinery the #50 workaround (`python -m coverage run`) already relies on.
- **Composes with CDG-012, doesn't duplicate it.** CDG-012 DV.2 is static (load `examples/*.json`, validate against node defs, no server); this battery is the live tier that POSTs the same graphs to the real model. Static proves *matches the node defs*; live proves *produces honest output*. CDG-012's KV_CACHE workflows drop into the battery's S9 slot and feed the same merged-coverage dataset.

### Negative Consequences
- **Slower and heavier than in-process live testing.** A launched ComfyUI subprocess, a 53GB load, operator-gated GPU coordination, and a coverage-combine step — real infra cost bought for the integration proof.
- **Coverage across a process boundary is more fragile than in-process.** Depends on `sitecustomize`/`COVERAGE_PROCESS_START` firing correctly and `sigterm=true` flushing on teardown; if #50's root cause recurs inside the server process, a fallback tracer is needed (Open Question).
- **Third-party-pack coupling for #36.** Reproducing #36 in its native habitat needs `ComfyUI-Easy-Use`'s For-loop node; a pack-independent fallback (two `/prompt` POSTs with a changed knob) is less faithful. (Open Question.)

## Alternatives Considered

### Option A: Reuse the existing in-process `live` marker (import `run_diffusion`/`DGemmaSampler`, run on real weights)
**Why rejected:** It is not black-box — it violates the operator's explicit independence commitment, and structurally it cannot catch #9/#36/#38, which are properties of the graph running *under ComfyUI* (node cache, scheduling, interrupt, websocket), not of `run_diffusion` in isolation. It is the right tool for the *seam* tests it already covers (`test_live_seams.py`), and stays — but it is a different tier from the integration proof this battery is.

### Option B: In-process `pytest-cov` for the battery's coverage
**Why rejected twice.** (1) It sees zero node-pack lines from a black-box run — the code executes in the *server* subprocess, invisible to the pytest-process tracer. (2) It rides the exact pytest-cov + torch C-tracer path #50 flags as intermittently `SystemError`-ing. `COVERAGE_PROCESS_START` in the subprocess solves both: it measures where the code actually runs, using the `coverage run` machinery the #50 workaround already trusts.

### Option C: No coverage for the E2E tier — accept it as an un-profiled black-box smoke suite
**Why rejected:** the operator's requirement explicitly names "coverage profiled and executed until there are all successful live integration tests," and CDG-012 DV.1 already establishes coverage-of-the-crossing-code as an acceptance bar. Dropping coverage here would leave the live path — the fake-planted boundaries (`from_pretrained`, the processor decode, the live-view push, the interrupt poll) — confirmed by nothing. The whole point of the live tier is that it reaches the rows a fake cannot falsify; not measuring that reach forfeits the tier's core value.

## Open Questions

- [ ] **#50 recurrence inside the server process.** The subprocess-coverage path is *designed* to sidestep the pytest-cov + torch C-tracer flake, but #50's root cause is unconfirmed. **Resolution trigger:** first real battery run under `coverage.process_startup()`; if the `SystemError` recurs, fall back to `COVERAGE_CORE=pytrace` (pure-Python tracer), then `concurrency=["thread","multiprocessing"]`. Recorded as open on *mechanism robustness*, not on whether coverage is measurable at all.
- [ ] **#38 assertion precision.** "The run stopped after `/interrupt`" needs a non-flaky observable (step-event count ceases + `/history` shows partial `steps_used`), and the tolerance on "extra steps after interrupt" depends on where the interrupt poll lands once #38 is fixed. **Resolution trigger:** set the tolerance when the #38 fix lands the poll (battery phase E3).
- [ ] **S8 loop-node coupling.** Whether to reproduce #36 via `ComfyUI-Easy-Use`'s For-loop node (faithful to the operator's observed habitat) or via two independent `/prompt` POSTs with a changed `entropy_bound` (pack-independent). **Resolution trigger:** implementer's call at battery phase E3, weighing faithfulness vs. third-party schema fragility.

## Supersession Relationships

**Supersedes:** none
**Superseded by:** TBD

## Implementation Notes

Implementation is banked in **issue #59** (scenario inventory S1–S9, six dependency-ordered phases E0–E4, runtime/VRAM budget, operator-gated preconditions). This ADR is the decision record for the *architecture*; the issue is the plan. Enforcement surfaces this ADR introduces, to be added to `ARCHITECTURE.md`'s enforcement-surface table (`NOT-YET-IMPLEMENTED` until the battery lands, per that doc's doc-and-code-move-together discipline):

| Surface | What it enforces |
|---|---|
| `e2e`-module import-guard test | The battery imports nothing from `dgemma`/`surfaces`/`consumers` (independence invariant) |
| `[tool.coverage.run]` + `sitecustomize`/`COVERAGE_PROCESS_START` | Node-pack coverage is measured inside the ComfyUI subprocess and merged with the unit data |
| Per-scenario green + combined-coverage readback banked to issue #59 / plan.md | "Done" = all scenarios green on the live model with the live path confirmed reaching the boundary rows |

**Named precondition defect (not a step this ADR takes):** the ComfyUI custom_nodes symlink `/srv/dev/ComfyUI/custom_nodes/ComfyUI-DiffusionGemma` currently points at `/srv/dev/cdg-feat21-review`, which no longer exists — a dead symlink that blocks the pack from loading. Repointing it at the canonical repo path touches the shared ComfyUI install and is operator-gated (issue #59 §5).

## References

- Issue #59 — the banked E2E battery plan (scenarios, phasing, budget, preconditions)
- ADR-CDG-012 §DV — the static delivery/verification contract this battery's live tier composes with
- `tests/README.md` — the two-halves (mocked / in-process `live`) convention this ADR adds a third tier to
- coverage.py multiprocess docs — `COVERAGE_PROCESS_START` / `coverage.process_startup()` subprocess measurement

---
