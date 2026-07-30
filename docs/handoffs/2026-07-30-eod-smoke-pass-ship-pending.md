# Handoff: 2026-07-30 EOD — #163 smoke PASS, 0.5.0 GO, ship call pending

**Supersedes** `2026-07-30-gate-day-0.5.0.md` (same day, earlier). **Ledger: #145.**

## State: `#163 gate: PASS for pre-0.5.0-release @ f40725d` — "0.5.0 mechanics are GO"

Smoke run 3 all-PASS (readback: #145 comment 2026-07-30 ~17:0x UTC): S-A gen 27s / peak 42.9GiB via restored spill path; S-B shipped kv-cache fixture byte-for-byte; **S-C first-ever success — Encode→Denoise→LogWriter wrote a valid 10-line JSONL**; S-D trace PNG. Evidence: /tmp/smoke-050 (rig standing, logs/run3, graphs/run3 — VOLATILE, lost on host reboot).

## Merged since the earlier handoff (all Opus-gated, all on `pre-0.5.0-release`)

- #173 → PR #177 @ 72d5dfc (rework after a gate FAIL): device_map="auto" restored for quant='none' + **spill-aware `_assert_no_meta_tensors`** (segment-aware hf_device_map exemption, bidirectionally mutation-verified). dd2767c's justification confirmed obsolete (d0bb93b's retie covers it).
- #174 → PR #176 @ 2ec1dd3: 5 examples repaired + all-examples conformance test. (.ui.json residual banked on #174.)
- #178 → PR #180 @ 94db0e3: kv_cache.py on transformers 5.13's REAL DynamicCache API (.layers[i].keys); **FakeDynamicCache now subclasses the real class** — cache-shape fake-drift structurally closed.
- #179 → PR #181 @ f40725d: examples repaired to the grounded live-validation rule (ComfyUI execution.py:896-913 — required widgets must be present regardless of defaults); conformance test blind spot closed.
- Suite on the line: 971/0.

## THE PENDING DECISION (operator, ship call)

Operator is playing with f40725d on shane-pc (pulled clean, diffstat reviewed as intentional). Open choice: **ship 0.5.0 now as gated** (recommended — tooltips #175 + #119 ride a fast 0.5.1) **or hold** for minimal tooltips + one more gate run. On "ship": mechanics = CHANGELOG, version bump, PR #135 disposition (likely superseded — its headline landed at 193edd3; close-or-refresh), merge `pre-0.5.0-release` → `main`, tag, registry publish, **re-check 0.4.1 registry activation (was left `Pending`)**, drop stash@{0} (self-labeled droppable after 0.5.0), close #169 (bifurcation-shadow failures — closes when release lands on main), close out ledger #145.

## In flight at handoff

- Read-only probe of shane-pc's ComfyUI API (192.168.137.1:8188 / 192.168.1.2:8188, operator-suggested): if reachable, **cross-host live testing opens** (#16 — INT4-on-Ampere, Windows install, the operator's true field environment in the #163 gate matrix). RULE: check /queue and coordinate with operator before ever submitting jobs — it is their live desktop.

## Session-discipline datapoints banked today (harness-tools)

HT#225 now carries 2 instances incl. a **phantom review** (agent confidently reported a posted PASS that never landed on the PR; merge refused because the git lane was contracted to verify the dashboard artifact first). Mitigation clause recorded on the issue and now standard in gate dispatches: "the posted comment IS the deliverable." HT#226 (bell liveness line) unimplemented, motivated by 4 gate-caught fake-vs-real bugs today (#173/#174/#178/#179).

## Post-0.5.0 queue (unchanged plus)

#119 hand-merge lane · #175 tooltips (minimal/full split, operator's call) · .ui.json residual · #114 hygiene · INT4-on-Turing probe (H0: kernel wall) · OD#40 ratification · #167 (week think) · #131/#15 GGUF promotion (audience path; llama.cpp PR #24423/#24427 research banked on #131) · #147 operator field-test · HT#226 implementation.
