# Handoff: 2026-07-30 late — #183 autoround regression fixed end-to-end; run-4 gate PASS @ a68e29d; 0.5.0 GO restored, ship call pending

**Supersedes** `2026-07-30-eod-smoke-pass-ship-pending.md` (its GO was vetoed mid-evening by the #183 field report; this handoff restores GO at a newer SHA). **Ledgers: #183 (fix arc), #145 (gate runs).**

## State: `pre-0.5.0-release @ a68e29d` — #163 gate run 4 ALL PASS (S-A bf16 / S-B kv-cache / S-C log-writer / S-D trace / S-A-autoround). 0.5.0 mechanics are GO pending the operator ship call.

## The #183 arc (compressed; full trail on the issue)

- Operator field report: autoround overflows VRAM, ~10x slower than bf16; later: hard cross-device failure under split; steers converged on **"load should be load"** — the quant-conditional placement divergence WAS the bug.
- Probe (banked, `docs/experiments/2026-07-30-autoround-unified-path-split-check/`): H0 falsified — INT4 + forced split dies in accelerate dispatch ("weight is on the meta device"); bf16 under the identical cap loads green. INT4-specific.
- Fix (PR #185, squash-merged @ a68e29d after Opus gate PASS, review 4824648747): unified load path (`device_map="auto"`, dtype-only per-quant residue, `.to("cuda")` deleted) + fail-loud 30 GiB pre-load VRAM precondition (grounded in measured 28.55 GiB footprint). Uniformity mutation guard: quant-conditional placement reddens by name.
- **The precondition already saved a run**: gate run 4's first autoround attempt ran with bf16 still resident (7.4 GiB free) — refused loud, exactly as designed; standalone re-run PASS. Both banked in `logs/run4/battery_results.json` (rig, volatile).
- INT4 numbers now: 33.0–33.1 GiB peak, ~1.5–2.2 s/step — faster per step than bf16's 2.2 and 9 GiB lighter.

## Checkpoint recovery (why the model exists locally)

HF cache was found gutted (only 4/18 files; ~30 GB of orphaned `.incomplete` partials — cleaned). Operator LAN-copied 5 shards from shane-pc; CRLF-mangled text files repaired from byte-canonical staging; **canonical dir: `/srv/dev/ComfyUI/models/text_encoders/diffusiongemma-26B-A4B-it-int4-AutoRound/` — 18/18 byte-exact vs HF manifest**. Staging removed. Rig reaches it by symlink.

## Ship mechanics (when the operator says go — unchanged from EOD plus #183 closures)

CHANGELOG · version bump · PR #135 disposition (likely superseded) · merge `pre-0.5.0-release` → `main` (carries the ARCHITECTURE.md trim → closes #134/#137, and the #183 fix) · tag · registry publish · re-check 0.4.1 registry activation (`Pending`) · drop stash@{0} · close #169, #183, #145.

## Operator decisions pending

- **Ship call** (the only gate left).
- #175 tooltips: in 0.5.0 or ride 0.5.1 (EOD recommendation was 0.5.1).
- **shane-pc caveat**: 24 GB RTX-3090 < the 30 GiB INT4 floor — autoround now refuses loudly there by design; block-wise onload is the deferred remedy (recorded on #183/#185). INT4 on shane-pc waits for that work.

## Banked this session (beyond #183)

#184 quant auto-detect (post-#183, unified block is its landing surface) · OD#41 operator-instincts-into-physics (schema-at-the-door payload structure + uniformity-pin DRY) · #167 schema-first signal for DGEMMA_RUN_RECORD · #182 closed (field-resolved, symlink was mis-created) · HT#225 instances 3–4 + masquerade-correction (mid-thought stops induced a duplicate ~100k-token dispatch).

## Residuals

`comfyui_detail.log` untracked on main (undispositioned, pre-session) · rig + run1–4 evidence + `/tmp` plan scratch all volatile until host reboot (durable copies: #183 comments, #145 readbacks, in-repo experiment entry) · PR #185 remote branch retained · #119 hand-merge lane unchanged · post-0.5.0 queue otherwise as in the EOD handoff.
