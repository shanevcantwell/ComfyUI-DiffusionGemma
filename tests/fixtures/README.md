# tests/fixtures/ — provenance

Fixtures here are byte-identical copies of real ComfyUI run output — not
synthesized — per issue #84's non-mocked-fixtures convention. A leading
comment inside each `.txt` would corrupt the composite blob's own header
parse (F-1's extractor asserts the file's first bytes ARE the timestamp),
so provenance lives here instead, one section per file.

## `count_numerals_2026-07-15T23-57-39_0000.txt`

- **Source:** `/srv/dev/ComfyUI/output/count_numerals_2026-07-15T23-57-39_0000.txt`
- **Copied:** 2026-07-16
- **Run settings:** `EntropyBoundScheduler`, `entropy_bound=0.05`, `t_min=0.4`,
  `t_max=0.8`, `num_inference_steps_requested=48`,
  `num_inference_steps_effective=48`; 12 steps captured. Prompt: "Generate 13
  individual numerals. Then sum the appearances of each numeral appearing in
  that set."
- **Format observed:** inline bold-markdown list, `*   **N:** k time(s)`
  (singular "time" at k=1) — run 1's tally format.

## `count_numerals_2026-07-15T23-59-14_0000.txt`

- **Source:** `/srv/dev/ComfyUI/output/count_numerals_2026-07-15T23-59-14_0000.txt`
- **Copied:** 2026-07-16
- **Run settings:** `EntropyBoundScheduler`, `entropy_bound=0.05`, `t_min=0.4`,
  `t_max=0.8`, `num_inference_steps_requested=48`,
  `num_inference_steps_effective=48`; 17 steps captured. Prompt: "Generate 2
  rows each of 13 individual numerals. Then sum the appearances of each
  numeral appearing in that set."
- **Format observed:** GFM pipe table, `| Numeral | <garbage header> |` — run
  2's tally format, consecutive run to run 1 above, format differed (design
  gate's grounding: "format differed between consecutive runs" is the design
  input this issue exists to handle). The final assembled table (last frame)
  is arithmetically consistent (`Total: 26` across two rows of 13); the
  real in-fixture revision event is `3: 1→2` at the final step (issue #84
  design-gate DECISION F-3).

## Composite blob shape (both files)

Neither file is `dgemma.loop.decode_frames()` output directly — each is a
`DGemmaTrace._format_summary`-shaped header (timestamp+prompt joined by
literal `\n\n` escape sequences, then `scheduler=…`/`steps=…`/
`committed_fraction per step…`/`mask-token corroboration…` lines joined by
real newlines) followed by the per-frame decoded texts `decode_frames`
produced, each frame boundary marked by the literal delimiter
`\n\nthought\n` (escaped-`\n\n` + the `thought` channel-label token +
one real newline) — the chat template's `thought` channel label bleeding
into `decode_frames`'s deliberately-raw (no excision) per-step decode.
`consumers/tally_audit.py`'s `extract_decoded_frames_from_composite_blob`
is the honest-failing extractor that reverses this shape into
`list[str]`, and is what issue #72 will supersede when the forward
schema'd-JSONL path lands (this extractor is legacy-txt-format support,
per issue #84's design-gate ratification comment).
