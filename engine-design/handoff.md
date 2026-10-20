# Code-harness handoff: start here

Implement the text engine described in this package, using the phase workflow below. Do not replace the work with another plan-only pass: I0 completes the explicitly missing source/fixture work, then I1 starts the implementation. The user authorized staged implementation and technical gates without making the user the C++ reviewer. The operator's early hands-on checkpoints are I2 and I3; public distribution waits for all current nodes and I5 qualification.

## Retrieve the isolated branch

From the user's existing CDG repository, inspect `git status --short` and `git worktree list --porcelain` first. Do not reset, stash, clean, merge or modify other work. Fetch only the new branch:

    git fetch origin refs/heads/design/dg-engine-20261019:refs/remotes/origin/design/dg-engine-20261019

If the local branch does not exist and the destination path is unused:

    git worktree add --track -b design/dg-engine-20261019 ../CDG-dg-engine origin/design/dg-engine-20261019

If the branch already exists, use its existing worktree or add it without `-b` at a fresh path after checking it is not checked out elsewhere. A detached worktree is not permission to create another integration branch. Do not force a branch reset to match remote. Compare any local divergence and record its disposition before writing.

The chat's local package-only worktree is not a full clone. Fetching the remote branch is the normal handoff path; every original CDG file remains in its base tree. All changes from this design are under `engine-design/` plus `.github/workflows/dg-engine-design.yml`.

## Verify the tooling, then dispatch I0

Run from the resulting worktree root:

    cmake -S engine-design -B ../dg-engine-design-checks
    cmake --build ../dg-engine-design-checks
    ctest --test-dir ../dg-engine-design-checks --output-on-failure --no-tests=error
    python engine-design/tools/workflow.py plan text
    python engine-design/tools/workflow.py packet text I0

Read `sources.md`, `requirements.json`, `architecture.md`, `protocol.md`, `decisions.md`, `verification.md`, `workflows/README.md` and `workflows/phase-scope.md`. Give Explore the missing-source inventory, not an unbounded repository search. Its durable output is the source lock, all-node-mode inventory and conflict dispositions. Assign a separate verifier and fresh model and systems/security reviewers. They must inspect source and evidence, not merely paraphrase this design.

The C header is a proposed facade with declarations only. Implement the smallest source-correct native model/step core first. Finalize export visibility, error ownership, allocation/fault semantics and exact phase transitions in the I1 contract review. Do not assume a header compiling proves an implementation can link or run. Do not mistake `src/llama-ext.h` for a stable stock-llama plugin ABI.

## Execute and promote each implementation phase

Use the cumulative test IDs in text.json and the concrete cases in verification.md plus phase-scope.md. Implement tests before claiming the corresponding capability. Bank raw failed and passing results with the actual code, model, tokenizer, converter, binary, backend and device identities. Match measurements to the exact candidate commit and a clean source worktree. Large traces use checksummed evidence-store pointers; do not inflate the Git repository with model weights or leave irreplaceable evidence only in a console.

After separate verifier/reviewer sessions complete, create a receipt following workflows/README.md and run:

    python engine-design/tools/workflow.py hashes text
    python engine-design/tools/workflow.py check text I0 /path/to/bank/I0.json --evidence-dir /path/to/bank --commit <tested-commit-sha>

The harness obtains the tested SHA from Git, not from the receipt. It checks that prerequisite candidate SHAs are ancestors and that report identities are genuine separate sessions. Every later gate reruns all cumulative required groups at its own candidate. Missing weights, busy GPUs, zero collection, skips, failed review or unavailable evidence remain BLOCKED, not PASS. Rework within scope belongs to the technical roles; changes to product intent return as a documented scope question, not a C++ quiz.

A GPU lease is required before a real-weight run. Do not kill or displace unrelated work. The initial numerical oracle is the exact HF reference, compared with unquantized GGUF before Q8/Q4. The draft PR implementations are comparison/donor arms, not correctness authorities. Preserve raw failures and calibrate error tolerances before testing candidate changes.

After a gate passes, bank its evidence pointer and create an immutable annotated implementation tag. Distinguish the tested candidate from a later evidence-only commit: bind the receipt to the candidate SHA, tag that candidate, and identify the evidence-only descendant explicitly. Do not claim a newly changed source commit was tested under its parent's SHA. Push only this branch and the exact phase tags; never `--tags`, force-push or overwrite existing tags. Use `Assisted-by: GPT-6 Astra Pro` when this design/code materially contributed, alongside the executing harness's actual identity.

## Image lane and upstream lane

    python engine-design/tools/workflow.py packet image M0

This is planning only. Image gate checking refuses because the workflow is disabled. Do not add image inference code without a later scope change. Text adapters must leave a path for libmtmd/Gemma4 image preprocessing and encoder inputs without implementing it now.

A distributable CDG engine and an upstream-acceptable patch set are separate deliverables. Keep model/conversion changes, minimal step APIs, controller/example and backend optimizations reviewable separately. Existing ADR-CDG-020 restrictions need explicit amendment/supersession when implementation adopts owned patches or a full native backend. Do not silently reinterpret that ADR. No upstream PR, issue/comment on the user's behalf, public binary release or license change is authorized by this handoff. Upstream contribution needs a qualified human owner; that role need not be the operator.
