# Working with Claude Code on this repo

## Critical rules for data sourcing

1. **All numbers in main.tex must be grep-traceable to source files**
   (STATE.md:LINE, experiments/path/file:LINE, or
   NEURIPS_EXPERIMENT_INVENTORY.md:LINE).

2. **Always use `grep -R` (capital R), not `grep -r`.**
   Several experiment subdirectories are symlinks to /data/short/...;
   lowercase -r does not follow directory symlinks and will produce
   false "not found" results.

3. **Never introduce numbers from memory** when drafting LaTeX.
   Always grep original files first. If a number cannot be
   grep-traced, it is presumed fabricated until proven otherwise.

4. **Commit messages must include source paths**, not platitudes
   like "all numbers checked" or "triple-verified".

5. **Before commit**: run `git rev-parse --show-toplevel` to confirm
   the correct worktree. This repo has multiple worktrees; cwd can
   reset silently after `git stash pop`.

6. **CC drafts, user reviews, then CC applies.** Never apply diffs
   pre-approval.

7. **After apply**: post compile log + PDF screenshots before
   declaring "done". "I applied the changes" is not sufficient.

## Known issues (not to be fixed without discussion)

- fig:overview (Figure 1 system diagram) overflows \textwidth by
  ~128pt. Pre-existing; deferred to camera-ready visual polish.

## Recent data integrity audits

- 2026-04-21: Full reverse-grep audit of main.tex passed.
  155/157 unique numbers sourced; 2 remaining are rounding/alignment
  artifacts (63.79 rounds from 63.7871; 24.40 pads from 24.4).
  No fabricated data detected.
