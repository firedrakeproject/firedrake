---
name: pr-description
description: Write the description for a Firedrake pull request, comparing only the target branch's pre-merge and post-merge states. Use when the user asks to "write a PR description", "describe this PR", "draft the PR body", or wants help before running `gh pr create`/`gh pr edit`.
argument-hint: <PR_NUMBER | branch | empty for current branch>
---

Read `AGENTS.md` at the repository root before writing.

A PR description compares exactly two states: the target branch before the merge, and the target
branch after it. It is not a log of how the branch got there, and it is not addressed to anyone who
watched it happen.

## 1. Resolve the diff

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Follow @../review-branch/SKILL.md ("Resolve `DEST`") to
find the target branch, then capture `git diff --stat <DEST>...<SRC>` and `git diff <DEST>...<SRC>`.
If a PR already exists for this branch, also pull its current title/body (`gh pr view <N> --json
title,body`) as a starting point rather than discarding it.

## 2. Gather the root cause

State only what the diff and the commit history actually establish — do not invent a rationale.

- Read the commit messages in `<DEST>..<SRC>` for the "why", not just the "what".
- If a defect is being fixed, identify its symptom (what a user or test would have observed) and its
  root cause (the specific assumption or code path that was wrong), not just the line that changed.
- If this branch fixes a regression, `git blame` the faulty line(s) to name the upstream PR.
- If anything is unclear from the diff and history alone, ask rather than guess.

## 3. Write for a reviewer with zero context

The reviewer sees only the diff's two endpoints — `<DEST>` and `<SRC>` — never the intermediate
states the branch passed through while it was being developed. A sentence that describes one of
those intermediate states sends the reviewer looking for code that appears nowhere in the diff, and
reads as a grammar error rather than as missing context.

Never write to the branch's author or to whoever produced it ("I found...", "we then tried...",
"after some debugging..."). Write as if narrating the diff itself to a stranger.

## 4. Get the verb tenses right

This is the part that goes wrong most often. Every sentence describes one of exactly two points in
time — before the merge, or after it — and the tense must say which:

| Referring to | Tense | Example |
|---|---|---|
| The defect, as it stood before this PR | past | "`refine_marked_elements` **scanned** every label stratum per fine cell." |
| What the old code should have done instead (state the expectation before what it did wrong) | "should" + infinitive | "The lookup **should** have used PETSc's value→stratum hash map." |
| The fixed behaviour, as it stands after merge | present, usually with "now" | "The rewrite **now** walks each stratum once." |
| A regression's origin, if applicable | past | "PR #NNNN **introduced** this when it switched to per-point lookups." |

Never use future tense ("this PR will fix...", "this change will make..."). By the time anyone reads
the description, the diff already exists — the fixed state is not a future promise, it is what
`<SRC>` already *is*. Write "now scatters", not "will scatter".

Do not narrate the PR itself as an agent ("this PR adds...", "this commit changes..."). Describe the
code's before/after behaviour directly; the diff is not the subject of its own sentences.

## 5. Structure

1. **Executive summary, first**: symptom, root cause, key fix, in that order, in a few sentences.
   Name the upstream PR by number if this fixes a regression.
2. Supporting detail only where the summary compresses something a reviewer needs to verify the fix
   is complete (e.g. why a particular loop was quadratic, not just that it was).
3. Any non-obvious collateral change — a rewritten comment, a renamed parameter, a docstring that
   moved — gets one line. Do not itemize mechanical changes (formatting, import reordering).
4. No test plan section. State that tests were added or updated; do not list the commands to run
   them.

## 6. Output

Print the drafted title and body. Do not run `gh pr create` or `gh pr edit` — opening or editing a
PR is a visible action on a shared system, and posting it is a separate step the user asks for
explicitly, same as every other GitHub-writing action.
