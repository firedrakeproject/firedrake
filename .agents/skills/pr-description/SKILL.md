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
A PR may already exist for this branch. Pull its current title and body (`gh pr view <N> --json
title,body`) and start from those rather than discard them.

## 2. Gather the root cause

State only what the diff and the commit history actually establish — do not invent a rationale.

- Read the commit messages in `<DEST>..<SRC>` for the "why", not just the "what".
- Name the symptom of a defect, and name its root cause. The symptom is what a user or a test
  observed. The root cause is the assumption or the code path that was wrong, not the line that
  changed.
- If this branch fixes a regression, `git blame` the faulty line(s) to name the upstream PR.
- If anything is unclear from the diff and history alone, ask rather than guess.

## 3. Write for a reviewer with zero context

The reviewer sees the diff's two endpoints alone: `<DEST>` and `<SRC>`. The branch passed through
other states while you developed it. The reviewer never sees those. A sentence about one of them
sends the reviewer looking for code that the diff does not hold. It reads as a grammar error, not
as missing context.

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

Never use future tense ("this PR will fix...", "this change will make..."). The diff already exists
by the time anyone reads the description. The fixed state is not a promise. It is what `<SRC>`
already *is*. Write "now scatters", not "will scatter".

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

Print the drafted title and body. Do not run `gh pr create` or `gh pr edit`. A PR is visible on a
shared system, so opening or editing one is a step the user asks for. Every other action that
writes to GitHub works the same way.
