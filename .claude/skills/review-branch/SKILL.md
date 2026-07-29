---
name: review-branch
description: Review a local Firedrake branch's changes against its target branch (origin/main or origin/release). Use when the user asks to "review this branch", "review my changes", or "check what I've done before pushing".
argument-hint: <branch | commit-ref | empty for HEAD>
---

Read `AGENTS.md` at the repository root before reviewing.

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that is not a single ref matching
`^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`.

## Resolve `DEST`

Firedrake branches target `main` or `release`, and feature branches are often stacked on other
feature branches. Resolve the base in this order:

1. If a PR exists for this branch, its base is authoritative:
   `gh pr list --head <SRC> --json number,baseRefName`. Use `baseRefName` as `DEST`.
2. Otherwise choose between the two long-lived branches by ancestry:

   ```
   MB=$(git merge-base origin/main <SRC>) && git merge-base --is-ancestor "$MB" origin/release && echo origin/release || echo origin/main
   ```

If `origin/main` does not resolve (`git rev-parse --verify -q origin/main` exits non-zero), first run
`git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main`,
then retry. On any other failure: abort and report — do not guess `DEST`.

## Capture the diff

State `DEST`, then:

- `git diff --stat <DEST>...<SRC>` to size the change
- `git diff <DEST>...<SRC>` to capture it

The three-dot form diffs against the merge base, which is what you want for a stacked branch. The
captured diff contains every file — do **not** re-run `git diff` per file. Options such as `--stat`
must precede the revision arguments.

If the working tree is dirty, say so and state whether you are reviewing committed work only
(`<SRC>` = `HEAD`) or including uncommitted changes (`git diff <DEST>` with no `<SRC>`).

## Review

Follow @../review-pr/review-procedure.md (Sections 4–7) to classify, verify, and compose findings.
Section 4's `gh pr diff` instruction does not apply — you already have the diff.

Print the report. Do not write files and do not post anything to GitHub.
