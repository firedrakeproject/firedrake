---
name: review-brief
description: Write a reviewer's brief for a mathematically heavy Firedrake pull request — what the method is, where each step lives in the diff, which invariants must hold, and which claims no test pins. Use when the user asks to "explain this PR for review", "write a review brief", "make this PR reviewable", or asks how to make a large numerical change easier to review.
argument-hint: <PR_NUMBER | branch | empty for current branch>
---

Produces a document that makes a numerically heavy change reviewable — not a summary, and not a
substitute for the reviewer's judgement.

A change of this kind arrives carrying context that exists only in the author's head: which
formulation was chosen and why, what it assumes, what must hold if it is right. Review stalls
because that context was never written down, not because the diff is unreadable. The brief writes it
down, and in doing so converts "trust me, the mathematics is right" into "here is the assertion that
fails if it isn't."

Whether a stated invariant is pinned by a test, whether a claimed generality is actually exercised,
and whether the code computes what the prose says it computes are all checkable without re-deriving
anything. Surfacing those three is the whole job.

## 1. Identify and gather

Follow @../review-pr/identify.md (Sections 1–3) to resolve the PR, fetch metadata, and establish the
base branch and stack position.

Then gather, in this order:

1. The PR title, body, and any linked issue or discussion.
2. The full diff (`gh pr diff <PR_NUMBER>`, one call).
3. The tests in the diff — read them completely. They are the evidence base for Section 6.
4. The surrounding code the change plugs into, enough to describe the interfaces honestly.

## 2. Interview the author before writing

The author is available. **Ask them rather than inventing rationale.** Do not guess at motivation,
provenance of a formula, or why one formulation was chosen over another — a plausible-sounding
invented justification is worse than an admitted gap, because a reader cannot tell the two apart.

Ask about anything you cannot establish from the diff and the literature:

- Where a formula, estimator, or algorithm comes from (paper, thesis, textbook, derived here).
- Why this formulation rather than an obvious alternative.
- What is known to be unsupported or approximate, and deliberately so.
- Which properties the author believes hold but has not tested.

Batch the questions into one round. Mark anything still unresolved as an open question in the brief
rather than filling it in.

## 3. Write the brief

Follow @brief-template.md for the structure.

Style rules, which matter more here than in an ordinary review:

- **Define every symbol and every term of art at first use.** If the brief uses "prolongation",
  "injection", "star forest", "goal-oriented", "sum factorisation", "macroelement" — define it, once,
  plainly. Assume fluency in Python, MPI, PETSc, and software architecture; introduce the finite
  element theory the change actually relies on rather than assuming it.
- **Never assert that the mathematics is correct.** Not "this is sound", not "the derivation checks
  out", not "correctly implements". You cannot verify it, and a confident endorsement is precisely
  the failure mode the brief exists to prevent. Describe what is computed, state what must hold, and
  point at the evidence.
- Distinguish three things explicitly and never blur them: what the PR **claims**, what the code
  **demonstrably does**, and what the tests **pin**.
- Prefer one concrete worked case (two cells, lowest degree, one refinement) over an abstract
  statement, wherever it fits.
- Cite literature specifically enough to look up: author, year, and which section to read.
- No praise. Do not call anything elegant, clever, or well-designed.

## 4. Output

Write the brief to `/tmp/.../review-brief-<PR_NUMBER>.md` in the scratchpad directory, and print the
path. Do **not** post it to GitHub as part of this skill.

Posting is a separate, explicitly requested step. If the user asks for it, post as a **single**
top-level PR comment (`gh pr comment`), never as inline review comments, and prepend the disclosure
line required by Firedrake's AI contribution policy — see `AGENTS.md` and the
[AI contribution policy](https://github.com/firedrakeproject/firedrake/wiki/AI-contribution-policy).
The brief is authored material published under the author's name: they must read it in full and
stand behind it before it goes anywhere near the PR.
