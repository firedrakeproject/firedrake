Structure for the brief. Keep every section; write "None." under one rather than dropping it — an
empty Section 7 is itself information.

---

# Reviewer's brief: #\<N> — \<title>

**Base:** `<base branch>` · **Size:** +A/−D across F files · **Stack:** \<position, or "standalone">

If the PR is stacked, name the parent PR and state in one line what the reviewer must take as given.

## 1. What this changes

One paragraph, no mathematics. What could Firedrake not do before, and what can it do now? If the
change is a refactor with no user-visible effect, say that plainly and state what motivated it.

## 2. Notation and prerequisites

A short table: symbol or term → meaning, in plain language. Every symbol used later in the brief
appears here. Include the terms of art, not only the symbols.

Then give, in two or three sentences, the minimum background needed to follow Section 3. Cite a
specific author, year, and section for a reader who wants the real treatment.

## 3. The method

What is actually computed, and why this formulation. Structure it as a short numbered procedure where
possible — reviewers follow steps better than prose.

State the assumptions the method depends on: mesh conformity, element family, degree, dimension, and
serial or distributed execution. Then say what happens when they do not hold. The answer is an
error, a silent degradation, or undefined behaviour.

Where a formulation was chosen over an alternative, say which alternative and why — from the author's
answers in the interview step, not from speculation.

## 4. Mathematics → code

A table mapping each step of Section 3 to where it lives: step → `file:line` → one-line note.

This is what lets the reviewer read the diff in a meaningful order instead of top to bottom.

## 5. Invariants

The properties that must hold if the implementation is right. For each: what it is, why it must hold,
and what a violation looks like from the outside (wrong answer, iteration-count growth, deadlock,
assertion failure).

A Firedrake change of this kind usually has several of the properties below. An operator is
symmetric or self-adjoint. A discrete quantity is conserved. The method is exact on a polynomial
subspace. Iteration counts are independent of the mesh or of the degree. Refinement converges at the
theoretical rate. Serial and MPI runs agree. A transfer round-trip is idempotent. Include only the
ones that genuinely apply.

## 6. What the tests pin

A table: invariant from Section 5 → the test that pins it (`file::test_name`) → what the test would
show if the invariant broke.

Read the tests before filling this in. A test that exercises a code path without asserting on the
property does **not** pin it — say so.

## 7. Claims with nothing pinning them

The most important section. List every property from Section 5 that has no test in Section 6. List
every mathematical claim in the PR body or the docstrings that no test exercises.

For each, state what a test would have to assert. This is the strongest lever the brief provides: a
specific test can be asked for without re-deriving anything.

A claim may be untested on purpose. It is too expensive, an upstream test covers it, or someone
checked it by hand once. Say so, and say which of the three. That is a legitimate answer, and it
belongs on the record.

## 8. Checks that need no re-derivation

Concrete questions answerable with software-engineering judgement alone. Draw them from this change,
not from a generic list. Typically:

- Do the collective calls line up on every rank, including early-return and exception paths?
- Are cache keys complete — does anything that changes the result also change the key?
- Does the public API have numpydoc docstrings and type hints, and do the docstrings describe what
  the code does?
- What is the complexity and memory behaviour as the mesh or degree grows?
- What breaks for downstream users, and is that acknowledged?
- Are new tests in the right existing file, and do they run under MPI where the code is
  parallel-sensitive?

## 9. Reviewability

Honest assessment of whether this PR can be reviewed as a unit. If not, propose split points as
specific commit or file groupings, and say which part carries the risk.

For a stack, give the order to review in and state what each PR assumes from its parent.

## 10. Open questions

Anything the author has not resolved, verbatim, unanswered. Do not fill these in with plausible
guesses.

---

*Footer:* generated with Claude Code (model, date), from `<PR_HEAD_SHA>`. Explanatory only — it does
not verify the mathematics, and nothing in it should be read as an endorsement of correctness.
