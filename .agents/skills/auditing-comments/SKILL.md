---
name: auditing-comments
description: Audits the comments and docstrings a branch adds, deleting the redundant ones and rewriting the code behind the rest. Use before opening a pull request or asking for review, after finishing a change, or when a reviewer says a comment is unclear, inaccurate, or misleading.
---

# Auditing comments

A comment is a debt the reviewer pays. This audit settles it before a human reads the
diff.

The rule: **a comment that explains what the code does is a bug report against the
code.** Fix the code. Rewriting the comment leaves the defect in place, and a reviewer
who catches one inaccurate comment stops trusting the rest of the branch.

## Workflow

Copy this checklist and check items off:

```
Comment audit:
- [ ] Step 1: List the comments the branch added
- [ ] Step 2: Judge each one against the four outcomes
- [ ] Step 3: Apply the edits
- [ ] Step 4: Re-run, and confirm the survivors are all "why" comments
- [ ] Step 5: Run the tests, because step 3 changed code
```

**Step 1.** Run it:

```bash
.agents/tools/fdk explain --range main...HEAD
```

Each block prints the added comment, then the code it annotates with the comment
stripped out. Read the stripped code first. That is what the reviewer sees.

**Step 2.** For each block, answer one question: *with the comment gone, does the code
still say this?* Then take the matching outcome below.

**Step 3.** Apply the edits. Prefer renaming and extracting over adding words.

**Step 4.** Run `fdk explain` again. Every surviving comment must say *why*, not *what*.

**Step 5.** Rewriting code breaks tests. Run them.

## The four outcomes

**1. The code already says it. Delete the comment.**

```python
# Loop over the cells and sum the volumes.        <- delete
total = sum(cell.volume for cell in cells)
```

**2. The code does not say it. Rewrite the code, not the comment.**

This is the common case and the one that costs review time. Name the intermediate,
extract the helper, or split the expression until the sentence is unnecessary.

```python
# WRONG: the comment carries the meaning
# result is -1 where the coarse cell has no child on this rank
result = np.full(size, -1, dtype=IntType)

# RIGHT: the code carries it
NO_CHILD_ON_THIS_RANK = -1
child_cells = np.full(size, NO_CHILD_ON_THIS_RANK, dtype=IntType)
```

**3. The comment is false. Delete the claim, then ask what the code was for.**

Check every identifier and every concept the comment names. If the code around it does
not touch that thing, the sentence is wrong. Do not soften it. Do not qualify it. A
half-true comment reads as carelessness and costs more than no comment.

Then keep going, because a false comment is seldom only a writing mistake. The sentence
was there to justify the code. If the justification is not true, the code may have no
reason to exist:

```python
# WRONG: the comment states a constraint that is not real
# Copy before assembling, because assembly renumbers the dofs.
work = coefficient.copy(deepcopy=True)
result = assemble(inner(work, v) * dx)
```

Assembly renumbers nothing. Once that is settled, the question is not how to word the
comment. It is what the copy was for. Nothing writes to `work`, so the copy guards
nothing:

```python
# RIGHT: the copy went with the claim that justified it
result = assemble(inner(coefficient, v) * dx)
```

Try this outcome before outcome 2. Rewriting code to carry a sentence is wasted work if
the code should not be there at all.

**4. The reason is genuinely not in the code. Keep the comment, and say only the
reason.**

Legitimate cases: a non-local invariant the caller must hold, why an obvious
alternative was rejected, a citation, a workaround for a named upstream bug. Say
*why*. Never restate *what*.

```python
# `distributeSection` overwrites the section it is handed, so let it build its
# own and only keep the root offsets it broadcasts.
remote_offsets, distributed_section = point_sf.distributeSection(root_section)
```

## Why rewriting the comment is the wrong move

A reviewer says a comment is unclear. The tempting answer is a better comment. It fails
twice over.

The first failure is that the replacement is usually still wrong. The sentence was
written from the same misunderstanding that produced the code, so a second attempt says
the same false thing at greater length. A reviewer who did not follow the first sentence
follows the second one less.

The second failure is worse. An inaccurate comment does not merely confuse a reader. It
puts the code under suspicion, and the reviewer starts asking why the code is there at
all. That question is usually a good one. Code defended by a false claim is often code
that nothing needed, and a reviewer pulling on the sentence is pulling on the real
defect.

So take the question seriously rather than deflecting it with better wording. When a
comment cannot be written truthfully, the reading to try first is that the code beneath
it should go.

## Notes

Judge only the lines the branch adds. Comments already on `main` are not this branch's
debt, and rewriting them enlarges the diff a reviewer has to read.

Tests and demos still want a short summary of what is being checked and why. This audit
targets explanatory comments inside function bodies, and docstrings that describe
mechanism rather than contract.

Renaming beats commenting, but renaming a public name is an API change. Inside a
function, rename freely.
