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

**3. The comment is false, or names something the code does not touch. Delete the
claim, then decide whether the code needs the rewrite from outcome 2.**

Check every identifier and every concept the comment names. If the code around it does
not touch that thing, the sentence is wrong. Do not soften it. Do not qualify it. A
half-true comment reads as carelessness and costs more than no comment.

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

## Worked example

This is a real exchange from Firedrake PR #5215. The comment:

```python
# Run createNetgenMesh before Mesh construction renumbers the plex.
```

The reviewer: *"But mesh construction doesn't renumber the plex no?"*

The comment was rewritten, which is the mistake this skill exists to prevent:

```python
# The netgen mesh follows the plex's local numbering, so build both from the
# same clone, leaving fine_mesh's own plex untouched.
```

The reviewer: *"Sorry but I understand the new comment even less. What is 'the plex's
local numbering'?"* — and then: *"any suggestion that we do anything to do with the
'numbering of the plex' is simply wrong. It's a very misleading comment **and it makes
me wonder why we are actually cloning the thing**."*

Two lessons. The second attempt was longer, more specific, and worse, because it was
still false. And the false comment did not merely confuse: it put the surrounding code
under suspicion, so the reviewer began to doubt the clone itself.

Outcome 3 then 2 was the fix. The claim about numbering was wrong, so it goes. What the
code actually needed was for the clone's purpose to be visible — a name saying what the
clone is for, so that no sentence has to justify it.

## Notes

Judge only the lines the branch adds. Comments already on `main` are not this branch's
debt, and rewriting them enlarges the diff a reviewer has to read.

Tests and demos still want a short summary of what is being checked and why. This audit
targets explanatory comments inside function bodies, and docstrings that describe
mechanism rather than contract.

Renaming beats commenting, but renaming a public name is an API change. Inside a
function, rename freely.
