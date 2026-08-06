### 4. Read the diff

- `gh pr diff <PR_NUMBER>` captures the whole change in one call. Do **not** re-run `git diff` or
  `gh pr diff` per file — the captured diff already contains every file. Page through it if large.
- Ignore generated and vendored content: `.cache/`, `build/`, `*.so`, `*.c`/`*.h` emitted by Cython
  from a `.pyx` in the same diff, and lockfiles. Do flag a mismatch between a `.pyx` change and its
  committed generated output, if any is committed.
- Surrounding unchanged code is context for understanding, not material for review. Report on it only
  when the change makes it wrong.

Act as a senior Firedrake developer. Classify each finding CRITICAL / HIGH / MEDIUM / Style / LOW.
Do not praise the PR.

**Enumerate exhaustively.** List every occurrence of each issue with `file:line`, not a
representative example. If a pattern appears in N places, report all N. Do not collapse repeats into
"and similar elsewhere"; do not stop once you have "enough" findings. Scan every changed hunk before
reporting.

### 5. Firedrake-specific checks

Apply the Anti-Patterns section of `AGENTS.md` in full. Beyond it, weight these:

**Parallel correctness (CRITICAL when violated).** These are the failures that do not show up in a
serial test run and cost days to diagnose:

- A collective call (`comm.allreduce`, `comm.bcast`, PETSc `Vec`/`Mat` assembly, `DMPlex`
  distribution, cache lookups on a `comm`) reachable on some ranks but not others — including via an
  early `return`, a `continue`, an exception path, or a `if comm.rank == 0` guard.
- A decision fed into code generation, or into a cache key, computed rank-locally when it must be
  collective. The symptom is `CompilationError: Generated code differs across ranks`; the fix is
  upstream of the generated source.
- Star-forest / halo handling that assumes leaves cover only owned points, or that ghost values are
  current without an explicit halo exchange.
- Anything that changes the number or ordering of DoFs without a matching update to the parallel
  numbering.

**Discretization generality (HIGH).** Branching on `cellname()`, degree, family, `mesh.comm.size`, or
extruded-vs-not, where a UFL/FInAT/PETSc abstraction already spans the cases. Also: a claim of
generality in a docstring that the implementation does not deliver (e.g. handles simplices only).

**Test coverage (HIGH when a behaviour change lands with no test).** `AGENTS.md` requires tests with
every PR. Check specifically:

- Does a behaviour change have a test that fails without the change? If you cannot identify one, say
  so as a finding — do not assume coverage exists because a nearby test touches the same module.
- Does a change to parallel-sensitive code have a `@pytest.mark.parallel` test? Serial-only coverage
  of a distribution/halo/redistribution change is a finding.
- Are new tests added to an existing test file covering that feature, rather than a new file?

**API drift (MEDIUM).** Firedrake, UFL and petsc4py signatures change; properties become methods and
back. If the diff calls an API in a form you have not seen used elsewhere in this codebase, read the
installed source to confirm the signature before either trusting or flagging it.

**Style (treat at par with MEDIUM).** `AGENTS.md` conventions are review blockers, not nits: numpydoc
docstrings on public APIs, type hints on new signatures, attributes declared in `__init__` or as
`functools.cached_property` rather than discovered via `hasattr`, no comments narrating what the code
used to do.

**Mathematical claims.** Changes of this kind rest on derivations that are not reconstructible from
the diff alone. Calibrate accordingly.

- Report a mathematical error only when you can name the specific term, index, sign, or scaling that
  is wrong **and** state what it should be. "This looks suspicious" is not a finding.
- Never report "I cannot verify this derivation" as a finding. Inability to check is not evidence of
  a defect. If a mathematical claim in the PR is load-bearing and unverifiable from the diff, that
  belongs in a reviewer brief (`/review-brief`), not in this report.
- A mismatch between what a docstring/comment claims and what the code computes **is** a finding, and
  is checkable without verifying the underlying mathematics.

### 6. Verify each finding before reporting

After generating the review, treat every finding at Style or above as tentative. For each one:
reopen the cited code and confirm it matches what the finding describes; reread it and confirm the
issue is real rather than a misread or speculation; confirm it is actionable. Drop findings that fail
any check. Report only those that survive.

For parallel-correctness findings specifically, state the concrete divergence — which ranks take
which path, and where they end up waiting. A collective-asymmetry finding you cannot walk through
rank-by-rank has not been verified.

### 7. Compose report

- Per finding: severity, `file:line`, description, suggested fix. Order CRITICAL → HIGH → MEDIUM →
  Style.
- State the base branch and, for a stacked PR, that the parent's changes are out of scope.
- If nothing at or above Style is found, say so explicitly.
- **LOW** — count, do not list. End with `(N LOW findings suppressed; ask to show them.)` when
  `N > 0`. List individual LOW items only when asked.
