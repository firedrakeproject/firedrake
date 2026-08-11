# Firedrake

Firedrake is an automated system for the portable solution of partial differential equations using
the finite element method (FEM). The codebase is primarily Python, relying heavily on code generation
and high-performance C backends to achieve scalability and speed.

Firedrake's full contribution process is documented at
[Contributing to Firedrake](https://firedrakeproject.org/contribute.html). In short, for AI-assisted
contributions: declare that AI was used and which tool; a human must lead the PR, understand every
change, and answer reviewer questions themselves rather than relaying them to the AI; the code must
have been run locally to confirm it works; and AI should not be used to close issues labelled
'good first issue'. The full, authoritative conditions are in the
[AI contribution policy](https://github.com/firedrakeproject/firedrake/wiki/AI-contribution-policy).

## Project Architecture

Firedrake solves variational problems discretized with finite elements through a coordinated
toolchain:

* **PETSc:** Firedrake relies on PETSc (specifically `DMPlex`) for scalable mesh management and
  parallel data distribution. PETSc's `PC`/`KSP`/`SNES` are used extensively as the underlying
  preconditioners and linear/nonlinear solvers, respectively.
* **UFL (Unified Form Language):** Users symbolically specify their variational problems and forms
  using UFL.
* **TSFC (Two-Stage Form Compiler):** TSFC automatically generates highly optimized C code to assemble
  the UFL integrals, in two stages:
  1. **Lowering to GEM:** TSFC lowers UFL into the GEM tensor language. GEM represents expressions over
     quadrature points involving compile-time pre-tabulated basis functions provided by **FInAT** and
     **FIAT**.
  2. **Lowering to Loopy:** The GEM expressions are then lowered into **loopy** kernels.
* **PyOP2:** Finally, the generated loopy kernels are wrapped and executed by PyOP2, which handles the
  parallel execution of loops over mesh cells and facets.

## Core Working Rules

* **Mathematical Root Causes:** Fix the underlying mathematical or architectural cause. Do not patch
  individual failing test cases.
* **Generality Over Complexity:** Rely on the mathematical generality of finite elements. Do not add
  special-case bookkeeping or branching.
* **Unified Abstractions:** Do not branch on cell type, polynomial degree, element family, or serial vs.
  MPI-parallel execution. Use the UFL/TSFC/PyOP2/PETSc abstraction that already handles it — see
  Anti-Patterns.
* **Preserve Style:** Match the naming and patterns of the package you are editing. Keep edits minimal
  and local to the requested change.
* **Avoid Duplication:** Reuse or extend nearby logic instead of duplicating it. Do not add speculative
  abstractions or broad refactors unless asked.
* **Verify API Shapes:** Read a Firedrake, UFL, or PETSc/petsc4py API's current signature from the
  installed source before calling it, unless you have just seen it used in this codebase. `fdk deps`
  finds where each component package lives.
* **Document The Present, Not The Past:** Do not describe a removed or rejected approach in a comment or
  docstring. Document only what the current code does. Checked by `fdk prose` — see Anti-Patterns.

## Coding Style And Conventions

* **Class Attributes:** Declare every attribute in `__init__`, or as a `functools.cached_property` for
  state that is expensive to compute eagerly. Do not discover an attribute via
  `hasattr`/`setattr`/`getattr`. See Anti-Patterns; the setup-guard case is checked by `fdk prose`.
* **No Python Mesh Loops:** Never iterate over degrees of freedom or cells with a Python `for` loop.
* **Prefer Code Generation/PETSc:** Implement mesh-level or DoF-level operations through PyOP2-driven
  kernels or DMPlex, via `petsc4py` or Firedrake's Cython wrappers. See Anti-Patterns.
* **NumPy For Vectorized Work Only:** Use NumPy for index computation and vectorized pre/post-processing.
  Do not iterate a large array element-by-element, or touch the same whole array repeatedly outside one
  vectorized expression.
* **Docstrings Are Always `numpydoc`:** Use numpydoc section headings (`Parameters`, `Returns`, `Raises`,
  `Notes`) in every docstring you write or touch, including private helpers and Cython functions in
  `firedrake/cython/*.pyx`. Never the old Sphinx field-list style (`:arg x:`, `:param x:`, `:returns:`,
  `:rtype:`), even where the surrounding file already uses it. Tests (`tests/**`) need only a one- or
  two-sentence summary. Checked by `fdk prose`.
* **Plain English (ASD-STE100) In Docstrings And Comments:** Short sentences, one idea each, active
  voice, subject named up front rather than buried in a relative clause. Checked by `fdk prose` — see
  the Clause-Stacked anti-pattern below.
* **Type Hints:** Add type hints to new function and method signatures, and to any parameter or
  return value you add to an existing one. The codebase is mid-migration and inconsistently typed
  elsewhere — do not retrofit a signature beyond what you touch.
* **Demos Are Literate Programs:** Keep `demos/<name>/<name>.py.rst` prose and code in step. A paragraph
  ending in `::` makes the following indented block executable; a `.. code-block:: python` directive
  renders in the docs but does not run. Prefer `::`.

## Testing Requirements

* Add tests that demonstrate the new feature or bug fix, in the existing test file for that module —
  do not create a new one. `fdk testfile <path>` finds it: Firedrake's test layout does not mirror
  its source layout, so the file's own basename is not a reliable guide.
* When behavior changes, update the affected tests and confirm parallel (MPI) runs match serial results.
* Add or update the narrowest test that proves the change.

## Pull Request Expectations

* All changes land through GitHub pull requests. Keep diffs focused.
* Before requesting review: `fdk lint`, `fdk prose --range main...HEAD`, and the relevant `fdk test`.

## Development Toolchain

### Agent Tools

`.agents/tools/fdk` is the required interface for tests and lint in this repo, not one option among
several: a bare `pytest` or `flake8` is denied outright, by a `PreToolUse` hook this checkout runs
by default. It is not on `PATH`, so call it by that path (or symlink it onto your own). Run
`.agents/tools/fdk help` for the full reference; the commands in daily use:

```bash
.agents/tools/fdk test <nprocs> [paths]                 # tests at a process count, one deduplicated summary
.agents/tools/fdk testraw <nprocs> [paths]              # as test, unfiltered, to read a traceback
.agents/tools/fdk baseline <nprocs> [paths]             # which failures this branch introduced
.agents/tools/fdk lint [paths]                          # make srclint, or flake8 on given paths
.agents/tools/fdk prose [--range base...head] [paths]   # prose rules, on an edit or over a branch
.agents/tools/fdk explain [--range base...head] [paths] # each added comment beside its code
.agents/tools/fdk testfile <path>                       # the test file(s) that cover a source file
.agents/tools/fdk show <file> <name>                    # one function or class, found by name
.agents/tools/fdk deps [name]                           # component packages: location, branch, dirty state
.agents/tools/fdk stack                                 # the PR stack, and how far each branch has drifted
.agents/tools/fdk pr <number> [--title ...] [--body-file ...]  # retitle/redescribe a PR
.agents/tools/fdk build / clean / py [args] / status
```

Use `fdk`, not the shell line it wraps: it always calls the virtual environment's interpreter,
filters parallel tests correctly, and needs no configuration in a normal checkout.

Run `fdk prose` on files you edit, and `fdk prose --range main...HEAD` on the whole branch before
requesting review. `check-prose.py --help` prints the `PostToolUse` hook config that runs it on
every edit automatically inside Claude Code; that is one way to drive it, not the only one.

### Environment Setup

* **Editable installs across the stack:** Install components in editable mode (see
  ["Editing subpackages"](https://firedrakeproject.org/install.html#editing-subpackages)) so source
  edits take effect without reinstalling. `fdk deps` reports each component's location, branch, and
  dirty state — check it before assuming a fix belongs in Firedrake itself.
* **`petsc4py`/PETSc version skew:** After switching the PETSc branch or commit under an existing venv,
  rebuild `petsc4py` (`pip install --no-build-isolation -e .`) before doing anything else. A stale
  `petsc4py` fails `import firedrake` with an `undefined symbol: ...` error that looks unrelated to
  PETSc.
* **Caching:** Generated TSFC kernels and compiled PyOP2 code are cached under
  `FIREDRAKE_TSFC_KERNEL_CACHE_DIR`/`PYOP2_CACHE_DIR` (default `$VIRTUAL_ENV/.cache/{tsfc,pyop2}`), set
  in-process by `firedrake.configuration.setup_cache_dirs()` on `import firedrake`. TSFC keys a cached
  kernel on the form and the compiler parameters, and PyOP2 keys compiled code on the generated source.
  Neither keys on the code generator, so an edit to `tsfc/`, `pyop2/`, FIAT or UFL leaves every kernel
  already on disk in place. `fdk test`, `fdk testraw` and `fdk baseline` compare a fingerprint of those
  sources against the caches and clear them when it has moved, so you do not have to remember `fdk clean`.
* **A stale kernel reads as a wrong answer, not as a stale kernel:** the run imports your edited Python
  and executes someone else's C. The result is a plausible number, a solver that diverges, or a test
  that fails on numerics. Treat any code-generation change whose test result you have not re-run
  from cleared caches as unmeasured.
* **Smoke test after install/rebuild:** `firedrake-check` runs a small grouped-by-process-count subset
  of the regression suite; use it to sanity-check an environment before a full test run.

### Testing

* **Attribute a failure before you analyse it.** Run `fdk baseline <nprocs> <paths>` on any failure
  you are asked to fix, before reading code. It reports which failures this branch introduced, which
  it fixed, and which it shares with the merge base. A failure the merge base also has is not yours.
* **A premise you were handed is not evidence.** "This passes on main", "this test is untouched, so
  the regression is in my change", and "the other configuration works" each name a fact that one
  command settles and that hours of reading cannot. Check the ones your search depends on first. A
  branch is often behind main as well: `git log --oneline HEAD..origin/main | wc -l` says how far, and
  a failure that main has already fixed is not a bug in this branch.
* Tests that must run under MPI are marked `@pytest.mark.parallel` (optionally
  `@pytest.mark.parallel(nprocs=N)` or `@pytest.mark.parallel([1, 3])`); an unmarked test's own nprocs
  is 1. Use `fdk test <nprocs> <paths>` or `fdk testraw`, never a bare `pytest`, on parallel-marked
  tests.
  `firedrake-run-split-tests <nprocs> <njobs> <pytest args> <paths>` shards a run the way CI does
  (`.github/workflows/core.yml`); run it from a scratch directory.
* Pick `<nprocs>` from what the target actually declares, not from a guess: `grep -n
  'pytest.mark.parallel' <path>` lists its markers. `fdk test`/`fdk testraw` select zero tests, and
  say so on stderr, when nothing in the given paths is marked for the `<nprocs>` you passed.
* Run the relevant subset, not a plain serial `pytest <dir>`: the tests that exercise the lines you
  changed, at the process counts where those lines are live.
* Reproduce narrowly first: run the single failing test node (`pytest path::test_name -k ...`) before
  the full module.

### Debugging

* **Generated kernels (niche, rarely needed):** Set `PYOP2_DEBUG=1` to compile generated C with
  `-O0 -g`, the prerequisite for `gdb`/`cgdb` on a compiled kernel.
* **Cross-rank code-generation mismatches:** `CompilationError: Generated code differs across ranks`
  dumps the mismatching per-rank source under `<cache_dir>/mismatching-kernels/src-rank*.c`. Fix the
  Python-level value that is computed differently per rank and fed into code generation — make that
  decision the same on every rank, rather than patching the generated source.
* **Parallel deadlocks (niche, rarely needed):** `PYOP2_SPMD_STRICT=1` adds barriers around
  `@collective` calls and cache access, to narrow down where ranks disagree about control flow.
* **Logging:** `firedrake.logging.set_log_level()` (or `PYOP2_LOG_LEVEL`) sets Firedrake/PyOP2 log
  verbosity, independent of PETSc's `-log_view`/`-info`.
* **PETSc-level diagnostics:** Pass PETSc options (`-ksp_view`, `-snes_view`, `-ksp_monitor`,
  `-log_view`, `-start_in_debugger`) through `solver_parameters` or the command line, as in any PETSc
  application.

### Reproducible Environments

* **Docker:** Pull one of the published images from
  [Docker Hub](https://hub.docker.com/u/firedrakeproject) (e.g. `firedrakeproject/firedrake:latest`,
  or `:dev-main`/`:dev-release` for the latest commit on each branch — see the
  [install docs](https://firedrakeproject.org/install.html#docker)) to rule out "works on my machine"
  environment drift before chasing a hard-to-reproduce bug.

## Anti-Patterns

These must be avoided when writing code, and flagged when reviewing it.

### Branching On Discretization Or Execution State

WRONG — Writing manual `if/else` logic to handle specific cell geometries or process counts, when the
same call already works uniformly across both branches:

```python
# Anti-pattern: special-casing mathematics instead of relying on abstractions
def stable_timestep(mesh, velocity, cfl=0.5):
    if mesh.ufl_cell().cellname() == "triangle":
        h = CellDiameter(mesh)
    elif mesh.ufl_cell().cellname() == "tetrahedron":
        h = CellDiameter(mesh)
    else:
        raise NotImplementedError("Unsupported cell type")

    DG0 = FunctionSpace(mesh, "DG", 0)
    dt_field = assemble(interpolate(cfl * h / velocity, DG0))
    local_dt = dt_field.dat.data_ro.min()
    if mesh.comm.rank == 0:
        # Anti-pattern: a collective call reachable from only one rank deadlocks
        # every other rank waiting to enter the same reduction
        dt = mesh.comm.allreduce(local_dt, op=MPI.MIN)
    else:
        dt = local_dt
    return dt
```

RIGHT — `CellDiameter` is defined uniformly for every cell type, so the branch on `cellname()` does
nothing but duplicate the same call. A per-cell timestep comes from interpolating into a `DG0` space,
and its global minimum from a PETSc `Vec`'s own collective `min()`, called unconditionally by every
rank rather than hand-rolled behind a `mesh.comm.rank` guard:

```python
def stable_timestep(mesh, velocity, cfl=0.5):
    h = CellDiameter(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    local_dt = assemble(interpolate(cfl * h / velocity, DG0))
    with local_dt.dat.vec_ro as v:
        _, dt_min = v.min()
    return dt_min
```

### Dynamic Attribute Assignment Outside Constructors

WRONG — Inventing a work `Function` the first time it happens to be needed, deep inside a method that
may be called many times (e.g. once per nonlinear iteration), instead of declaring it up front:

```python
class ResidualMonitor:
    def __init__(self, F, u):
        self.F = F
        self.u = u

    def __call__(self, snes, it, rnorm):
        # Anti-pattern: scratch Function discovered via hasattr, inside a hot callback
        if not hasattr(self, "_work"):
            self._work = Function(self.u.function_space())
        assemble(self.F, tensor=self._work)
        print(f"iteration {it}: |F| = {norm(self._work)}")
```

RIGHT — Laziness itself is fine — allocating a `Function` is not free, and this monitor may never be
attached to a solve — but express it with `functools.cached_property` rather than ad hoc
`hasattr`/`setattr`. The attribute is declared once, in the class body, and is computed and memoized
automatically on first access:

```python
from functools import cached_property


class ResidualMonitor:
    def __init__(self, F, u):
        self.F = F
        self.u = u

    @cached_property
    def _work(self):
        return Function(self.u.function_space())

    def __call__(self, snes, it, rnorm):
        assemble(self.F, tensor=self._work)
        print(f"iteration {it}: |F| = {norm(self._work)}")
```

### Using `hasattr` As A Setup Guard

WRONG — Using `hasattr` to infer whether one-time setup has already run, by probing for state that
setup is expected to have built:

```python
class KSPWrapper:
    def solve(self, pc, b, x):
        # Anti-pattern: hasattr stands in for "has `_ksp` been built yet?"
        if not hasattr(self, "_ksp"):
            self._ksp = PETSc.KSP().create(comm=pc.comm)
            self._ksp.setOperators(*pc.getOperators())
        self._ksp.solve(b, x)
```

RIGHT — Declare a boolean attribute that describes the state directly, and check that instead:

```python
class KSPWrapper:
    def __init__(self):
        self._initialized = False

    def solve(self, pc, b, x):
        if not self._initialized:
            self._ksp = PETSc.KSP().create(comm=pc.comm)
            self._ksp.setOperators(*pc.getOperators())
            self._initialized = True
        self._ksp.solve(b, x)
```

This is exactly the pattern used by `PCSNESBase` (`firedrake/preconditioners/base.py`), the base class
every Firedrake `PCBase`/`SNESBase` preconditioner inherits: its `__init__` sets
`self.initialized = False`, and `setUp()` dispatches to `initialize()` or `update()` based on that flag
rather than probing for the presence of state built by `initialize()`. A boolean records intent and is
trivially greppable; `hasattr` is indistinguishable from "I forgot to initialize this" until it fails.

### Python-Level Looping Over Mesh-Bound Array Data

WRONG — Pulling mesh/DoF data into a Python `for` loop, whether or not NumPy is involved. The problem
is the loop, not NumPy: a genuinely vectorized NumPy expression over the same array would be fine, but
would still bypass PyOP2/code-generation for anything mesh-bound:

```python
# Anti-pattern: Python-level loop over mesh coordinates instead of a code-generated kernel
coords = mesh.coordinates.dat.data_ro
for i in range(coords.shape[0]):
    do_heavy_math(coords[i])
```

RIGHT — First choice: express the operation in UFL and let TSFC/PyOP2 generate the C code:

```python
# Correct: let PyOP2 and code generation handle the loop
expr = ufl_expression(mesh.coordinates)
assemble(expr * dx)
```

If the computation is not expressible as a UFL form at all (e.g. `do_heavy_math` is some arbitrary,
non-variational, per-DoF transform), there are two sanctioned escape hatches, in order of preference:

1. A PyOP2 direct loop via `firedrake.par_loop`, still generated, cached, and parallelized like any
   other kernel, just without going through a UFL form:

   ```python
   # Correct: a par_loop is still a code-generated, PyOP2-managed kernel
   domain = "{[i]: 0 <= i < A.dofs}"
   instructions = """
   for i
       A[i] = fmax(A[i], B[0])
   end
   """
   par_loop((domain, instructions), dx, {"A": (A, RW), "B": (B, READ)})
   ```

2. A compiled Cython loop over the raw DoF array, following the same pattern Firedrake's own
   `firedrake/cython/` wrappers use for mesh-topology bookkeeping (see below), for the rare case where
   even a `par_loop` kernel is too restrictive (e.g. the transform needs a general-purpose C library
   call that loopy cannot express):

   ```cython
   # heavy_math.pyx, compiled ahead of time -- not a plain Python loop
   import numpy as np
   cimport numpy as np

   def apply_heavy_math(np.ndarray[np.float64_t, ndim=2, mode="c"] coords):
       cdef Py_ssize_t i, n = coords.shape[0]
       for i in range(n):
           heavy_math_c(&coords[i, 0])
   ```

   called from Python as `apply_heavy_math(mesh.coordinates.dat.data)`.

Compute-heavy operations that could otherwise be code-generated bypass Firedrake's parallelization,
cache-optimization, and MPI scaling capabilities, acting as massive performance bottlenecks.

This rule is about Python-level loops. Explicit loops over mesh entities (cells, facets, closures) are
the norm, not an exception, inside Firedrake's own Cython wrappers in `firedrake/cython/`
(`dmcommon.pyx`, `extrusion_numbering.pyx`, `mgimpl.pyx`, `patchimpl.pyx`, ...), which exist precisely
to implement mesh-topology bookkeeping that has no UFL/TSFC representation — e.g. `create_cell_closure()`
in `dmcommon.pyx` loops `for c in range(cStart, cEnd)` to build the FIAT-ordered closure map that later
code generation depends on. These loops are compiled and typed (`cdef`/`PetscInt`,
`@cython.boundscheck(False)`), operating directly on DMPlex point ranges rather than interpreted Python
objects — that combination, not mere placement in a `.pyx` file, is what makes them acceptable. Do not
use this as license to write a plain Python loop over `.dat.data` and call it fine because "Firedrake
has C-level loops elsewhere."

### Clause-Stacked Docstrings And Comments

WRONG — the subject hides inside a relative clause the reader must unwind before finding the verb:

```python
def scale_boundary_nodes(u, factor):
    """Give the nodes a boundary condition constrains their scaled values."""
```

RIGHT — subject named up front, one short sentence, active voice:

```python
def scale_boundary_nodes(u, factor):
    """Scale the values of the nodes that a boundary condition constrains."""
```

### Documenting Code That Is Not There

A reader has only the file in front of them. A comment can describe a removed approach. It can also
argue against a branch the code does not take. Either one sends the reader looking for something
that is not there.

WRONG — the first sentence describes deleted code, and the second argues with an absent branch:

```python
def cell_average(u):
    # This no longer divides by the number of cells, which was wrong when
    # the cells had different sizes. A test for an empty mesh here would
    # return a nan.
    return assemble(u*dx) / assemble(1*dx)
```

RIGHT — say what the present code does, and state the condition it relies on:

```python
def cell_average(u):
    # Divide by the measured volume, so that cells of different sizes
    # contribute in proportion. The caller passes a non-empty mesh.
    return assemble(u*dx) / assemble(1*dx)
```

Some words give this away on sight: "used to", "previously", "no longer", "instead of", "we removed",
"this replaces". Watch equally for "would" when its subject is code that does not exist. An argument
against a branch that nobody can see is still a description of the past.
