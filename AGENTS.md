# Firedrake

Firedrake is an automated system for the portable solution of partial differential equations using
the finite element method (FEM). The codebase is primarily Python, relying heavily on code generation
and high-performance C backends to achieve scalability and speed.

Two properties are treated as defining features of Firedrake, not optional extras: **composability**
(function spaces, forms, boundary conditions, solvers, and preconditioners are expected to combine
freely, without special-casing particular combinations) and **differentiability** (via `pyadjoint`,
essentially any computation built from Firedrake's own operations — assembly, interpolation,
variational solves, boundary condition application — can be taped and differentiated end-to-end).
Differentiability is expected to fall out of composability: a new feature built from already-annotated
Firedrake operations should be differentiable for free, with no extra work. A feature that instead
reaches past those operations into a lower-level, unannotated API silently breaks this guarantee —
forward runs stay numerically correct, but the adjoint quietly goes wrong, and nothing fails until
pyadjoint is specifically exercised.

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
* **pyadjoint:** Firedrake integrates with `pyadjoint` for algorithmic differentiation. Annotated
  operations (assembly, interpolation, variational solves, boundary condition application, ...) are
  recorded on a `Tape` as a DAG of `Block`s; a `ReducedFunctional` composed with one or more `Control`s
  can then be evaluated, differentiated (adjoint or tangent-linear), and checked with a `taylor_test`.
  Taping happens at the level of Firedrake's own operations, not the underlying numerics, so any new
  feature assembled purely from already-annotated building blocks is differentiable automatically.

## Core Working Rules

* **Mathematical Root Causes:** Bug fixes must address the underlying core mathematical or
  architectural issue. Do not merely patch particular failing test cases or edge cases.
* **Generality Over Complexity:** Avoid increasing code complexity with complicated bookkeeping or
  special-case logic. Firedrake relies on the mathematical generality of finite elements.
* **Unified Abstractions:** Proper Firedrake code avoids branching on the wide range of discretizations
  (e.g., cell type, polynomial degree, or finite element family) or execution states (serial vs. MPI
  parallel). Rely on UFL, TSFC, PyOP2, and PETSc abstractions to handle these variations
  transparently.
* **Preserve Style:** Preserve Firedrake style and naming conventions. Keep edits minimal and local to
  the requested change. Match existing patterns in the package you are modifying.
* **Avoid Duplication:** Avoid unnecessary code duplication. Prefer reusing or extending nearby logic
  when it keeps behavior clear and local. Do not add speculative abstractions or broad refactors unless
  explicitly requested.
* **Do Not Trust Memorized API Shapes:** Firedrake, UFL, and PETSc/petsc4py APIs change over time —
  properties become methods, arguments get renamed, call signatures get deprecated. An LLM's trained
  knowledge reflects a snapshot that may already be stale, and will confidently reproduce the old,
  no-longer-correct form (e.g. calling a now-method as a bare attribute, or vice versa). Before calling
  an API you have not just seen used in this codebase, verify its actual current signature by reading
  the installed Firedrake/UFL/PETSc source rather than relying on memorized patterns.
* **Document The Present, Not The Past:** When fixing code that was wrong, do not leave comments or
  prose explaining what the removed, incorrect approach used to do or why it was wrong. Keep comments
  and documentation focused on the current, correct code; a reader should never need the history of
  what used to be there to understand why the present code is right.
* **Composability And Differentiability:** New features are expected to compose with existing ones
  without special-casing, and, via `pyadjoint`, to remain differentiable when built from already-taped
  operations. Prefer the annotated, top-level API (e.g. `firedrake.assemble`) over a lower-level
  equivalent that bypasses it (e.g. calling an `Interpolator`'s own `.assemble()` directly) even when
  both give the same forward numbers — the lower-level call silently drops out of the tape, and no test
  will notice unless it specifically exercises pyadjoint. When a change could plausibly sit on the tape,
  verify differentiability explicitly with a `taylor_test`, not just a forward-value check.

## Coding Style And Conventions

* **Class Attributes:** Every attribute a class can hold must be declared in one visible place, either
  initialized in the constructor (`__init__`) or, for state that is expensive or unnecessary to compute
  eagerly, declared as a `functools.cached_property`. Avoid discovering an attribute's existence via
  `hasattr`/`setattr`/`getattr` scattered across methods — laziness is fine, ad hoc laziness is not.
* **No Python Mesh Loops:** The Firedrake style strictly avoids using Python `for` loops to iterate
  over degrees of freedom (DoFs) or cells in a mesh.
* **Prefer Code Generation/PETSc:** All mesh-level or DoF-level operations must be implemented using
  PyOP2-driven kernels or DMPlex operations. These should be accessed either through `petsc4py` or
  Firedrake's internal Cython wrappers.
* **NumPy Is Fine, Repeatedly Touching Whole Arrays Is Not:** NumPy is the right tool for index
  computations, small metadata configurations, and vectorized pre/post-processing. The anti-pattern is
  not "using NumPy" but iterating a large array element-by-element (in a Python `for` loop) or
  otherwise touching the same whole array repeatedly outside of a single vectorized expression — that
  is what defeats NumPy's own performance model, on top of bypassing PyOP2/code-generation for
  mesh-bound data.
* **Docstrings:** All public-facing APIs must include properly formatted `numpydoc`-style docstrings.
* **Type Hints:** New code should include type hints on function/method signatures.
* **Keep Annotation Out Of Plain Modules:** pyadjoint bookkeeping never appears inside a method body in
  `firedrake/*.py`. Differentiable types instead have a `*Mixin` in `firedrake/adjoint_utils/` exposing
  one `_ad_annotate_<name>` decorator per method that needs taping, applied where the method is defined
  (`@SomeMixin._ad_annotate_foo` on `def foo`). The decorator wraps the whole method, so the real
  implementation stays pyadjoint-agnostic. Add a new decorator to the `Mixin` rather than calling `_ad_*`
  from `firedrake/*.py` directly.

## Testing Requirements

* **Pull Requests:** All PRs must include comprehensive tests demonstrating that the new feature works
  or the bug is fixed.
* If behavior changes, update the relevant test blocks and ensure that parallel runs (MPI) yield
  correct and identical mathematical results to serial runs.
* Keep tests targeted. Add or update the narrowest test that proves the behavior you changed.
* Do not create new test files for this. Add the new test(s) to the existing test file(s) that already
  cover the feature or module being changed.

## Pull Request Expectations

* All changes are expected to arrive through GitHub Pull Requests.
* Keep diffs reviewable and focused.
* Before concluding work, ensure `make srclint` passes, and verify that the relevant subset of the
  pytest test suite succeeds locally.

## Development Toolchain

### Environment Setup

* **Editable installs across the stack:** A bug can live in Firedrake or in any of its component
  packages (PETSc, petsc4py, UFL, FIAT, FInAT, TSFC, PyOP2, loopy). Follow the
  ["Editing subpackages"](https://firedrakeproject.org/install.html#editing-subpackages) instructions
  in the install docs to get a component installed in editable mode so source edits take effect without
  reinstalling, and check which branch/commit of each component is actually active before assuming a
  fix belongs in Firedrake itself.
* **`petsc4py`/PETSc version skew:** `petsc4py` is a compiled extension built against one specific
  PETSc checkout. If you switch the PETSc branch/commit underneath an existing venv (e.g. to bisect a
  PETSc-side issue) without rebuilding `petsc4py` against it, `import firedrake` fails with a confusing
  `undefined symbol: ...` error from `petsc4py`'s `.so` — not a Firedrake traceback, and easy to
  misattribute to whatever you were just changing. Rebuild `petsc4py` (and re-run
  `pip install --no-build-isolation -e .` for it) after switching PETSc, rather than debugging the
  symptom.
* **Caching:** Generated TSFC kernels and compiled PyOP2 code are cached on disk, under
  `FIREDRAKE_TSFC_KERNEL_CACHE_DIR`/`PYOP2_CACHE_DIR`. These are not pre-set shell variables — do not
  expect `echo $PYOP2_CACHE_DIR` to show anything. `firedrake.configuration.setup_cache_dirs()` sets
  them in-process, defaulting to `$VIRTUAL_ENV/.cache/{tsfc,pyop2}`, as one of the first things
  `import firedrake` does (right after PETSc initialization, before PyOP2 loads) unless you already
  exported them yourself beforehand. This also means that if PETSc initialization itself fails (e.g.
  the version-skew symptom above), these variables never get set at all. If a code-generation change
  does not seem to take effect, or you suspect a stale kernel, run `firedrake-clean` before re-testing
  (it prints the actual paths in use).
* **Smoke test after install/rebuild:** `firedrake-check` runs a small grouped-by-process-count subset
  of the regression suite; use it to sanity-check an environment before investing time in a full test
  run.

### Testing

* **Parallel tests:** Tests that must run under MPI are marked `@pytest.mark.parallel` (optionally
  `@pytest.mark.parallel(nprocs=N)` or `@pytest.mark.parallel([1, 3])` for multiple process counts), via
  the `mpi-pytest` plugin. Plain `pytest test_foo.py` does exercise them: for each parallel test it
  self-forks an `mpiexec` subprocess with the right `nprocs`, one test at a time, which is slow and
  produces one nested pytest report per test. To instead run every `nprocs=3` test in `test_foo.py`
  together, directly under a single outer `mpiexec`, filter on the `parallel[match]` marker that the
  plugin attaches to tests whose `nprocs` equals the launched communicator size:
  ```bash
  mpiexec -n 3 python -m pytest -m "parallel[match]" test_foo.py
  ```
  Tests requiring a different `nprocs` are collected but skipped (not run) by this invocation; do not
  conclude a parallel code path is untested just because a plain, unmarked `pytest` run was green.
* **Splitting for CI:** `firedrake-run-split-tests` shards the suite by process count for CI; look at
  it (and `.github/workflows/pr.yml`/`core.yml`) if a failure only reproduces in CI and not locally.
* **Narrow reproduction first:** Run the single failing test node (`pytest path::test_name -k ...`)
  before the full module; the suite is large and full-module reruns are slow to iterate against.

### Debugging

* **Generated kernels:** By default, generated C is compiled optimized and
  without debug symbols, so a debugger attached to the Python process cannot meaningfully step through
  it. Set `PYOP2_DEBUG=1` to compile with `-O0 -g` instead, which is the prerequisite for using
  `gdb`/`cgdb` on the compiled kernel at all.
* **Cross-rank code-generation mismatches:** If a parallel run raises `CompilationError: Generated code
  differs across ranks`, the mismatching per-rank source is dumped under
  `<cache_dir>/mismatching-kernels/src-rank*.c`. Diffing the two sources only tells you *what* differs;
  the actual fix is almost always upstream of that, in whatever Python-level parameter or branch is
  computed differently per rank and fed into code generation (e.g. a rank-local decision that should be
  a collective/global one) — make that decision the same on every rank, rather than patching the
  generated source or the difference itself.
* **Parallel deadlocks (niche, rarely needed):** `PYOP2_SPMD_STRICT=1` adds barriers around calls
  marked `@collective` and around cache access, trading overhead for a much narrower failure point when
  ranks disagree about control flow.
* **Logging:** `firedrake.logging.set_log_level()` (or the `PYOP2_LOG_LEVEL` environment variable)
  raises verbosity of Firedrake's/PyOP2's own logger, independent of PETSc's `-log_view`/`-info`.
* **PETSc-level diagnostics:** Since the linear/nonlinear solve ultimately runs through petsc4py,
  standard PETSc options (`-ksp_view`, `-snes_view`, `-ksp_monitor`, `-log_view`, `-start_in_debugger`)
  can be passed through Firedrake's `solver_parameters` or the command line exactly as in a plain PETSc
  application.

### PyAdjoint / Differentiability

Lessons from pyadjoint debugging that generalize to any record/replay system (caches, dependency
graphs, memoization). Ordered from quick diagnostic wins to deeper architectural gotchas:

* **Read the warning before the crash.** `Adjoint value is None, is the functional independent of the
  control variable?` (`pyadjoint/control.py`) names the actual gap; a `ZeroDivisionError`/`nan` that
  `taylor_test` raises afterwards is usually just that gap's downstream consequence.
* **Inspect the recorded structure directly instead of guessing from the symptom.**
  `pyadjoint.get_working_tape().get_blocks()` lists every `Block` in order; each one's
  `.get_dependencies()`/`.get_outputs()` `.saved_output` shows exactly which step is missing a
  dependency it should have.
* **A "too perfect" verification result is a red flag, not a pass.** `taylor_test` residuals all
  ~1e-16 usually mean the functional is accidentally *exactly* linear in the control at that point
  (e.g. linear PDE + linear BC + zero initial condition) — introduce real nonlinearity (square the
  control) to get a meaningful check.
* **A dependency check by structural type, not concept, silently skips conforming objects.**
  `firedrake.Constant` is a `ufl.Terminal`, not a `Coefficient`; `AssembleBlock` finds dependencies via
  `form.coefficients()`, so a bare `Constant` records **zero** dependencies. Use a `Function` on a
  `"R"` space instead.
* **State re-derived lazily from construction-time args must stay synced on every mutation.**
  `DirichletBCBlock` rebuilds from `self._ad_args`, fixed at `__init__`; without keeping `_ad_args`
  current on every `set_value`, reuse silently keeps re-deriving from the *original* value (Taylor
  rate 1, a wrong gradient, not 2).
* **A performance cache must resync every axis that varies between reuses, not just the obvious one.**
  `NonlinearVariationalSolveBlock`'s cached solver refreshes form coefficients automatically but not
  its own embedded `DirichletBC` objects — recomputation under a perturbed control silently reused
  stale per-step BC data until `_forward_solve` was fixed to resync them too.

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
