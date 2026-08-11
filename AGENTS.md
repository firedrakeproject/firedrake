# Firedrake

Firedrake is an automated system for the portable solution of partial differential equations using
the finite element method (FEM). The codebase is primarily Python, with code generation and
high-performance C backends for scalability and speed.

Firedrake's full contribution process is documented at
[Contributing to Firedrake](https://firedrakeproject.org/contribute.html). For AI-assisted
contributions: declare that AI was used and which tool; a human leads the PR, understands every
change, and answers reviewer questions directly; the code has been run locally; AI does not close
issues labelled 'good first issue'. Full conditions:
[AI contribution policy](https://github.com/firedrakeproject/firedrake/wiki/AI-contribution-policy).

## Project Architecture

Firedrake's toolchain, in order:

* **PETSc:** `DMPlex` manages meshes and parallel data distribution. `PC`/`KSP`/`SNES` are the
  preconditioners and linear/nonlinear solvers.
* **UFL (Unified Form Language):** Users write variational problems and forms in UFL.
* **TSFC (Two-Stage Form Compiler):** Generates optimized C to assemble UFL integrals, in two stages:
  1. **Lowering to GEM:** TSFC lowers UFL into GEM, a tensor language for expressions over quadrature
     points, using basis functions pre-tabulated by **FInAT** and **FIAT**.
  2. **Lowering to Loopy:** GEM lowers into **loopy** kernels.
* **PyOP2:** Wraps and executes the loopy kernels, parallelizing loops over mesh cells and facets.

## Core Working Rules

* **Mathematical Root Causes:** Fix the underlying mathematical or architectural cause. Do not patch
  individual failing test cases.
* **Generality Over Complexity:** Rely on the mathematical generality of finite elements. Do not add
  special-case bookkeeping or branching.
* **Unified Abstractions:** Do not branch on cell type, polynomial degree, element family, or serial
  vs. MPI-parallel execution. Use the UFL/TSFC/PyOP2/PETSc abstraction that already handles it — see
  Anti-Patterns.
* **Preserve Style:** Match the naming and patterns of the package you are editing. Keep edits minimal
  and local to the requested change.
* **Avoid Duplication:** Reuse or extend nearby logic instead of duplicating it. Do not add speculative
  abstractions or broad refactors unless asked.
* **Do Not Trust Memorized API Shapes:** Read a Firedrake, UFL, or PETSc/petsc4py API's current
  signature from the installed source before calling it, unless you have just seen it used in this
  codebase — see Anti-Patterns.
* **Document The Present, Not The Past:** Do not describe a removed or rejected approach in a comment
  or docstring. Document only what the current code does — see Anti-Patterns.

## Coding Style And Conventions

* **Class Attributes:** Declare every attribute in `__init__`, or as a `functools.cached_property` for
  state that is expensive to compute eagerly. Do not discover an attribute via
  `hasattr`/`setattr`/`getattr` — see Anti-Patterns.
* **No Python Mesh Loops:** Never iterate over degrees of freedom or cells with a Python `for` loop —
  see Anti-Patterns.
* **Prefer Code Generation/PETSc:** Implement mesh-level or DoF-level operations through PyOP2-driven
  kernels or DMPlex, via `petsc4py` or Firedrake's Cython wrappers.
* **NumPy For Vectorized Work Only:** Use NumPy for index computation and vectorized pre/post-processing.
  Do not iterate a large array element-by-element, or touch the same whole array repeatedly outside one
  vectorized expression.
* **Docstrings:** All public-facing APIs use `numpydoc`-style docstrings.
* **Type Hints:** Add type hints to new function/method signatures.

## Testing Requirements

* Add tests that demonstrate the new feature or bug fix, in the existing test file for that module.
* When behavior changes, update the affected tests and confirm parallel (MPI) runs match serial
  results.
* Add or update the narrowest test that proves the change.

## Pull Request Expectations

* All changes land through GitHub pull requests. Keep diffs focused.
* Before requesting review: `make srclint`, the relevant test subset, and the
  [pre-submission checklist](https://firedrakeproject.org/contribute.html#pre-submission-checklist) in
  `docs/source/contribute.rst`.

## Development Toolchain

### Environment Setup

* **Editable installs:** Install PETSc, petsc4py, UFL, FIAT, FInAT, TSFC, PyOP2, and loopy in editable
  mode (see ["Editing subpackages"](https://firedrakeproject.org/install.html#editing-subpackages)) so
  source edits take effect without reinstalling. Check each component's active branch/commit before
  assuming a fix belongs in Firedrake.
* **`petsc4py`/PETSc version skew:** Rebuild `petsc4py` (`pip install --no-build-isolation -e .`) after
  switching the PETSc branch/commit under an existing venv. A stale `petsc4py` fails `import firedrake`
  with an `undefined symbol: ...` error, not a Firedrake traceback.
* **Caching:** TSFC kernels and PyOP2 code are cached under
  `FIREDRAKE_TSFC_KERNEL_CACHE_DIR`/`PYOP2_CACHE_DIR` (default `$VIRTUAL_ENV/.cache/{tsfc,pyop2}`), set
  by `firedrake.configuration.setup_cache_dirs()` on `import firedrake`. Run `firedrake-clean` if a
  code-generation change does not take effect.
* **Smoke test:** `firedrake-check` runs a process-count-grouped subset of the regression suite; use it
  before a full run.

### Testing

* **Parallel tests:** Tests needing MPI are marked `@pytest.mark.parallel` (`nprocs=N`, or a list for
  multiple counts), run via the `mpi-pytest` plugin. Plain `pytest test_foo.py` self-forks one
  `mpiexec` subprocess per parallel test, one nested report each. Run every test at a given `nprocs`
  together, under one outer `mpiexec`, filtered on `parallel[match]`:
  ```bash
  mpiexec -n 3 python -m pytest -m "parallel[match]" test_foo.py
  ```
  Tests at other `nprocs` are collected but skipped. A green unmarked `pytest` run is not evidence
  that the parallel tests passed.
* **Splitting for CI:** `firedrake-run-split-tests` shards the suite by process count for CI. Check it
  and `.github/workflows/pr.yml`/`core.yml` if a failure reproduces only in CI.
* **Narrow reproduction first:** Run the single failing test node (`pytest path::test_name -k ...`)
  before the full module.

### Debugging

* **Generated kernels (niche):** Set `PYOP2_DEBUG=1` to compile generated C with `-O0 -g`, needed for
  `gdb`/`cgdb` on a compiled kernel.
* **Cross-rank code-generation mismatches:** `CompilationError: Generated code differs across ranks`
  dumps the mismatching per-rank source under `<cache_dir>/mismatching-kernels/src-rank*.c`. Fix the
  Python-level value computed differently per rank and fed into code generation, not the generated
  source.
* **Parallel deadlocks (niche):** `PYOP2_SPMD_STRICT=1` adds barriers around `@collective` calls and
  cache access, to narrow down where ranks disagree on control flow.
* **Logging:** `firedrake.logging.set_log_level()` (or `PYOP2_LOG_LEVEL`) sets Firedrake/PyOP2 log
  verbosity, independent of PETSc's `-log_view`/`-info`.
* **PETSc-level diagnostics:** Pass PETSc options (`-ksp_view`, `-snes_view`, `-ksp_monitor`,
  `-log_view`, `-start_in_debugger`) through `solver_parameters` or the command line, as in any PETSc
  application.

### Reproducible Environments

* **Docker:** Pull a published image from
  [Docker Hub](https://hub.docker.com/u/firedrakeproject) (`firedrakeproject/firedrake:latest`, or
  `:dev-main`/`:dev-release` — see the
  [install docs](https://firedrakeproject.org/install.html#docker)) to rule out environment drift
  before chasing a hard-to-reproduce bug.

## Anti-Patterns

Each pattern below is a WRONG/RIGHT pair to read.

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

RIGHT — `CellDiameter` is defined uniformly for every cell type, so the branch on `cellname()`
duplicates the same call. A per-cell timestep comes from interpolating into a `DG0` space, and its
global minimum from a PETSc `Vec`'s own collective `min()`, called unconditionally by every rank:

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

RIGHT — Express the laziness with `functools.cached_property` instead of ad hoc
`hasattr`/`setattr`. The attribute is declared once, in the class body, computed and memoized on
first access:

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

`PCSNESBase` (`firedrake/preconditioners/base.py`), the base class inherited by every
`PCBase`/`SNESBase` preconditioner, uses exactly this pattern: `__init__` sets
`self.initialized = False`, and `setUp()` dispatches to `initialize()` or `update()` on that flag. A
boolean is greppable; `hasattr` is indistinguishable from a forgotten initialization until it fails.

### Python-Level Looping Over Mesh-Bound Array Data

WRONG — Pulling mesh/DoF data into a Python `for` loop. The problem is the loop, not NumPy — a
vectorized NumPy expression over the same array is fine, but still bypasses PyOP2/code-generation for
mesh-bound data:

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

2. A compiled Cython loop over the raw DoF array — the same pattern used by Firedrake's own
   `firedrake/cython/` wrappers for mesh-topology bookkeeping (see below) — for the rare case where a
   `par_loop` kernel cannot express the transform (e.g. it needs a general-purpose C library call):

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

This rule is about Python-level loops. Firedrake's own Cython wrappers in `firedrake/cython/`
(`dmcommon.pyx`, `extrusion_numbering.pyx`, `mgimpl.pyx`, `patchimpl.pyx`, ...) loop over mesh entities
routinely, for mesh-topology bookkeeping with no UFL/TSFC representation — e.g. `create_cell_closure()`
in `dmcommon.pyx` loops `for c in range(cStart, cEnd)` to build the closure map that code generation
depends on. These loops are compiled and typed (`cdef`/`PetscInt`, `@cython.boundscheck(False)`), on
DMPlex point ranges, not interpreted Python objects. A plain Python loop over `.dat.data` is not fine
merely because "Firedrake has C-level loops elsewhere."

### Documenting Code That Is Not There

A reader has only the file in front of them. A comment can describe a removed approach, or argue
against a branch that the code does not take. Either one sends the reader looking for something that
is not there.

WRONG — the comment explains what the old scaling used to do and why it broke, instead of the code
in front of the reader:

```python
def pc_stiffness_scaling(mesh, kappa):
    # This no longer scales by the mean of kappa. That broke for a kappa that
    # varied by orders of magnitude across the mesh, denormalizing h_scaling
    # to 1 for a constant kappa.
    h = CellDiameter(mesh)
    return h * h * kappa
```

RIGHT — say what the present line does:

```python
def pc_stiffness_scaling(mesh, kappa):
    # Scale by the local kappa, so cells do not end up over- or
    # under-weighted when kappa spans several orders of magnitude.
    h = CellDiameter(mesh)
    return h * h * kappa
```

Some words give this away on sight: "used to", "previously", "no longer", "instead of", "we removed",
"this replaces". Watch equally for "would" when its subject is code that does not exist.

### Grammar

Follow ASD-STE100 (Simplified Technical English): short sentences, one idea each, active voice, the
subject named up front, and every relative pronoun kept explicit.

WRONG — a dropped relative pronoun, and nouns stacked in place of a verb:

```python
# The kernel caches key on the form being compiled.
```

"caches key on" strings two nouns and a verbalised noun together, so a reader must guess which word
is the verb. "the form being compiled" drops "that is" — the reduced participle reads as an
adjective on "form" until the reader notices that it is a clause.

RIGHT — the pronoun restored, and the stack broken with a plain verb:

```python
# The kernel caches are indexed by the form that is being compiled.
```

### A `python` That Is Not The Environment's

WRONG — `python` resolves to whatever is first on `$PATH`, silently the system interpreter if the venv
was never activated:

```bash
python -m pytest tests/firedrake/regression/test_helmholtz.py
```

RIGHT — activate the venv first, so `python` resolves to the one with Firedrake installed:

```bash
. venv-firedrake/bin/activate
python -m pytest tests/firedrake/regression/test_helmholtz.py
```

### Linting One File And Calling It Clean

WRONG — checks only the file just edited:

```bash
flake8 firedrake/mesh.py
```

RIGHT — `make srclint` also lints `firedrake/scripts`, `scripts`, `tests`, `pyop2`, `pyop2/scripts`,
and `tsfc`, which is what CI enforces:

```bash
make srclint
```

### Recalling An API Instead Of Reading It

Firedrake, UFL, and PETSc/petsc4py move faster than training data; a remembered signature can already
be wrong.

WRONG — a signature written from memory, not from the version actually installed:

```python
snes.setConvergenceTest(my_test)
```

If `petsc4py` renamed a keyword or moved the method since the version an LLM was trained on, this
fails with a `TypeError` far from its real cause.

RIGHT — read the signature installed in the venv, which may be an editable checkout of an unreleased
branch:

```bash
python -c "from petsc4py import PETSc; help(PETSc.SNES.setConvergenceTest)"
```
