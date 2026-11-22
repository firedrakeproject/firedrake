# Firedrake

Firedrake solves partial differential equations with the finite element method. Reviewers read
every line of a pull request, so each change should be easy to follow and easy to justify.

## Project Architecture

* PETSc: `DMPlex` holds the mesh and its parallel distribution. `KSP`, `SNES` and `PC` are the
  solvers.
* UFL: the language in which users write variational forms.
* TSFC: compiles UFL to GEM, using elements tabulated by FIAT and FInAT, and then to loopy.
* pyop3: runs the loopy kernels over the cells and facets of the mesh.

## Core Working Rules

* Fix bugs at their root cause, in the component where the cause is (PETSc, UFL, FIAT, or Firedrake).
* Do not branch on the cell type, polynomial degree, element family or number of processes. Use the
  abstraction that already handles all cases. This also applies to tests, and existing code that
  branches is not an example to follow.
* Check each condition in one place, by its name: `mesh.extruded`, not `m.offset is None`.
* Let edge cases go through the general code path.
* Reuse existing helpers, the exceptions in `firedrake/exceptions.py`, and PETSc functionality.
* Keep each pull request focused. Put unrelated refactors in a separate pull request.
* Report or fix every bug that you find, also bugs outside the scope of your change. Open an issue
  for each bug that you do not fix.
* Read the signature of an API in the installed source before you call it.
* Describe the current code in comments and docstrings, not earlier or rejected versions of it.

## Coding Style And Conventions

* Prefer simple constructs, for example `if` instead of `match`.
* Keep `try` blocks small, and catch a specific exception.
* Do not use exceptions for control flow.
* Use f-strings.
* Put imports at the top of the module, unless the import breaks a circular dependency.
* Raise an exception for invalid input. Do not skip or overwrite things silently.
* Use `assert` only for internal invariants.
* Initialize every attribute in `__init__`.
* Use `functools.cached_property` for lazy values and `@property` for derived values.
* Do not use `hasattr` or `getattr(obj, "x", None)` to find attributes.
* Only add private attributes or helpers when several places use them.
* Call each collective operation on every rank.
* Send MPI messages on `pyop3.mpi.temp_internal_comm(comm)`, not on the communicator that the user
  gave.
* Do not loop over mesh or DoF data in Python. Prefer UFL, then `par_loop`, then Cython.
* Write comments and docstrings for a reader who knows the finite element method but not this
  code. Use plain words, define each symbol and abbreviation, and give a short example when the
  behaviour is not obvious.
* Keep every statement in a comment or docstring correct.
* Use the same terms in the names of functions, arguments and variables as in their docstrings and
  in the surrounding code. Use the existing UFL, FInAT or PETSc term for a concept, not a new synonym.
* Remove commented-out code, debug prints and `breakpoint()` calls.
* Write numpydoc docstrings for new and changed functions.
* Add type hints to new signatures.
* Write comments and documentation in ASD-STE100: complete sentences, explicit `that` and `which`,
  no stacked nouns, and no dangling modifiers.
* Write comments and docstring summaries in the imperative mood: "Return the mesh", not "Returns
  the mesh".
* Use `can` or `will` for what has not happened yet: "Whether `expr` can be assembled in one
  kernel", not "Whether one kernel assembles `expr`".

## Testing Requirements

* Compare the result against an exact value or an independently computed value, so that the test
  fails without the fix. A test that only checks that the code runs is not sufficient.
* Test behaviour, not implementation details. Do not check kernel flop counts, call counts, internal
  types or warning messages.
* Do not hardcode numbers copied from a previous run.
* Use the public API (`assemble`, `interpolate`, `solve`) with the smallest input that shows the bug.
  Do not monkeypatch internals.
* Check one property in each test. If two cases need different assertions, write two tests.
* Use one `parametrize` decorator for each parameter.
* Put constants in the test body.
* When cases differ only in which function they call, parametrize over the function.
* Name tests after the property that they check, using UFL and FInAT terminology.
* Add tests to the existing test file for the module.
* Use the skip markers from `tests/firedrake/conftest.py`.
* Write one parallel test with `@pytest.mark.parallel([1, 3])` instead of separate serial and
  parallel tests.

## Pull Request Expectations

1. Read the full diff and check it against this file.
2. Run `make srclint`.
3. Run the test file for the changed module, at each process count that it uses.
4. Before you report a test failure, record the branch and commit of every editable component. A
   failure that also happens without your change can come from a component on the wrong branch.
5. In the pull request description, state the goal of the pull request. Explain each new feature
   and each bug that it fixes.
6. Target `release` for bug fixes and documentation, and `main` for new features.
7. Declare the AI tool that you used, and add the `LLM used` label.

## Development Toolchain

### Environment Setup

* Install UFL, FIAT and loopy in editable mode. TSFC and pyop3 are part of this repository.
* Build PETSc in place.
* After you change PETSc, run `make` in `$PETSC_DIR` and then reinstall petsc4py, not in editable
  mode: `pip install --no-build-isolation --no-deps --no-cache-dir $PETSC_DIR/src/binding/petsc4py`.
  An old petsc4py makes `import firedrake` fail with `undefined symbol`.
* Firedrake keeps generated kernels in a disk cache. If a change to FIAT, TSFC, or pyop3 seems to have no
  effect, the old kernels can still be in use: run `firedrake-clean` to delete the cache.
* To compare two versions, set `FIREDRAKE_CACHE_DIR` to a temporary directory, and delete it
  afterwards.

### Testing

Run parallel tests once for each process count, with the `parallel[match]` filter. Run pytest
directly for one process and under `mpiexec` for more. Running plain `pytest` on parallel tests
does not test them correctly.

```bash
python -m pytest -m "parallel[match]" test_foo.py
mpiexec -n 3 python -m pytest -m "parallel[match]" test_foo.py
```

### Debugging

* `PYOP3_DEBUG=1` compiles kernels with debug symbols.
* `PYOP3_SPMD_STRICT=1` helps to find where ranks disagree.
* "Generated code differs across ranks" means that a Python value used in code generation differs
  between ranks.

### Reproducible Environments

Use a `firedrakeproject/firedrake` Docker image to rule out problems with the local environment.

## Anti-Patterns

### Branching On Discretization Or Execution State

```python
# WRONG
family = "DP" if mesh.ufl_cell() == interval else "DQ"
# RIGHT
family = "DG"
```

### Dynamic Attribute Assignment Outside Constructors

```python
# WRONG
if not hasattr(self, "_work"):
    self._work = Function(V)
redist = getattr(mesh, "redist", None)

# RIGHT
@cached_property
def _work(self):
    return Function(self.V)
```

### Using `hasattr` As A Setup Guard

Set `self.initialized = False` in `__init__` and check that flag, as `PCSNESBase` does in
`firedrake/preconditioners/base.py`.

### Python-Level Looping Over Mesh-Bound Array Data

```python
# WRONG
for i, x in enumerate(mesh.coordinates.dat.data_ro):
    f.dat.data[i] = g(x)
# RIGHT
f.interpolate(g(SpatialCoordinate(mesh)))
```

### Overloaded Parametrized Tests

```python
# WRONG
@pytest.mark.parametrize("cell, family, wrapper, flop_limit", [
    (cell, family, wrapper, None)
    for cell, family in (("triangle", "DP"), ("quadrilateral", "DQ"))
    for wrapper in (VectorFunctionSpace, TensorFunctionSpace)
] + [("triangle", "Bernardi-Raugel", VectorFunctionSpace, 8392)])
def test_wrapped_interpolation(cell, family, wrapper, flop_limit):
    ...
    if flop_limit is None:
        assert errornorm(expr, u) < 1e-12
    else:
        assert kernel.flop_count <= flop_limit


# RIGHT
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("make_expr", [
    pytest.param(lambda x, y: as_vector([x, y**2]), id="vector"),
    pytest.param(lambda x, y: as_tensor([[x, y**2], [x * y, y]]), id="tensor"),
])
def test_interpolation_reproduces_polynomials(mesh, make_expr):
    expr = make_expr(*SpatialCoordinate(mesh))
    V = TensorFunctionSpace(mesh, "DG", 2, shape=expr.ufl_shape)
    u = Function(V).interpolate(expr)
    assert errornorm(expr, u) < 1e-12
```
