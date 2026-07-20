# Speeding up `dmcommon.entity_orientations` (de-Python-ified)

## Motivation

Profiling construction of a `BoxMesh(79,79,79)` (≈2.96M tets, 512k P1
DOFs) showed mesh/topology setup — not the linear solve — dominating wall-clock.
Within it, `entity_orientations` cost **≈19.7 s**.

The cost was *not* the arithmetic. The per-entity loop runs `numCells ×
numEntities` ≈ 45M times, and each iteration made several **Python** calls:

- `dm.getCellType(p)` — a petsc4py method, invoked up to **4×/entity**
- `math.factorial(...)` — a Python call per entity
- `cell_closure[cell, e]` / output writes — `np.ndarray` scalar indexing

## Change

`firedrake/cython/dmcommon.pyx`:

- `_compute_orientation`: one C `DMPlexGetCellType` (already declared in
  `petschdr.pxi`) fetches the `DMPolytopeType` once; the dispatch then branches
  on the C enum (`DM_POLYTOPE_*`) instead of calling `dm.getCellType` repeatedly.
  `cell_closure` is taken as a typed memoryview (`PetscInt[:, ::1]`), with
  `boundscheck`/`wraparound` off.
- New `_fact` C factorial replaces `math.factorial` in
  `_compute_orientation_simplex` and `_compute_orientation_interval_tensor_product`.
- `entity_orientations`: reads/writes through typed memoryviews.

This is a single, unified code path for **all** cell types (simplex and
tensor-product), and — crucially — it **keeps the GIL and all error handling**
(the `RuntimeError`/`ValueError` raises on malformed input are preserved).
No new build dependency; `setup.py` is unchanged.

## Result (N=79, 2.96M tets, 512k DOF)

| version | time (min of 5) | speedup | correctness |
|---|---|---|---|
| baseline (`main`)        | 19.7 s  | 1×    | —              |
| this PR (C-path)         | 2.11 s  | **9.3×** | bit-identical |

Correctness: output is `np.array_equal` to the unpatched baseline. End-to-end
solves verified on both tet (simplex path) and quad (tensor-product path) meshes.

## Reproduce

```bash
source .../venv-firedrake/bin/activate
# baseline: on main, build, then save a reference:
BENCH_N=79 BENCH_SAVE=/tmp/ref.npy python bench/bench_entity_orientations.py
# this PR: after `python setup.py build_ext --inplace`:
BENCH_N=79 BENCH_CHECK=/tmp/ref.npy python bench/bench_entity_orientations.py
```

## Scope / follow-up

- This PR deliberately contains **only** the safe, single-process win that
  keeps the GIL. A separate, more invasive change (dropping the GIL +
  multithreading the loop with `prange`/OpenMP over cells) takes the simplex
  path from this 9.3× to ~30× on one thread and ~180× on 16 threads, but adds
  an OpenMP build dependency and interacts with MPI process placement, so it is
  kept out of this PR.
- The same de-Python-ification pattern applies to the other hot topology
  kernels (`closure_ordering`, `facet_numbering`).
