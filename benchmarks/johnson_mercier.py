#!/usr/bin/env python
"""Measure simplex Johnson--Mercier code generation and assembly."""

import argparse
import cProfile
import os
import pstats
import tempfile
import time
import types

import numpy


def isolate_caches() -> None:
    """Point the TSFC and PyOP2 caches at a fresh directory.

    Notes
    -----
    Must run before ``import firedrake``, which fills these variables in if
    they are unset.  Johnson--Mercier assembly is almost entirely code
    generation, so a warm disk cache hides the quantity being measured.
    """
    cache = tempfile.mkdtemp(prefix="johnson-mercier-")
    os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = os.path.join(cache, "tsfc")
    os.environ["PYOP2_CACHE_DIR"] = os.path.join(cache, "pyop2")


def build_form(dim: int, size: int) -> tuple[object, object]:
    """Build the JM mass-plus-divergence form on a simplex mesh.

    Parameters
    ----------
    dim
        Topological dimension.
    size
        Number of mesh cells along each axis.

    Returns
    -------
    form
        The bilinear form.
    space
        The Johnson--Mercier function space it is posed on.
    """
    from firedrake import (FunctionSpace, TestFunction, TrialFunction,
                           UnitCubeMesh, UnitSquareMesh, div, dx, inner)
    mesh = (UnitSquareMesh, UnitCubeMesh)[dim - 2](*(size,) * dim)
    space = FunctionSpace(mesh, "Johnson-Mercier", 1)
    u = TrialFunction(space)
    v = TestFunction(space)
    return (inner(u, v) + inner(div(u), div(v))) * dx, space


def compile_target(form: object) -> object:
    """Compile a form through TSFC.

    Parameters
    ----------
    form
        The bilinear form.

    Returns
    -------
    object
        Compiled TSFC kernel.
    """
    from tsfc import compile_form
    return compile_form(form, parameters={"mode": "spectral"})[0]


def temporary_metrics(
        kernel: object) -> tuple[int, int, int, int, int, int]:
    """Separate writable intermediates from immutable tables.

    Parameters
    ----------
    kernel
        Compiled TSFC kernel.

    Returns
    -------
    scalar_count
        Number of scalar temporaries.
    mutable_array_count
        Number of writable array temporaries.
    mutable_elements
        Total entries in writable array temporaries.
    largest_mutable_array
        Entries in the largest writable array temporary.
    table_count
        Number of read-only initialized arrays.
    table_elements
        Total entries in read-only initialized arrays.

    Notes
    -----
    Loopy represents compile-time quadrature and tabulation data as
    initialized temporary variables.  Those arrays are kernel inputs in the
    finite element algorithm, not writable contraction intermediates, so
    combining them would overstate the working set created by factorization.
    """
    temporaries = kernel.ast.default_entrypoint.temporary_variables.values()
    temporaries = tuple(temporaries)
    mutable = [
        temporary for temporary in temporaries
        if temporary.shape
        and not (temporary.read_only and temporary.initializer is not None)
    ]
    tables = [
        temporary for temporary in temporaries
        if temporary.shape
        and temporary.read_only and temporary.initializer is not None
    ]
    mutable_sizes = [
        numpy.prod(temporary.shape, dtype=int) for temporary in mutable
    ]
    table_sizes = [
        numpy.prod(temporary.shape, dtype=int) for temporary in tables
    ]
    return (
        sum(not temporary.shape for temporary in temporaries),
        len(mutable_sizes),
        sum(mutable_sizes),
        max(mutable_sizes, default=0),
        len(table_sizes),
        sum(table_sizes),
    )


def time_kernel(form: object, repeats: int) -> float:
    """Time repeated calls to the compiled cell kernel.

    Parameters
    ----------
    form
        The bilinear form.
    repeats
        Number of calls to average over.

    Returns
    -------
    float
        Mean seconds per call to the generated code.

    Notes
    -----
    Calling the compiled function directly measures the cell loop without
    the Python and PETSc work that surrounds a call to ``assemble``.
    """
    from firedrake import assemble
    from pyop2.global_kernel import GlobalKernel, compile_global_kernel

    calls = []
    original = GlobalKernel.__call__

    def record(self, comm, *arguments):
        calls.append((self, comm, arguments))
        return original(self, comm, *arguments)

    GlobalKernel.__call__ = record
    try:
        assemble(form)
    finally:
        GlobalKernel.__call__ = original

    kernel, comm, arguments = max(
        calls, key=lambda call: call[0].local_kernel.num_flops)
    execute = compile_global_kernel(kernel, comm)

    execute(*arguments)
    start = time.perf_counter()
    for _ in range(repeats):
        execute(*arguments)
    return (time.perf_counter() - start) / repeats


def measure(dim: int, size: int, repeats: int) -> types.SimpleNamespace:
    """Time compilation, cold assembly and the generated cell kernel.

    Parameters
    ----------
    dim
        Topological dimension.
    size
        Number of mesh cells along each axis.
    repeats
        Number of kernel calls to average over.

    Returns
    -------
    types.SimpleNamespace
        Timings, problem size and compiled kernel metrics.
    """
    from firedrake import assemble
    form, space = build_form(dim, size)

    start = time.perf_counter()
    kernel = compile_target(form)
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    assemble(form)
    cold = time.perf_counter() - start

    warm = time_kernel(form, repeats)

    (nscalar, nmutable, nmutable_elements, largest_mutable,
     ntables, ntable_elements) = temporary_metrics(kernel)
    return types.SimpleNamespace(
        dim=dim,
        dofs=space.dim(),
        cells=space.mesh().num_cells(),
        compile_time=compile_time,
        cold=cold,
        warm=warm,
        flops=kernel.flop_count,
        nscalar=nscalar,
        nmutable=nmutable,
        nmutable_elements=nmutable_elements,
        largest_mutable=largest_mutable,
        ntables=ntables,
        ntable_elements=ntable_elements,
        ast_lines=len(str(kernel.ast).splitlines()),
    )


def profile(dim: int, size: int, count: int) -> None:
    """Print the hottest calls in a cold assembly.

    Parameters
    ----------
    dim
        Topological dimension.
    size
        Number of mesh cells along each axis.
    count
        Number of lines of profile output to print.
    """
    from firedrake import assemble
    form, _ = build_form(dim, size)
    profiler = cProfile.Profile()
    profiler.enable()
    assemble(form)
    profiler.disable()
    print(f"<!-- cold assemble profile, dim {dim} -->")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(count)


def main() -> None:
    """Print benchmark measurements as copyable Markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warm-cache", action="store_true",
                        help="reuse the on-disk kernel caches")
    parser.add_argument("--profile", type=int, default=0, metavar="LINES",
                        help="profile a cold assemble instead of timing it")
    args = parser.parse_args()

    if not args.warm_cache:
        isolate_caches()

    if args.profile:
        for dim in args.dims:
            profile(dim, args.size, args.profile)
        return

    print("<!-- generated by benchmarks/johnson_mercier.py -->")
    print("| dim | cells | dofs | compile (s) | assemble cold (s) | "
          "kernel (s) | Gflop/s | flops | scalar temps | mutable arrays | "
          "mutable elements | mutable bytes | largest mutable | tables | "
          "table elements | AST lines |")
    print("| ---: " * 16 + "|")
    for dim in args.dims:
        run = measure(dim, args.size, args.repeats)
        print(f"| {run.dim} | {run.cells} | {run.dofs} | "
              f"{run.compile_time:.6f} | {run.cold:.6f} | {run.warm:.6f} | "
              f"{run.flops * run.cells / run.warm / 1e9:.2f} | "
              f"{run.flops:.0f} | {run.nscalar} | {run.nmutable} | "
              f"{run.nmutable_elements} | {8 * run.nmutable_elements} | "
              f"{run.largest_mutable} | {run.ntables} | "
              f"{run.ntable_elements} | {run.ast_lines} |")


if __name__ == "__main__":
    main()
