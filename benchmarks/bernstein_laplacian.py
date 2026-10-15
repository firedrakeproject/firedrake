#!/usr/bin/env python
"""Measure Bernstein Laplacian code generation on simplices."""

import argparse
import ctypes
from math import comb
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import tempfile
import time

import loopy as lp
import numpy

from finat.ufl import FiniteElement, VectorElement
from tsfc import compile_form
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dx, grad, inner
from ufl.cell import Cell


def compile_target(
        cell: Cell, degree: int, scheme: str) -> tuple[object, float]:
    """Compile a Bernstein Laplacian bilinear form.

    Parameters
    ----------
    cell
        Reference simplex.
    degree
        Polynomial degree.
    scheme
        Quadrature scheme.

    Returns
    -------
    kernel
        Compiled TSFC kernel.
    elapsed
        Compilation time in seconds.
    """
    mesh = Mesh(VectorElement("CG", cell, 1))
    space = FunctionSpace(mesh, FiniteElement("Bernstein", cell, degree))
    u = TrialFunction(space)
    v = TestFunction(space)
    form = inner(grad(u), grad(v)) * dx(scheme=scheme)
    start = time.perf_counter()
    kernel, = compile_form(form, parameters={"mode": "spectral"})
    return kernel, time.perf_counter() - start


def temporary_metrics(kernel: object) -> tuple[int, int, int, int, int]:
    """Measure statically allocated Loopy temporaries.

    Parameters
    ----------
    kernel
        Compiled TSFC kernel.

    Returns
    -------
    scalar_count
        Number of scalar temporaries.
    array_count
        Number of array temporaries.
    stored_values
        Total scalar and array entries.
    largest_array
        Entries in the largest array temporary.
    maximum_rank
        Largest temporary tensor rank.
    """
    temporaries = tuple(
        kernel.ast.default_entrypoint.temporary_variables.values())
    shapes = [temporary.shape for temporary in temporaries]
    storage_shapes = [temporary.shape for temporary in temporaries
                      if temporary.base_storage is None]
    array_sizes = [int(numpy.prod(shape)) for shape in shapes if shape]
    storage_sizes = [int(numpy.prod(shape)) if shape else 1
                     for shape in storage_shapes]
    return (
        sum(not shape for shape in shapes),
        len(array_sizes),
        sum(storage_sizes),
        max(array_sizes, default=0),
        max(map(len, shapes), default=0),
    )


def direct_runtime(
        kernel: object, cell: Cell, degree: int, calls: int,
        repeats: int) -> tuple[numpy.ndarray, float]:
    """Compile and time direct calls to a generated C kernel.

    Parameters
    ----------
    kernel
        Compiled TSFC kernel.
    cell
        Reference simplex.
    degree
        Bernstein polynomial degree.
    calls
        Kernel calls in each timed sample.
    repeats
        Number of timed samples.

    Returns
    -------
    output
        Tensor produced by one kernel call.
    elapsed
        Median seconds per direct kernel call.
    """
    dimension = cell.topological_dimension
    ndofs = comb(degree + dimension, dimension)
    coordinates = numpy.vstack((
        numpy.zeros((1, dimension)), numpy.eye(dimension)
    )).astype(numpy.float64)
    output = numpy.zeros((ndofs, ndofs), dtype=numpy.float64)
    source = lp.generate_code_v2(kernel.ast).device_code()

    with tempfile.TemporaryDirectory(prefix="bernstein-kernel-") as directory:
        directory = Path(directory)
        source_path = directory / "kernel.c"
        library_path = directory / "kernel.so"
        source_path.write_text(source)
        compiler = shlex.split(os.environ.get("CC", "cc"))
        subprocess.run(
            [*compiler, "-O3", "-march=native", "-shared", "-fPIC",
             str(source_path), "-lm", "-o", str(library_path)],
            check=True,
        )
        library = ctypes.CDLL(str(library_path))
        function = getattr(library, kernel.ast.default_entrypoint.name)
        pointer = ctypes.POINTER(ctypes.c_double)
        function.argtypes = (pointer, pointer)
        output_pointer = output.ctypes.data_as(pointer)
        coordinate_pointer = coordinates.ctypes.data_as(pointer)

        output.fill(0)
        function(output_pointer, coordinate_pointer)
        reference = output.copy()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(calls):
                function(output_pointer, coordinate_pointer)
            samples.append((time.perf_counter() - start) / calls)
    return reference, statistics.median(samples)


def main() -> None:
    """Print compiler metrics as copyable Markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell", choices=("triangle", "tetrahedron"),
        default="tetrahedron")
    parser.add_argument("--degrees", nargs="+", type=int, default=(10,))
    parser.add_argument(
        "--schemes", nargs="+", choices=("collapsed", "canonical"),
        default=("collapsed", "canonical"))
    parser.add_argument("--runtime-calls", type=int, default=5)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    args = parser.parse_args()
    cell = Cell(args.cell)

    print("<!-- generated by benchmarks/bernstein_laplacian.py -->")
    print("| cell | degree | scheme | compile (s) | runtime (ms/call) | "
          "max error | flops | scalar temps | "
          "array temps | allocated values | bytes | largest | max rank | "
          "AST lines |")
    print("| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | "
          "---: | ---: | ---: | ---: | ---: | ---: |")
    for degree in args.degrees:
        rows = []
        for scheme in args.schemes:
            kernel, elapsed = compile_target(cell, degree, scheme)
            source = str(kernel.ast)
            nscalar, narray, nstored, largest, max_rank = \
                temporary_metrics(kernel)
            output, runtime = direct_runtime(
                kernel, cell, degree, args.runtime_calls,
                args.runtime_repeats)
            rows.append((scheme, output, runtime, elapsed, kernel.flop_count,
                         nscalar, narray, nstored, largest, max_rank,
                         len(source.splitlines())))
        reference = next((output for scheme, output, *_ in rows
                          if scheme == "canonical"), None)
        for (scheme, output, runtime, elapsed, flops, nscalar, narray,
             nstored, largest, max_rank, ast_lines) in rows:
            error = (numpy.max(numpy.abs(output - reference))
                     if reference is not None else numpy.nan)
            print(
                f"| {args.cell} | {degree} | {scheme} | {elapsed:.6f} | "
                f"{1000 * runtime:.6f} | {error:.3e} | {flops:.0f} | "
                f"{nscalar} | {narray} | "
                f"{nstored} | {8 * nstored} | {largest} | {max_rank} | "
                f"{ast_lines} |"
            )


if __name__ == "__main__":
    main()
