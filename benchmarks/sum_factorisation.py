#!/usr/bin/env python
"""Measure tensor-product sum-factorisation code generation and assembly.

The cases follow ``docs/notebooks/10-sum-factorisation.py``: a Laplacian on
hexahedra, and a curl-curl form on the H(curl) conforming NCE element.  Each
is compiled in three representations, which separate what sum factorisation
contributes from what collocated quadrature contributes:

vanilla
    No factorisation, the O(p^9) reference.
spectral
    Sum factorisation on a canonical Gauss--Legendre rule, O(p^7).
gll
    Sum factorisation on the collocated Gauss--Lobatto--Legendre rule, whose
    identity tabulations reduce the same form to O(p^5).
"""

import argparse
import cProfile
import pstats
import time
import types

from metrics import (dump_kernel, isolate_caches, kernel_metrics,
                     time_kernel)


OPERATORS = {}


def build_operators() -> None:
    """Populate the case table with UFL operators.

    Notes
    -----
    UFL is imported through Firedrake, so the table cannot be built until
    the caches have been redirected.
    """
    from firedrake import curl, dot, grad
    OPERATORS.update(
        CG=lambda u, v: dot(grad(u), grad(v)),
        NCE=lambda u, v: dot(curl(u), curl(v)),
    )


def gauss_lobatto_legendre_line_rule(degree: int) -> object:
    """Build the GLL rule on the reference interval.

    Parameters
    ----------
    degree
        Polynomial degree the rule integrates.

    Returns
    -------
    object
        FInAT quadrature rule.
    """
    import FIAT
    import finat
    fiat_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule(
        FIAT.ufc_simplex(1), degree + 1)
    points = finat.point_set.GaussLobattoLegendrePointSet(
        fiat_rule.get_points())
    return finat.quadrature.QuadratureRule(points, fiat_rule.get_weights())


def gauss_lobatto_legendre_cube_rule(dimension: int, degree: int) -> object:
    """Build the GLL rule on the reference hypercube.

    Parameters
    ----------
    dimension
        Topological dimension of the cube.
    degree
        Polynomial degree the rule integrates.

    Returns
    -------
    object
        FInAT tensor product quadrature rule.
    """
    import finat
    rule = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        rule = finat.quadrature.TensorProductQuadratureRule(
            [rule, gauss_lobatto_legendre_line_rule(degree)])
    return rule


MODES = {
    "vanilla": dict(mode="vanilla", variant=None, collocated=False),
    "spectral": dict(mode="spectral", variant=None, collocated=False),
    "gll": dict(mode="spectral", variant="spectral", collocated=True),
}


def build_form(mesh: object, family: str, degree: int, mode: str,
               operator: str) -> tuple[object, object]:
    """Build one benchmark form on a hexahedral mesh.

    Parameters
    ----------
    mesh
        Extruded hexahedral mesh.
    family
        Element family, a key of ``OPERATORS``.
    degree
        Polynomial degree.
    mode
        Representation, a key of ``MODES``.
    operator
        ``"bilinear"`` for the operator itself, ``"action"`` for its action
        on a coefficient.

    Returns
    -------
    form
        The form to compile.
    space
        The function space it is posed on.

    Notes
    -----
    Timing the bilinear form measures the insertion of a dense element
    matrix as much as the local assembly that produced it, and at high
    degree the insertion dominates.  Its action assembles a vector, so the
    contraction being factorized is what the timing sees.
    """
    from firedrake import (FiniteElement, Function, FunctionSpace,
                           TestFunction, TrialFunction, action, dx)
    settings = MODES[mode]
    element = FiniteElement(family, mesh.ufl_cell(), degree=degree,
                            variant=settings["variant"])
    space = FunctionSpace(mesh, element)
    u = TrialFunction(space)
    v = TestFunction(space)
    measure = dx
    if settings["collocated"]:
        measure = dx(scheme=gauss_lobatto_legendre_cube_rule(
            mesh.topological_dimension, degree))
    form = OPERATORS[family](u, v) * measure
    if operator == "action":
        form = action(form, Function(space))
    return form, space


def build_mesh(size: int) -> object:
    """Extrude a quadrilateral mesh into hexahedra.

    Parameters
    ----------
    size
        Number of cells along each axis.

    Returns
    -------
    object
        The extruded mesh.
    """
    from firedrake import ExtrudedMesh, UnitSquareMesh
    return ExtrudedMesh(UnitSquareMesh(size, size, quadrilateral=True), size)


def measure(mesh: object, family: str, degree: int, mode: str,
            operator: str, repeats: int,
            dump: str = "") -> types.SimpleNamespace:
    """Compile and time one case.

    Parameters
    ----------
    mesh
        Extruded hexahedral mesh.
    family
        Element family, a key of ``OPERATORS``.
    degree
        Polynomial degree.
    mode
        Representation, a key of ``MODES``.
    operator
        ``"bilinear"`` or ``"action"``.
    repeats
        Number of kernel calls to average over, zero to skip execution.
    dump
        Directory to write the loopy and C sources into, empty to skip.

    Returns
    -------
    types.SimpleNamespace
        Compiled kernel metrics, compile time, and mean kernel time.
    """
    from tsfc import compile_form
    form, space = build_form(mesh, family, degree, mode, operator)
    parameters = {"mode": MODES[mode]["mode"]}

    start = time.perf_counter()
    kernel, = compile_form(form, parameters=parameters)
    compile_time = time.perf_counter() - start

    run = kernel_metrics(kernel)
    run.family = family
    run.degree = degree
    run.mode = mode
    run.operator = operator
    run.dofs = space.dim()
    run.cells = space.mesh().num_cells()
    run.compile_time = compile_time
    run.warm = (time_kernel(form, parameters, repeats)
                if repeats else float("nan"))
    if dump:
        dump_kernel(kernel, form, parameters, dump,
                    f"{family}{degree}-{mode}-{operator}")
    return run


def report(runs: list[types.SimpleNamespace]) -> None:
    """Print measurements as copyable Markdown.

    Parameters
    ----------
    runs
        Measurements to tabulate.
    """
    print("| family | degree | mode | form | dofs | compile (s) | kernel (s) | "
          "Gflop/s | flops | scalar temps | mutable arrays | "
          "mutable elements | largest mutable | tables | table elements | "
          "AST lines |")
    print("| ---: " * 16 + "|")
    for run in runs:
        print(f"| {run.family} | {run.degree} | {run.mode} | "
              f"{run.operator} | {run.dofs} | "
              f"{run.compile_time:.6f} | {run.warm:.6f} | "
              f"{run.flops * run.cells / run.warm / 1e9:.2f} | "
              f"{run.flops:.0f} | {run.nscalar} | {run.nmutable} | "
              f"{run.nmutable_elements} | {run.largest_mutable} | "
              f"{run.ntables} | {run.ntable_elements} | {run.ast_lines} |")


def profile(mesh: object, family: str, degree: int, mode: str,
            operator: str, count: int) -> None:
    """Print the hottest calls in one compilation.

    Parameters
    ----------
    mesh
        Extruded hexahedral mesh.
    family
        Element family, a key of ``OPERATORS``.
    degree
        Polynomial degree.
    mode
        Representation, a key of ``MODES``.
    operator
        ``"bilinear"`` or ``"action"``.
    count
        Number of lines of profile output to print.
    """
    from tsfc import compile_form
    form, _ = build_form(mesh, family, degree, mode, operator)
    profiler = cProfile.Profile()
    profiler.enable()
    compile_form(form, parameters={"mode": MODES[mode]["mode"]})
    profiler.disable()
    print(f"<!-- compile profile, {family} degree {degree} {mode} -->")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(count)


def main() -> None:
    """Print benchmark measurements as copyable Markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", nargs="+", default=("CG",),
                        choices=("CG", "NCE"))
    parser.add_argument("--degrees", nargs="+", type=int,
                        default=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--modes", nargs="+", default=tuple(MODES),
                        choices=tuple(MODES))
    parser.add_argument("--forms", nargs="+", default=("action",),
                        choices=("bilinear", "action"))
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10,
                        help="kernel calls to average, 0 to compile only")
    parser.add_argument("--warm-cache", action="store_true",
                        help="reuse the on-disk kernel caches")
    parser.add_argument("--dump", default="", metavar="DIR",
                        help="write the loopy and C sources of each kernel")
    parser.add_argument("--profile", type=int, default=0, metavar="LINES",
                        help="profile compilation instead of timing it")
    args = parser.parse_args()

    if not args.warm_cache:
        isolate_caches("sum-factorisation-")
    build_operators()
    mesh = build_mesh(args.size)

    if args.profile:
        for family in args.family:
            for degree in args.degrees:
                for mode in args.modes:
                    for operator in args.forms:
                        profile(mesh, family, degree, mode, operator,
                                args.profile)
        return

    print("<!-- generated by benchmarks/sum_factorisation.py -->")
    report([
        measure(mesh, family, degree, mode, operator, args.repeats,
                args.dump)
        for family in args.family
        for degree in args.degrees
        for mode in args.modes
        for operator in args.forms
    ])


if __name__ == "__main__":
    main()
