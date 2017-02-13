from __future__ import absolute_import, print_function, division
from firedrake import *
import pytest
import numpy as np
import itertools


def run_injection(mtype, vector, space, degree, ref_per_level=1):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    if ref_per_level > 2:
        nref = 1
    else:
        nref = 2
    mh = MeshHierarchy(m, nref, refinements_per_level=ref_per_level)
    mesh = mh[-1]
    if vector:
        V = VectorFunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression(("pow(x[0], d)", ), d=degree)
        elif mtype == "square":
            expr = Expression(("pow(x[0], d) - pow(x[1], d)",
                               "pow(x[0], d) + pow(x[1], d)"), d=degree)
    else:
        V = FunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        expr = Expression("pow(x[0], d)", d=degree)

    actual = Function(V)
    actual.interpolate(expr)

    for mesh in reversed(mh[:-1]):
        V = FunctionSpace(mesh, V.ufl_element())
        expect = Function(V).interpolate(expr)
        tmp = Function(V)
        inject(actual, tmp)
        actual = tmp
        assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.parametrize(["mtype", "vector", "fs", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
@pytest.mark.parametrize("ref_per_level", [1, 2, 3])
def test_injection(mtype, vector, fs, degree, ref_per_level):
    if fs == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if fs == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_injection(mtype, vector, fs, degree, ref_per_level)


@pytest.mark.parallel(nprocs=2)
def test_cg_injection_square_parallel():
    for degree in range(1, 4):
        run_injection("square", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_injection_square_parallel():
    for degree in range(0, 3):
        run_injection("square", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_injection_square_parallel():
    for degree in range(1, 4):
        run_injection("square", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_injection_square_parallel():
    for degree in range(0, 3):
        run_injection("square", True, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_injection_interval_parallel():
    for degree in range(1, 4):
        run_injection("interval", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_injection_interval_parallel():
    for degree in range(0, 3):
        run_injection("interval", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_injection_interval_parallel():
    for degree in range(1, 4):
        run_injection("interval", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_injection_interval_parallel():
    for degree in range(0, 3):
        run_injection("interval", True, "DG", degree)


def run_extruded_injection(mtype, vector, space, degree, ref_per_level=1):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2, refinements_per_level=ref_per_level)

    emh = ExtrudedMeshHierarchy(mh, layers=3)
    mesh = emh[-1]
    if vector:
        V = VectorFunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression(("pow(x[0], d)", "pow(x[1], d)"), d=degree)
        elif mtype == "square":
            expr = Expression(("pow(x[0], d) - pow(x[1], d)",
                               "pow(x[0], d) + pow(x[1], d)",
                               "pow(x[2], d)"), d=degree)
    else:
        V = FunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        expr = Expression("pow(x[0], d)", d=degree)

    actual = Function(V)
    actual.interpolate(expr)

    for mesh in reversed(emh[:-1]):
        V = FunctionSpace(mesh, V.ufl_element())
        expect = Function(V).interpolate(expr)
        tmp = Function(V)
        inject(actual, tmp)
        actual = tmp
        assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.parametrize(["mtype", "vector", "space", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
@pytest.mark.parametrize("ref_per_level", [1, 2])
def test_extruded_injection(mtype, vector, space, degree, ref_per_level):
    if space == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if space == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_extruded_injection(mtype, vector, space, degree, ref_per_level)


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg_injection_square_parallel():
    for d in range(0, 3):
        run_extruded_injection("square", False, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_dg_injection_square_parallel():
    for d in range(0, 3):
        run_extruded_injection("square", True, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_cg_injection_square_parallel():
    for d in range(1, 4):
        run_extruded_injection("square", False, "CG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_cg_injection_square_parallel():
    for d in range(1, 4):
        run_extruded_injection("square", True, "CG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg_injection_interval_parallel():
    for d in range(0, 3):
        run_extruded_injection("interval", False, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_dg_injection_interval_parallel():
    for d in range(0, 3):
        run_extruded_injection("interval", True, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_cg_injection_interval_parallel():
    for d in range(1, 4):
        run_extruded_injection("interval", False, "CG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_cg_injection_interval_parallel():
    for d in range(1, 4):
        run_extruded_injection("interval", True, "CG", d)


def run_mixed_injection():
    m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    mesh = mh[-1]
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)

    W = V*P

    expr = Expression(("x[0]*x[1]", "-x[1]*x[0]", "x[0] - x[1]"))

    actual = Function(W)

    actual.interpolate(expr)

    for mesh in reversed(mh[:-1]):
        W = FunctionSpace(mesh, W.ufl_element())
        expect = Function(W).interpolate(expr)
        tmp = Function(W)
        inject(actual, tmp)
        actual = tmp
        for e, a in zip(expect.split(), actual.split()):
            assert np.allclose(e.dat.data_ro, a.dat.data_ro)


def test_mixed_injection():
    run_mixed_injection()


@pytest.mark.parallel(nprocs=2)
def test_mixed_injection_parallel():
    run_mixed_injection()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
