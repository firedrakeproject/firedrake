from __future__ import absolute_import, print_function, division
from firedrake import *
import pytest
import numpy as np
import itertools


def run_restriction(mtype, vector, space, degree, ref_per_level=1):
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
        if mtype == "interval":
            c = Constant((1, ))
        elif mtype == "square":
            c = Constant((1, 1))
    else:
        V = FunctionSpace(mesh, space, degree)
        c = Constant(1)

    actual = assemble(dot(c, TestFunction(V))*dx)

    for mesh in reversed(mh[:-1]):
        V = FunctionSpace(mesh, V.ufl_element())
        v = TestFunction(V)
        expect = assemble(dot(c, v)*dx)
        tmp = Function(V)
        restrict(actual, tmp)
        actual = tmp
        assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.parametrize(["mtype", "degree", "vector", "fs"],
                         itertools.product(("interval", "square"),
                                           range(0, 4),
                                           [False, True],
                                           ["CG", "DG"]))
@pytest.mark.parametrize("ref_per_level", [1, 2, 3])
def test_restriction(mtype, degree, vector, fs, ref_per_level):
    if fs == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if fs == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_restriction(mtype, vector, fs, degree, ref_per_level)


@pytest.mark.parallel(nprocs=2)
def test_cg_restriction_square_parallel():
    for degree in range(1, 4):
        run_restriction("square", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_restriction_square_parallel():
    for degree in range(0, 3):
        run_restriction("square", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_restriction_square_parallel():
    for degree in range(1, 4):
        run_restriction("square", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_restriction_square_parallel():
    for degree in range(0, 3):
        run_restriction("square", True, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_restriction_interval_parallel():
    for degree in range(1, 4):
        run_restriction("interval", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_restriction_interval_parallel():
    for degree in range(0, 3):
        run_restriction("interval", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_restriction_interval_parallel():
    for degree in range(1, 4):
        run_restriction("interval", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_restriction_interval_parallel():
    for degree in range(0, 3):
        run_restriction("interval", True, "DG", degree)


def run_extruded_restriction(mtype, vector, space, degree, ref_per_level=1):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2, refinements_per_level=ref_per_level)

    emh = ExtrudedMeshHierarchy(mh, layers=3)
    mesh = emh[-1]
    if vector:
        V = VectorFunctionSpace(mesh, space, degree)
        if mtype == "interval":
            c = Constant((1, 1))
        elif mtype == "square":
            c = Constant((1, 1, 1))
    else:
        V = FunctionSpace(mesh, space, degree)
        c = Constant(1)

    actual = assemble(dot(c, TestFunction(V))*dx)

    for mesh in reversed(emh[:-1]):
        V = FunctionSpace(mesh, V.ufl_element())
        v = TestFunction(V)
        expect = assemble(dot(c, v)*dx)
        tmp = Function(V)
        restrict(actual, tmp)
        actual = tmp
        assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.parametrize(["mtype", "vector", "space", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
@pytest.mark.parametrize("ref_per_level", [1, 2])
def test_extruded_restriction(mtype, vector, space, degree, ref_per_level):
    if space == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if space == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_extruded_restriction(mtype, vector, space, degree, ref_per_level)


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg_restriction_square_parallel():
    for d in range(0, 3):
        run_extruded_restriction("square", False, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_dg_restriction_square_parallel():
    for d in range(0, 3):
        run_extruded_restriction("square", True, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_cg_restriction_square_parallel():
    for d in range(1, 4):
        run_extruded_restriction("square", False, "CG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_cg_restriction_square_parallel():
    for d in range(1, 4):
        run_extruded_restriction("square", True, "CG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg_restriction_interval_parallel():
    for d in range(0, 3):
        run_extruded_restriction("interval", False, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_dg_restriction_interval_parallel():
    for d in range(0, 3):
        run_extruded_restriction("interval", True, "DG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_cg_restriction_interval_parallel():
    for d in range(1, 4):
        run_extruded_restriction("interval", False, "CG", d)


@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_cg_restriction_interval_parallel():
    for d in range(1, 4):
        run_extruded_restriction("interval", True, "CG", d)


def run_mixed_restriction():
    m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    mesh = mh[-1]
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)

    W = V*P

    c = Constant((1, 1))

    v, p = TestFunctions(W)
    actual = assemble(dot(c, v)*dx + p*dx)

    for mesh in reversed(mh[:-1]):
        W = FunctionSpace(mesh, W.ufl_element())
        v, p = TestFunctions(W)
        expect = assemble(dot(c, v)*dx + p*dx)
        tmp = Function(W)
        restrict(actual, tmp)
        actual = tmp
        for e, a in zip(expect.split(), actual.split()):
            assert np.allclose(e.dat.data_ro, a.dat.data_ro)


def test_mixed_restriction():
    run_mixed_restriction()


@pytest.mark.parallel(nprocs=2)
def test_mixed_restriction_parallel():
    run_mixed_restriction()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
