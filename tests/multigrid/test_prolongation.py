from firedrake import *
import pytest
import numpy as np
import itertools


def run_prolongation(mtype, vector, space, degree):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    if vector:
        V = VectorFunctionSpaceHierarchy(mh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression(("pow(x[0], d)", ), d=degree)
        elif mtype == "square":
            expr = Expression(("pow(x[0], d) - pow(x[1], d)",
                               "pow(x[0], d) + pow(x[1], d)"), d=degree)
    else:
        V = FunctionSpaceHierarchy(mh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression("pow(x[0], d)", d=degree)
        elif mtype == "square":
            expr = Expression("pow(x[0], d) - pow(x[1], d)", d=degree)

    expected = FunctionHierarchy(V)

    for e in expected:
        e.interpolate(expr)

    actual = FunctionHierarchy(V)

    actual[0].assign(expected[0])

    for i in range(len(actual)-1):
        actual.prolong(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize(["mtype", "degree", "vector", "fs"],
                         itertools.product(("interval", "square"),
                                           range(0, 4),
                                           [False, True],
                                           ["CG", "DG"]))
def test_prolongation(mtype, degree, vector, fs):
    if fs == "CG" and degree == 0:
        pytest.skip("CG 0 makes no sense")
    run_prolongation(mtype, vector, fs, degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_prolongation_square_parallel():
    for degree in range(1, 4):
        run_prolongation("square", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_prolongation_square_parallel():
    for degree in range(0, 4):
        run_prolongation("square", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_prolongation_square_parallel():
    for degree in range(1, 4):
        run_prolongation("square", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_prolongation_square_parallel():
    for degree in range(0, 4):
        run_prolongation("square", True, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_prolongation_interval_parallel():
    for degree in range(1, 4):
        run_prolongation("interval", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_prolongation_interval_parallel():
    for degree in range(0, 4):
        run_prolongation("interval", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_prolongation_interval_parallel():
    for degree in range(1, 4):
        run_prolongation("interval", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_prolongation_interval_parallel():
    for degree in range(0, 4):
        run_prolongation("interval", True, "DG", degree)


def run_extruded_dg0_prolongation(mtype):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    emh = ExtrudedMeshHierarchy(mh, layers=3)

    V = FunctionSpaceHierarchy(emh, 'DG', 0)

    expected = FunctionHierarchy(V)

    for e in expected:
        # Exactly represented on coarsest grid
        e.interpolate(Expression("3"))

    actual = FunctionHierarchy(V)

    actual[0].assign(expected[0])

    for i in range(len(actual)-1):
        actual.prolong(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize("mtype", ["interval", "square"])
def test_extruded_dg0_prolongation(mtype):
    run_extruded_dg0_prolongation(mtype)


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg0_prolongation_square_parallel():
    run_extruded_dg0_prolongation("square")


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg0_prolongation_interval_parallel():
    run_extruded_dg0_prolongation("interval")


def run_mixed_prolongation():
    m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    V = VectorFunctionSpaceHierarchy(mh, "CG", 2)
    P = FunctionSpaceHierarchy(mh, "CG", 1)

    W = V*P

    expected = FunctionHierarchy(W)

    for e in expected:
        # Exactly represented on coarsest grid
        e.interpolate(Expression(("x[0]*x[1]", "-x[1]*x[0]",
                                  "x[0] - x[1]")))

    actual = FunctionHierarchy(W)

    actual[0].assign(expected[0])

    for i in range(len(actual)-1):
        actual.prolong(i)

    for e, a in zip(expected, actual):
        for e_, a_ in zip(e.split(), a.split()):
            assert np.allclose(e_.dat.data, a_.dat.data)


def test_mixed_prolongation():
    run_mixed_prolongation()


@pytest.mark.parallel(nprocs=2)
def test_mixed_prolongation_parallel():
    run_mixed_prolongation()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
