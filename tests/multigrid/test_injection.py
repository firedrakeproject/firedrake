from firedrake import *
import pytest
import numpy as np
import itertools


def run_injection(mtype, vector, space, degree):
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
        expr = Expression("pow(x[0], d)", d=degree)

    expected = FunctionHierarchy(V)

    for e in expected:
        e.interpolate(expr)

    actual = FunctionHierarchy(V)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.inject(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize(["mtype", "vector", "fs", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
def test_injection(mtype, vector, fs, degree):
    if fs == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if fs == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_injection(mtype, vector, fs, degree)


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


def run_extruded_injection(mtype, vector, space, degree):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    emh = ExtrudedMeshHierarchy(mh, layers=3)
    if vector:
        V = VectorFunctionSpaceHierarchy(emh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression(("pow(x[0], d)", "pow(x[1], d)"), d=degree)
        elif mtype == "square":
            expr = Expression(("pow(x[0], d) - pow(x[1], d)",
                               "pow(x[0], d) + pow(x[1], d)",
                               "pow(x[2], d)"), d=degree)
    else:
        V = FunctionSpaceHierarchy(emh, space, degree)
        # Exactly represented on coarsest grid
        expr = Expression("pow(x[0], d)", d=degree)

    expected = FunctionHierarchy(V)

    for e in expected:
        # Exactly represented on coarsest grid
        e.interpolate(expr)

    actual = FunctionHierarchy(V)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.inject(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize(["mtype", "vector", "space", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
def test_extruded_injection(mtype, vector, space, degree):
    if space == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if space == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_extruded_injection(mtype, vector, space, degree)


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

    V = VectorFunctionSpaceHierarchy(mh, "CG", 2)
    P = FunctionSpaceHierarchy(mh, "CG", 1)

    W = V*P

    expected = FunctionHierarchy(W)

    for e in expected:
        # Exactly represented on coarsest grid
        e.interpolate(Expression(("x[0]*x[1]", "-x[1]*x[0]",
                                  "x[0] - x[1]")))

    actual = FunctionHierarchy(W)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.inject(i)

    for e, a in zip(expected, actual):
        for e_, a_ in zip(e.split(), a.split()):
            assert np.allclose(e_.dat.data, a_.dat.data)


def test_mixed_injection():
    run_mixed_injection()


@pytest.mark.parallel(nprocs=2)
def test_mixed_injection_parallel():
    run_mixed_injection()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
