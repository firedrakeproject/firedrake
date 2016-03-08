from firedrake import *
import pytest
import numpy as np
import itertools


def run_restriction(mtype, vector, space, degree):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    if vector:
        V = VectorFunctionSpaceHierarchy(mh, space, degree)
        if mtype == "interval":
            c = Constant((1, ))
        elif mtype == "square":
            c = Constant((1, 1))
    else:
        V = FunctionSpaceHierarchy(mh, space, degree)
        c = Constant(1)

    expected = FunctionHierarchy(V)

    for e in expected:
        v = TestFunction(e.function_space())
        e.assign(assemble(dot(c, v)*dx(domain=e.function_space().mesh())))

    actual = FunctionHierarchy(V)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.restrict(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize(["mtype", "degree", "vector", "fs"],
                         itertools.product(("interval", "square"),
                                           range(0, 4),
                                           [False, True],
                                           ["CG", "DG"]))
def test_restriction(mtype, degree, vector, fs):
    if fs == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if fs == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_restriction(mtype, vector, fs, degree)


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


def run_extruded_restriction(mtype, vector, space, degree):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    emh = ExtrudedMeshHierarchy(mh, layers=3)

    if vector:
        V = VectorFunctionSpaceHierarchy(emh, space, degree)
        if mtype == "interval":
            c = Constant((1, 1))
        elif mtype == "square":
            c = Constant((1, 1, 1))
    else:
        V = FunctionSpaceHierarchy(emh, space, degree)
        c = Constant(1)

    expected = FunctionHierarchy(V)

    for e in expected:
        v = TestFunction(e.function_space())
        e.assign(assemble(dot(c, v)*dx(domain=e.function_space().mesh())))

    actual = FunctionHierarchy(V)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.restrict(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize(["mtype", "vector", "space", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
def test_extruded_restriction(mtype, vector, space, degree):
    if space == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if space == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_extruded_restriction(mtype, vector, space, degree)


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

    V = VectorFunctionSpaceHierarchy(mh, "CG", 2)
    P = FunctionSpaceHierarchy(mh, "CG", 1)

    W = V*P

    expected = FunctionHierarchy(W)

    for e in expected:
        v, p = TestFunctions(e.function_space())
        c = Constant((1, 1))

        _dx = dx(domain=e.function_space().mesh())
        e.assign(assemble(dot(c, v)*_dx + p*_dx))

    actual = FunctionHierarchy(W)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.restrict(i)

    for e, a in zip(expected, actual):
        for e_, a_ in zip(e.split(), a.split()):
            assert np.allclose(e_.dat.data, a_.dat.data)


def test_mixed_restriction():
    run_mixed_restriction()


@pytest.mark.parallel(nprocs=2)
def test_mixed_restriction_parallel():
    run_mixed_restriction()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
