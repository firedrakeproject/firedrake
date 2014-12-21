from firedrake import *
import pytest
import numpy as np


def run_restriction(vector, space, degree):
    m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    if vector:
        V = VectorFunctionSpaceHierarchy(mh, space, degree)
        c = Constant((1, 1))
    else:
        V = FunctionSpaceHierarchy(mh, space, degree)
        c = Constant(1)

    expected = FunctionHierarchy(V)

    for e in expected:
        v = TestFunction(e.function_space())
        e.assign(assemble(dot(c, v)*e.function_space().mesh()._dx))

    actual = FunctionHierarchy(V)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.restrict(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_cg_restriction(degree):
    run_restriction(False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_restriction_parallel():
    for degree in range(1, 4):
        run_restriction(False, "CG", degree)


@pytest.mark.parametrize("degree", range(0, 4))
def test_dg_restriction(degree):
    run_restriction(False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_restriction_parallel():
    for degree in range(0, 4):
        run_restriction(False, "DG", degree)


@pytest.mark.parametrize("degree", range(1, 4))
def test_vector_cg_restriction(degree):
    run_restriction(True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_restriction_parallel():
    for degree in range(1, 4):
        run_restriction(True, "CG", degree)


@pytest.mark.parametrize("degree", range(0, 4))
def test_vector_dg_restriction(degree):
    run_restriction(True, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_restriction_parallel():
    for degree in range(0, 4):
        run_restriction(True, "DG", degree)


def run_extruded_dg0_restriction():
    m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    emh = ExtrudedMeshHierarchy(mh, layers=3)

    V = FunctionSpaceHierarchy(emh, 'DG', 0)

    expected = FunctionHierarchy(V)

    for e in expected:
        v = TestFunction(e.function_space())
        e.assign(assemble(v*e.function_space().mesh()._dx))

    actual = FunctionHierarchy(V)

    actual[-1].assign(expected[-1])

    for i in reversed(range(1, len(actual))):
        actual.restrict(i)

    for e, a in zip(expected, actual):
        assert np.allclose(e.dat.data, a.dat.data)


def test_extruded_dg0_restriction():
    run_extruded_dg0_restriction()


@pytest.mark.parallel(nprocs=2)
def test_extruded_dg0_restriction_parallel():
    run_extruded_dg0_restriction()


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

        dx = e.function_space().mesh()._dx
        e.assign(assemble(dot(c, v)*dx + p*dx))

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
