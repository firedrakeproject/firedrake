from firedrake import *
import numpy as np
import pytest


def test_scalar_constant():
    for m in [UnitIntervalMesh(5), UnitSquareMesh(2, 2), UnitCubeMesh(2, 2, 2)]:
        c = Constant(1, cell=m.ufl_cell())
        assert abs(assemble(c*m._dx) - 1.0) < 1e-10


def test_scalar_constant_assign():
    for m in [UnitIntervalMesh(5), UnitSquareMesh(2, 2), UnitCubeMesh(2, 2, 2)]:
        c = Constant(1, cell=m.ufl_cell())
        assert abs(assemble(c*m._dx) - 1.0) < 1e-10
        c.assign(4)
        assert abs(assemble(c*m._dx) - 4.0) < 1e-10


@pytest.mark.parametrize(('init', 'new_vals'),
                         ((1, ([1, 1], "x", [[1, 1], [1, 1]])),
                          ([1, 1], ([1, "x"], "x", 1, [[1, 1], [1, 1]])),
                          ([[1], [1]], ([1, "x"], "x", 1, [[1, 1], [1, 1]]))))
def test_constant_assign_mismatch(init, new_vals):
    c = Constant(init)
    for v in new_vals:
        with pytest.raises(ValueError):
            c.assign(v)


def test_vector_constant_2d():
    m = UnitSquareMesh(1, 1)
    n = FacetNormal(m)

    c = Constant([1, -1])
    # Mesh is:
    # ,---.
    # |\  |
    # | \ |
    # |  \|
    # `---'
    # Normal is in (1, 1) direction
    assert abs(assemble(dot(c('+'), n('+'))*dS)) < 1e-10
    assert abs(assemble(dot(c('-'), n('+'))*dS)) < 1e-10

    # Normal is in (-1, -1) direction
    assert abs(assemble(dot(c('+'), n('-'))*dS)) < 1e-10
    assert abs(assemble(dot(c('-'), n('-'))*dS)) < 1e-10

    c.assign([1, 1])
    assert abs(assemble(dot(c('+'), n('+'))*dS) - 2) < 1e-10
    assert abs(assemble(dot(c('-'), n('+'))*dS) - 2) < 1e-10

    # Normal is in (-1, -1) direction
    assert abs(assemble(dot(c('+'), n('-'))*dS) + 2) < 1e-10
    assert abs(assemble(dot(c('-'), n('-'))*dS) + 2) < 1e-10


def test_constant_scalar_assign_distributes():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant(11)

    f.assign(c)

    assert np.allclose(f.dat.data_ro, 11)


def test_constant_vector_assign_works():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant([10, 11])

    f.assign(c)

    assert np.allclose(f.dat.data_ro[:, 0], 10)
    assert np.allclose(f.dat.data_ro[:, 1], 11)


def test_constant_vector_assign_to_scalar_error():
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant([10, 11])

    with pytest.raises(ValueError):
        f.assign(c)


def test_constant_vector_assign_to_vector_mismatch_error():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant([10, 11, 12])

    with pytest.raises(ValueError):
        f.assign(c)


def test_constant_assign_to_mixed():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    W = V*V

    f = Function(W)
    c = Constant([10, 11])

    f.assign(c)

    for d in f.dat.data_ro:
        assert np.allclose(d[:, 0], 10)
        assert np.allclose(d[:, 1], 11)


def test_constant_multiplies_function():
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = Function(V)

    u.assign(10)
    f = Function(V)

    c = Constant(11)
    f.assign(u * c)

    assert np.allclose(f.dat.data_ro, 110)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
