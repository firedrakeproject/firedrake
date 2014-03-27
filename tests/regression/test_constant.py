from firedrake import *
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

    c.assign([1, 1])
    assert abs(assemble(dot(c('+'), n('+'))*dS) - 2) < 1e-10


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
