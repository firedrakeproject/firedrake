import pytest
import numpy as np
from firedrake import *


def gen_mesh(cell):
    """Generate a mesh according to the cell provided."""
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == tetrahedron:
        return UnitCubeMesh(1, 1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell  not recognized" % cell)


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_scalar_field_left_inverse(cell):
    """Tests the SLATE expression A.inv * A = I"""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    Result = assemble(A.inv * A)
    nnode = V.node_count
    assert (np.array(Result.M.values) - np.identity(nnode) <= 1e-13).all()


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_scalar_field_right_inverse(cell):
    """Tests the SLATE expression A * A.inv = I"""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    Result = assemble(A * A.inv)
    nnode = V.node_count
    assert (np.array(Result.M.values) - np.identity(nnode) <= 1e-13).all()


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_symmetry(cell):
    """Tests that the SLATE matrices associated with
    symmetric bilinear forms are indeed symmetric."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx + inner(grad(u), grad(v))*dx

    A = Tensor(form)
    M1 = assemble(A)
    M2 = assemble(A.T)
    assert (np.array(M1.M.values) - np.array(M2.M.values) <= 1e-13).all()


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_negation(cell):
    """Tests that subtracting two matrices results in the
    zero matrix."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    M = assemble(A - A)
    assert (np.array(M.M.values) <= 1e-13).all()


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_add_the_negative(cell):
    """Tests that adding the negative of an expression gives
    you zero."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    M = assemble(A + -A)
    assert (np.array(M.M.values) <= 1e-13).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
