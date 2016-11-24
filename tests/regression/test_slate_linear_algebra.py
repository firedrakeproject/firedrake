import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[interval, triangle, tetrahedron, quadrilateral])
def gen_mesh(request):
    """Generate a mesh according to the cell provided."""
    cell = request.param
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == tetrahedron:
        return UnitCubeMesh(1, 1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell not recognized" % cell)


def test_left_inverse(gen_mesh):
    """Tests the SLATE expression A.inv * A = I"""
    V = FunctionSpace(gen_mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    Result = assemble(A.inv * A)
    nnode = V.node_count
    assert (Result.M.values - np.identity(nnode) <= 1e-13).all()


def test_right_inverse(gen_mesh):
    """Tests the SLATE expression A * A.inv = I"""
    V = FunctionSpace(gen_mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    Result = assemble(A * A.inv)
    nnode = V.node_count
    assert (Result.M.values - np.identity(nnode) <= 1e-13).all()


def test_symmetry(gen_mesh):
    """Tests that the SLATE matrices associated with a
    symmetric bilinear form is symmetric.
    """
    V = FunctionSpace(gen_mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx + inner(grad(u), grad(v))*dx

    A = Tensor(form)
    M1 = assemble(A)
    M2 = assemble(A.T)
    assert (M1.M.values - M2.M.values <= 1e-13).all()


def test_subtract_to_zero(gen_mesh):
    """Tests that subtracting two identical matrices results
    in the zero matrix.
    """
    V = FunctionSpace(gen_mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    M = assemble(A - A)
    assert (M.M.values <= 1e-13).all()


def test_add_the_negative(gen_mesh):
    """Adding the negative of a matrix gives
    you the zero matrix.
    """
    V = FunctionSpace(gen_mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    M = assemble(A + -A)
    assert (M.M.values <= 1e-13).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
