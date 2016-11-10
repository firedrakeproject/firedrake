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


@pytest.mark.parametrize("degree", range(0, 4))
@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_scalar_field_left_inverse(degree, cell):
    """Tests the SLATE expression A.inv * A = I"""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "DG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = slate.Matrix(form)
    Result = assemble(A.inv * A)
    nnode = len(Result.M.values)
    assert (np.array(Result.M.values) - np.identity(nnode) <= 1e-13).all()


@pytest.mark.parametrize("degree", range(0, 4))
@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_scalar_field_right_inverse(degree, cell):
    """Tests the SLATE expression A * A.inv = I"""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "DG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = slate.Matrix(form)
    Result = assemble(A * A.inv)
    nnode = len(Result.M.values)
    assert (np.array(Result.M.values) - np.identity(nnode) <= 1e-13).all()
