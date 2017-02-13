from __future__ import absolute_import, print_function, division
import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return m


def test_left_inverse(mesh):
    """Tests the SLATE expression A.inv * A = I"""
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    Result = assemble(A.inv * A)
    nnode = V.node_count
    assert (Result.M.values - np.identity(nnode) <= 1e-13).all()


def test_right_inverse(mesh):
    """Tests the SLATE expression A * A.inv = I"""
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    Result = assemble(A * A.inv)
    nnode = V.node_count
    assert (Result.M.values - np.identity(nnode) <= 1e-13).all()


def test_symmetry(mesh):
    """Tests that the SLATE matrices associated with a
    symmetric bilinear form is symmetric.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx + inner(grad(u), grad(v))*dx

    A = Tensor(form)
    M1 = assemble(A)
    M2 = assemble(A.T)
    assert (M1.M.values - M2.M.values <= 1e-13).all()


def test_subtract_to_zero(mesh):
    """Tests that subtracting two identical matrices results
    in the zero matrix.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    M = assemble(A - A)
    assert (M.M.values <= 1e-13).all()


def test_add_the_negative(mesh):
    """Adding the negative of a matrix gives
    you the zero matrix.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = u*v*dx

    A = Tensor(form)
    M = assemble(A + -A)
    assert (M.M.values <= 1e-13).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
