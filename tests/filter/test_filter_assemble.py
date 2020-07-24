import pytest
from firedrake import *
from firedrake import ufl_expr
import numpy as np


def test_filter_one_form_lagrange():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5))

    fltr = Function(V).assign(1., subset=V.boundary_node_subset((1, )))

    rhs0 = assemble(f * v * dx)
    rhs1 = assemble(f * Masked(v, fltr) * dx)

    expected = np.multiply(rhs0.dat.data, fltr.dat.data)

    assert np.allclose(rhs1.dat.data, expected)


def test_filter_one_form_bdm():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'BDM', 1)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).project(as_vector([8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/7) * cos(2 * pi * y + pi/11)]))

    fltr = Function(V).assign(Function(V).project(as_vector([1., 2.])), subset=V.boundary_node_subset((1, )))

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Masked(v, fltr)) * dx)

    expected = np.multiply(rhs0.dat.data, fltr.dat.data)

    assert np.allclose(rhs1.dat.data, expected)


def test_filter_one_form_mixed():

    mesh = UnitSquareMesh(2, 2)

    BDM = FunctionSpace(mesh, 'BDM', 1)
    CG = FunctionSpace(mesh, 'CG', 1)

    V = BDM * CG
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).project(as_vector([8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/7) * cos(2 * pi * y + pi/11),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/13) * cos(2 * pi * y + pi/17)]))

    fltr = Function(V)
    fltr.sub(0).assign(Function(BDM).project(as_vector([1., 2.])), subset=BDM.boundary_node_subset((1, )))
    fltr.sub(1).assign(Constant(1.), subset=CG.boundary_node_subset((1, )))

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Masked(v, fltr)) * dx)

    for i in range(len(V)):
        expected = np.multiply(rhs0.dat.data[i], fltr.dat.data[i])
        assert np.allclose(rhs1.dat.data[i], expected)


def test_filter_two_form_lagrange():

    mesh = UnitSquareMesh(1, 1, quadrilateral=True)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    subset_1 = V.boundary_node_subset((1, ))
    fltr_b = Function(V).assign(Constant(1.), subset=subset_1)
    fltr_d = Function(V).assign(Constant(1.), subset=V.node_set.difference(subset_1))

    v_d = Masked(v, fltr_d)
    v_b = Masked(v, fltr_b)
    u_d = Masked(u, fltr_d)
    u_b = Masked(u, fltr_b)

    # Mass matrix
    a = dot(grad(u), grad(v)) * dx
    A = assemble(a)
    expected = np.array([[ 2/3, -1/6, -1/3, -1/6],
                         [-1/6,  2/3, -1/6, -1/3],
                         [-1/3, -1/6, 2/3, -1/6],
                         [-1/6, -1/3, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows)
    a = dot(grad(u), grad(v_d)) * dx
    A = assemble(a)
    expected = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [-1/3, -1/6, 2/3, -1/6],
                         [-1/6, -1/3, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    a = dot(grad(u_d), grad(v_d)) * dx
    A = assemble(a)
    expected = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    # Boundary mass matrix
    a = dot(grad(u_d), grad(v_d)) * dx + u * v_b * ds(1)
    A = assemble(a)
    expected = np.array([[1/3, 1/6, 0, 0],
                         [1/6, 1/3, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    # Boundary mass matrix
    # Test action/derivative
    a = dot(grad(u_d), grad(v_d)) * dx + u * v_b * ds(1)
    u_ = Function(V)
    a = ufl_expr.action(a, u_)
    a = ufl_expr.derivative(a, u_)
    A = assemble(a)
    expected = np.array([[1/3, 1/6, 0, 0],
                         [1/6, 1/3, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    # Boundary mass matrix
    # Test action/derivative
    # derivative with du=Masked(...)
    a = dot(grad(u), grad(v_d)) * dx
    u_ = Function(V)
    a = ufl_expr.action(a, u_)
    a = ufl_expr.derivative(a, u_, du=u_d)
    a += u * v_b * ds(1)
    A = assemble(a)
    expected = np.array([[1/3, 1/6, 0, 0],
                         [1/6, 1/3, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))
