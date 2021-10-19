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

    Vsub = BoundarySubspace(V, 1)

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Projected(v, Vsub)) * dx)

    expected = np.multiply(rhs0.dat.data, Vsub._subspaces[0].dat.data)

    assert np.allclose(rhs1.dat.data, expected)


def test_filter_one_form_lagrange_action():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5))

    Vsub = BoundarySubspace(V, 1)

    fsub = Function(V)
    fsub.dat.data[:] = f.dat.data[:] * Vsub._subspaces[0].dat.data[:]
    rhs0 = assemble(inner(fsub, Projected(v, Vsub)) * dx)
    a = inner(Projected(u, Vsub), Projected(v, Vsub)) * dx
    rhs1 = assemble(action(a, f))

    assert np.allclose(rhs1.dat.data, rhs0.dat.data)


def test_filter_one_form_bdm():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'BDM', 1)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).project(as_vector([8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/7) * cos(2 * pi * y + pi/11)]))

    subdomain = (1, )
    g = Function(V).project(as_vector([1., 2.]))
    g = Function(V).assign(g, V.boundary_node_subset(subdomain))
    Vsub = DofSubspace(V, g)

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Projected(v, Vsub)) * dx)

    expected = np.multiply(rhs0.dat.data, Vsub.dat.data)

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

    g = Function(V)
    g.sub(0).assign(Function(BDM).project(as_vector([1., 2.])), subset=BDM.boundary_node_subset((1, )))
    g.sub(1).assign(Constant(1.), subset=CG.boundary_node_subset((1, )))

    Vsub = DofSubspace(V, g)

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Projected(v, Vsub)) * dx)

    for i in range(len(V)):
        expected = np.multiply(rhs0.dat.data[i], Vsub.dat.data[i])
        assert np.allclose(rhs1.dat.data[i], expected)


def test_filter_one_form_mixed_action():

    mesh = UnitSquareMesh(2, 2)

    BDM = FunctionSpace(mesh, 'BDM', 1)
    CG = FunctionSpace(mesh, 'CG', 1)

    V = BDM * CG
    v = TestFunction(V)
    u = TrialFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).project(as_vector([8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/7) * cos(2 * pi * y + pi/11),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/13) * cos(2 * pi * y + pi/17)]))

    g = Function(V)
    g.sub(0).assign(Function(BDM).project(as_vector([1., 2.])), subset=BDM.boundary_node_subset((1, )))
    g.sub(1).assign(Constant(1.), subset=CG.boundary_node_subset((1, )))

    Vsub = DofSubspace(V, g)
    fsub = Function(V)
    for i in range(len(V)):
        fsub.dat.data[i][:] = f.dat.data[i][:] * Vsub.dat.data[i][:]
    rhs0 = assemble(inner(fsub, Projected(v, Vsub)) * dx)
    a = inner(Projected(u, Vsub), Projected(v, Vsub)) * dx
    rhs1 = assemble(action(a, f))

    for i in range(len(V)):
        assert np.allclose(rhs1.dat.data[i], rhs0.dat.data[i])


def test_filter_two_form_lagrange():

    mesh = UnitSquareMesh(1, 1, quadrilateral=True)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    subdomain = V.boundary_node_subset((1, ))
    Vsub_b = DofSubspace(V, Function(V).assign(Constant(1.), subdomain))
    V0 = ComplementSubspace(Vsub_b)

    v_b = Projected(v, Vsub_b)
    u_b = Projected(u, Vsub_b)
    #v_d = v - v_b
    #u_d = u - u_b
    v_d = Projected(v, V0)
    u_d = Projected(u, V0)

    # Mass matrix
    a = inner(grad(u), grad(v)) * dx
    A = assemble(a)
    expected = np.array([[ 2/3, -1/6, -1/3, -1/6],
                         [-1/6,  2/3, -1/6, -1/3],
                         [-1/3, -1/6, 2/3, -1/6],
                         [-1/6, -1/3, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows)
    a = inner(grad(u), grad(v_d)) * dx
    A = assemble(a)
    expected = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [-1/3, -1/6, 2/3, -1/6],
                         [-1/6, -1/3, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    a = inner(grad(u_d), grad(v_d)) * dx
    A = assemble(a)
    expected = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    # Boundary mass matrix
    a = inner(grad(u_d), grad(v_d)) * dx + inner(u, v_b) * ds(1)
    A = assemble(a)
    expected = np.array([[1/3, 1/6, 0, 0],
                         [1/6, 1/3, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))

    # Mass matrix (remove boundary rows/cols)
    # Boundary mass matrix
    # Test action/derivative
    a = inner(grad(u_d), grad(v_d)) * dx + inner(u, v_b) * ds(1)
    u_ = Function(V)
    a = ufl_expr.action(a, u_)
    a = ufl_expr.derivative(a, u_)
    A = assemble(a)
    expected = np.array([[1/3, 1/6, 0, 0],
                         [1/6, 1/3, 0, 0],
                         [0, 0, 2/3, -1/6],
                         [0, 0, -1/6, 2/3]])
    assert(np.allclose(A.M.values, expected))
