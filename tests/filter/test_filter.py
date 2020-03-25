import pytest
from firedrake import *
from firedrake import ufl_expr
from pyop2 import op2
import numpy as np


def test_filter_one_form_lagrange():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5))

    nodes = V.boundary_nodes(1, "topological")
    subset = op2.Subset(V.node_set, nodes)
    fltr = Function(V).assign(1., subset=subset)

    rhs0 = assemble(f * v * dx)
    rhs1 = assemble(f * Filtered(v, fltr) * dx)

    expected = np.multiply(rhs0.dat.data, fltr.dat.data)

    assert np.allclose(rhs1.dat.data, expected)


def test_filter_one_form_bdm():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'BDM', 1)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).project(as_vector([8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5),
                                       8.0 * pi * pi * cos(2 * pi *x + pi/7) * cos(2 * pi * y + pi/11)]))

    nodes = V.boundary_nodes(1, "topological")
    subset = op2.Subset(V.node_set, nodes)
    fltr = Function(V).assign(Function(V).project(as_vector([1., 2.])), subset=subset)

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Filtered(v, fltr)) * dx)

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
    nodes = CG.boundary_nodes(1, "topological")
    subset = op2.Subset(CG.node_set, nodes)
    fltr.sub(1).assign(1., subset=subset)

    rhs0 = assemble(inner(f, v) * dx)
    #rhs1 = assemble(inner(f, Filtered(v, fltr)) * dx)

    print(rhs0.dat.data)
    #print(rhs1.dat.data)


def test_filter_two_form_lagrange():

    mesh = UnitSquareMesh(1, 1, quadrilateral=True)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    nodes = V.boundary_nodes(1, "topological")
    subset = op2.Subset(V.node_set, nodes)
    fltr_b = Function(V).assign(1., subset=subset)
    fltr_d = Function(V).interpolate(1. - fltr_b)

    v_d = Filtered(v, fltr_d)
    v_b = Filtered(v, fltr_b)
    u_d = Filtered(u, fltr_d)
    u_b = Filtered(u, fltr_b)

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


def test_filter_poisson():

    mesh = UnitSquareMesh(2**8, 2**8)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    # Analytical solution
    x, y = SpatialCoordinate(mesh)
    g = Function(V).interpolate(cos(2 * pi * x) * cos(2 * pi * y))
    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x) * cos(2 * pi * y))

    # Solve with DirichletBC
    a0 = dot(grad(v), grad(u)) * dx
    L0 = f * v * dx
    bc = DirichletBC(V, g, [1, 2, 3, 4])
    u0 = Function(V)
    solve(a0 == L0, u0, bcs = [bc, ], solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Solve with Filtered; no DirichletBC
    nodes = V.boundary_nodes([1, 2, 3, 4], "topological")
    subset = op2.Subset(V.node_set, nodes)
    fltr_b = Function(V).assign(1., subset=subset)
    fltr_d = Function(V).interpolate(1. - fltr_b)

    v_d = Filtered(v, fltr_d)
    v_b = Filtered(v, fltr_b)
    u_d = Filtered(u, fltr_d)
    u_b = Filtered(u, fltr_b)
    g_b = Filtered(g, fltr_b)

    #a1 = dot(grad(v_d), grad(u)) * dx + u * v_b * ds
    #L1 = f * v_d * dx + g * v_b *ds

    a1 = dot(grad(u_d), grad(v_d)) * dx + u_b * v_b * ds
    L1 = f * v_d * dx - dot(grad(g_b), grad(v_d)) * dx + g * v_b * ds

    u1 = Function(V)
    solve(a1 == L1, u1, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Assert u1 == u0
    assert(sqrt(assemble(dot(u1 - u0, u1 - u0) * dx)) < 1e-15)
    assert(sqrt(assemble(dot(u1 - g, u1 - g) * dx)) < 1e-4)


