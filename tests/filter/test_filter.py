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
    nodes = BDM.boundary_nodes(1, "topological")
    subset = op2.Subset(BDM.node_set, nodes)
    fltr.sub(0).assign(Function(BDM).project(as_vector([1., 2.])), subset=subset)
    nodes = CG.boundary_nodes(1, "topological")
    subset = op2.Subset(CG.node_set, nodes)
    fltr.sub(1).assign(1., subset=subset)

    rhs0 = assemble(inner(f, v) * dx)
    rhs1 = assemble(inner(f, Filtered(v, fltr)) * dx)

    for i in range(len(V)):
        expected = np.multiply(rhs0.dat.data[i], fltr.dat.data[i])
        assert np.allclose(rhs1.dat.data[i], expected)


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

    # Mass matrix (remove boundary rows/cols)
    # Boundary mass matrix
    # Test action/derivative
    # derivative with du=Filtered(...)
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


def test_filter_stokes():
    # Modified a demo problem found at:
    # https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/07-geometric-multigrid.ipynb

    degree = 8

    solver_parameters={"ksp_type": "gmres",
                       "pc_type": "ilu",
                       #"pc_factor_shift_type": "inblocks",
                       #"ksp_monitor": None,
                       "pmat_type": "aij"}

    # Unrotated domain
    mesh = RectangleMesh(15, 10, 1.5, 1)
    
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    
    nu = Constant(1)
    x, y = SpatialCoordinate(mesh)
    
    t = conditional(y < 0.5, y - 1/4, y - 3/4)
    gbar = conditional(Or(And(1/6 < y,
                              y < 1/3),
                          And(2/3 < y,
                              y < 5/6)),
                       1, 
                       0)

    value = as_vector([gbar*(1 - (12*t)**2), 0])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)),
           DirichletBC(W.sub(0).sub(1), zero(1), (3, 4))]
    
    a = (nu * inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx(degree=degree)
    L = inner(Constant((0, 0)), v) * dx(degree=degree)
    wh0 = Function(W)

    solve(a == L, wh0, bcs=bcs, solver_parameters=solver_parameters)

    # Rotated
    mesh = RectangleMesh(15, 10, 1.5, 1)
    Vc = mesh.coordinates.function_space()
    x, y = SpatialCoordinate(mesh)
    theta = pi / 3
    f = Function(Vc).interpolate(as_vector([cos(theta) * x - sin(theta) * y,
                                            sin(theta) * x + cos(theta) * y]))
    mesh.coordinates.assign(f)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    
    #u, p = TrialFunctions(W)
    #v, q = TestFunctions(W)
    
    nu = Constant(1)
    x, y = SpatialCoordinate(mesh)

    r = -sin(theta) * x + cos(theta) * y
    t = conditional(r < 0.5, r - 1/4, r - 3/4)
    gbar = conditional(Or(And(1/6 < r,
                              r < 1/3),
                          And(2/3 < r,
                              r < 5/6)),
                       1, 
                       0)

    gbarx = gbar*(1 - (12*t)**2)
    value = as_vector([cos(theta) * gbarx, sin(theta) * gbarx])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)), ]
    
    nodes = V.boundary_nodes((3, 4), "topological")
    subset = op2.Subset(V.node_set, nodes)
    fltr0 = Function(W)
    fltr0.assign(Constant([1., 1., 1]))
    fltr0.sub(0).assign(Function(V).interpolate(Constant([cos(theta), sin(theta)])), subset=subset)
    fltr1 = Function(W)
    fltr1.sub(0).assign(Function(V).interpolate(Constant([-sin(theta), cos(theta)])), subset=subset)

    
    # Domain equations
    ud, pd = split(Filtered(TrialFunction(W), fltr0))
    vd, qd = split(Filtered(TestFunction(W), fltr0))

    # Boundary equations
    ub, pb = split(Filtered(TrialFunction(W), fltr1))
    vb, qb = split(Filtered(TestFunction(W), fltr1))

    a = (nu * inner(grad(ud), grad(vd)) - pd * div(vd) + div(ud) * qd) * dx(degree=degree) + dot(ub, vb) * ds
    L = inner(Constant((0, 0)), vd) * dx(degree=degree)
    wh1 = Function(W)

    solve(a == L, wh1, bcs=bcs, solver_parameters=solver_parameters)


    u0, p0 = wh0.split()
    u1, p1 = wh1.split()

    ux0 = u0.dat.data[:, [0]]
    uy0 = u0.dat.data[:, [1]]
    u0 = np.concatenate((cos(theta) * ux0 - sin(theta) * uy0,
                         sin(theta) * ux0 + cos(theta) * uy0), axis=1)

    print(np.linalg.norm(u1.dat.data - u0)/np.linalg.norm(u0))
    print(np.linalg.norm(p1.dat.data - p0.dat.data)/np.linalg.norm(p0.dat.data))

    import matplotlib.pyplot as plt
    u, p = wh1.split()
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    arrows = quiver(u, axes=axes[0])
    axes[0].set_aspect("equal")
    axes[0].set_title("Velocity")
    fig.colorbar(arrows, ax=axes[0], fraction=0.032, pad=0.02)

    triangles = tripcolor(p, axes=axes[1], cmap='coolwarm')
    axes[1].set_aspect("equal")
    axes[1].set_title("Pressure")
    fig.colorbar(triangles, ax=axes[1], fraction=0.032, pad=0.02)

    plt.savefig('temp.pdf')

