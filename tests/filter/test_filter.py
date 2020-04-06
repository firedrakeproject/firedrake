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

    # Define common parts
    mesh = RectangleMesh(15, 10, 1.5, 1)
    
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    x, y = SpatialCoordinate(mesh)
    nu = Constant(1)

    def get_gbar(r):
        t = conditional(r < 0.5, r - 1/4, r - 3/4)
        gbar = conditional(Or(And(1/6 < r,
                                  r < 1/3),
                              And(2/3 < r,
                                  r < 5/6)),
                           1, 
                           0)
        return gbar * (1 - (12 * t)**2)

    solver_parameters={"ksp_type": "gmres",
                       "pc_type": "ilu",
                       "pmat_type": "aij"}

    # Unrotated domain
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    
    value = as_vector([get_gbar(y), 0])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)),
           DirichletBC(W.sub(0).sub(1), zero(1), (3, 4))]
    a = (nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    L = inner(Constant((0, 0)), v) * dx
    wh0 = Function(W)
    solve(a == L, wh0, bcs=bcs, solver_parameters=solver_parameters)

    # Rotated domain
    theta = pi / 6
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([cos(theta) * x - sin(theta) * y,
                                            sin(theta) * x + cos(theta) * y]))
    mesh.coordinates.assign(f)
    normal = Constant([-sin(theta), cos(theta)])
    tangent = Constant([cos(theta), sin(theta)])

    r = -sin(theta) * x + cos(theta) * y
    gbar = get_gbar(r)
    value = as_vector([cos(theta) * gbar, sin(theta) * gbar])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)), ]
    
    # Define filters
    nodes = V.boundary_nodes((3, 4), "topological")
    subset = op2.Subset(V.node_set, nodes)
    nodes_p = Q.boundary_nodes((3, 4), "topological")
    subset_p = op2.Subset(Q.node_set, nodes_p)
    fltr0 = Function(W)
    fltr0.assign(Constant([1., 1., 1]))
    fltr0.sub(0).assign(Constant([0., 0.]), subset=subset)
    fltr1 = Function(W)
    fltr1.sub(0).assign(Constant([1., 1.]), subset=subset)

    # Trial/Test function
    u = TrialFunction(W)
    v = TestFunction(W)
    ubar, p = split(u)

    # Filter: domain
    ubar0, p0 = split(Filtered(u, fltr0))
    vbar0, q0 = split(Filtered(v, fltr0))

    # Filter: boundary {3, 4}
    ubar1, p1 = split(Filtered(u, fltr1))
    vbar1, q1 = split(Filtered(v, fltr1))

    # Filter: boundary {1, 2}
    # not used; use DirichletBC

    # Define form
    ubar1t = dot(ubar1, tangent) * tangent
    vbar1t = dot(vbar1, tangent) * tangent
    # Unsymmetrised form
    # a0 = (nu * inner(grad(ubar), grad(vbar0)) - p * div(vbar0) - div(ubar) * q0) * dx
    # a1 = (nu * inner(grad(ubar), grad(vbar1t)) - p * div(vbar1t) - div(ubar) * q1) * dx + dot(ubar, normal) * dot(vbar1, normal) * ds
    # Symmetrised form
    #a0 = (nu * inner(grad(ubar0 + ubar1t), grad(vbar0)) - (p0 + p1) * div(vbar0) - div(ubar0 + ubar1t) * q0) * dx
    #a1 = (nu * inner(grad(ubar0 + ubar1t), grad(vbar1t)) - (p0 + p1) * div(vbar1t) - div(ubar0 + ubar1t) * q1) * dx + dot(ubar1, normal) * dot(vbar1, normal) * ds
    a0 = (nu * inner(grad(ubar0 + ubar1t), grad(vbar0)) - p0 * div(vbar0) - div(ubar0 + ubar1t) * q0) * dx
    a1 = (nu * inner(grad(ubar0 + ubar1t), grad(vbar1t)) - p0 * div(vbar1t)) * dx + dot(ubar1, normal) * dot(vbar1, normal) * ds
    a = a0 + a1
    L = inner(Constant((0, 0)), vbar0) * dx
    wh1 = Function(W)
    solve(a == L, wh1, bcs=bcs, solver_parameters=solver_parameters)

    # Postprocess
    ubar_target, p_target = wh0.split()
    ubar, p = wh1.split()
    ubar_target_data = np.concatenate((cos(theta) * ubar_target.dat.data[:, [0]] - sin(theta) * ubar_target.dat.data[:, [1]],
                                       sin(theta) * ubar_target.dat.data[:, [0]] + cos(theta) * ubar_target.dat.data[:, [1]]), axis=1)
    assert(np.linalg.norm(ubar.dat.data - ubar_target_data)/np.linalg.norm(ubar_target_data) < 1e-14)
    assert(np.linalg.norm(p.dat.data - p_target.dat.data)/np.linalg.norm(p_target.dat.data) < 1e-13)

    # Plot solution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    arrows = quiver(ubar, axes=axes[0])
    axes[0].set_xlim(-0.6, 1.8)
    axes[0].set_aspect("equal")
    axes[0].set_title("Velocity")
    fig.colorbar(arrows, ax=axes[0], fraction=0.032, pad=0.02)

    triangles = tripcolor(p, axes=axes[1], cmap='coolwarm')
    axes[1].set_aspect("equal")
    axes[1].set_title("Pressure")
    fig.colorbar(triangles, ax=axes[1], fraction=0.032, pad=0.02)

    plt.savefig('temp.pdf')
