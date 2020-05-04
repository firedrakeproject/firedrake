import pytest
from firedrake import *
from firedrake import ufl_expr
import numpy as np


def _rotate_mesh(mesh, theta):
    x, y = SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    coords = Function(Vc).interpolate(as_vector([cos(theta) * x - sin(theta) * y,
                                                 sin(theta) * x + cos(theta) * y]))
    return Mesh(coords)


def test_filter_one_form_lagrange():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x + pi/3) * cos(2 * pi * y + pi/5))

    fltr = Function(V).assign(1., subset=V.boundary_node_subset((1, )))

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

    fltr = Function(V).assign(Function(V).project(as_vector([1., 2.])), subset=V.boundary_node_subset((1, )))

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
    fltr.sub(0).assign(Function(BDM).project(as_vector([1., 2.])), subset=BDM.boundary_node_subset((1, )))
    fltr.sub(1).assign(Constant(1.), subset=CG.boundary_node_subset((1, )))

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

    subset_1 = V.boundary_node_subset((1, ))
    fltr_b = Function(V).assign(Constant(1.), subset=subset_1)
    fltr_d = Function(V).assign(Constant(1.), subset=V.node_set.difference(subset_1))

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
    subset_1234 = V.boundary_node_subset((1, 2, 3, 4))
    fltr_b = Function(V).assign(Constant(1.), subset=subset_1234)
    fltr_d = Function(V).assign(Constant(1.), subset=V.node_set.difference(subset_1234))

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
    subset_34 = V.boundary_node_subset((3, 4))
    fltr0 = Function(W)
    fltr0.assign(Constant([1., 1., 1]))
    fltr0.sub(0).assign(Constant([0., 0.]), subset=subset_34)
    fltr1 = Function(W)
    fltr1.sub(0).assign(Constant([1., 1.]), subset=subset_34)

    # Trial/Test function
    u = TrialFunction(W)
    v = TestFunction(W)
    #ubar, p = split(u)

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

    plt.savefig('temp11.pdf')


def test_filter_stokes2():
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
    theta_rot = pi / 6
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([cos(theta_rot) * x - sin(theta_rot) * y,
                                            sin(theta_rot) * x + cos(theta_rot) * y]))
    mesh.coordinates.assign(f)

    r = -sin(theta_rot) * x + cos(theta_rot) * y
    gbar = get_gbar(r)
    value = as_vector([cos(theta_rot) * gbar, sin(theta_rot) * gbar])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)), ]
    
    # Define filters
    theta = theta_rot
    subset_34 = V.boundary_node_subset((3, 4))
    fltr0 = Function(W)
    fltr0.assign(Constant([1., 1., 1]))
    fltr0.sub(0).assign(Constant([0., 0.]), subset=subset_34)
    fltr1 = Function(W)
    fltr1.sub(0).assign(Constant([1., 1.]), subset=subset_34)
    fltr2 = Function(W)
    fltr2.sub(0).assign(Constant([cos(theta) * cos(theta), cos(theta) * sin(theta)]), subset=subset_34)
    fltr3 = Function(W)
    fltr3.sub(0).assign(Constant([cos(theta) * sin(theta), sin(theta) * sin(theta)]), subset=subset_34)

    # Trial/Test function
    u = TrialFunction(W)
    v = TestFunction(W)

    # Filter: domain
    ubar0, p0 = split(Filtered(u, fltr0))
    vbar0, q0 = split(Filtered(v, fltr0))

    # Filter: boundary {3, 4}
    u1 = Filtered(u, fltr1)
    v1 = Filtered(v, fltr1)
    ex = as_vector([1, 0, 0])
    ey = as_vector([0, 1, 0])
    uxi = dot(Filtered(u, fltr2), as_vector([1, 1, 1])) * ex + dot(Filtered(u, fltr3), as_vector([1, 1, 1])) * ey
    ueta = u1 - uxi
    uxibar = as_vector([uxi[0], uxi[1]])
    uetabar = as_vector([ueta[0], ueta[1]])
    vxi = dot(Filtered(v, fltr2), as_vector([1, 1, 1])) * ex + dot(Filtered(v, fltr3), as_vector([1, 1, 1])) * ey
    veta = v1 - vxi
    vxibar = as_vector([vxi[0], vxi[1]])
    vetabar = as_vector([veta[0], veta[1]])

    # Filter: boundary {1, 2}
    # not used; use DirichletBC

    # Unsymmetrised form
    # a0 = (nu * inner(grad(ubar0 + uxibar), grad(vbar0)) - p0 * div(vbar0) - div(ubar0 + uxibar) * q0) * dx
    # a1 = (nu * inner(grad(ubar0 + uxibar), grad(vxibar)) - p0 * div(vxibar)) * dx + inner(uetabar, vetabar) * ds
    # a = a0 + a1
    # Symmetrised form (with detail)
    # a0 = (nu * inner(grad(ubar0 + uxibar), grad(vbar0)) - inner(p0, div(vbar0)) - inner(div(ubar0 + uxibar), q0)) * dx
    # a1 = (nu * inner(grad(ubar0 + uxibar), grad(vxibar)) - inner(p0, div(vxibar))) * dx + inner(uetabar, vetabar) * ds
    # a = a0 + a1
    # Symmetrised form (simplified)
    u_ = ubar0 + uxibar
    v_ = vbar0 + vxibar
    a = (nu * inner(grad(u_), grad(v_)) - inner(p0, div(v_)) - inner(div(u_), q0)) * dx + inner(uetabar, vetabar) * ds

    L = inner(Constant((0, 0)), v_) * dx
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

    plt.savefig('temp13.pdf')


def test_filter_stokes3():
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

    # Rotated domain
    theta_rot = pi / 6
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([cos(theta_rot) * x - sin(theta_rot) * y,
                                            sin(theta_rot) * x + cos(theta_rot) * y]))
    mesh.coordinates.assign(f)

    xprime = cos(theta_rot) * x + sin(theta_rot) * y
    r = -sin(theta_rot) * x + cos(theta_rot) * y
    gbar = get_gbar(r)
    value = as_vector([cos(theta_rot) * gbar, sin(theta_rot) * gbar])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)), ]
    
    # Define filters
    theta = pi / 2 * xprime / 1.5
    subset_34 = V.boundary_node_subset((3, 4))
    fltr0 = Function(W)
    fltr0.assign(Constant([1., 1., 1]))
    fltr0.sub(0).assign(Constant([0., 0.]), subset=subset_34)
    fltr1 = Function(W)
    fltr1.sub(0).assign(Constant([1., 1.]), subset=subset_34)
    fltr2 = Function(W)
    #fltr2.sub(0).assign(Constant([cos(theta) * cos(theta), cos(theta) * sin(theta)]), subset=subset_34)
    fltr2.sub(0).assign(Function(V).interpolate(as_vector([cos(theta) * cos(theta), cos(theta) * sin(theta)])), subset=subset_34)
    fltr3 = Function(W)
    #fltr3.sub(0).assign(Constant([cos(theta) * sin(theta), sin(theta) * sin(theta)]), subset=subset_34)
    fltr3.sub(0).assign(Function(V).interpolate(as_vector([cos(theta) * sin(theta), sin(theta) * sin(theta)])), subset=subset_34)

    # Trial/Test function
    u = TrialFunction(W)
    v = TestFunction(W)

    # Filter: domain
    ubar0, p0 = split(Filtered(u, fltr0))
    vbar0, q0 = split(Filtered(v, fltr0))

    # Filter: boundary {3, 4}
    u1 = Filtered(u, fltr1)
    v1 = Filtered(v, fltr1)
    ex = as_vector([1, 0, 0])
    ey = as_vector([0, 1, 0])
    uxi = dot(Filtered(u, fltr2), as_vector([1, 1, 1])) * ex + dot(Filtered(u, fltr3), as_vector([1, 1, 1])) * ey
    ueta = u1 - uxi
    uxibar = as_vector([uxi[0], uxi[1]])
    uetabar = as_vector([ueta[0], ueta[1]])
    vxi = dot(Filtered(v, fltr2), as_vector([1, 1, 1])) * ex + dot(Filtered(v, fltr3), as_vector([1, 1, 1])) * ey
    veta = v1 - vxi
    vxibar = as_vector([vxi[0], vxi[1]])
    vetabar = as_vector([veta[0], veta[1]])

    u_ = ubar0 + uxibar
    v_ = vbar0 + vxibar
    a = (nu * inner(grad(u_), grad(v_)) - inner(p0, div(v_)) - inner(div(u_), q0)) * dx + inner(uetabar, vetabar) * ds

    L = inner(Constant((0, 0)), v_) * dx
    wh1 = Function(W)
    solve(a == L, wh1, bcs=bcs, solver_parameters=solver_parameters)

    # Postprocess
    ubar, p = wh1.split()

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

    plt.savefig('temp13.pdf')


def _poisson_analytical(V, xi, eta, which):
    if which == 'solution':
        return sin(2 * pi * xi) * sin(2 * pi * eta)
    elif which == 'force':
        return 8.0 * pi * pi * sin(2 * pi * xi) * sin(2 * pi * eta)


def _poisson_get_forms_original(V, x, y, xi, eta, f, n):
    u = TrialFunction(V)
    v = TestFunction(V)
    normal = FacetNormal(V.mesh())
    alpha = 100
    h = 1./(2**n)
    a = dot(grad(v), grad(u)) * dx - dot(grad(u), normal) * v * ds - dot(grad(v), normal) * u * ds + alpha / h * u * v * ds
    L = f * v * dx
    return a, L


def _poisson_get_forms_hermite(V, xi, eta, e_xi, e_eta, f):

    # Define op2.subsets to be used when defining filters
    subset_1234 = V.boundary_node_subset((1, 2, 3, 4))
    subset_12 = V.boundary_node_subset((1, 2))
    subset_34 = V.boundary_node_subset((3, 4))
    subset_value = V.node_subset(derivative_order=0)  # subset of value nodes
    subset_deriv = V.node_subset(derivative_order=1)  # subset of derivative nodes

    # Define filters
    function0 = Function(V).project(xi)
    function1 = Function(V).project(eta)
    # -- domain nodes
    fltr0 = Function(V)
    fltr0.assign(Constant(1.))
    fltr0.assign(Constant(0.), subset=subset_1234)
    # -- boundary normal derivative nodes
    fltr0.assign(function0, subset=subset_12.difference(subset_34).intersection(subset_deriv))
    fltr0.assign(function1, subset=subset_34.difference(subset_12).intersection(subset_deriv))
    #fltr0.assign(Constant(1.), subset=subset_12.difference(subset_34).intersection(subset_deriv))
    #fltr0.assign(Constant(1.), subset=subset_34.difference(subset_12).intersection(subset_deriv))
    # -- boundary tangent derivative nodes 
    fltr1 = Function(V)
    #fltr1.assign(function1, subset=subset_12.intersection(subset_deriv))
    #fltr1.assign(function0, subset=subset_34.intersection(subset_deriv))
    fltr1.assign(Constant(1.), subset=subset_12.intersection(subset_deriv))
    fltr1.assign(Constant(1.), subset=subset_34.intersection(subset_deriv))
    # -- boundary value nodes
    fltr3 = Function(V)
    fltr3.assign(Constant(1.), subset=subset_1234)

    # Filter test function
    u = TrialFunction(V)
    v = TestFunction(V)

    Lij = as_tensor([Function(V) for _ in range(10)])
    


    v0 = Filtered(v, fltr0)
    v1 = Filtered(v, fltr1)
    v2 = Filtered(v, fltr3)
    from tsfc.finatinterface import create_element
    finat_element = create_element(V.ufl_element())
    print(finat_element.index_shape)
    print(Lij.ufl_shape)
    print(type(Lij))
    v0 = Filtered(v, Lij)
    a = dot(grad(v0), grad(u)) * dx + \
        dot(grad(v1), e_eta) * dot(grad(u), e_eta) * ds((1, 2)) + \
        dot(grad(v1), e_xi) * dot(grad(u), e_xi) * ds((3, 4)) + \
        v2 * u * ds((1, 2, 3, 4))
    L = f * v0 * dx
    return a, L


def _poisson(n, el_type, degree, perturb):
    mesh = UnitSquareMesh(2**n, 2**n)
    if perturb:
        V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
        eps = Constant(1 / 2**(n+1))

        x, y = SpatialCoordinate(mesh)
        new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                                 y - eps*sin(8*pi*x)*sin(8*pi*y)]))
        mesh = Mesh(new)

    # Rotate mesh (Currently only works with unrotated mesh)
    theta = pi / 6 * 0
    mesh = _rotate_mesh(mesh, theta)

    V = FunctionSpace(mesh, el_type, degree)
    x, y = SpatialCoordinate(mesh)

    # Rotate coordinates
    xi = cos(theta) * x + sin(theta) * y
    eta = -sin(theta) * x + cos(theta) * y

    # Rotate base
    e_xi = Constant([cos(theta), sin(theta)])
    e_eta = Constant([-sin(theta), cos(theta)])

    # normal and tangential components are not separable, 
    # have to do it algebraically

    # Define forms
    f = Function(V).project(_poisson_analytical(V, xi, eta, 'force'))
    a, L = _poisson_get_forms_hermite(V, xi, eta, e_xi, e_eta, f)

    # Solve
    sol = Function(V)
    solve(a == L, sol, bcs=[], solver_parameters={"mat_type": "aij",
                                                  #"snes_monitor": None,
                                                  #"snes_test_jacobian": True,
                                                  #"snes_test_jacobian_display": True,
                                                  "ksp_type": 'preonly',
                                                  "pc_type": 'lu'
                                                  #"ksp_type": "gmres",
                                                  #"ksp_rtol": 1.e-12,
                                                  #"ksp_atol": 1.e-12,
                                                  #"ksp_max_it": 500000,
                                                  })

    # Postprocess
    g_form = _poisson_analytical(V, xi, eta, 'solution')
    g = Function(V).project(g_form)
    err = sqrt(assemble(dot(sol - g, sol - g) * dx))
    berr = sqrt(assemble(dot(sol - g_form, sol - g_form) * ds))
    print("error            : ", err)
    print("error on boundary: ", berr)
    """
    # Plot solution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True)
    triangles = tripcolor(sol, axes=axes, cmap='coolwarm')
    axes.set_aspect("equal")
    axes.set_title("Pressure")
    fig.colorbar(triangles, ax=axes[0], fraction=0.032, pad=0.02)

    plt.savefig('temphermite.pdf')
    """
    return err, berr

@pytest.mark.skip(reason="not yet supported")
def test_filter_poisson_zany():
    #err, berr = _poisson(5, 'Hermite', 3, True)
    #assert(berr < 1e-8)
    """
    for el, deg, convrate in [('CG', 3, 4),
                              ('CG', 4, 5),
                              ('CG', 5, 6)]:
    for el, deg, convrate in [('Hermite', 3, 3.8),
                              ('Bell', 5, 4.8),
                              ('Argyris', 5, 4.8)]:
        diff = np.array([poisson(i, el, deg, True) for i in range(3, 8)])
        conv = np.log2(diff[:-1] / diff[1:])
        print(conv)
        #assert (np.array(conv) > convrate).all()
    """
    for el, deg, convrate in [('Hermite', 3, 3.8),]:
        diff = np.array([_poisson(i, el, deg, True)[0] for i in range(3, 8)])
        conv = np.log2(diff[:-1] / diff[1:])
        print(conv)
        #assert (np.array(conv) > convrate).all()
