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
    a0 = inner(grad(u), grad(v)) * dx
    L0 = inner(f, v) * dx
    bc = DirichletBC(V, g, [1, 2, 3, 4])
    u0 = Function(V)
    solve(a0 == L0, u0, bcs = [bc, ], solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Solve with Masked; no DirichletBC
    Vsub = Subspace(V, Constant(1.), (1, 2, 3, 4))
    v_b = Masked(v, Vsub)
    u_b = Masked(u, Vsub)
    g_b = Masked(g, Vsub)
    v_d = v - v_b 
    u_d = u - u_b 

    # Unsymmetrised form:
    # a1 = inner(grad(u), grad(v_d)) * dx + inner(u, v_b) * ds
    # L1 = inner(f, v_d) * dx + inner(g, v_b) * ds

    # Symmetrised form:
    a1 = inner(grad(u_d), grad(v_d)) * dx + inner(u_b, v_b) * ds
    L1 = inner(f, v_d) * dx - inner(grad(g_b), grad(v_d)) * dx + inner(g, v_b) * ds

    u1 = Function(V)
    solve(a1 == L1, u1, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Assert u1 == u0
    assert(sqrt(assemble(dot(u1 - u0, u1 - u0) * dx)) < 1e-13)
    assert(sqrt(assemble(dot(u1 - g, u1 - g) * dx)) < 1e-4)
    # Check that Dirichlet B.C. is strongly applied
    assert(sqrt(assemble(dot(u1 - g, u1 - g) * ds)) < 1e-13)


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

    flux = as_vector([get_gbar(y), 0])
    bcs = [DirichletBC(W.sub(0), Function(V).interpolate(flux), (1, 2)),
           DirichletBC(W.sub(0).sub(1), zero(1), (3, 4))]
    a = (nu * inner(grad(u), grad(v)) - inner(p, div(v)) - inner(div(u), q)) * dx
    L = inner(Constant((0, 0)), v) * dx
    sol_target = Function(W)
    solve(a == L, sol_target, bcs=bcs, solver_parameters=solver_parameters)

    # Rotated domain
    theta = pi / 6
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([cos(theta) * x - sin(theta) * y,
                                            sin(theta) * x + cos(theta) * y]))
    mesh.coordinates.assign(f)
    xprime = Constant([cos(theta), sin(theta)])
    yprime = Constant([-sin(theta), cos(theta)])

    r = -sin(theta) * x + cos(theta) * y
    gbar = get_gbar(r)
    flux = as_vector([cos(theta) * gbar, sin(theta) * gbar])
    bcs = [DirichletBC(W.sub(0), Function(V).interpolate(flux), (1, 2)), ]
    
    # Trial/Test function
    up = TrialFunction(W)
    vq = TestFunction(W)

    g = Function(W)
    g.sub(0).assign(Constant([1., 1.]), subset=V.boundary_node_subset((3, 4)))
    Wsub = Subspace(W, g)
    up34 = Masked(up, Wsub)
    vq34 = Masked(vq, Wsub)

    u, p = split(up)
    v, q = split(vq)
    u34, p34 = split(up34)
    v34, q34 = split(vq34)
    u00, p0 = u - u34, p - p34
    v00, q0 = v - v34, q - q34

    v0 = v00 + dot(v34, xprime) * xprime
    u0 = u00 + dot(u34, xprime) * xprime
    v1 = dot(v34, yprime) * yprime
    u1 = dot(u34, yprime) * yprime

    # Define form
    a = (nu * inner(grad(u0), grad(v0)) - inner(p0, div(v0)) - inner(div(u0), q0)) * dx + inner(u1, v1) * ds
    L = inner(Constant((0, 0)), v0) * dx

    # Solve
    sol = Function(W)
    solve(a == L, sol, bcs=bcs, solver_parameters=solver_parameters)

    # Postprocess
    a0, b0 = sol_target.split()
    a1, b1 = sol.split()
    a0_, b0_ = a0.dat.data, b0.dat.data
    a1_, b1_ = a1.dat.data, b1.dat.data
    a0_ = np.concatenate((cos(theta) * a0_[:, [0]] - sin(theta) * a0_[:, [1]],
                          sin(theta) * a0_[:, [0]] + cos(theta) * a0_[:, [1]]), axis=1)
    assert(np.linalg.norm(a1_ - a0_)/np.linalg.norm(a0_) < 1e-14)
    assert(np.linalg.norm(b1_ - b0_)/np.linalg.norm(b0_) < 1e-13)

    # Plot solution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    arrows = quiver(a1, axes=axes[0])
    axes[0].set_xlim(-0.6, 1.8)
    axes[0].set_aspect("equal")
    axes[0].set_title("Velocity")
    fig.colorbar(arrows, ax=axes[0], fraction=0.032, pad=0.02)
    triangles = tripcolor(b1, axes=axes[1], cmap='coolwarm')
    axes[1].set_aspect("equal")
    axes[1].set_title("Pressure")
    fig.colorbar(triangles, ax=axes[1], fraction=0.032, pad=0.02)

    plt.savefig('test_subspace_stokes.pdf')


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
    fltr0.assign(Constant(1.))
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
    ubar0, p0 = split(Masked(u, fltr0))
    vbar0, q0 = split(Masked(v, fltr0))

    # Filter: boundary {3, 4}
    u1 = Masked(u, fltr1)
    v1 = Masked(v, fltr1)
    ex = as_vector([1, 0, 0])
    ey = as_vector([0, 1, 0])
    uxi = dot(Masked(u, fltr2), as_vector([1, 1, 1])) * ex + dot(Masked(u, fltr3), as_vector([1, 1, 1])) * ey
    ueta = u1 - uxi
    uxibar = as_vector([uxi[0], uxi[1]])
    uetabar = as_vector([ueta[0], ueta[1]])
    vxi = dot(Masked(v, fltr2), as_vector([1, 1, 1])) * ex + dot(Masked(v, fltr3), as_vector([1, 1, 1])) * ey
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

    plt.savefig('temp12.pdf')


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

    r = -sin(theta_rot) * x + cos(theta_rot) * y
    gbar = get_gbar(r)
    value = as_vector([cos(theta_rot) * gbar, sin(theta_rot) * gbar])
    bcs = [DirichletBC(W.sub(0), interpolate(value, V), (1, 2)), ]
    
    # Define filters
    xprime = cos(theta_rot) * x + sin(theta_rot) * y
    theta = pi / 2 * xprime / 1.5
    subset_34 = V.boundary_node_subset((3, 4))
    fltr0 = Function(W)
    fltr0.assign(Constant(1.))
    fltr0.sub(0).assign(Constant([0., 0.]), subset=subset_34)
    fltr1 = Function(W)
    fltr1.sub(0).assign(Constant([1., 1.]), subset=subset_34)
    fltr2 = Function(W)
    fltr2.sub(0).assign(Function(V).interpolate(as_vector([cos(theta) * cos(theta), cos(theta) * sin(theta)])), subset=subset_34)
    fltr3 = Function(W)
    fltr3.sub(0).assign(Function(V).interpolate(as_vector([cos(theta) * sin(theta), sin(theta) * sin(theta)])), subset=subset_34)

    # Trial/Test function
    u = TrialFunction(W)
    v = TestFunction(W)

    # Filter: domain
    ubar0, p0 = split(Masked(u, fltr0))
    vbar0, q0 = split(Masked(v, fltr0))

    # Filter: boundary {3, 4}
    u1 = Masked(u, fltr1)
    v1 = Masked(v, fltr1)
    ex = as_vector([1, 0, 0])
    ey = as_vector([0, 1, 0])
    uxi = dot(Masked(u, fltr2), as_vector([1, 1, 1])) * ex + dot(Masked(u, fltr3), as_vector([1, 1, 1])) * ey
    ueta = u1 - uxi
    uxibar = as_vector([uxi[0], uxi[1]])
    uetabar = as_vector([ueta[0], ueta[1]])
    vxi = dot(Masked(v, fltr2), as_vector([1, 1, 1])) * ex + dot(Masked(v, fltr3), as_vector([1, 1, 1])) * ey
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


def _poisson_get_forms_hermite(V, xi, eta, f):

    # Define op2.subsets to be used when defining filters
    subset_1234 = V.boundary_node_subset((1, 2, 3, 4))
    subset_12 = V.boundary_node_subset((1, 2))
    subset_34 = V.boundary_node_subset((3, 4))
    subset_value = V.node_subset(derivative_order=0)  # subset of value nodes
    subset_deriv = V.node_subset(derivative_order=1)  # subset of derivative nodes

    # Define filters
    # -- domain nodes
    fltr0 = Function(V)
    fltr0.assign(Constant(1.))
    fltr0.assign(Constant(0.), subset=subset_1234)
    # -- boundary normal derivative nodes
    fltr0.assign(project(xi, V), subset=subset_12.difference(subset_34).intersection(subset_deriv))
    fltr0.assign(project(eta, V), subset=subset_34.difference(subset_12).intersection(subset_deriv))
    # -- boundary tangent derivative nodes 
    fltr1 = Function(V).assign(project(eta, V), subset=subset_12.intersection(subset_deriv))
    fltr2 = Function(V).assign(project(xi, V), subset=subset_34.intersection(subset_deriv))
    # -- boundary value nodes
    fltr3 = Function(V).assign(project(Constant(1.), V), subset=subset_1234.intersection(subset_value))

    # Filter test function
    u = TrialFunction(V)
    v = TestFunction(V)

    v0 = Masked(v, fltr0)
    v1 = Masked(v, fltr1)
    v2 = Masked(v, fltr2)
    v3 = Masked(v, fltr3)
    u0 = Masked(u, fltr0)
    u1 = Masked(u, fltr1)
    u2 = Masked(u, fltr2)
    u3 = Masked(u, fltr3)
    a = dot(grad(v0), grad(u)) * dx + \
        dot(grad(v1), grad(u1))* ds((1, 2)) + \
        dot(grad(v2), grad(u2))* ds((3, 4)) + \
        v3 * u3 * ds
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
    theta_rot = pi / 6 * 0
    mesh = _rotate_mesh(mesh, theta_rot)

    V = FunctionSpace(mesh, el_type, degree)
    x, y = SpatialCoordinate(mesh)

    # Rotate coordinates
    xi = cos(theta_rot) * x + sin(theta_rot) * y
    eta = -sin(theta_rot) * x + cos(theta_rot) * y

    # normal and tangential components are not separable, 
    # have to do it algebraically

    # Define forms
    f = Function(V).project(_poisson_analytical(V, xi, eta, 'force'))
    a, L = _poisson_get_forms_hermite(V, xi, eta, f)
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

#@pytest.mark.skip(reason="not yet supported")
def test_filter_poisson_zany():
    err, berr = _poisson(1, 'Hermite', 3, True)
    print("err:", err)
    print("berr:", berr)
    assert(berr < 1e-8)
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
