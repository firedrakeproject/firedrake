import pytest
from firedrake import *
from firedrake import ufl_expr
import numpy as np


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
    Vsub = BoundarySubspace(V, (1, 2, 3, 4))
    V0 = ComplementSubspace(Vsub)
    v_b = Projected(v, Vsub)
    u_b = Projected(u, Vsub)
    g_b = Projected(g, Vsub)
    v_d = v - v_b #Projected(v, V0)
    u_d = u - u_b #Projected(u, V0)

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
    # Modified a demo problem at:
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
           DirichletBC(W.sub(0).sub(1), 0, (3, 4))]
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

    #g = Function(W)
    #g.sub(0).assign(Constant([1., 1.]), subset=V.boundary_node_subset((3, 4)))
    Wsub = BoundarySubspace(W.sub(0), (3, 4))
    up34 = Projected(up, Wsub)
    vq34 = Projected(vq, Wsub)

    W0 = ComplementSubspace(Wsub)
    up34_ = Projected(up, W0)
    vq34_ = Projected(vq, W0)

    u, p = split(up)
    v, q = split(vq)
    u34, p34 = split(up34)
    v34, q34 = split(vq34)
    #u00, p0 = u - u34, p - p34
    #v00, q0 = v - v34, q - q34
    u00, p0 = split(up - up34)
    v00, q0 = split(vq - vq34)
    #u00, p0 = split(up34_)
    #v00, q0 = split(vq34_)
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


def test_filter_stokes_rot():
    # Modified a demo problem at:
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
                       "ksp_rtol": 1.e-15,
                       "pc_type": "ilu",
                       "pmat_type": "aij"}

    # Unrotated domain
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    flux = as_vector([get_gbar(y), 0])
    bcs = [DirichletBC(W.sub(0), Function(V).interpolate(flux), (1, 2)),
           DirichletBC(W.sub(0).sub(1), 0, (3, 4))]
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
    vq = TestFunction(W)
    up = TrialFunction(W)

    v, q = split(vq)
    u, p = split(up)
    
    normal = FacetNormal(mesh)
    V4 = BoundaryComponentSubspace(W.sub(0), (3, 4), normal)

    vq4 = Projected(vq, V4)
    up4 = Projected(up, V4)

    v4, q4 = split(vq4)
    u4, p4 = split(up4)

    v0, q0 = v - v4, q - q4
    u0, p0 = u - u4, p - p4

    bcs = [DirichletBC(W.sub(0), Function(V).interpolate(flux), (1, 2)), ]

    a = nu * inner(grad(u0), grad(v0)) * dx \
        - inner(p0, div(v0)) * dx \
        - inner(div(u0), q0) * dx \
        + inner(u4, v4) * ds((3, 4))
    #L = inner(Constant(0), q0) * dx
    L = inner(Constant((0, 0)), v0) * dx

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

    plt.savefig('test_subspace_stokes_rot.pdf')
