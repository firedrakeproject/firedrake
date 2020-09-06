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


def _poisson_get_forms_original(V, f, n):
    u = TrialFunction(V)
    v = TestFunction(V)
    normal = FacetNormal(V.mesh())
    alpha = 100
    h = 1./(2**n)
    a = dot(grad(v), grad(u)) * dx - dot(grad(u), normal) * v * ds - dot(grad(v), normal) * u * ds + alpha / h * u * v * ds
    L = f * v * dx
    return a, L


def _poisson(n, el_type, degree, perturb):
    # Modified code examples in R. C. Kirby and L. Mitchell 2019

    mesh = UnitSquareMesh(2**n, 2**n)
    if perturb:
        V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
        eps = Constant(1 / 2**(n+1))

        x, y = SpatialCoordinate(mesh)
        new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                                 y - eps*sin(8*pi*x)*sin(8*pi*y)]))
        mesh = Mesh(new)

    # Rotate mesh
    theta = pi / 6
    mesh = _rotate_mesh(mesh, theta)

    V = FunctionSpace(mesh, el_type, degree)
    x, y = SpatialCoordinate(mesh)

    # Rotate coordinates
    xprime = cos(theta) * x + sin(theta) * y
    yprime = -sin(theta) * x + cos(theta) * y

    # 
    if True:
        g = cos(2 * pi * xprime) * cos(2 * pi * yprime)
        f = 8.0 * pi * pi * cos(2 * pi * xprime) * cos(2 * pi * yprime)
    else:
        g = sin(2 * pi * xprime) * sin(2 * pi * yprime)
        f = 8.0 * pi * pi * sin(2 * pi * xprime) * sin(2 * pi * yprime)

    gV = Function(V).project(g, solver_parameters={"ksp_rtol": 1.e-16})

    #a, L = _poisson_get_forms_original(V, f, n)
    u = TrialFunction(V)
    v = TestFunction(V)

    V1 = BoundarySubspace(V, (1,2,3,4))

    ub = Masked(u, V1)
    vb = Masked(v, V1)
    gb = Masked(gV, V1)

    # Make sure to project with very small tolerance.
    ud = Masked(u, V1.complement)
    vd = Masked(v, V1.complement)

    a = inner(grad(ud), grad(vd)) * dx + inner(ub, vb)* ds
    L = inner(f, vd) * dx - inner(grad(gb), grad(vd)) * dx + inner(gb, vb) * ds

    # Solve
    sol = Function(V)
    solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-13})

    # Postprocess
    err = sqrt(assemble(dot(sol - g, sol - g) * dx))
    berr = sqrt(assemble(dot(sol - gV, sol - gV) * ds))
    berr2 = sqrt(assemble(dot(sol, sol) * ds))
    print("error            : ", err)
    print("error on boundary: ", berr)
    print("error on boundary2: ", berr2)
    """
    # Plot solution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True)
    #triangles = tripcolor(sol, axes=axes, cmap='coolwarm')
    triangles = tripcolor(Function(V).interpolate(sol_exact), axes=axes, cmap='coolwarm')
    axes.set_aspect("equal")
    axes.set_title("Pressure")
    fig.colorbar(triangles, ax=axes, fraction=0.032, pad=0.02)

    plt.savefig('temphermite.pdf')
    """
    return err, berr


def test_subspace_rotated_subspace_poisson_zany():
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
    import time
    a=time.time()
    for el, deg, convrate in [('Hermite', 3, 4.0),]:
        errs = []
        for i in range(4, 9):
            err, berr = _poisson(i, el, deg, True)
            errs.append(err)
            assert(berr < 1.05e-12)
        errs = np.array(errs)
        conv = np.log2(errs[:-1] / errs[1:])
        print(conv)
        assert (np.array(conv) > convrate).all()
    b=time.time()
    print("time consumed:", b - a)


def test_subspace_rotated_subspace_stokes():


    mesh = Mesh("./docs/notebooks/stokes-control.msh")

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    vq = TestFunction(W)
    up = TrialFunction(W)

    v, q = split(vq)
    u, p = split(up)
    
    """
    v0, q0 = v, q
    u0, p0 = u, p
    """

    normal = FacetNormal(mesh)
    V4 = BoundaryComponentSubspace(W.sub(0), 4, normal)

    vq4 = Masked(vq, V4)
    up4 = Masked(up, V4)

    v4, q4 = split(vq4)
    u4, p4 = split(up4)

    v0, q0 = v - v4, q - q4
    u0, p0 = u - u4, p - p4


    nu = Constant(1)     # Viscosity coefficient

    x, y = SpatialCoordinate(mesh)
    u_inflow = as_vector([y * (10 - y) / 25.0, 0])

    bcs = [DirichletBC(W.sub(0), (0, 0), (3, 5)),
           DirichletBC(W.sub(0), interpolate(u_inflow, V), 1), ]
           #DirichletBC(W.sub(0), (0, 0), (4, ))]

    a = nu * inner(grad(u0), grad(v0)) * dx - inner(p0, div(v0)) * dx - inner(q0, div(u0)) * dx + inner(u4, v4) * ds(4)
    L = Constant(0) * q0 * dx

    w = Function(W)
    solve(a == L, w, bcs=bcs, solver_parameters={"pc_type": "lu", "mat_type": "aij",
                                                 "pc_factor_shift_type": "inblocks"})

    # Plot solution
    u, p = w.split()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
    arrows = quiver(u, axes=axes[0])
    axes[0].set_xlim(-0.5, 30.5)
    axes[0].set_aspect("equal")
    axes[0].set_title("Velocity")
    fig.colorbar(arrows, ax=axes[0], fraction=0.032, pad=0.02)
    triangles = tripcolor(p, axes=axes[1], cmap='coolwarm')
    axes[1].set_xlim(-0.5, 30.5)
    axes[1].set_aspect("equal")
    axes[1].set_title("Pressure")
    fig.colorbar(triangles, ax=axes[1], fraction=0.032, pad=0.02)

    plt.savefig('test_subspace_rotated_subspace_stokes.pdf')
