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
    theta = pi / 6 * 1
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
    V1 = BoundarySubspace(V, (1, 2, 3, 4))

    ub = Projected(u, V1)
    vb = Projected(v, V1)
    gb = Projected(gV, V1)

    # Make sure to project with very small tolerance.
    ud = u-ub #Projected(u, V1.complement)
    vd = v-vb #Projected(v, V1.complement)

    a = inner(grad(ud), grad(vd)) * dx + inner(ub, vb) * ds((1,2,3,4))
    L = inner(f, vd) * dx - inner(grad(gb), grad(vd)) * dx + inner(gb, vb) * ds((1,2,3,4))

    # Solve
    sol = Function(V)
    solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-13})

    # Postprocess
    err = sqrt(assemble(dot(sol - g, sol - g) * dx))
    berr = sqrt(assemble(dot(sol - gV, sol - gV) * ds))
    berr2 = sqrt(assemble(dot(sol, sol) * ds))
    print("error            : ", err)
    print("error on boundary: ", berr)
    #print("error on boundary2: ", berr2)
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
            assert(berr < 1.e-12)
        errs = np.array(errs)
        conv = np.log2(errs[:-1] / errs[1:])
        print(conv)
        assert (np.array(conv) > convrate).all()
    b=time.time()
    print("time consumed:", b - a)


@pytest.mark.parallel(nprocs=3)
def test_subspace_rotated_subspace_stokes():
    # Modified a demo problem at:
    # https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/06-pde-constrained-optimisation.ipynb
    mesh = Mesh("./docs/notebooks/stokes-control.msh")
    theta = pi / 6
    mesh = _rotate_mesh(mesh, theta)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    vq = TestFunction(W)
    up = TrialFunction(W)

    v, q = split(vq)
    u, p = split(up)
    
    normal = FacetNormal(mesh)
    V4 = BoundaryComponentSubspace(W.sub(0), (3, 4, 5), normal)
    vq4 = Projected(vq, V4)
    up4 = Projected(up, V4)

    v4, q4 = split(vq4)
    u4, p4 = split(up4)

    v0, q0 = v - v4, q - q4
    u0, p0 = u - u4, p - p4

    nu = Constant(1)     # Viscosity coefficient

    x, y = SpatialCoordinate(mesh)
    yprime = -sin(theta) * x + cos(theta) * y
    gbar = yprime * (10 - yprime) / 25.0
    u_inflow = as_vector([cos(theta) * gbar, sin(theta) * gbar])

    bcs = [DirichletBC(W.sub(0), interpolate(u_inflow, V), 1), ]

    a = nu * inner(grad(u0), grad(v0)) * dx \
        - inner(p0, div(v0)) * dx \
        - inner(div(u0), q0) * dx \
        + inner(u4, v4) * ds((3, 4, 5))
    L = inner(Constant(0), q0) * dx

    w = Function(W)
    solve(a == L, w, bcs=bcs, solver_parameters={"pc_type": "lu", "mat_type": "aij",
                                                 "pc_factor_shift_type": "inblocks"})
    # Plot
    #uplot, pplot = w.split()
    #plot_velocity("test_subspace_rotated_subspace_stokes.pdf", uplot, theta, [-5, 27], None, [0, 1.], 20.)


@pytest.mark.parallel(nprocs=3)
def test_subspace_rotated_subspace_swe():
    # Modified a demo problem at:
    # https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/06-pde-constrained-optimisation.ipynb

    len_x = 30
    len_y = 10
    mesh = Mesh("./docs/notebooks/stokes-control.msh")
    # Rotate mesh
    theta = pi / 6
    mesh = _rotate_mesh(mesh, theta)
    #mesh = RectangleMesh(30, 10, len_x, len_y)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    vq = TestFunction(W)
    uh = Function(W)
    uh_ = Function(W)

    v, q = split(vq)
    u, h = split(uh)
    u_, h_ = split(uh_)
    
    normal = FacetNormal(mesh)
    V4 = BoundaryComponentSubspace(W.sub(0), (3, 4, 5), normal)

    vq4 = Projected(vq, V4)
    uh4 = Projected(uh, V4)
    v4, q4 = split(vq4)
    u4, h4 = split(uh4)
    v0, q0 = v - v4, q - q4
    u0, h0 = u - u4, h - h4

    nu = Constant(1)     # Viscosity coefficient
    g = Constant(1)
    CD = Constant(1)
    H = Constant(1)
    dt = 0.2
    eps = Constant(1e-12)

    T = 2 * nu * sym(grad(u)) - 2 / 3 * nu * div(u) * Identity(2)

    F = inner(u - u_, v0) / dt * dx + \
        inner(dot(u, grad(u)), v0) * dx + \
        inner(g * grad(h), v0) * dx + \
        inner(T, grad(v0)) * dx + \
        inner(CD * sqrt(dot(u, u) + eps) * u / (H + h), v0) * dx + \
        inner(h - h_, q0) / dt * dx - \
        inner((H + h) * u, grad(q0)) * dx + \
        inner(u, v4) * ds((3, 4, 5))

    x, y = SpatialCoordinate(mesh)
    u_inflow = as_vector([y * (10 - y) / 25.0, 0])

    #bcs = [DirichletBC(W.sub(0), interpolate(u_inflow, V), 1), ]

    
    bcs = [DirichletBC(W.sub(1), 0, (1, 2, 3, 4, 5))]


    xprime = cos(theta) * x + sin(theta) * y
    yprime = -sin(theta) * x + cos(theta) * y
    uh.sub(1).assign(Function(Q).interpolate(0.0005 * xprime * (5 - xprime) * conditional(real(xprime) < 5, 1, 0) *  \
                                                      yprime * (10 - yprime) ))



    t = 0
    while t < 10:
        uplot, hplot = uh.split()
        #plot_velocity('swe_velocity_%05.2f.pdf' % t, uplot, theta, [0, 30], [0, 15], [0,1], 0.02, t)
        #plot_surface(hplot, 'swe_surface_%05.2f.pdf' % t, t)

        uh_.assign(uh)
        solve(F == 0, uh, bcs=bcs, solver_parameters={"ksp_type": "gmres"})
        t += dt
        print(t)

"""
import matplotlib.pyplot as plt


def plot_velocity(name, uplot, theta, xlim, ylim, clim, scale, t=None, fontsize=36):
    len_x=30
    len_y=10
    x0 = 0
    x1 = len_x * cos(theta)
    x3 = -len_y * sin(theta)
    x2 = x1 + x3
    y0 = 0
    y1 = len_x * sin(theta)
    y3 = len_y * cos(theta)
    y2 = y1 + y3

    #fig, axes = plt.subplots(figsize=(16, 12), nrows=1, sharex=True, sharey=True)
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, sharex=True, sharey=True)
    axes.plot([x0, x1, x2, x3, x0], [y0, y1, y2, y3, y0], color='gray')
    arrows = quiver(uplot, axes=axes, clim=clim, scale=scale)
    if xlim:
        axes.set_xlim(*xlim)
    if ylim:
        axes.set_ylim(*ylim)
    axes.set_aspect("equal")
    #plt.xticks(fontsize=28)
    #plt.yticks(fontsize=28)
    if t is not None:
        axes.set_title("u: t = %05.2f" % t, fontsize=fontsize)
    else:
        pass
        #axes.set_title("Velocity", fontsize=36)
    plt.axis('off')
    cbar = fig.colorbar(arrows, ax=axes, fraction=0.032, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(name, transparent=True)
    plt.close(fig)


def plot_height():
    pass
    #triangles = tripcolor(hplot, axes=axes[1], cmap='coolwarm', vmin=-0.02, vmax=0.02)
    #axes[0].set_xlim(-5, 27)
    #axes[1].set_aspect("equal")
    #axes[1].set_title("t = %05.2f: Relative height" % t)
    #fig.colorbar(triangles, ax=axes[1], fraction=0.032, pad=0.02)

def plot_surface(hplot, name, t=None):
    fig = plt.figure(figsize=(12, 8))
    axes = fig.add_subplot(111, projection='3d')
    triangles = trisurf(hplot, axes=axes, cmap='coolwarm', vmin=-0.005, vmax=0.005)
    axes.set_xlim(5, 20)
    axes.set_zlim(-0.05, 0.05)
    axes.view_init(elev=30., azim=-20)
    plt.axis('off')
    if t:
        axes.set_title("t = %05.2f: Relative height" % t, fontsize=20)
    cbar = fig.colorbar(triangles, ax=axes, fraction=0.032, pad=0.02)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(name)
    plt.close(fig)
"""


if __name__ == "__main__":
    test_subspace_rotated_subspace_stokes()
    test_subspace_rotated_subspace_swe()
