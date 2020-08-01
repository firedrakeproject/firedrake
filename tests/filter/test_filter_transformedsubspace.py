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


def _poisson_get_forms_hermite(V, xprime, yprime, f):

    # Define op2.subsets to be used when defining filters
    subset_1234 = V.boundary_node_subset((1, 2, 3, 4))
    subset_12 = V.boundary_node_subset((1, 2))
    subset_34 = V.boundary_node_subset((3, 4))
    subset_value = V.node_subset(derivative_order=0)  # subset of value nodes
    subset_deriv = V.node_subset(derivative_order=1)  # subset of derivative nodes

    # Define filters
    # -- domain nodes
    g0 = Function(V)
    g0.assign(Constant(1.))
    g0.assign(Constant(0.), subset=subset_1234)
    # -- boundary normal derivative nodes
    g1 = Function(V)
    g1.assign(project(xprime, V), subset=subset_12.difference(subset_34).intersection(subset_deriv))
    g1.assign(project(yprime, V), subset=subset_34.difference(subset_12).intersection(subset_deriv))
    # -- boundary tangent derivative nodes 
    g2 = Function(V).assign(project(yprime, V), subset=subset_12.intersection(subset_deriv))
    g3 = Function(V).assign(project(xprime, V), subset=subset_34.intersection(subset_deriv))
    # -- boundary value nodes
    g4 = Function(V).assign(project(Constant(1.), V), subset=subset_1234.intersection(subset_value))

    # Filter test function
    u = TrialFunction(V)
    v = TestFunction(V)

    V0 = Subspace(V, g0)
    V1 = TransformedSubspace(V, g1)
    V2 = TransformedSubspace(V, g2)
    V3 = TransformedSubspace(V, g3)
    V4 = Subspace(V, g4)

    v0 = Masked(v, V0)
    v1 = Masked(v, V1)
    v2 = Masked(v, V2)
    v3 = Masked(v, V3)
    v4 = Masked(v, V4)
    u0 = Masked(u, V0)
    u1 = Masked(u, V1)
    u2 = Masked(u, V2)
    u3 = Masked(u, V3)
    u4 = Masked(u, V4)
    a = dot(grad(v0), grad(u)) * dx + \
        dot(grad(v1), grad(u)) * dx + \
        dot(grad(v2), grad(u2))* ds((1, 2)) + \
        dot(grad(v3), grad(u3))* ds((3, 4)) + \
        v4 * u4 * ds
    L = f * (v0 + v1) * dx
    A = assemble(a)
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
    theta = pi / 6
    mesh = _rotate_mesh(mesh, theta)

    V = FunctionSpace(mesh, el_type, degree)
    x, y = SpatialCoordinate(mesh)

    # Rotate coordinates
    xprime = cos(theta) * x + sin(theta) * y
    yprime = -sin(theta) * x + cos(theta) * y

    # Define forms
    f = Function(V).project(_poisson_analytical(V, xprime, yprime, 'force'))
    a, L = _poisson_get_forms_hermite(V, xprime, yprime, f)
    # Solve
    sol = Function(V)
    solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'preonly', "pc_type": 'lu'})

    # Postprocess
    sol_exact = _poisson_analytical(V, xprime, yprime, 'solution')
    err = sqrt(assemble(dot(sol - sol_exact, sol - sol_exact) * dx))
    berr = sqrt(assemble(dot(sol - sol_exact, sol - sol_exact) * ds))
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
def test_subspace_transformedsubspace_poisson_zany():
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
        errs = []
        for i in range(4, 8):
            err, berr = _poisson(i, el, deg, True)
            errs.append(err)
            assert(berr < 1e-8)
        errs = np.array(errs)
        conv = np.log2(errs[:-1] / errs[1:])
        print(conv)
        assert (np.array(conv) > convrate).all()
