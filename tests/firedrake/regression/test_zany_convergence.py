"""Test the convergence of solutions with zany elements."""

import numpy as np
import pytest
from firedrake import *


def helmholtz(n, el_type, degree, perturb):
    """Solve Helmholtz's equation on the unit square

      - div grad u(x, y) + u(x,y) = f(x, y)

    with the analytical solution

      u(x, y) = cos(x[0]*3*pi)*cos(x[1]*3*pi)

    and return the H1 norm of the error.
    """
    mesh = UnitSquareMesh(2**n, 2**n)
    if perturb:
        V = mesh.coordinates.function_space()
        eps = Constant(1 / 2**(n+1))

        x, y = SpatialCoordinate(mesh)
        new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                                 y - eps*sin(8*pi*x)*sin(8*pi*y)]))
        mesh = Mesh(new)

    V = FunctionSpace(mesh, el_type, degree)
    x = SpatialCoordinate(V.mesh())
    degree = V.ufl_element().degree()

    # Define variational problem
    k = Constant(3)
    u_exact = cos(x[0]*pi*k) * cos(x[1]*pi*k)

    lmbda = Constant(1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx + lmbda * inner(u, v)*dx
    L = a(v, u_exact)

    # Compute solution
    sol = Function(V)
    solve(a == L, sol)

    # Return the H1 norm
    return sqrt(assemble(a(sol - u_exact, sol - u_exact)))


# Test H1 convergence on Hermite, HCT, Bell, and Argyris
# Morley is omitted since it only can be used on 4th-order problems.
# It is, somewhat oddly, a suitable C^1 nonconforming element but
# not a suitable C^0 nonconforming one.
@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Hermite', 3, 3),
                          ('Bell', 5, 4),
                          ('Argyris', 5, 5),
                          ('Argyris', 6, 6)])
@pytest.mark.parametrize("perturb", [False, True], ids=["Regular", "Perturbed"])
def test_firedrake_helmholtz_scalar_convergence(el, deg, convrate, perturb):
    l = 4
    diff = np.array([helmholtz(i, el, deg, perturb) for i in range(l, l+2)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate - 0.3).all()


def discrete_kirchhoff_error(n):
    """Return the L2 error of the discrete Hessian of the discrete Kirchhoff triangle.

    The discrete Kirchhoff triangle replaces the gradient of the deflection by
    its interpolant into the rotated Bernardi-Raugel space. The discrete
    Hessian is the gradient of this interpolant. The plate is clamped on the
    boundary of the unit square, which has an ``n`` by ``n`` mesh.
    """
    mesh = UnitSquareMesh(n, n)
    W = FunctionSpace(mesh, "Reduced-Hermite", 3)
    Theta = FunctionSpace(mesh, "Rotated-Bernardi-Raugel", 1)

    w = TrialFunction(W)
    v = TestFunction(W)
    discrete_grad_w = interpolate(grad(w), Theta)
    discrete_grad_v = interpolate(grad(v), Theta)
    a = inner(grad(discrete_grad_w), grad(discrete_grad_v)) * dx

    x, y = SpatialCoordinate(mesh)
    w_exact = x**2 * (1 - x)**2 * y**2 * (1 - y)**2
    L = inner(div(grad(div(grad(w_exact)))), v) * dx

    w_h = Function(W)
    bcs = DirichletBC(W, 0, "on_boundary")
    solve(a == L, w_h, bcs=bcs)

    # The Hessian of w_exact has degree 6, so a quadrature rule of degree 12
    # integrates the squared error exactly.
    error = grad(interpolate(grad(w_h), Theta)) - grad(grad(w_exact))
    return sqrt(assemble(inner(error, error) * dx(degree=12)).real)


def test_discrete_kirchhoff_hessian_convergence():
    # The discrete gradient is piecewise linear, apart from the edge bubbles,
    # so the error of its gradient converges at first order.
    errors = np.array([discrete_kirchhoff_error(n) for n in (4, 8, 16)])
    rates = np.log2(errors[:-1] / errors[1:])
    assert (rates > 0.8).all()
