"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

import numpy as np
import pytest
from firedrake import *


def helmholtz(n, el_type, degree, perturb):
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
