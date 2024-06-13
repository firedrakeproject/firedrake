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
        V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
        eps = Constant(1 / 2**(n+1))

        x, y = SpatialCoordinate(mesh)
        new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                                 y - eps*sin(8*pi*x)*sin(8*pi*y)]))
        mesh = Mesh(new)

    V = FunctionSpace(mesh, el_type, degree)
    x = SpatialCoordinate(V.mesh())

    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    f.project((1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2))
    a = (inner(grad(u), grad(v)) + lmbda * inner(u, v)) * dx
    L = inner(f, v) * dx

    # Compute solution
    assemble(a)
    assemble(L)
    sol = Function(V)
    solve(a == L, sol, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    # Analytical solution
    f.project(cos(x[0]*pi*2)*cos(x[1]*pi*2))
    return sqrt(assemble(inner(sol - f, sol - f) * dx))


# Test convergence on Hermite, Bell, and Argyris
# Morley is omitted since it only can be used on 4th-order problems.
# It is, somewhat oddly, a suitable C^1 nonconforming element but
# not a suitable C^0 nonconforming one.
@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Hermite', 3, 3.8),
                          ('HCT', 3, 3.8),
                          ('Bell', 5, 4.8),
                          ('Argyris', 5, 4.8)])
@pytest.mark.parametrize("perturb", [False, True], ids=["Regular", "Perturbed"])
def test_firedrake_helmholtz_scalar_convergence(el, deg, convrate, perturb):
    diff = np.array([helmholtz(i, el, deg, perturb) for i in range(1, 4)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate).all()
