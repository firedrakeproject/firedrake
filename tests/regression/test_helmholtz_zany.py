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


def helmholtz(x, el_type, degree, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(mesh, el_type, degree)

    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    f.project(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    a = (dot(grad(v), grad(u)) + lmbda * v * u) * dx
    L = f * v * dx

    # Compute solution
    assemble(a)
    assemble(L)
    x = Function(V)
    solve(a == L, x, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    # Analytical solution
    f.project(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    return sqrt(assemble(dot(x - f, x - f) * dx)), x, f


# Test convergence on Hermite, Bell, and Argyris
# Morley is omitted since it only can be used on 4th-order problems.
# It is, somewhat oddly, a suitable C^1 nonconforming element but
# not a suitable C^0 nonconforming one.
@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Hermite', 3, 3.8),
                          ('Bell', 5, 4.8),
                          ('Argyris', 5, 5)])
def test_firedrake_helmholtz_scalar_convergence(el, deg, convrate):
    diff = np.array([helmholtz(i, el, deg)[0] for i in range(1, 4)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
