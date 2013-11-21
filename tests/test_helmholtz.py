"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

import pytest

from firedrake import *


def helmholtz(x, degree=2):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    f.interpolate(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    a = (dot(grad(v), grad(u)) + lmbda * v * u) * dx
    L = f * v * dx

    # Compute solution
    assemble(a)
    assemble(L)
    x = Function(V)
    solve(a == L, x, solver_parameters={'ksp_type': 'cg'})

    # Analytical solution
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    return sqrt(assemble(dot(x - f, x - f) * dx)), x, f


def test_firedrake_helmholtz():
    import numpy as np
    diff = np.array([helmholtz(i)[0] for i in range(3, 6)])
    print "l2 error norms:", diff
    conv = np.log2(diff[:-1] / diff[1:])
    print "convergence order:", conv
    assert (np.array(conv) > 2.8).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
