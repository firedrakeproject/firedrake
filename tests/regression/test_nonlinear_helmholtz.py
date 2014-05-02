"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

import pytest

from firedrake import *


def run_test(x, parameters={}):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(mesh, "CG", 2)

    # Define variational problem
    lmbda = 1
    u = Function(V)
    v = TestFunction(V)
    f = Function(V)
    f.interpolate(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    a = (dot(grad(v), grad(u)) + lmbda * v * u) * dx
    L = f * v * dx

    # Compute solution
    solve(a - L == 0, u, solver_parameters=parameters)

    f.interpolate(Expression("cos(x[0]*2*pi)*cos(x[1]*2*pi)"))

    return sqrt(assemble(dot(u - f, u - f) * dx))


def run_convergence_test(parameters={}):
    import numpy as np
    diff = np.array([run_test(i, parameters) for i in range(3, 6)])
    return np.log2(diff[:-1] / diff[1:])


@pytest.mark.parametrize('params', [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}])
def test_l2_conv(params):
    assert (run_convergence_test(parameters=params) > 2.8).all()


@pytest.mark.xfail
@pytest.mark.parallel
def test_l2_conv_parallel():
    from mpi4py import MPI
    l2_conv = run_convergence_test()
    print '[%d]' % MPI.COMM_WORLD.rank, 'convergence rate:', l2_conv
    assert (l2_conv > 2.8).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
