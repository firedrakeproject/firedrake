"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

# Begin demo
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
    diff = [run_test(i, parameters) for i in range(3, 8)]
    from math import log
    import numpy as np
    conv = [log(diff[i] / diff[i + 1], 2) for i in range(len(diff) - 1)]
    return np.array(conv)

if __name__ == '__main__':
    l2_diff = run_test(5)
    print "L2 difference to analytic solution: %s" % l2_diff
