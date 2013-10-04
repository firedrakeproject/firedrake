"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

# Begin demo
from firedrake import *


def run_test(x, degree=2):
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
    solve(a == L, x)

    # Analytical solution
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    return sqrt(assemble(dot(x - f, x - f) * dx)), x, f


def run_convergence_test():
    diff = []
    for i in range(3, 8):
        tmp, _, _ = run_test(i)
        diff.append(tmp)
    conv = []
    from math import log
    import numpy as np
    for i in range(len(diff) - 1):
        conv.append(log(diff[i] / diff[i + 1], 2))
    return np.array(conv)

if __name__ == '__main__':
    conv = run_convergence_test()
    print 'L2 convergence %s' % conv
    _, x, f = run_test(5, degree=1)
    # Save solution in VTK format
    output = File("helmholtz.pvd")
    output << x
    output << f
