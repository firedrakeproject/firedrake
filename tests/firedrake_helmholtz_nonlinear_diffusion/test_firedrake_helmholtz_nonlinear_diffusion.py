"""This demo program solves Helmholtz's equation

  - div D(u) grad u(x, y) + kappa u(x,y) = f(x, y)

with

   D(u) = 1 + alpha * u**2

   alpha = 0.1
   kappa = 1

on the unit square with source f given by

   f(x, y) = -8*pi^2*alpha*cos(2*pi*x)*cos(2*pi*y)^3*sin(2*pi*x)^2
             - 8*pi^2*alpha*cos(2*pi*x)^3*cos(2*pi*y)*sin(2*pi*y)^2
             + 8*pi^2*(alpha*cos(2*pi*x)^2*cos(2*pi*y)^2 + 1)
               *cos(2*pi*x)*cos(2*pi*y)
             + kappa*cos(2*pi*x)*cos(2*pi*y)

and the analytical solution

  u(x, y) = cos(x*2*pi)*cos(y*2*pi)
"""

# Begin demo


from firedrake import *


def run_test(x):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(mesh, "CG", 1)

    # Define variational problem
    kappa = 1
    alpha = 0.1
    u = Function(V)
    v = TestFunction(V)
    f = Function(V)
    D = 1 + alpha * u * u
    f.interpolate(
        Expression("-8*pi*pi*%(alpha)s*cos(2*pi*x[0])*cos(2*pi*x[1])\
                   *cos(2*pi*x[1])*cos(2*pi*x[1])*sin(2*pi*x[0])*sin(2*pi*x[0])\
                   - 8*pi*pi*%(alpha)s*cos(2*pi*x[0])*cos(2*pi*x[0])\
                   *cos(2*pi*x[0])*cos(2*pi*x[1])*sin(2*pi*x[1])*sin(2*pi*x[1])\
                   + 8*pi*pi*(%(alpha)s*cos(2*pi*x[0])*cos(2*pi*x[0])\
                   *cos(2*pi*x[1])*cos(2*pi*x[1]) + 1)*cos(2*pi*x[0])*cos(2*pi*x[1])\
                   + %(kappa)s*cos(2*pi*x[0])*cos(2*pi*x[1])"
                   % {'alpha': alpha, 'kappa': kappa}))
    a = (dot(grad(v), D * grad(u)) + kappa * v * u) * dx
    L = f * v * dx

    solve(a - L == 0, u)

    f.interpolate(Expression("cos(x[0]*2*pi)*cos(x[1]*2*pi)"))

    return sqrt(assemble(dot(u - f, u - f) * dx))


def run_convergence_test():
    import numpy as np
    l2_diff = [run_test(i) for i in range(3, 8)]

    from math import log
    conv = [log(l2_diff[i] / l2_diff[i + 1], 2)
            for i in range(len(l2_diff) - 1)]
    return np.array(l2_diff), np.array(conv)

l2_diff, l2_conv = run_convergence_test()
if __name__ == '__main__':
    print "L2 difference to analytic solution: %s" % l2_diff
    print "Convergence ratios: %s" % l2_conv
