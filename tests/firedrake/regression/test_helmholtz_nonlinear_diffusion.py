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

import pytest

from firedrake import *


def helmholtz(r, quadrilateral=False, parameters={}):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    # Define variational problem
    kappa = Constant(1)
    alpha = Constant(0.1)
    u = Function(V)
    v = TestFunction(V)
    f = Function(V)
    D = 1 + alpha * u * u
    f.interpolate(-8*pi*pi*alpha*cos(2*pi*x[0])*cos(2*pi*x[1])
                  * cos(2*pi*x[1])*cos(2*pi*x[1])*sin(2*pi*x[0])*sin(2*pi*x[0])
                  - 8*pi*pi*alpha*cos(2*pi*x[0])*cos(2*pi*x[0])
                  * cos(2*pi*x[0])*cos(2*pi*x[1])*sin(2*pi*x[1])*sin(2*pi*x[1])
                  + 8*pi*pi*(alpha*cos(2*pi*x[0])*cos(2*pi*x[0])
                             * cos(2*pi*x[1])*cos(2*pi*x[1]) + 1)
                  * cos(2*pi*x[0])*cos(2*pi*x[1])
                  + kappa*cos(2*pi*x[0])*cos(2*pi*x[1]))
    a = (inner(D * grad(u), grad(v)) + kappa * inner(u, v)) * dx
    L = inner(f, v) * dx

    solve(a - L == 0, u, solver_parameters=parameters)

    f.interpolate(cos(x[0]*2*pi)*cos(x[1]*2*pi))

    return sqrt(assemble(inner(u - f, u - f) * dx))


def run_convergence_test(quadrilateral=False, parameters={}):
    import numpy as np
    l2_diff = np.array([helmholtz(i, quadrilateral, parameters) for i in range(3, 6)])
    return np.log2(l2_diff[:-1] / l2_diff[1:])


def run_l2_conv():
    assert (run_convergence_test() > 1.8).all()


def test_l2_conv_serial():
    run_l2_conv()


@pytest.mark.parallel
def test_l2_conv_parallel():
    run_l2_conv()


def run_l2_conv_on_quadrilaterals():
    assert (run_convergence_test(quadrilateral=True) > 1.8).all()


def test_l2_conv_on_quadrilaterals_serial():
    run_l2_conv_on_quadrilaterals()


@pytest.mark.parallel
def test_l2_conv_on_quadrilaterals_parallel():
    run_l2_conv_on_quadrilaterals()
