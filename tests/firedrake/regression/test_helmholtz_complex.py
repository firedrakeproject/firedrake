"""This demo program solves Helmholtz's equation

  - div grad u(x, y) - u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (pi**2 - 1)*exp(1j*x[0]*pi)

and the analytical solution

  u(x, y) = exp(1j*x[0]*pi)
"""

import numpy as np
import pytest

from firedrake import *


def helmholtz(r, quadrilateral=False, degree=2, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    lmbda = -1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    u_exact = exp(1j*x[0]*pi)
    f.interpolate((pi*pi-1)*u_exact)
    a = (inner(grad(u), grad(v)) + lmbda * inner(u, v)) * dx + inner(1j*u, pi*v)*ds(1) - inner(1j*u, pi*v)*ds(2)
    L = inner(f, v) * dx

    # Compute solution
    assemble(a)
    assemble(L)
    sol = Function(V)
    solve(a == L, sol, solver_parameters={'pc_type': 'lu'})

    # Analytical solution
    f.interpolate(u_exact)
    return sqrt(assemble(inner(sol - f, sol - f) * dx)), sol, f


def run_firedrake_helmholtz():
    diff = np.array([helmholtz(i)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


@pytest.mark.skipreal
def test_firedrake_helmholtz_serial():
    run_firedrake_helmholtz()


@pytest.mark.skipreal
@pytest.mark.parallel
def test_firedrake_helmholtz_parallel():
    run_firedrake_helmholtz()


@pytest.mark.skipreal
@pytest.mark.parametrize(('testcase', 'convrate'),
                         [((1, (4, 6)), 1.9),
                          ((2, (3, 6)), 2.9),
                          ((3, (2, 4)), 3.9)])
def test_firedrake_helmholtz_scalar_convergence_on_quadrilaterals(testcase, convrate):
    degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        val = helmholtz(ii, quadrilateral=True, degree=degree)[0]
        l2err[ii - start] = val.real
        assert np.allclose(val.imag, 0)
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()
