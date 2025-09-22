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


def helmholtz(r, quadrilateral=False, degree=1, variant=None, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "CR", degree, variant=variant)

    x, y = SpatialCoordinate(mesh)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    uex = cos(x*pi*2)*cos(y*pi*2)
    f = -div(grad(uex)) + uex

    a = (inner(grad(u), grad(v)) + inner(u, v))*dx
    L = inner(f, v)*dx(degree=12)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    # Compute solution
    sol = Function(V)
    solve(a == L, sol, solver_parameters=params)
    # Error norm
    return sqrt(assemble(dot(sol - uex, sol - uex) * dx)), sol, uex


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [((1, (4, 6)), 1.9),
                          ((3, (2, 4)), 3.9),
                          ((5, (2, 4)), 5.7)])
@pytest.mark.parametrize("variant", ("point", "integral"))
def test_firedrake_helmholtz_scalar_convergence(variant, testcase, convrate):
    degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start] = helmholtz(ii, degree=degree, variant=variant)[0]
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()
