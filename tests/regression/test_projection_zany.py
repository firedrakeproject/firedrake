"""This does L^2 projection

on the unit square of a function

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

using elements with nonstandard pullbacks
"""

import numpy as np
import pytest

from firedrake import *


def do_projection(n, el_type, degree):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**n, 2**n)

    V = FunctionSpace(mesh, el_type, degree)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(mesh)
    f = sin(x*pi)*sin(2*pi*y)
    a = inner(u, v) * dx
    L = inner(f, v) * dx

    # Compute solution
    x = Function(V)
    solve(a == L, x, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner(x - f, x - f) * dx))


@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Morley', 2, 2.4),
                          ('Hermite', 3, 3),
                          ('Bell', 5, 4),
                          ('Argyris', 5, 4.9)])
def test_firedrake_projection_scalar_convergence(el, deg, convrate):
    diff = np.array([do_projection(i, el, deg) for i in range(1, 4)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate).all()
