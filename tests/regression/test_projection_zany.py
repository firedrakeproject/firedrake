"""This does L^2 projection

on the unit square of a function

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

using elements with nonstandard pullbacks
"""

import numpy as np
import pytest

from firedrake import *


@pytest.fixture
def hierarchy(request):
    msh = UnitSquareMesh(2, 2)
    return MeshHierarchy(msh, 2)


def do_projection(mesh, el_type, degree):
    V = FunctionSpace(mesh, el_type, degree)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)

    f = np.prod([sin((1+i) * x[i]*pi) for i in range(len(x))])
    f = f * Constant(np.ones(u.ufl_shape))

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    # Compute solution
    fh = Function(V)
    solve(a == L, fh, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner(fh - f, fh - f) * dx))


@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Johnson-Mercier', 1, 1.8),
                          ('Morley', 2, 2.4),
                          ('HCT-red', 3, 2.7),
                          ('HCT', 3, 3),
                          ('Hermite', 3, 3),
                          ('Bell', 5, 4),
                          ('Argyris', 5, 4.9)])
def test_projection_zany_convergence_2d(hierarchy, el, deg, convrate):
    diff = np.array([do_projection(m, el, deg) for m in hierarchy])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate).all()
