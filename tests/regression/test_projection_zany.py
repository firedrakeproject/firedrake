"""This does L^2 projection

on the unit square of a function

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

using elements with nonstandard pullbacks
"""

import numpy as np
import pytest

from firedrake import *


relative_magnitudes = lambda x: np.array(x)[1:] / np.array(x)[:-1]
convergence_orders = lambda x: -np.log2(relative_magnitudes(x))


@pytest.fixture
def hierarchy(request):
    msh = UnitSquareMesh(2, 2)
    return MeshHierarchy(msh, 4)


def do_projection(mesh, el_type, degree):
    V = FunctionSpace(mesh, el_type, degree)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)

    f = np.prod([sin(x[i]*pi) for i in range(len(x))])
    f = f * Constant(np.ones(u.ufl_shape))

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    # Compute solution
    fh = Function(V)
    solve(a == L, fh)

    return sqrt(assemble(inner(fh - f, fh - f) * dx))


@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Johnson-Mercier', 1, 1.8),
                          ('Morley', 2, 2.4),
                          ('HCT-red', 3, 2.7),
                          ('HCT', 3, 3.7),
                          ('HCT', 4, 4.8),
                          ('Hermite', 3, 3),
                          ('Bell', 5, 4),
                          ('Argyris', 5, 6),
                          ('Argyris', 6, 6.8)])
def test_projection_zany_convergence_2d(hierarchy, el, deg, convrate):
    l = 1 if deg > 3 else 2
    diff = np.array([do_projection(m, el, deg) for m in hierarchy[l:l+3]])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate).all()


@pytest.mark.parametrize(('element', 'degree'),
                         [('HCT-red', 3),
                          ('HCT', 3),
                          ('HCT', 4),
                          ('Argyris', 5),
                          ('Argyris', 6)])
def test_mass_conditioning(element, degree, hierarchy):
    mass_cond = []
    for msh in hierarchy[1:4]:
        V = FunctionSpace(msh, element, degree)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(u, v)*dx
        B = assemble(a, mat_type="aij").M.handle
        A = B.convert("dense").getDenseArray()
        kappa = np.linalg.cond(A)

        mass_cond.append(kappa)

    assert max(relative_magnitudes(mass_cond)) < 1.1
