"""This does L^2 projection

on the unit square of a function

  f(x, y) = LegendreP(degree + 2, x + y)

using elements with nonstandard pullbacks
"""

import numpy as np
import pytest
from firedrake import *

relative_magnitudes = lambda x: np.array(x)[1:] / np.array(x)[:-1]
convergence_orders = lambda x: -np.log2(relative_magnitudes(x))


def jrc(a, b, n):
    """Jacobi recurrence coefficients"""
    an = (2*n+1+a+b)*(2*n+2+a+b) / (2*(n+1)*(n+1+a+b))
    bn = (a+b)*(a-b)*(2*n+1+a+b) / (2*(n+1)*(n+1+a+b)*(2*n+a+b))
    cn = (n+a)*(n+b)*(2*n+2+a+b) / ((n+1)*(n+1+a+b)*(2*n+a+b))
    return an, bn, cn


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

    m = degree + 2
    z = sum(x)
    p0 = 1
    p1 = z
    for n in range(1, m):
        an, bn, cn = jrc(0, 0, n)
        ptemp = p1
        p1 = (an * z + bn) * p1 - cn * p0
        p0 = ptemp

    f = p1 * Constant(np.ones(u.ufl_shape))
    a = inner(u, v) * dx
    L = inner(f, v) * dx

    # Compute solution
    fh = Function(V)
    solve(a == L, fh)
    return sqrt(assemble(inner(fh - f, fh - f) * dx))


@pytest.mark.parametrize(('el', 'deg', 'convrate'),
                         [('Johnson-Mercier', 1, 1.8),
                          ('Morley', 2, 2.4),
                          ('PS6', 2, 2.4),
                          ('PS12', 2, 2.4),
                          ('HCT-red', 3, 2.7),
                          ('HCT', 3, 3.7),
                          ('HCT', 4, 4.8),
                          ('Hermite', 3, 3.8),
                          ('Bell', 5, 4.7),
                          ('Argyris', 5, 5.8),
                          ('Argyris', 6, 6.7)])
def test_projection_zany_convergence_2d(hierarchy, el, deg, convrate):
    diff = np.array([do_projection(m, el, deg) for m in hierarchy[2:]])
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
