"""This does L^2 projection

on the unit square of a function

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

using some C^0 macro elements.
"""

import numpy as np
import pytest
from firedrake import *


# compute h1 projection of f into u's function
# space, store the result in u.
def h1_proj(u, f, bcs=None):
    v = TestFunction(u.function_space())
    F = (inner(grad(u-f), grad(v)) * dx
         + inner(u-f, v) * dx)
    solve(F == 0, u,
          bcs=bcs,
          solver_parameters={"snes_type": "ksponly",
                             "ksp_type": "preonly",
                             "pc_type": "lu"})


def h1_proj_bc(u, f):
    h1_proj(u, f, bcs=DirichletBC(u.function_space(),
                                  f, "on_boundary"))


def proj_bc(u, f):
    u.project(f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


def do_op(n, op, deg, variant):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**n, 2**n)

    el = FiniteElement("CG", triangle, deg, variant=variant)
    V = FunctionSpace(mesh, el)

    u = Function(V)
    x, y = SpatialCoordinate(mesh)

    f = sin(x*pi)*sin(2*pi*y)

    op(u, f)

    return sqrt(assemble(inner(u - f, u - f) * dx(degree=2*deg)))


@pytest.mark.parametrize('op', (lambda x, y: x.interpolate(y),
                                lambda x, y: x.project(y),
                                proj_bc, h1_proj, h1_proj_bc))
@pytest.mark.parametrize(('deg', 'variant', 'convrate'),
                         [(2, None, 2.7),
                          (2, 'alfeld', 2.8),
                          (1, 'iso(2)', 1.9),
                          (1, 'iso(3)', 1.9)])
def test_firedrake_projection_scalar_convergence(op, deg, variant, convrate):
    diff = np.array([do_op(i, op, deg, variant) for i in range(3, 5)])
    conv = np.log2(diff[:-1] / diff[1:])
    # test *eventual* convergence rate
    assert conv[-1] > convrate
