"""This does L^2 projection

on the unit square of a function

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

using some C^0 macro elements.
"""

import numpy as np
import pytest
from firedrake import *


def interp(u, f):
    u.interpolate(f)


def proj(u, f):
    u.project(f)


def proj_bc(u, f):
    u.project(f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


def h1_proj(u, f, bcs=None):
    # compute h1 projection of f into u's function
    # space, store the result in u.
    v = TestFunction(u.function_space())
    F = (inner(grad(u-f), grad(v)) * dx
         + inner(u-f, v) * dx)
    solve(F == 0, u,
          bcs=bcs,
          solver_parameters={"snes_type": "ksponly",
                             "ksp_type": "preonly",
                             "pc_type": "cholesky"})


def h1_proj_bc(u, f):
    h1_proj(u, f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


@pytest.fixture
def hierarchy():
    base_mesh = UnitSquareMesh(2**3, 2**3)
    return MeshHierarchy(base_mesh, 1)


@pytest.mark.parametrize('op', (interp, proj, proj_bc, h1_proj, h1_proj_bc))
@pytest.mark.parametrize(('deg', 'variant', 'convrate'),
                         [(2, None, 2.7),
                          (2, 'alfeld', 2.8),
                          (1, 'iso(2)', 1.9),
                          (1, 'iso(3)', 1.9)])
def test_projection_scalar_convergence(op, hierarchy, deg, variant, convrate):
    errors = []
    for msh in hierarchy:
        V = FunctionSpace(msh, "CG", degree=deg, variant=variant)
        u = Function(V)
        x, y = SpatialCoordinate(msh)
        f = sin(x*pi)*sin(2*pi*y)
        op(u, f)
        errors.append(sqrt(assemble(inner(u - f, u - f) * dx(degree=2*deg))))

    diff = np.array(errors)
    conv = np.log2(diff[:-1] / diff[1:])
    # test *eventual* convergence rate
    assert conv[-1] > convrate
