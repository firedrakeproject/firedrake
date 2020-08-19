"""This does L^2 projection

on the unit square of a function

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

using elements with nonstandard pullbacks
"""

import numpy as np
import pytest

from firedrake import *


def do_projection(n, degree):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**n, 2**n, quadrilateral=True)

    # V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
    # eps = Constant(0.5 / 2**(n+1))

    # x, y = SpatialCoordinate(mesh)
    # new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
    #                                          y - eps*sin(8*pi*x)*sin(8*pi*y)]))
    # mesh = Mesh(new)

    V = FunctionSpace(mesh, "Sdirect", degree)

    # Define variational problem

    x, y = SpatialCoordinate(mesh)
    f = sin(x*pi)*sin(2*pi*y)
    u = project(f, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    # Compute solution
    return errornorm(f, u)


@pytest.mark.parametrize(('deg', 'convrate'),
                         [(2, 2.5),
                          (3, 3.8),
                          (4, 4.8),
                          (5, 4.9)])
def test_firedrake_projection_scalar_convergence(deg, convrate):
    diff = np.array([do_projection(i, deg) for i in range(2, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    print(diff)
    print(conv)
    assert np.array(conv)[-1] > convrate
