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

    V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
    eps = Constant(0.1 / 2**(n+1))

    x, y = SpatialCoordinate(mesh)
    # Interpolation is not a thing yet for FInAT bases without underlying
    # FIAT elements.  Hopefully soon
    new = Function(V).project(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                         y - eps*sin(8*pi*x)*sin(8*pi*y)]))
    mesh = Mesh(new)

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
                          (4, 4.7),
                          (5, 5.7)])
def test_firedrake_projection_scalar_convergence(deg, convrate):
    diff = np.array([do_projection(i, deg) for i in range(3, 7)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > convrate).all()
