from firedrake import *
from firedrake.mesh import _from_cell_list
import numpy as np


def test_two_triangles():
    parameters['pyop2_options']['debug'] = True
    nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    eles1 = [[0, 1, 2], [0, 2, 3]]
    eles2 = [[0, 1, 3], [1, 2, 3]]
    mesh1 = Mesh(_from_cell_list(2, eles1, nodes, COMM_WORLD), reorder=False)
    mesh2 = Mesh(_from_cell_list(2, eles2, nodes, COMM_WORLD), reorder=False)
    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    dx1 = dx(domain=mesh1)

    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    u1 = interpolate(x1, V1)
    u2 = interpolate(y2, V2)
    np.testing.assert_almost_equal(assemble(u1*u2*dx1), assemble(x1*y1*dx1))
    np.testing.assert_almost_equal(assemble(u1*u2*u2*dx1), assemble(x1*y1*y1*dx1))

    V1 = FunctionSpace(mesh1, "CG", 2)
    V2 = FunctionSpace(mesh2, "DG", 3)
    u1 = interpolate(x1*y1, V1)
    u2 = interpolate(y2+x2**3, V2)
    np.testing.assert_almost_equal(assemble(u1*u2*dx1), assemble((x1*y1)*(y1+x1**3)*dx))
    u1 = project(u2, V1)
    v1 = project(y1+x1**3, V1)
    np.testing.assert_almost_equal(u1.dat.data, v1.dat.data)
