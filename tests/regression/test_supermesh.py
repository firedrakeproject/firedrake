from firedrake import *
from firedrake.mesh import _from_cell_list


def test_two_triangles():
    parameters['pyop2_options']['debug'] = True
    nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    eles1 = [[0, 1, 2], [0, 2, 3]]
    eles2 = [[0, 1, 3], [1, 2, 3]]
    mesh1 = Mesh(_from_cell_list(2, eles1, nodes, COMM_WORLD), reorder=False)
    mesh2 = Mesh(_from_cell_list(2, eles2, nodes, COMM_WORLD), reorder=False)
    V1 = FunctionSpace(mesh1, "DG", 3)
    x1, y1 = SpatialCoordinate(mesh1)
    u1 = interpolate(x1, V1)
    V2 = FunctionSpace(mesh2, "DG", 2)
    x2, y2 = SpatialCoordinate(mesh2)
    u2 = interpolate(y2, V2)
    F = u1 * u2 * u2 * dx(domain=mesh1)
    print(assemble(F))
