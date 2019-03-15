from firedrake import *
from firedrake.supermeshing import *
import pytest


@pytest.fixture(params=[2, "q", 3])
def mesh(request):
    if request.param == 2:
        return UnitSquareMesh(2, 3)
    if request.param == "q":
        return UnitSquareMesh(2, 3, quadrilateral=True)
    if request.param == 3:
        return UnitCubeMesh(3, 2, 1)


def test_intersection_finder(mesh):
    mesh_A = mesh
    mesh_B = mesh

    intersections = intersection_finder(mesh_A, mesh_B)

    for cell_A in range(mesh_A.num_cells()):
        print("intersections[%d] = %s" % (cell_A, intersections[cell_A]))
        assert cell_A in intersections[cell_A]
