from firedrake import *
import pytest


@pytest.fixture(params=["interval", "square-x-periodic", "square",
                        "quad-square"])
def base_mesh(request):
    if request.param == "interval":
        return PeriodicUnitIntervalMesh(4)
    elif request.param == "square-x-periodic":
        return PeriodicUnitSquareMesh(4, 4, direction="x")
    elif request.param == "square":
        return PeriodicUnitSquareMesh(4, 4)
    elif request.param == "quad-square":
        return PeriodicUnitSquareMesh(4, 4, quadrilateral=True)


def test_mixed_periodic(base_mesh):
    mesh = ExtrudedMesh(base_mesh, layers=4, layer_height=0.25)
    V1 = FunctionSpace(mesh, "DG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    V2_broken = FunctionSpace(mesh, BrokenElement(V2.ufl_element()))
    MixedFunctionSpace((V1, V2_broken))
