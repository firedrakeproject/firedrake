from firedrake import *
import pytest
import numpy as np


@pytest.fixture(params=["interval", "square", "quad-square"])
def uniform_mesh(request):
    if request.param == "interval":
        base = UnitIntervalMesh(4)
    elif request.param == "square":
        base = UnitSquareMesh(5, 4)
    elif request.param == "quad-square":
        base = UnitSquareMesh(4, 6, quadrilateral=True)
    return ExtrudedMesh(base, layers=10, layer_height=0.1,
                        extrusion_type="uniform")


@pytest.fixture(params=["circlemanifold",
                        "icosahedron",
                        "cubedsphere"])
def hedgehog_mesh(request):
    if request.param == "circlemanifold":
        # Circumference of 1
        base = CircleManifoldMesh(ncells=3, radius=1/np.sqrt(27))
    elif request.param == "icosahedron":
        # Surface area of 1
        base = IcosahedralSphereMesh(np.sin(2*np.pi/5) * np.sqrt(1/(5*np.sqrt(3))), refinement_level=0)
    elif request.param == "cubedsphere":
        # Surface area of 1
        base = CubedSphereMesh(radius=1/(2*np.sqrt(2)), refinement_level=0)

    return ExtrudedMesh(base, layers=5, layer_height=0.2, extrusion_type="radial_hedgehog")


def test_uniform_extrusion_volume(uniform_mesh):
    assert np.allclose(assemble(Constant(1, domain=uniform_mesh)*dx), 1.0)


def test_hedgehog_extrusion_volume(hedgehog_mesh):
    assert np.allclose(assemble(Constant(1, domain=hedgehog_mesh)*dx), 1.0)
