import numpy
from firedrake import *


def test_extrude_uniform_mesh_volume():
    mesh = UnitSquareMesh(4, 4)

    extmesh = ExtrudedMesh(mesh, 4, layer_height=[0.1, 0.2, 0.3, 0.4], extrusion_type="uniform")

    assert numpy.allclose(assemble(1*dx(domain=extmesh)), 1.0)


def test_extrude_variable_uniform_mesh_volume():
    mesh = IntervalMesh(2, 2)
    mesh.coordinates.dat.data[2] = 3
    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [2, 3]],
                           layer_height=[0.1, 0.2, 0.3, 0.4, 0.5])

    assert numpy.allclose(assemble(1*dx(domain=extmesh)), (0.1 + 0.2) + 2 * (0.3 + 0.4 + 0.5))


def test_extrude_radial_mesh_volume():
    radius = 1000

    mesh = IcosahedralSphereMesh(radius=radius, refinement_level=4)

    # layer heights sum to 1.5 / radius
    layer_heights = numpy.array([0.2] * 5 + [0.1] * 5) / radius
    extmesh = ExtrudedMesh(mesh, 10, layer_height=layer_heights, extrusion_type="radial")

    exact = 4 * pi * ((radius + 1.5/radius)**3 - radius**3) / 3
    assert numpy.allclose(assemble(1*dx(domain=extmesh)), exact, rtol=4e-3)
