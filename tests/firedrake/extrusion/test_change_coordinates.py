from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np


def test_extruded_change_coordinates():
    # This test exists to ensure the code in the manual works.

    # start extruded change coordinates
    base_mesh = IntervalMesh(16, 0, 2*pi)
    # Make a height 1 mesh.
    unit_extruded_mesh = ExtrudedMesh(base_mesh, layers=10)

    base_fs = FunctionSpace(base_mesh, "CG", 1)

    x, = SpatialCoordinate(base_mesh)
    # You could set this field any way you like.
    bathymetry = assemble(interpolate(0.2*sin(x), base_fs))

    # Now we transfer the bathymetry field into a depth-averaged field.
    extruded_element = FiniteElement("R", "interval", 0)
    extruded_space = FunctionSpace(unit_extruded_mesh,
                                   TensorProductElement(base_fs.ufl_element(),
                                                        extruded_element))
    extruded_bathymetry = Function(extruded_space)
    extruded_bathymetry.dat.data_wo[:] = bathymetry.dat.data_ro[:]

    # Build a new coordinate field by change of coordinates.
    x, y = SpatialCoordinate(unit_extruded_mesh)
    new_coordinates = assemble(
        interpolate(
            as_vector([x, extruded_bathymetry + y * (1-extruded_bathymetry)]),
            unit_extruded_mesh.coordinates.function_space()
        )
    )
    # Finally build the mesh you are actually after.
    mesh = Mesh(new_coordinates)
    # end extruded change coordinates

    assert np.allclose(mesh.coordinates.dat.data_ro[:, 1].min(), -0.2)
