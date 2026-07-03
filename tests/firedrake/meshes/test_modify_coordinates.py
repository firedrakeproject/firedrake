from firedrake import *
import numpy as np


def test_scale_coordinates():

    m = UnitSquareMesh(4, 4)
    m.coordinates *= 2

    assert np.allclose(assemble(Constant(1)*dx(domain=m)), 4.0)


def test_addto_coordinates():

    m = UnitSquareMesh(4, 4)
    X = SpatialCoordinate(m)
    xf = Function(m.coordinates.function_space())
    xf.interpolate(X)
    m.coordinates += xf

    assert np.allclose(assemble(Constant(1)*dx(domain=m)), 4.0)
