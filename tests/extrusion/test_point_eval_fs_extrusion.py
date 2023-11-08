from os.path import abspath, dirname
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


def extrude(m):
    mesh = ExtrudedMesh(m, 1)
    return mesh


@pytest.fixture
def mesh_quad():
    m = IntervalMesh(1, -0.2, 1.4)
    return extrude(m)


@pytest.fixture
def mesh_prism():
    m = UnitTriangleMesh()
    m.coordinates.dat.data[:] = [[0.1, 0.0], [1.2, 0.0], [0.0, 0.9]]
    return extrude(m)


@pytest.fixture
def mesh_hex():
    m = UnitSquareMesh(1, 1, quadrilateral=True)
    for row in m.coordinates.dat.data:
        row[:] = [1.1*row[0] - 0.1*row[1],
                  0.1*row[0] + 1.0*row[1]]
    return extrude(m)


@pytest.mark.parametrize(('family', 'degree', 'vfamily', 'vdegree'),
                         [('CG', 3, 'DG', 2),
                          ('DG', 3, 'CG', 2)])
def test_quad(mesh_quad, family, degree, vfamily, vdegree):
    V = FunctionSpace(mesh_quad, family, degree, vfamily=vfamily, vdegree=vdegree)
    xs = SpatialCoordinate(mesh_quad)
    f = Function(V).interpolate((xs[0] - 0.5)*(xs[0] - xs[1]))
    assert np.allclose(0.02, f([0.6, 0.4]))
    assert np.allclose(0.45, f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('RTCF', 2),
                          ('RTCE', 2)])
def test_quad_vector(mesh_quad, family, degree):
    x, y = SpatialCoordinate(mesh_quad)
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_quad, family, degree)
        f = Function(V).interpolate(as_vector([0.2 + y, 0.8*x + 0.2*y]))
    else:
        V = FunctionSpace(mesh_quad, family, degree)
        f = Function(V).project(as_vector([0.2 + y, 0.8*x + 0.2*y]))

    assert np.allclose([0.6, 0.56], f([0.6, 0.4]))
    assert np.allclose([1.1, 0.18], f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree', 'vfamily', 'vdegree'),
                         [('CG', 3, 'DG', 2),
                          ('DG', 3, 'CG', 2)])
def test_prism(mesh_prism, family, degree, vfamily, vdegree):
    x, y, z = SpatialCoordinate(mesh_prism)
    V = FunctionSpace(mesh_prism, family, degree, vfamily=vfamily, vdegree=vdegree)
    f = Function(V).interpolate((x - 0.5)*(y - z))
    assert np.allclose(0.01, f([0.6, 0.4, 0.3]))
    assert np.allclose(0.06, f([0.2, 0.6, 0.8]))


@pytest.mark.parametrize('args',
                         [('CG', 1),
                          ('DG', 1),
                          EnrichedElement(
                              HDiv(TensorProductElement(
                                  FiniteElement('RT', triangle, 2),
                                  FiniteElement('DG', interval, 1))),
                              HDiv(TensorProductElement(
                                  FiniteElement('DG', triangle, 1),
                                  FiniteElement('CG', interval, 2)))),
                          EnrichedElement(
                              HCurl(TensorProductElement(
                                  FiniteElement('RT', triangle, 2),
                                  FiniteElement('CG', interval, 2))),
                              HCurl(TensorProductElement(
                                  FiniteElement('CG', triangle, 2),
                                  FiniteElement('DG', interval, 1))))])
def test_prism_vector(mesh_prism, args):

    x, y, z = SpatialCoordinate(mesh_prism)
    if isinstance(args, tuple):
        V = VectorFunctionSpace(mesh_prism, *args)
        f = Function(V).interpolate(as_vector([0.2 + y, 0.8*x + 0.2*z, y]))
    else:
        V = FunctionSpace(mesh_prism, args)
        f = Function(V).project(as_vector([0.2 + y, 0.8*x + 0.2*z, y]))

    assert np.allclose([0.6, 0.54, 0.4], f([0.6, 0.4, 0.3]))
    assert np.allclose([0.8, 0.32, 0.6], f([0.2, 0.6, 0.8]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2)])
def test_hex(mesh_hex, family, degree):
    x, y, z = SpatialCoordinate(mesh_hex)
    V = FunctionSpace(mesh_hex, family, degree)
    f = Function(V).interpolate((x - 0.5)*(y - z))
    assert np.allclose(+0.01, f([0.6, 0.4, 0.3]))
    assert np.allclose(-0.06, f([0.4, 0.7, 0.1]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('NCF', 2),
                          ('NCE', 2)])
def test_hex_vector(mesh_hex, family, degree):

    x, y, z = SpatialCoordinate(mesh_hex)
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_hex, family, degree)
        f = Function(V).interpolate(as_vector([0.2 + y, 0.8*x + 0.2*z, y]))
    else:
        V = FunctionSpace(mesh_hex, family, degree)
        f = Function(V).project(as_vector([0.2 + y, 0.8*x + 0.2*z, y]))

    assert np.allclose([0.6, 0.54, 0.4], f([0.6, 0.4, 0.3]))
    assert np.allclose([0.9, 0.34, 0.7], f([0.4, 0.7, 0.1]))
