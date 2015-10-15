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
    f = Function(V).interpolate(Expression("(x[0] - 0.5)*(x[0] - x[1])"))
    assert np.allclose(0.02, f([0.6, 0.4]))
    assert np.allclose(0.45, f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('RTCF', 2),
                          ('RTCE', 2)])
def test_quad_vector(mesh_quad, family, degree):
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_quad, family, degree)
        f = Function(V).interpolate(Expression(("0.2 + x[1]", "0.8*x[0] + 0.2*x[1]")))
    else:
        V = FunctionSpace(mesh_quad, family, degree)
        f = Function(V).project(Expression(("0.2 + x[1]", "0.8*x[0] + 0.2*x[1]")))

    assert np.allclose([0.6, 0.56], f([0.6, 0.4]))
    assert np.allclose([1.1, 0.18], f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree', 'vfamily', 'vdegree'),
                         [('CG', 3, 'DG', 2),
                          ('DG', 3, 'CG', 2)])
def test_prism(mesh_prism, family, degree, vfamily, vdegree):
    V = FunctionSpace(mesh_prism, family, degree, vfamily=vfamily, vdegree=vdegree)
    f = Function(V).interpolate(Expression("(x[0] - 0.5)*(x[1] - x[2])"))
    assert np.allclose(0.01, f([0.6, 0.4, 0.3]))
    assert np.allclose(0.06, f([0.2, 0.6, 0.8]))


@pytest.mark.parametrize('args',
                         [('CG', 1),
                          ('DG', 1),
                          EnrichedElement(
                              HDiv(OuterProductElement(
                                  FiniteElement('RT', triangle, 2),
                                  FiniteElement('DG', interval, 1))),
                              HDiv(OuterProductElement(
                                  FiniteElement('DG', triangle, 1),
                                  FiniteElement('CG', interval, 2)))),
                          EnrichedElement(
                              HCurl(OuterProductElement(
                                  FiniteElement('RT', triangle, 2),
                                  FiniteElement('CG', interval, 2))),
                              HCurl(OuterProductElement(
                                  FiniteElement('CG', triangle, 2),
                                  FiniteElement('DG', interval, 1))))])
def test_prism_vector(mesh_prism, args):
    if isinstance(args, tuple):
        V = VectorFunctionSpace(mesh_prism, *args)
        f = Function(V).interpolate(Expression(("0.2 + x[1]", "0.8*x[0] + 0.2*x[2]", "x[1]")))
    else:
        V = FunctionSpace(mesh_prism, args)
        f = Function(V).project(Expression(("0.2 + x[1]", "0.8*x[0] + 0.2*x[2]", "x[1]")))

    assert np.allclose([0.6, 0.54, 0.4], f([0.6, 0.4, 0.3]))
    assert np.allclose([0.8, 0.32, 0.6], f([0.2, 0.6, 0.8]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2)])
def test_hex(mesh_hex, family, degree):
    V = FunctionSpace(mesh_hex, family, degree)
    f = Function(V).interpolate(Expression("(x[0] - 0.5)*(x[1] - x[2])"))
    assert np.allclose(+0.01, f([0.6, 0.4, 0.3]))
    assert np.allclose(-0.06, f([0.4, 0.7, 0.1]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('NCF', 2),
                          ('NCE', 2)])
def test_hex_vector(mesh_hex, family, degree):
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_hex, family, degree)
        f = Function(V).interpolate(Expression(("0.2 + x[1]", "0.8*x[0] + 0.2*x[2]", "x[1]")))
    else:
        V = FunctionSpace(mesh_hex, family, degree)
        f = Function(V).project(Expression(("0.2 + x[1]", "0.8*x[0] + 0.2*x[2]", "x[1]")))

    assert np.allclose([0.6, 0.54, 0.4], f([0.6, 0.4, 0.3]))
    assert np.allclose([0.9, 0.34, 0.7], f([0.4, 0.7, 0.1]))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
