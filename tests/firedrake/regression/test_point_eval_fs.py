from os.path import abspath, dirname
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


@pytest.fixture
def mesh_interval():
    return IntervalMesh(1, -0.2, 1.4)


@pytest.fixture
def mesh_triangle():
    m = UnitTriangleMesh()
    m.coordinates.dat.data[:] = [[0.1, 0.0], [1.2, 0.0], [0.0, 0.9]]
    return m


@pytest.fixture
def mesh_quadrilateral():
    m = UnitSquareMesh(1, 1, quadrilateral=True)
    for row in m.coordinates.dat.data:
        row[:] = [1.1*row[0] - 0.1*row[1],
                  0.1*row[0] + 1.0*row[1]]
    return m


@pytest.fixture
def mesh_tetrahedron():
    m = UnitTetrahedronMesh()
    m.coordinates.dat.data[:] = [[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [0.4, 1.0, 0.0],
                                 [0.5, 0.6, 1.0]]
    return m


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2)])
def test_interval(mesh_interval, family, degree):
    V = FunctionSpace(mesh_interval, family, degree)
    x = SpatialCoordinate(mesh_interval)
    f = Function(V).interpolate((x[0] - 0.5)*(x[0] - 0.5))
    assert np.allclose(0.01, f([0.6]))
    assert np.allclose(0.25, f([1.0]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2)])
def test_interval_vector(mesh_interval, family, degree):
    V = VectorFunctionSpace(mesh_interval, family, degree, dim=2)
    x = SpatialCoordinate(mesh_interval)
    f = Function(V).interpolate(as_vector(((x[0] - 0.5)*(x[0] - 0.5), x[0]*x[0])))
    assert np.allclose([0.01, 0.36], f([0.6]))
    assert np.allclose([0.25, 1.00], f([1.0]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2),
                          ('Bernstein', 2)])
def test_triangle(mesh_triangle, family, degree):
    V = FunctionSpace(mesh_triangle, family, degree)
    # Bernstein currently requires projection
    x = SpatialCoordinate(mesh_triangle)
    f = Function(V).project((x[0] - 0.5)*(x[1] - 0.2))
    assert np.allclose(+0.02, f([0.6, 0.4]))
    assert np.allclose(-0.35, f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('RT', 2),
                          ('BDM', 2),
                          ('BDFM', 2),
                          ('N1curl', 2),
                          ('N2curl', 2)])
def test_triangle_vector(mesh_triangle, family, degree):
    x = SpatialCoordinate(mesh_triangle)
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_triangle, family, degree)
        f = Function(V).interpolate(as_vector((0.2 + x[1], 0.8*x[0] + 0.2*x[1])))
    else:
        V = FunctionSpace(mesh_triangle, family, degree)
        f = Function(V).project(as_vector((0.2 + x[1], 0.8*x[0] + 0.2*x[1])))

    assert np.allclose([0.6, 0.56], f([0.6, 0.4]))
    assert np.allclose([1.1, 0.18], f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1)])
def test_triangle_tensor(mesh_triangle, family, degree):
    V = TensorFunctionSpace(mesh_triangle, family, degree)
    x = SpatialCoordinate(mesh_triangle)
    f = Function(V).interpolate(as_tensor(((x[1], 0.2 + x[0]), (0.8*x[0], 0.2*x[1]))))

    assert np.allclose([[0.4, 0.8], [0.48, 0.08]], f([0.6, 0.4]))
    assert np.allclose([[0.9, 0.2], [0.00, 0.18]], f([0.0, 0.9]))


def test_triangle_mixed(mesh_triangle):
    V1 = FunctionSpace(mesh_triangle, "DG", 1)
    V2 = FunctionSpace(mesh_triangle, "RT", 2)
    V = V1 * V2
    f = Function(V)
    f1, f2 = f.subfunctions
    x = SpatialCoordinate(mesh_triangle)
    f1.interpolate(x[0] + 1.2*x[1])
    f2.project(as_vector((x[1], 0.8 + x[0])))

    # Single point
    actual = f.at([0.6, 0.4])
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    assert np.allclose(1.08, actual[0])
    assert np.allclose([0.4, 1.4], actual[1])

    # Multiple points
    actual = f.at([0.6, 0.4], [0.0, 0.9], [0.3, 0.5])
    assert len(actual) == 3
    assert np.allclose(1.08, actual[0][0])
    assert np.allclose([0.4, 1.4], actual[0][1])
    assert np.allclose(1.08, actual[1][0])
    assert np.allclose([0.9, 0.8], actual[1][1])
    assert np.allclose(0.90, actual[2][0])
    assert np.allclose([0.5, 1.1], actual[2][1])


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2)])
def test_quadrilateral(mesh_quadrilateral, family, degree):
    V = FunctionSpace(mesh_quadrilateral, family, degree)
    x = SpatialCoordinate(mesh_quadrilateral)
    f = Function(V).interpolate((x[0] - 0.5)*(x[1] - 0.2))
    assert np.allclose(+0.02, f([0.6, 0.4]))
    assert np.allclose(-0.35, f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('RTCF', 2),
                          ('RTCE', 2)])
def test_quadrilateral_vector(mesh_quadrilateral, family, degree):
    x = SpatialCoordinate(mesh_quadrilateral)
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_quadrilateral, family, degree)
        f = Function(V).interpolate(as_vector((0.2 + x[1], 0.8*x[0] + 0.2*x[1])))
    else:
        V = FunctionSpace(mesh_quadrilateral, family, degree)
        f = Function(V).project(as_vector((0.2 + x[1], 0.8*x[0] + 0.2*x[1])))

    assert np.allclose([0.6, 0.56], f([0.6, 0.4]))
    assert np.allclose([1.1, 0.18], f([0.0, 0.9]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 2),
                          ('DG', 2)])
def test_tetrahedron(mesh_tetrahedron, family, degree):
    V = FunctionSpace(mesh_tetrahedron, family, degree)
    x = SpatialCoordinate(mesh_tetrahedron)
    f = Function(V).interpolate((x[0] - 0.5)*(x[1] - x[2]))
    assert np.allclose(+0.01, f([0.6, 0.4, 0.3]))
    assert np.allclose(-0.06, f([0.4, 0.7, 0.1]))


@pytest.mark.parametrize(('family', 'degree'),
                         [('CG', 1),
                          ('DG', 1),
                          ('N1F', 2),
                          ('N2F', 2),
                          ('N1E', 2),
                          ('N2E', 2)])
def test_tetrahedron_vector(mesh_tetrahedron, family, degree):
    x = SpatialCoordinate(mesh_tetrahedron)
    if family in ['CG', 'DG']:
        V = VectorFunctionSpace(mesh_tetrahedron, family, degree)
        f = Function(V).interpolate(as_vector((0.2 + x[1], 0.8*x[0] + 0.2*x[2], x[1])))
    else:
        V = FunctionSpace(mesh_tetrahedron, family, degree)
        f = Function(V).project(as_vector((0.2 + x[1], 0.8*x[0] + 0.2*x[2], x[1])))

    assert np.allclose([0.6, 0.54, 0.4], f([0.6, 0.4, 0.3]))
    assert np.allclose([0.9, 0.34, 0.7], f([0.4, 0.7, 0.1]))


def test_point_eval_forces_writes():
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)

    assert np.allclose([0.0], f.at((0.3, 0.3)))
    f.assign(1)
    assert np.allclose([1.0], f.at((0.3, 0.3)))


def test_point_reset_works():
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)

    assert np.allclose([0.0], f.at((0.3, 0.3)))
    f.assign(1)
    m.clear_spatial_index()
    assert np.allclose([1.0], f.at((0.3, 0.3)))


def test_changing_coordinates_invalidates_spatial_index():
    mesh = UnitSquareMesh(2, 2)
    mesh.init()

    saved_spatial_index = mesh.spatial_index
    mesh.coordinates.assign(mesh.coordinates * 2)
    assert mesh.spatial_index != saved_spatial_index
