from os.path import abspath, dirname
import numpy as np
import pytest

from firedrake import *
from firedrake.mesh import VertexOnlyMeshMissingPointsError

cwd = abspath(dirname(__file__))


def test_1d_args():
    mesh = UnitIntervalMesh(1)
    f = mesh.coordinates

    # one point
    assert np.allclose(0.2, f._at(0.2))
    assert np.allclose(0.2, f._at((0.2,)))
    assert np.allclose(0.2, f._at([0.2]))
    assert np.allclose(0.2, f._at(np.array([0.2])))

    # multiple points as arguments
    assert np.allclose([[0.2], [0.3]], f._at(0.2, 0.3))
    assert np.allclose([[0.2], [0.3]], f._at((0.2,), (0.3,)))
    assert np.allclose([[0.2], [0.3]], f._at([0.2], [0.3]))
    assert np.allclose([[0.2], [0.3]], f._at(np.array(0.2), np.array(0.3)))
    assert np.allclose([[0.2], [0.3]], f._at(np.array([0.2]), np.array([0.3])))

    # multiple points as tuple
    assert np.allclose([[0.2], [0.3]], f._at((0.2, 0.3)))
    assert np.allclose([[0.2], [0.3]], f._at(((0.2,), (0.3,))))
    assert np.allclose([[0.2], [0.3]], f._at(([0.2], [0.3])))
    assert np.allclose([[0.2], [0.3]], f._at((np.array(0.2), np.array(0.3))))
    assert np.allclose([[0.2], [0.3]], f._at((np.array([0.2]), np.array([0.3]))))

    # multiple points as list
    assert np.allclose([[0.2], [0.3]], f._at([0.2, 0.3]))
    assert np.allclose([[0.2], [0.3]], f._at([(0.2,), (0.3,)]))
    assert np.allclose([[0.2], [0.3]], f._at([[0.2], [0.3]]))
    assert np.allclose([[0.2], [0.3]], f._at([np.array(0.2), np.array(0.3)]))
    assert np.allclose([[0.2], [0.3]], f._at([np.array([0.2]), np.array([0.3])]))

    # multiple points as numpy array
    assert np.allclose([[0.2], [0.3]], f._at(np.array([0.2, 0.3])))
    assert np.allclose([[0.2], [0.3]], f._at(np.array([[0.2], [0.3]])))


def test_2d_args():
    mesh = UnitSquareMesh(1, 1)
    f = mesh.coordinates

    # one point
    assert np.allclose([0.2, 0.4], f._at(0.2, 0.4))
    assert np.allclose([0.2, 0.4], f._at((0.2, 0.4)))
    assert np.allclose([0.2, 0.4], f._at([0.2, 0.4]))
    assert np.allclose([0.2, 0.4], f._at(np.array([0.2, 0.4])))

    # multiple points as arguments
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at((0.2, 0.4), (0.3, 0.5)))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at([0.2, 0.4], [0.3, 0.5]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at(np.array([0.2, 0.4]),
                                                       np.array([0.3, 0.5])))

    # multiple points as tuple
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at(((0.2, 0.4), (0.3, 0.5))))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at(([0.2, 0.4], [0.3, 0.5])))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at((np.array([0.2, 0.4]),
                                                        np.array([0.3, 0.5]))))

    # multiple points as list
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at([(0.2, 0.4), (0.3, 0.5)]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at([[0.2, 0.4], [0.3, 0.5]]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at([np.array([0.2, 0.4]),
                                                        np.array([0.3, 0.5])]))

    # multiple points as numpy array
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f._at(np.array([[0.2, 0.4],
                                                                [0.3, 0.5]])))


def test_dont_raise():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    f = Function(V).interpolate(2.0 * x[0])

    # raise exception without dont_raise
    with pytest.raises(PointNotInDomainError):
        f._at(-1)

    # dont_raise=True
    assert f._at(-1, dont_raise=True) is None

    actual = f._at([-1, 0, 0.5], dont_raise=True)
    assert actual[0] is None
    assert np.allclose([0.0, 1.0], actual[1:])


def test_dont_raise_vector():
    mesh = UnitIntervalMesh(1)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    x = SpatialCoordinate(mesh)
    f = Function(V).interpolate(as_vector((x[0], 2.0 * x[0])))

    # raise exception without dont_raise
    with pytest.raises(PointNotInDomainError):
        f._at(-1)

    # dont_raise=True
    assert f._at(-1, dont_raise=True) is None

    actual = f._at([-1, 1], dont_raise=True)
    assert actual[0] is None
    assert np.allclose([1.0, 2.0], actual[1])


def test_dont_raise_mixed():
    mesh = UnitSquareMesh(1, 1)
    V1 = FunctionSpace(mesh, "DG", 1)
    V2 = FunctionSpace(mesh, "RT", 2)
    V = V1 * V2
    f = Function(V)
    f1, f2 = f.subfunctions
    x = SpatialCoordinate(mesh)
    f1.interpolate(x[0] + 1.2*x[1])
    f2.project(as_vector((x[1], 0.8 + x[0])))

    # raise exception without dont_raise
    with pytest.raises(PointNotInDomainError):
        # Point has to be well outside the mesh since the tolerance is of order
        # cell size
        f._at([2.2, 0.5])

    # dont_raise=True
    assert f._at([2.2, 0.5], dont_raise=True) is None


@pytest.mark.parallel(nprocs=3)
def test_nascent_parallel_support():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 2)
    x = SpatialCoordinate(mesh)
    f = Function(V).interpolate((x[0] + 0.2)*x[1])

    assert np.allclose(0.0576, f._at([0.12, 0.18]))
    assert np.allclose(1.0266, f._at([0.98, 0.87]))
    assert np.allclose([0.2176, 0.2822], f._at([0.12, 0.68], [0.63, 0.34]))


def test_tolerance():
    mesh = UnitSquareMesh(1, 1)
    old_tol = mesh.tolerance
    f = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(SpatialCoordinate(mesh))
    assert np.allclose([0.2, 0.4], f._at(0.2, 0.4))
    # tolerance of mesh is not changed
    assert mesh.tolerance == old_tol
    # Outside mesh, but within tolerance
    assert np.allclose([-0.1, 0.4], f._at((-0.1, 0.4), tolerance=0.2))
    # tolerance of mesh is changed
    assert mesh.tolerance == 0.2
    # works if mesh tolerance is changed
    mesh.tolerance = 1e-11
    with pytest.raises(PointNotInDomainError):
        f._at((-0.1, 0.4))
    assert np.allclose([-1e-12, 0.4], f._at((-1e-12, 0.4)))


@pytest.fixture(scope="module")
def mesh_and_points():
    points = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    mesh = UnitSquareMesh(10, 10)
    evaluator = PointEvaluator(mesh, points)
    return mesh, evaluator


@pytest.mark.parallel([1, 3])
def test_point_evaluator_scalar(mesh_and_points):
    mesh, evaluator = mesh_and_points
    points = evaluator.points
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(x + y)

    # Test standard scalar function evaluation at points
    f_at_points = evaluator.evaluate(f)
    assert np.allclose(f_at_points, [0.2, 0.4, 0.6])

    # Test standard scalar function with missing points
    eval_missing = PointEvaluator(mesh, np.append(points, [[1.5, 1.5]], axis=0), missing_points_behaviour="ignore")
    f_at_points_missing = eval_missing.evaluate(f)
    assert np.isnan(f_at_points_missing[-1])


@pytest.mark.parallel([1, 3])
def test_point_evaluator_vector_tensor_mixed(mesh_and_points):
    mesh, evaluator = mesh_and_points
    V_vec = VectorFunctionSpace(mesh, "CG", 1)
    f_vec = Function(V_vec)
    x, y = SpatialCoordinate(mesh)
    f_vec.interpolate(as_vector([x, y]))
    f_vec_at_points = evaluator.evaluate(f_vec)
    vec_expected = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    assert np.allclose(f_vec_at_points, vec_expected)

    V_tensor = TensorFunctionSpace(mesh, "CG", 1, shape=(2, 3))
    f_tensor = Function(V_tensor)
    f_tensor.interpolate(as_matrix([[x, y, x*y], [y, x, x*y]]))
    f_tensor_at_points = evaluator.evaluate(f_tensor)
    tensor_expected = np.array([[[0.1, 0.1, 0.01], [0.1, 0.1, 0.01]],
                                [[0.2, 0.2, 0.04], [0.2, 0.2, 0.04]],
                                [[0.3, 0.3, 0.09], [0.3, 0.3, 0.09]]])
    assert np.allclose(f_tensor_at_points, tensor_expected)

    V_mixed = V_vec * V_tensor
    f_mixed = Function(V_mixed)
    f_vec, f_tensor = f_mixed.subfunctions
    f_vec.interpolate(as_vector([x, y]))
    f_tensor.interpolate(as_matrix([[x, y, x*y], [y, x, x*y]]))
    f_mixed_at_points = evaluator.evaluate(f_mixed)
    assert np.allclose(f_mixed_at_points[0], vec_expected)
    assert np.allclose(f_mixed_at_points[1], tensor_expected)


@pytest.mark.parallel(3)
def test_point_evaluator_nonredundant(mesh_and_points):
    mesh = mesh_and_points[0]
    if mesh.comm.rank == 0:
        points = [[0.1, 0.1]]
    elif mesh.comm.rank == 1:
        points = [[0.4, 0.4], [0.5, 0.5]]
    else:
        points = [[0.8, 0.8]]
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(x + y)
    evaluator = PointEvaluator(mesh, points, redundant=False)
    f_at_points = evaluator.evaluate(f)
    if mesh.comm.rank == 0:
        assert np.allclose(f_at_points, [0.2])
    elif mesh.comm.rank == 1:
        assert np.allclose(f_at_points, [0.8, 1.0])
    else:
        assert np.allclose(f_at_points, [1.6])


def test_point_evaluator_moving_mesh(mesh_and_points):
    mesh, evaluator = mesh_and_points
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(x + y)

    mesh.coordinates.dat.data[:, 0] += 1.0

    with pytest.raises(VertexOnlyMeshMissingPointsError):
        # The VOM is reimmersed, but the points
        # are now outside of the mesh.
        f_at_points = evaluator.evaluate(f)

    mesh.coordinates.dat.data[:, 0] -= 1.0
    f_at_points = evaluator.evaluate(f)
    assert np.allclose(f_at_points, [0.2, 0.4, 0.6])


def test_point_evaluator_tolerance():
    mesh = UnitSquareMesh(1, 1)
    old_tol = mesh.tolerance
    f = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(SpatialCoordinate(mesh))
    ev = PointEvaluator(mesh, [[0.2, 0.4]])
    assert np.allclose([0.2, 0.4], ev.evaluate(f))
    # tolerance of mesh is not changed
    assert mesh.tolerance == old_tol
    # Outside mesh, but within tolerance
    ev = PointEvaluator(mesh, [[-0.1, 0.4]], tolerance=0.2)
    assert np.allclose([-0.1, 0.4], ev.evaluate(f))
    # tolerance of mesh is changed
    assert mesh.tolerance == 0.2
    # works if mesh tolerance is changed
    mesh.tolerance = 1e-11
    with pytest.raises(VertexOnlyMeshMissingPointsError):
        ev.evaluate(f)
    ev = PointEvaluator(mesh, [[-1e-12, 0.4]])
    assert np.allclose([-1e-12, 0.4], ev.evaluate(f))


def test_point_evaluator_inputs_1d():
    mesh = UnitIntervalMesh(1)
    f = mesh.coordinates

    # one point
    for input in [0.2, (0.2,), [0.2], np.array([0.2])]:
        e = PointEvaluator(mesh, input)
        assert np.allclose(0.2, e.evaluate(f))

    # multiple points as tuples/list
    for input in [
        (0.2, 0.3), ((0.2,), (0.3,)), ([0.2], [0.3]),
        (np.array(0.2), np.array(0.3)), (np.array([0.2]), np.array([0.3]))
    ]:
        e2 = PointEvaluator(mesh, input)
        assert np.allclose([[0.2, 0.3]], e2.evaluate(f))
        e3 = PointEvaluator(mesh, list(input))
        assert np.allclose([[0.2, 0.3]], e3.evaluate(f))

    # multiple points as numpy array
    for input in [np.array([0.2, 0.3]), np.array([[0.2], [0.3]])]:
        e = PointEvaluator(mesh, input)
        assert np.allclose([[0.2, 0.3]], e.evaluate(f))

    # test incorrect inputs
    for input in [[[0.2, 0.3]], ([0.2, 0.3], [0.4, 0.5]), np.array([[0.2, 0.3]])]:
        with pytest.raises(ValueError):
            PointEvaluator(mesh, input)


def test_point_evaluator_inputs_2d():
    mesh = UnitSquareMesh(1, 1)
    f = mesh.coordinates

    # one point
    for input in [(0.2, 0.4), [0.2, 0.4], [[0.2, 0.4]], np.array([0.2, 0.4])]:
        e = PointEvaluator(mesh, input)
        assert np.allclose([0.2, 0.4], e.evaluate(f))

    # multiple points as tuple
    for input in [
        ((0.2, 0.4), (0.3, 0.5)), ([0.2, 0.4], [0.3, 0.5]),
        (np.array([0.2, 0.4]), np.array([0.3, 0.5]))
    ]:
        e1 = PointEvaluator(mesh, input)
        assert np.allclose([[0.2, 0.4], [0.3, 0.5]], e1.evaluate(f))
        e2 = PointEvaluator(mesh, list(input))
        assert np.allclose([[0.2, 0.4], [0.3, 0.5]], e2.evaluate(f))

    # multiple points as numpy array
    e = PointEvaluator(mesh, np.array([[0.2, 0.4], [0.3, 0.5]]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], e.evaluate(f))

    # test incorrect inputs
    for input in [0.2, [0.2]]:
        with pytest.raises(ValueError):
            PointEvaluator(mesh, input)
