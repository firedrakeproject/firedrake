from os.path import abspath, dirname
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


def test_1d_args():
    mesh = UnitIntervalMesh(1)
    f = mesh.coordinates

    # one point
    assert np.allclose(0.2, f.at(0.2))
    assert np.allclose(0.2, f.at((0.2,)))
    assert np.allclose(0.2, f.at([0.2]))
    assert np.allclose(0.2, f.at(np.array([0.2])))

    # multiple points as arguments
    assert np.allclose([[0.2], [0.3]], f.at(0.2, 0.3))
    assert np.allclose([[0.2], [0.3]], f.at((0.2,), (0.3,)))
    assert np.allclose([[0.2], [0.3]], f.at([0.2], [0.3]))
    assert np.allclose([[0.2], [0.3]], f.at(np.array(0.2), np.array(0.3)))
    assert np.allclose([[0.2], [0.3]], f.at(np.array([0.2]), np.array([0.3])))

    # multiple points as tuple
    assert np.allclose([[0.2], [0.3]], f.at((0.2, 0.3)))
    assert np.allclose([[0.2], [0.3]], f.at(((0.2,), (0.3,))))
    assert np.allclose([[0.2], [0.3]], f.at(([0.2], [0.3])))
    assert np.allclose([[0.2], [0.3]], f.at((np.array(0.2), np.array(0.3))))
    assert np.allclose([[0.2], [0.3]], f.at((np.array([0.2]), np.array([0.3]))))

    # multiple points as list
    assert np.allclose([[0.2], [0.3]], f.at([0.2, 0.3]))
    assert np.allclose([[0.2], [0.3]], f.at([(0.2,), (0.3,)]))
    assert np.allclose([[0.2], [0.3]], f.at([[0.2], [0.3]]))
    assert np.allclose([[0.2], [0.3]], f.at([np.array(0.2), np.array(0.3)]))
    assert np.allclose([[0.2], [0.3]], f.at([np.array([0.2]), np.array([0.3])]))

    # multiple points as numpy array
    assert np.allclose([[0.2], [0.3]], f.at(np.array([0.2, 0.3])))
    assert np.allclose([[0.2], [0.3]], f.at(np.array([[0.2], [0.3]])))


def test_2d_args():
    mesh = UnitSquareMesh(1, 1)
    f = mesh.coordinates

    # one point
    assert np.allclose([0.2, 0.4], f.at(0.2, 0.4))
    assert np.allclose([0.2, 0.4], f.at((0.2, 0.4)))
    assert np.allclose([0.2, 0.4], f.at([0.2, 0.4]))
    assert np.allclose([0.2, 0.4], f.at(np.array([0.2, 0.4])))

    # multiple points as arguments
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at((0.2, 0.4), (0.3, 0.5)))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at([0.2, 0.4], [0.3, 0.5]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at(np.array([0.2, 0.4]),
                                                      np.array([0.3, 0.5])))

    # multiple points as tuple
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at(((0.2, 0.4), (0.3, 0.5))))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at(([0.2, 0.4], [0.3, 0.5])))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at((np.array([0.2, 0.4]),
                                                       np.array([0.3, 0.5]))))

    # multiple points as list
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at([(0.2, 0.4), (0.3, 0.5)]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at([[0.2, 0.4], [0.3, 0.5]]))
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at([np.array([0.2, 0.4]),
                                                       np.array([0.3, 0.5])]))

    # multiple points as numpy array
    assert np.allclose([[0.2, 0.4], [0.3, 0.5]], f.at(np.array([[0.2, 0.4],
                                                                [0.3, 0.5]])))


def test_dont_raise():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    f = Function(V).interpolate(2.0 * x[0])

    # raise exception without dont_raise
    with pytest.raises(PointNotInDomainError):
        f.at(-1)

    # dont_raise=True
    assert f.at(-1, dont_raise=True) is None

    actual = f.at([-1, 0, 0.5], dont_raise=True)
    assert actual[0] is None
    assert np.allclose([0.0, 1.0], actual[1:])


def test_dont_raise_vector():
    mesh = UnitIntervalMesh(1)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    x = SpatialCoordinate(mesh)
    f = Function(V).interpolate(as_vector((x[0], 2.0 * x[0])))

    # raise exception without dont_raise
    with pytest.raises(PointNotInDomainError):
        f.at(-1)

    # dont_raise=True
    assert f.at(-1, dont_raise=True) is None

    actual = f.at([-1, 1], dont_raise=True)
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
        f.at([2.2, 0.5])

    # dont_raise=True
    assert f.at([2.2, 0.5], dont_raise=True) is None


@pytest.mark.parallel(nprocs=3)
def test_nascent_parallel_support():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 2)
    x = SpatialCoordinate(mesh)
    f = Function(V).interpolate((x[0] + 0.2)*x[1])

    assert np.allclose(0.0576, f.at([0.12, 0.18]))
    assert np.allclose(1.0266, f.at([0.98, 0.87]))
    assert np.allclose([0.2176, 0.2822], f.at([0.12, 0.68], [0.63, 0.34]))


def test_tolerance():
    mesh = UnitSquareMesh(1, 1)
    old_tol = mesh.tolerance
    f = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(SpatialCoordinate(mesh))
    assert np.allclose([0.2, 0.4], f.at(0.2, 0.4))
    # tolerance of mesh is not changed
    assert mesh.tolerance == old_tol
    # Outside mesh, but within tolerance
    assert np.allclose([-0.1, 0.4], f.at((-0.1, 0.4), tolerance=0.2))
    # tolerance of mesh is changed
    assert mesh.tolerance == 0.2
    # works if mesh tolerance is changed
    mesh.tolerance = 1e-11
    with pytest.raises(PointNotInDomainError):
        f.at((-0.1, 0.4))
    assert np.allclose([-1e-12, 0.4], f.at((-1e-12, 0.4)))
