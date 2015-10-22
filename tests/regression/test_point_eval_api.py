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


def test_fill_value():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V).interpolate(Expression("2.0 * x[0]"))

    # raise exception without fill_value
    with pytest.raises(PointNotInDomainError):
        f.at(-1)

    # set fill_value
    assert np.allclose(2.0, f.at(-1, fill_value=2))
    assert np.allclose([2.0, 0.0, 1.0], f.at([-1, 0, 0.5], fill_value=2))


def test_fill_value_vector():
    mesh = UnitIntervalMesh(1)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    f = Function(V).interpolate(Expression(("x[0]", "2.0 * x[0]")))

    # raise exception without fill_value
    with pytest.raises(PointNotInDomainError):
        f.at(-1)

    # set fill_value
    assert np.allclose([3.0, 4.0], f.at(-1, fill_value=[3, 4]))
    assert np.allclose([[3.0, 4.0], [1.0, 2.0]], f.at([-1, 1], fill_value=[3, 4]))

    # broadcast fill_value
    assert np.allclose([2.0, 2.0], f.at(-1, fill_value=2))
    assert np.allclose([[2.0, 2.0], [1.0, 2.0]], f.at([-1, 1], fill_value=2))


def test_fill_value_mixed():
    mesh = UnitSquareMesh(1, 1)
    V1 = FunctionSpace(mesh, "DG", 1)
    V2 = FunctionSpace(mesh, "RT", 2)
    V = V1 * V2
    f = Function(V)
    f1, f2 = f.split()
    f1.interpolate(Expression("x[0] + 1.2*x[1]"))
    f2.project(Expression(("x[1]", "0.8 + x[0]")))

    # raise exception without fill_value
    with pytest.raises(PointNotInDomainError):
        f.at([1.2, 0.5])

    assert '(nan, array([ nan,  nan]))' == str(f.at([1.2, 0.5], fill_value=np.nan))
    assert '(1.0, array([ 2.,  2.]))' == str(f.at([1.2, 0.5], fill_value=(1, 2)))
    assert '(1.0, array([ 2.1,  2.2]))' == str(f.at([1.2, 0.5], fill_value=(1, [2.1, 2.2])))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
