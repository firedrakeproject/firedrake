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
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    f = Function(V).interpolate(Expression(("x[0]", "2.0 * x[0]")))

    # raise exception without fill_value
    with pytest.raises(PointNotInDomainError):
        f.at(-1)

    # set fill_value
    assert np.allclose([3.0, 4.0], f.at(-1, fill_value=[3, 4]))

    # broadcast fill_value
    assert np.allclose([2.0, 2.0], f.at(-1, fill_value=2))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
