import numpy as np
import pytest

import pyop3 as op3


@pytest.fixture
def scalar_arrayref():
    base = np.arange(10)
    return op3.arrayref.ArrayReference(base, [0, 2, 4, 6, 8])


@pytest.fixture
def vector_arrayref():
    base = np.arange(20)
    return op3.arrayref.ArrayReference(base, [0, 1, 4, 5, 8, 9, 12, 13, 16, 17], block_shape=(2,))


def test_scalar_arrayref_sizes(scalar_arrayref):
    assert scalar_arrayref.size == 5
    assert scalar_arrayref.shape == (5,)
    assert scalar_arrayref.block_shape == ()


def test_vector_arrayref_sizes(vector_arrayref):
    assert vector_arrayref.size == 10
    assert vector_arrayref.shape == (5, 2)
    assert vector_arrayref.block_shape == (2,)


def test_scalar_arrayref_asarray(scalar_arrayref):
    assert np.equal(np.asarray(scalar_arrayref), np.arange(10)[::2]).all()


def test_vector_arrayref_asarray(vector_arrayref):
    assert np.equal(np.asarray(vector_arrayref), np.arange(20).reshape((10, 2))[::2]).all()


def test_scalar_arrayref_assign_ellipsis_scalar(scalar_arrayref):
    scalar_arrayref[...] = 666
    assert (scalar_arrayref.base == [666, 1, 666, 3, 666, 5, 666, 7, 666, 9]).all()


def test_scalar_arrayref_assign_single_scalar(scalar_arrayref):
    scalar_arrayref[2] = 666
    assert (scalar_arrayref.base == [0, 1, 2, 3, 666, 5, 6, 7, 8, 9]).all()


def test_vector_arrayref_assign_ellipsis_scalar(vector_arrayref):
    vector_arrayref[...] = 666
    expected = [666, 666, 2, 3, 666, 666, 6, 7, 666, 666, 10, 11, 666, 666, 14, 15, 666, 666, 18, 19]
    assert (vector_arrayref.base == expected).all()


def test_vector_arrayref_assign_single_scalar(vector_arrayref):
    vector_arrayref[2] = 666
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 666, 666, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert (vector_arrayref.base == expected).all()


def test_vector_arrayref_assign_ellipsis_block(vector_arrayref):
    vector_arrayref[...] = [666, 777]
    expected = [666, 777, 2, 3, 666, 777, 6, 7, 666, 777, 10, 11, 666, 777, 14, 15, 666, 777, 18, 19]
    assert (vector_arrayref.base == expected).all()


def test_vector_arrayref_assign_single_block(vector_arrayref):
    vector_arrayref[2] = [666, 777]
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 666, 777, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert (vector_arrayref.base == expected).all()
