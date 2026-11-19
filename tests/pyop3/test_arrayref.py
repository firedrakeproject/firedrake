import numpy as np
import pytest

import pyop3 as op3


@pytest.fixture
def scalar_arrayref():
    base = np.arange(10)
    return op3.arrayref.ArrayReference(base, np.asarray([0, 2, 4, 6, 8], dtype=int))


@pytest.fixture
def vector_arrayref():
    base = np.arange(10)
    return op3.arrayref.ArrayReference(
        base, np.asarray([[0, 1], [4, 5], [8, 9]], dtype=int)
    )


def test_scalar_arrayref_sizes(scalar_arrayref):
    assert scalar_arrayref.size == 5
    assert scalar_arrayref.shape == (5,)
    assert scalar_arrayref.ndim == 1


def test_vector_arrayref_sizes(vector_arrayref):
    assert vector_arrayref.size == 6
    assert vector_arrayref.shape == (3, 2)
    assert vector_arrayref.ndim == 2


def test_scalar_arrayref_asarray(scalar_arrayref):
    assert np.equal(np.asarray(scalar_arrayref), np.arange(10)[::2]).all()


def test_vector_arrayref_asarray(vector_arrayref):
    assert np.equal(np.asarray(vector_arrayref), np.arange(10).reshape((5, 2))[::2]).all()


@pytest.mark.parametrize(
    "indices,value,expected",
    [
        (Ellipsis, 666, [666, 1, 666, 3, 666, 5, 666, 7, 666, 9]),
        (2, 666, [0, 1, 2, 3, 666, 5, 6, 7, 8, 9]),
    ],
    ids=["ellipsis", "scalar"],
)
def test_scalar_arrayref_setitem(scalar_arrayref, indices, value, expected):
    scalar_arrayref[indices] = value
    assert (scalar_arrayref.base == expected).all()


@pytest.mark.parametrize(
    "indices,value,expected",
    [
        (Ellipsis, 666, [666, 666, 2, 3, 666, 666, 6, 7, 666, 666]),
        (2, 666, [0, 1, 2, 3, 4, 5, 6, 7, 666, 666]),
        (Ellipsis, [666, 777], [666, 777, 2, 3, 666, 777, 6, 7, 666, 777]),
        (2, [666, 777], [0, 1, 2, 3, 4, 5, 6, 7, 666, 777]),
        ((Ellipsis, 1), 666, [0, 666, 2, 3, 4, 666, 6, 7, 8, 666]),
        ((1, 1), 666, [0, 1, 2, 3, 4, 666, 6, 7, 8, 9]),
    ],
    ids=[
        "ellipsis-scalar",
        "scalar-scalar",
        "ellipsis-block",
        "scalar-block",
        "(ellipsis,scalar)",
        "(scalar,scalar)",
    ],
)
def test_vector_arrayref_setitem(vector_arrayref, indices, value, expected):
    vector_arrayref[indices] = value
    assert (vector_arrayref.base == expected).all()
