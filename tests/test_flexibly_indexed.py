from itertools import product

import numpy
import pytest

import gem
import tsfc
import tsfc.loopy


parameters = tsfc.loopy.LoopyContext()
parameters.names = {}


def convert(expression, multiindex):
    assert not expression.free_indices
    element = gem.Indexed(expression, multiindex)
    element, = gem.optimise.remove_componenttensors((element,))
    subscript = tsfc.loopy.expression(element, parameters)
    # Convert a pymbolic subscript expression to a rank tuple. For example
    # the subscript:
    #
    #     Subscript(Variable('A'), (Sum((3,)), Sum((5,))))
    #
    # will yield a rank of (3, 5).
    return sum((idx.children for idx in subscript.index), start=())


@pytest.fixture(scope='module')
def vector():
    return gem.Variable('u', (12,))


@pytest.fixture(scope='module')
def matrix():
    return gem.Variable('A', (10, 12))


def test_reshape(vector):
    expression = gem.reshape(vector, (3, 4))
    assert expression.shape == (3, 4)

    actual = [convert(expression, multiindex)
              for multiindex in numpy.ndindex(expression.shape)]

    assert [(i,) for i in range(12)] == actual


def test_view(matrix):
    expression = gem.view(matrix, slice(3, 8), slice(5, 12))
    assert expression.shape == (5, 7)

    actual = [convert(expression, multiindex)
              for multiindex in numpy.ndindex(expression.shape)]

    assert list(product(range(3, 8), range(5, 12))) == actual


def test_view_view(matrix):
    expression = gem.view(gem.view(matrix, slice(3, 8), slice(5, 12)),
                          slice(4), slice(3, 6))
    assert expression.shape == (4, 3)

    actual = [convert(expression, multiindex)
              for multiindex in numpy.ndindex(expression.shape)]

    assert list(product(range(3, 7), range(8, 11))) == actual


def test_view_reshape(vector):
    expression = gem.view(gem.reshape(vector, (3, 4)), slice(2), slice(1, 3))
    assert expression.shape == (2, 2)

    actual = [convert(expression, multiindex)
              for multiindex in numpy.ndindex(expression.shape)]

    assert [(1,), (2,), (5,), (6,)] == actual


def test_reshape_shape(vector):
    expression = gem.reshape(gem.view(vector, slice(5, 11)), (3, 2))
    assert expression.shape == (3, 2)

    actual = [convert(expression, multiindex)
              for multiindex in numpy.ndindex(expression.shape)]

    assert [(i,) for i in range(5, 11)] == actual


def test_reshape_reshape(vector):
    expression = gem.reshape(gem.reshape(vector, (4, 3)), (2, 2), (3,))
    assert expression.shape == (2, 2, 3)

    actual = [convert(expression, multiindex)
              for multiindex in numpy.ndindex(expression.shape)]

    assert [(i,) for i in range(12)] == actual


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
