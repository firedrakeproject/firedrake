from __future__ import absolute_import, print_function, division
from six.moves import range

import gem
import numpy
import pytest
import tsfc


parameters = tsfc.coffee.Bunch()
parameters.names = {}


def convert(expression, multiindex):
    assert not expression.free_indices
    element = gem.Indexed(expression, multiindex)
    element, = gem.optimise.remove_componenttensors((element,))
    return tsfc.coffee.expression(element, parameters).rank


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


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
