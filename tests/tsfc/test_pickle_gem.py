import pickle
import gem
import numpy
import pytest


@pytest.mark.parametrize('protocol', range(3))
def test_pickle_gem(protocol):
    f = gem.VariableIndex(gem.Indexed(gem.Variable('facet', (2,), dtype=gem.uint_type), (1,)))
    q = gem.Index()
    r = gem.Index()
    _1 = gem.Indexed(gem.Literal(numpy.random.rand(3, 6, 8)), (f, q, r))
    _2 = gem.Indexed(gem.view(gem.Variable('w', (None, None)), slice(8), slice(1)), (r, 0))
    expr = gem.ComponentTensor(gem.IndexSum(gem.Product(_1, _2), (r,)), (q,))

    unpickled = pickle.loads(pickle.dumps(expr, protocol))
    assert repr(expr) == repr(unpickled)


@pytest.mark.parametrize('protocol', range(3))
def test_pickle_jagged_index(protocol):
    p = gem.Index(name='p', extent=4)
    q = gem.JaggedIndex(name='q', extent=4, parents=(p,))
    expr = gem.IndexSum(gem.Indexed(gem.Variable('A', (4, 4)), (p, q)), (p, q))

    unpickled = pickle.loads(pickle.dumps(expr, protocol))
    assert repr(expr) == repr(unpickled)
    up, uq = unpickled.multiindex
    assert isinstance(uq, gem.JaggedIndex)
    assert uq.extent == 4
    assert uq.parents == (up,)


@pytest.mark.parametrize('protocol', range(3))
def test_listtensor(protocol):
    expr = gem.ListTensor([gem.Variable('x', ()), gem.Zero()])

    unpickled = pickle.loads(pickle.dumps(expr, protocol))
    assert expr == unpickled


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
