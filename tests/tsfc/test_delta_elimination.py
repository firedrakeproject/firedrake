import numpy
import pytest

from gem.gem import (Delta, Identity, Index, Indexed, Literal, Variable,
                     VariableIndex, one, uint_type)
from gem.optimise import delta_elimination, remove_componenttensors


def test_delta_elimination():
    i = Index()
    j = Index()
    k = Index()
    I = Identity(3)

    sum_indices = (i, j)
    factors = [Delta(i, j), Delta(i, k), Indexed(I, (j, k))]

    sum_indices, factors = delta_elimination(sum_indices, factors)
    factors = remove_componenttensors(factors)

    assert sum_indices == []
    assert factors == [one, one, Indexed(I, (k, k))]


def test_delta_elimination_indirect_only():
    i = Index()
    k = Index()
    A = Variable("A", (3,))
    columns = Literal(numpy.array([2, 0, 1]), dtype=uint_type)
    indirect = VariableIndex(Indexed(columns, (k,)))

    # An indirect Delta cancels either way round: no later pass can lower one.
    factors = [Delta(indirect, i), Indexed(A, (i,))]
    cancelled, gathered = delta_elimination((i,), factors, indirect_only=True)
    assert cancelled == []
    assert remove_componenttensors(gathered) == [one, Indexed(A, (indirect,))]

    # A Delta between two plain indices is left to monomial collection.
    factors = [Delta(i, k), Indexed(A, (i,))]
    cancelled, kept = delta_elimination((i,), factors, indirect_only=True)
    assert cancelled == [i]
    assert kept == factors

    assert delta_elimination((i,), factors)[0] == []


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
