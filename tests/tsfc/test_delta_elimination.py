import pytest

from gem.gem import Delta, Identity, Index, Indexed, Variable, one
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


def test_delta_elimination_protected():
    i = Index()
    k = Index()
    A = Variable("A", (3,))
    factors = [Delta(i, k), Indexed(A, (i,))]

    # Cancelling the Delta gathers A along the protected index instead.
    cancelled, _ = delta_elimination((i,), factors)
    assert cancelled == []

    # Protecting it keeps the Delta, so A is still gathered along i.
    cancelled, kept = delta_elimination((i,), factors, protected={k})
    assert cancelled == [i]
    assert kept == factors


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
