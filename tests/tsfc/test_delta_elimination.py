import pytest

from gem.gem import Delta, Identity, Index, Indexed, one
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


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
