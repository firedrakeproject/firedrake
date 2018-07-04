import pytest
from gem import gem


def test_expressions():
    x = gem.Variable("x", (3, 4))
    y = gem.Variable("y", (4, ))
    i = gem.Index()
    j = gem.Index()

    xij = x[i, j]
    yj = y[j]

    assert xij == gem.Indexed(x, (i, j))
    assert yj == gem.Indexed(y, (j, ))

    assert xij + yj == gem.Sum(xij, yj)
    assert xij * yj == gem.Product(xij, yj)
    assert xij - yj == gem.Sum(xij, gem.Product(gem.Literal(-1), yj))
    assert xij / yj == gem.Division(xij, yj)

    with pytest.raises(AssertionError):
        xij + 1

    with pytest.raises(AssertionError):
        xij + y
