import pytest
import gem


def test_expressions():
    x = gem.Variable("x", (3, 4))
    y = gem.Variable("y", (4, ))
    i, j = gem.indices(2)

    xij = x[i, j]
    yj = y[j]

    assert xij == gem.Indexed(x, (i, j))
    assert yj == gem.Indexed(y, (j, ))

    assert xij + yj == gem.Sum(xij, yj)
    assert xij * yj == gem.Product(xij, yj)
    assert xij - yj == gem.Sum(xij, gem.Product(gem.Literal(-1), yj))
    assert xij / yj == gem.Division(xij, yj)

    assert xij + 1 == gem.Sum(xij, gem.Literal(1))
    assert 1 + xij == gem.Sum(gem.Literal(1), xij)

    assert (xij + y).shape == (4, )

    assert (x @ y).shape == (3, )

    assert x.T.shape == (4, 3)

    with pytest.raises(ValueError):
        xij.T @ y

    with pytest.raises(ValueError):
        xij + "foo"


def test_as_gem():
    with pytest.raises(ValueError):
        gem.as_gem([1, 2])

    assert gem.as_gem(1) == gem.Literal(1)
