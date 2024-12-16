import pytest

from gem.gem import Variable, Zero, Conditional, \
    LogicalAnd, Index, Indexed, Product


def test_conditional_simplification():
    a = Variable("A", ())
    b = Variable("B", ())

    expr = Conditional(LogicalAnd(b, a), a, a)

    assert expr == a


def test_conditional_zero_folding():
    b = Variable("B", ())
    a = Variable("A", (3, ))
    i = Index()
    expr = Conditional(LogicalAnd(b, b),
                       Product(Indexed(a, (i, )),
                               Zero()),
                       Zero())

    assert expr == Zero()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
