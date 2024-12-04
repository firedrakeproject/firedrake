from functools import partial

import pytest

import gem
from gem.node import traversal
from gem.refactorise import ATOMIC, COMPOUND, OTHER, Monomial, collect_monomials


def test_refactorise():
    f = gem.Variable('f', (3,))
    u = gem.Variable('u', (3,))
    v = gem.Variable('v', ())

    i = gem.Index()
    f_i = gem.Indexed(f, (i,))
    u_i = gem.Indexed(u, (i,))

    def classify(atomics_set, expression):
        if expression in atomics_set:
            return ATOMIC

        for node in traversal([expression]):
            if node in atomics_set:
                return COMPOUND

        return OTHER
    classifier = partial(classify, {u_i, v})

    # \sum_i 5*(2*u_i + -1*v)*(u_i + v*f)
    expr = gem.IndexSum(
        gem.Product(
            gem.Literal(5),
            gem.Product(
                gem.Sum(gem.Product(gem.Literal(2), u_i),
                        gem.Product(gem.Literal(-1), v)),
                gem.Sum(u_i, gem.Product(v, f_i))
            )
        ),
        (i,)
    )

    expected = [
        Monomial((i,),
                 (u_i, u_i),
                 gem.Literal(10)),
        Monomial((i,),
                 (u_i, v),
                 gem.Product(gem.Literal(5),
                             gem.Sum(gem.Product(f_i, gem.Literal(2)),
                                     gem.Literal(-1)))),
        Monomial((),
                 (v, v),
                 gem.Product(gem.Literal(5),
                             gem.IndexSum(gem.Product(f_i, gem.Literal(-1)),
                                          (i,)))),
    ]

    actual, = collect_monomials([expr], classifier)
    assert expected == list(actual)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
