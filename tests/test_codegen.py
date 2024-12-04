import pytest

from gem import impero_utils
from gem.gem import Index, Indexed, IndexSum, Product, Variable


def test_loop_fusion():
    i = Index()
    j = Index()
    Ri = Indexed(Variable('R', (6,)), (i,))

    def make_expression(i, j):
        A = Variable('A', (6,))
        s = IndexSum(Indexed(A, (j,)), (j,))
        return Product(Indexed(A, (i,)), s)

    e1 = make_expression(i, j)
    e2 = make_expression(i, i)

    def gencode(expr):
        impero_c = impero_utils.compile_gem([(Ri, expr)], (i, j))
        return impero_c.tree

    assert len(gencode(e1).children) == len(gencode(e2).children)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
