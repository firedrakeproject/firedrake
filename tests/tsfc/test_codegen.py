import numpy
import pytest

import gem
from finat.physically_mapped import MappedTabulation
from gem import impero_utils
from gem.flop_count import count_flops
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


def sparse_map_flops(lengths, ncolumns=4, npoints=2):
    """Count the flops of applying a sparse basis map.

    Parameters
    ----------
    lengths : list of int
        Number of nonzeros in each row of the map.
    ncolumns : int
        Number of reference basis functions.
    npoints : int
        Number of points the reference tabulation holds.

    Returns
    -------
    int
        Flops the lowered tabulation costs.
    """
    rows = []
    for row, length in enumerate(lengths):
        entries = [gem.Zero()] * ncolumns
        for column in range(length):
            entries[column] = gem.Variable(f"c_{row}_{column}", ())
        rows.append(entries)
    M = gem.ListTensor(numpy.asarray(rows, dtype=object))
    table = gem.Literal(numpy.ones((ncolumns, npoints)))

    mapped = MappedTabulation(M, {None: table})[None]
    i, j = gem.indices(2)
    expr, = impero_utils.preprocess_gem([Indexed(mapped, (i, j))])
    result = Indexed(Variable("A", (len(lengths), npoints)), (i, j))
    return count_flops(impero_utils.compile_gem([(result, expr)], (i, j)))


@pytest.mark.parametrize("lengths", [[3, 1, 1], [1, 3, 1], [1, 1, 3]])
def test_sparse_basis_map_is_one_rectangle(lengths):
    """Apply a sparse basis map as one rectangular contraction.

    Every row contracts over the same number of entries. The cost follows
    the longest row, and not the arrangement of the nonzeros.
    """
    assert sparse_map_flops(lengths) == sparse_map_flops([3, 3, 3])
    assert sparse_map_flops([2, 2, 2]) < sparse_map_flops([3, 3, 3])


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
