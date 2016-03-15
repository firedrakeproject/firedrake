from tsfc import driver, impero_utils, scheduling
from tsfc.gem import Index, Indexed, IndexSum, Product, Variable


def test_loop_fusion():
    i = Index()
    j = Index()
    Ri = Indexed(Variable('R', (6,)), (i,))

    def make_expression(i, j):
        A = Variable('A', (6,))
        s = IndexSum(Indexed(A, (j,)), j)
        return Product(Indexed(A, (i,)), s)

    e1 = make_expression(i, j)
    e2 = make_expression(i, i)

    apply_ordering = driver.make_index_orderer((i, j))
    get_indices = lambda expr: apply_ordering(expr.free_indices)

    def gencode(expr):
        ops = scheduling.emit_operations([(Ri, expr)], get_indices)
        impero_c = impero_utils.process(ops, get_indices)
        return impero_c.tree

    assert len(gencode(e1).children) == len(gencode(e2).children)
