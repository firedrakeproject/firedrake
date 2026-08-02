import numpy
import pytest

from gem import gem, impero_utils
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


def test_jagged_index_codegen(monkeypatch):
    import islpy as isl
    import loopy as lp
    import tsfc.loopy

    # Execute the generated code so we check the numbers, not just the loop bounds
    monkeypatch.setattr(tsfc.loopy, "target", lp.ExecutableCTarget())

    n = 4
    extent = n + 1
    npts = 3
    ndof = (n + 1) * (n + 2) // 2

    rng = numpy.random.default_rng(7)
    # Table zero-padded outside the simplex lattice p + q > n, and a
    # clamped Morton index table for the coefficient gather
    B = rng.random((extent, extent, npts))
    morton = numpy.zeros((extent, extent), dtype=gem.uint_type)
    for p_, q_ in numpy.ndindex(morton.shape):
        if p_ + q_ > n:
            B[p_, q_] = 0.0
        else:
            morton[p_, q_] = (p_ + q_) * (p_ + q_ + 1) // 2 + q_
    c = rng.random(ndof)

    i = Index(name="i", extent=npts)
    p = Index(name="p", extent=extent)
    q = gem.JaggedIndex(name="q", extent=extent, parents=(p,))

    dof = gem.VariableIndex(Indexed(gem.Literal(morton, dtype=gem.uint_type), (p, q)))
    integrand = Product(Indexed(Variable("c", (ndof,)), (dof,)),
                        Indexed(gem.Literal(B), (p, q, i)))
    expr = IndexSum(integrand, (p, q))

    u = Variable("u", (npts,))
    impero_c = impero_utils.compile_gem([(Indexed(u, (i,)), expr)], (i, p, q))
    args = [lp.GlobalArg("u", dtype=numpy.float64, shape=(npts,)),
            lp.GlobalArg("c", dtype=numpy.float64, shape=(ndof,))]
    knl, _ = tsfc.loopy.generate(impero_c, args, numpy.float64)

    # The jagged loop must have a domain parametrized by its parent iname
    assert any(dom.get_var_names(isl.dim_type.param)
               for dom in knl.default_entrypoint.domains)

    u_out = numpy.zeros(npts)
    knl(c=c, u=u_out)
    u_ref = numpy.tensordot(c[morton], B, axes=((0, 1), (0, 1)))
    assert numpy.allclose(u_out, u_ref, rtol=1e-14)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
