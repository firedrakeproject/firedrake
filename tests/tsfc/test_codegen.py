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


@pytest.mark.parametrize("cellname,degree", [("triangle", 3), ("tetrahedron", 2)])
def test_duffy_scatter_and_contract(monkeypatch, cellname, degree):
    """Route B of the simplex sum-factorization milestone 2 design:
    ``tsfc.fem._scatter_to_dof_index`` (the `translate_argument` path) and
    ``tsfc.fem._contract_dof_index`` (the `translate_coefficient` path)
    must reproduce the standard dense FIAT tabulation, via the Morton
    dof numbering FIAT already uses, from
    `finat.spectral.Legendre.duffy_evaluation`'s lattice-indexed,
    sum-factorized tabulation.
    """
    import loopy as lp
    import tsfc.loopy
    from FIAT.reference_element import UFCTetrahedron, UFCTriangle
    from finat.quadrature import make_quadrature
    from finat.spectral import Legendre
    from gem import impero_utils
    from gem.gem import Index, Indexed, Variable
    from gem.optimise import remove_componenttensors
    from tsfc.fem import _contract_dof_index, _scatter_to_dof_index

    # Execute the generated code so we check the numbers, not just the loop bounds
    monkeypatch.setattr(tsfc.loopy, "target", lp.ExecutableCTarget())

    cell = {"triangle": UFCTriangle, "tetrahedron": UFCTetrahedron}[cellname]()
    element = Legendre(cell, degree)
    ndof = element.space_dimension()

    quad_rule = make_quadrature(cell, 2 * degree, scheme="collapsed")
    point_set = quad_rule.point_set
    point_indices = point_set.indices
    point_shape = tuple(index.extent for index in point_indices)

    entity = (cell.get_dimension(), 0)
    multiindex, duffy_dict = element.duffy_evaluation(1, point_set, entity)
    dense_dict = element._element.tabulate(1, point_set.points)

    rng = numpy.random.default_rng(1)
    coefficients = rng.random(ndof)

    for alpha, table_expr in duffy_dict.items():
        dense = dense_dict[alpha].reshape((ndof,) + point_shape)

        # translate_argument path: flat-dof-indexed dense table
        scattered = _scatter_to_dof_index(multiindex, {alpha: table_expr}, element)[alpha]
        r = Index(extent=ndof)
        table, = remove_componenttensors([Indexed(scattered, (r,))])
        u = Variable("u", (ndof,) + point_shape)
        impero_c = impero_utils.compile_gem(
            [(Indexed(u, (r,) + point_indices), table)], (r,) + point_indices)
        args = [lp.GlobalArg("u", dtype=numpy.float64, shape=(ndof,) + point_shape)]
        knl, _ = tsfc.loopy.generate(impero_c, args, numpy.float64)
        u_out = numpy.zeros((ndof,) + point_shape)
        knl(u=u_out)
        assert numpy.allclose(u_out, dense, rtol=1e-12, atol=1e-12)

        # translate_coefficient path: contraction against a coefficient vector
        c = Variable("c", (ndof,))
        contracted = _contract_dof_index(multiindex, {alpha: table_expr}, element, c)[alpha]
        value, = remove_componenttensors([Indexed(contracted, ())])
        v = Variable("v", point_shape)
        impero_c = impero_utils.compile_gem(
            [(Indexed(v, point_indices), value)], point_indices)
        args = [lp.GlobalArg("v", dtype=numpy.float64, shape=point_shape),
                lp.GlobalArg("c", dtype=numpy.float64, shape=(ndof,))]
        knl, _ = tsfc.loopy.generate(impero_c, args, numpy.float64)
        v_out = numpy.zeros(point_shape)
        knl(v=v_out, c=coefficients)
        v_ref = numpy.tensordot(coefficients, dense, axes=(0, 0))
        assert numpy.allclose(v_out, v_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
