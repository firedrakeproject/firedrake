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


@pytest.mark.parametrize("element_name", ["Legendre", "IntegratedLegendre"])
@pytest.mark.parametrize("cellname,degree", [("triangle", 3), ("tetrahedron", 2)])
def test_duffy_scatter_and_contract(cellname, degree, element_name):
    """Route B of the simplex sum-factorization milestone 2 design:
    `finat.duffy.DuffyElement.basis_evaluation` (the `translate_argument`
    path) and `finat.duffy.DuffyElement.duffy_contraction` (the
    `translate_coefficient` path) must reproduce the standard dense FIAT
    tabulation, via the dof numbering FIAT already uses, from
    `duffy_evaluation`'s lattice-indexed, sum-factorized tabulation.
    `Legendre` (continuity=None) reads exactly one lattice point per dof
    (lattice-lexicographic order); `IntegratedLegendre` (continuity="C0")
    additionally exercises the `FIAT.expansions.C0_basis` recombination,
    where each dof combines a handful of lattice points.

    Verified via `gem.interpreter.evaluate` rather than a compiled loopy
    kernel: the GEM expressions built here (in particular
    `duffy_contraction`'s `gem.VariableIndex`-based index arithmetic)
    schedule fine once merged into a real PyOP2 wrapper kernel (as confirmed
    via `firedrake.assemble` on real forms), but are not guaranteed
    schedulable by loopy in isolation -- scheduling an isolated,
    unwrapped kernel is not a configuration real Firedrake usage ever
    exercises. The GEM interpreter checks the same numerical correctness
    without depending on loopy scheduling at all.
    """
    from FIAT.reference_element import UFCTetrahedron, UFCTriangle
    from finat.quadrature import make_quadrature
    from finat.spectral import Legendre, IntegratedLegendre
    from gem.gem import Index, Indexed, Variable
    from gem.interpreter import evaluate
    from gem.optimise import remove_componenttensors

    cell = {"triangle": UFCTriangle, "tetrahedron": UFCTetrahedron}[cellname]()
    element_cls = {"Legendre": Legendre, "IntegratedLegendre": IntegratedLegendre}[element_name]
    element = element_cls(cell, degree)
    ndof = element.space_dimension()

    quad_rule = make_quadrature(cell, 2 * degree, scheme="collapsed")
    point_set = quad_rule.point_set
    point_indices = point_set.indices
    point_shape = tuple(index.extent for index in point_indices)

    entity = (cell.get_dimension(), 0)
    dense_dict = element._element.tabulate(1, point_set.points)

    rng = numpy.random.default_rng(1)
    coefficients = rng.random(ndof)

    # translate_argument path: dispatched transparently through basis_evaluation
    scattered_dict = element.basis_evaluation(1, point_set, entity)
    for alpha, dense in dense_dict.items():
        dense = dense.reshape((ndof,) + point_shape)

        r = Index(extent=ndof)
        table, = remove_componenttensors([Indexed(scattered_dict[alpha], (r,))])
        u_out, = evaluate([table])
        assert u_out.fids == (r,) + point_indices
        assert numpy.allclose(u_out.arr, dense, rtol=1e-12, atol=1e-12)

    # translate_coefficient path: contraction against a coefficient vector,
    # dispatched through duffy_contraction
    c = Variable("c", (ndof,))
    for order in (0, 1):
        contracted_dict = element.duffy_contraction(order, point_set, entity, c, epsilon=0.0)
        for alpha, contracted in contracted_dict.items():
            dense = dense_dict[alpha].reshape((ndof,) + point_shape)
            value, = remove_componenttensors([Indexed(contracted, ())])
            v_out, = evaluate([value], bindings={c: coefficients})
            assert v_out.fids == point_indices
            v_ref = numpy.tensordot(coefficients, dense, axes=(0, 0))
            assert numpy.allclose(v_out.arr, v_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
