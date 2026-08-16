import numpy
import pytest

import gem
import gem.coffee
import tsfc.spectral
from gem.gem import one
from gem.contraction import estimate_cost
from gem.refactorise import MonomialSum
from ufl import (Coefficient, Mesh, FunctionSpace, TestFunction, TrialFunction,
                 TensorProductCell, dx, action, interval, triangle,
                 quadrilateral, curl, dot, div, grad, inner)
from finat.ufl import (FiniteElement, VectorElement, EnrichedElement,
                       TensorProductElement, HCurlElement, HDivElement)

from tsfc import compile_form
from tsfc.spectral import _plans


def helmholtz(cell, degree):
    m = Mesh(VectorElement('CG', cell, 1))
    V = FunctionSpace(m, FiniteElement('CG', cell, degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    return (u*v + dot(grad(u), grad(v)))*dx


def split_mixed_poisson(cell, degree):
    m = Mesh(VectorElement('CG', cell, 1))
    if cell.cellname in ['interval * interval', 'quadrilateral']:
        hdiv_element = FiniteElement('RTCF', cell, degree)
    elif cell.cellname == 'triangle * interval':
        U0 = FiniteElement('RT', triangle, degree)
        U1 = FiniteElement('DG', triangle, degree - 1)
        V0 = FiniteElement('CG', interval, degree)
        V1 = FiniteElement('DG', interval, degree - 1)
        Wa = HDivElement(TensorProductElement(U0, V1))
        Wb = HDivElement(TensorProductElement(U1, V0))
        hdiv_element = EnrichedElement(Wa, Wb)
    elif cell.cellname == 'quadrilateral * interval':
        hdiv_element = FiniteElement('NCF', cell, degree)
    RT = FunctionSpace(m, hdiv_element)
    DG = FunctionSpace(m, FiniteElement('DQ', cell, degree - 1))
    sigma = TrialFunction(RT)
    u = TrialFunction(DG)
    tau = TestFunction(RT)
    v = TestFunction(DG)
    return [dot(sigma, tau) * dx, div(tau) * u * dx, div(sigma) * v * dx]


def split_vector_laplace(cell, degree):
    m = Mesh(VectorElement('CG', cell, 1))
    if cell.cellname in ['interval * interval', 'quadrilateral']:
        hcurl_element = FiniteElement('RTCE', cell, degree)
    elif cell.cellname == 'triangle * interval':
        U0 = FiniteElement('RT', triangle, degree)
        U1 = FiniteElement('CG', triangle, degree)
        V0 = FiniteElement('CG', interval, degree)
        V1 = FiniteElement('DG', interval, degree - 1)
        Wa = HCurlElement(TensorProductElement(U0, V0))
        Wb = HCurlElement(TensorProductElement(U1, V1))
        hcurl_element = EnrichedElement(Wa, Wb)
    elif cell.cellname == 'quadrilateral * interval':
        hcurl_element = FiniteElement('NCE', cell, degree)
    RT = FunctionSpace(m, hcurl_element)
    CG = FunctionSpace(m, FiniteElement('Q', cell, degree))
    sigma = TrialFunction(CG)
    u = TrialFunction(RT)
    tau = TestFunction(CG)
    v = TestFunction(RT)
    return [dot(u, grad(tau))*dx, dot(grad(sigma), v)*dx, dot(curl(u), curl(v))*dx]


def count_flops(form):
    kernel, = compile_form(form, parameters=dict(mode='spectral'))
    flops = kernel.flop_count
    return flops


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 5),
                          (TensorProductCell(interval, interval), 5),
                          (TensorProductCell(triangle, interval), 7),
                          (TensorProductCell(quadrilateral, interval), 7)])
def test_lhs(cell, order):
    degrees = list(range(3, 8))
    if cell == TensorProductCell(triangle, interval):
        degrees = list(range(3, 6))
    flops = [count_flops(helmholtz(cell, degree))
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 3),
                          (TensorProductCell(interval, interval), 3),
                          (TensorProductCell(triangle, interval), 5),
                          (TensorProductCell(quadrilateral, interval), 4)])
def test_rhs(cell, order):
    degrees = list(range(3, 8))
    if cell == TensorProductCell(triangle, interval):
        degrees = list(range(3, 6))
    flops = [count_flops(action(helmholtz(cell, degree)))
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 5),
                          (TensorProductCell(interval, interval), 5),
                          (TensorProductCell(triangle, interval), 7),
                          (TensorProductCell(quadrilateral, interval), 7)
                          ])
def test_mixed_poisson(cell, order):
    degrees = numpy.arange(3, 8)
    if cell == TensorProductCell(triangle, interval):
        degrees = numpy.arange(3, 6)
    flops = [[count_flops(form)
              for form in split_mixed_poisson(cell, int(degree))]
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops).T) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 3),
                          (TensorProductCell(interval, interval), 3),
                          (TensorProductCell(triangle, interval), 5),
                          (TensorProductCell(quadrilateral, interval), 4)
                          ])
def test_mixed_poisson_action(cell, order):
    degrees = numpy.arange(3, 8)
    if cell == TensorProductCell(triangle, interval):
        degrees = numpy.arange(3, 6)
    flops = [[count_flops(action(form))
              for form in split_mixed_poisson(cell, int(degree))]
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops).T) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 5),
                          (TensorProductCell(interval, interval), 5),
                          (TensorProductCell(triangle, interval), 7),
                          (TensorProductCell(quadrilateral, interval), 7)
                          ])
def test_vector_laplace(cell, order):
    degrees = numpy.arange(3, 8)
    if cell == TensorProductCell(triangle, interval):
        degrees = numpy.arange(3, 6)
    flops = [[count_flops(form)
              for form in split_vector_laplace(cell, int(degree))]
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops).T) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 3),
                          (TensorProductCell(interval, interval), 3),
                          (TensorProductCell(triangle, interval), 5),
                          (TensorProductCell(quadrilateral, interval), 4)
                          ])
def test_vector_laplace_action(cell, order):
    degrees = numpy.arange(3, 8)
    if cell == TensorProductCell(triangle, interval):
        degrees = numpy.arange(3, 6)
    flops = [[count_flops(action(form))
              for form in split_vector_laplace(cell, int(degree))]
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops).T) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


def test_shared_physically_mapped_tabulation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Share a mapped tabulation between both argument axes.

    Parameters
    ----------
    monkeypatch
        Pytest fixture used to disable the sharing pass for comparison.

    Notes
    -----
    Johnson--Mercier has six mapped basis outputs in two dimensions.  The
    seventh writable vector holds geometry data, and the eighth holds the
    coefficients of the basis transformation.  Algebra shared while
    constructing those outputs belongs inside their common basis-row loop
    and must therefore remain scalar.
    """
    mesh = Mesh(VectorElement("CG", triangle, 1))
    element = FiniteElement("Johnson-Mercier", triangle, 1)
    space = FunctionSpace(mesh, element)
    u = TrialFunction(space)
    v = TestFunction(space)
    form = (inner(u, v) + inner(div(u), div(v))) * dx

    optimized, = compile_form(form, parameters={"mode": "spectral"})
    monkeypatch.setattr(
        gem.coffee, "_share_linear_maps",
        lambda monomial_sum, indices: monomial_sum)
    baseline, = compile_form(form, parameters={"mode": "spectral"})

    optimized_shapes = [
        temporary.shape for temporary in
        optimized.ast.default_entrypoint.temporary_variables.values()
        if not (temporary.read_only
                and temporary.initializer is not None)]
    baseline_shapes = [
        temporary.shape for temporary in
        baseline.ast.default_entrypoint.temporary_variables.values()
        if not (temporary.read_only
                and temporary.initializer is not None)]

    assert optimized.flop_count < baseline.flop_count
    assert sum(not shape for shape in optimized_shapes) \
        < sum(not shape for shape in baseline_shapes)
    assert [shape for shape in optimized_shapes if shape] == [(15,)] * 8


def test_linear_map_representation_is_costed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not preserve a linear map when expansion costs fewer FLOPs.

    Parameters
    ----------
    monkeypatch
        Pytest fixture used to force the expanded representation for
        comparison.

    Notes
    -----
    A Bernstein tabulation is separable, so preserving every one-axis sum
    produces a compact loop nest but can hide profitable scalar
    factorization.  Plan selection must compare that representation with the
    expanded polynomial instead of committing to either transformation.

    """
    mesh = Mesh(VectorElement("CG", triangle, 1))
    element = FiniteElement("Bernstein", triangle, 2)
    space = FunctionSpace(mesh, element)
    u = TrialFunction(space)
    v = TestFunction(space)
    form = inner(grad(u), grad(v)) * dx(scheme="canonical")

    optimized, = compile_form(form, parameters={"mode": "spectral"})
    collect_monomials = tsfc.spectral.collect_monomials
    monkeypatch.setattr(
        tsfc.spectral, "collect_monomials",
        lambda expressions, classifier, _: collect_monomials(
            expressions, classifier))
    expanded, = compile_form(form, parameters={"mode": "spectral"})

    assert optimized.flop_count <= expanded.flop_count


def test_sum_factorisation_order() -> None:
    """Select the least-cost quadrature contraction ordering."""
    i, j, q0, q1 = (gem.Index(extent=4) for _ in range(4))
    inner = gem.Indexed(gem.Variable("inner", (4, 4)), (i, q0))
    outer = gem.Indexed(
        gem.Variable("outer", (4, 4, 4)), (i, j, q1))
    variable = gem.Indexed(
        gem.Variable("result", (4, 4)), (i, j))
    monomial_sum = MonomialSum()
    monomial_sum.add((q0, q1), (inner * outer,), one)

    separate, _ = _plans(((variable, monomial_sum),), (q1, q0))
    (_, expression), = separate
    candidates = [
        estimate_cost((tsfc.spectral.sum_factorise(
            variable, ordering, monomial_sum),))
        for ordering in ((q1, q0), (q0, q1))
    ]

    assert estimate_cost((expression,)) == min(candidates)


def test_shared_contraction_ordering_bounds_storage(monkeypatch) -> None:
    """Share one contraction ordering when separate ones cost storage."""
    cell = TensorProductCell(quadrilateral, interval)
    mesh = Mesh(VectorElement('Q', cell, 1))
    space = FunctionSpace(mesh, FiniteElement('NCE', cell, 3))
    u = TrialFunction(space)
    v = TestFunction(space)
    form = action(dot(curl(u), curl(v)) * dx, Coefficient(space))

    def stored(kernel):
        temporaries = kernel.ast.default_entrypoint.temporary_variables
        return sum(
            numpy.prod(temporary.shape, dtype=int)
            for temporary in temporaries.values()
            if temporary.shape and not (temporary.read_only
                                        and temporary.initializer is not None))

    chosen, = compile_form(form, parameters={'mode': 'spectral'})

    # Deny the selection its shared-ordering candidate, leaving the plan
    # that lets every assignment minimise its own arithmetic.
    plans = tsfc.spectral._plans
    monkeypatch.setattr(
        tsfc.spectral, '_plans',
        lambda assignments, quadrature_indices: 2 * plans(
            assignments, quadrature_indices)[:1])
    separate, = compile_form(form, parameters={'mode': 'spectral'})

    assert stored(chosen) < stored(separate)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
