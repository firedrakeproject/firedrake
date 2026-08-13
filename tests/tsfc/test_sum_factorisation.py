import numpy
import pytest

import gem
import tsfc.spectral
from gem.gem import one
from gem.refactorise import MonomialSum
from ufl import (Mesh, FunctionSpace, TestFunction, TrialFunction,
                 TensorProductCell, dx, action, interval, triangle,
                 quadrilateral, curl, dot, div, grad, inner)
from finat.ufl import (FiniteElement, VectorElement, EnrichedElement,
                       TensorProductElement, HCurlElement, HDivElement)

from tsfc import compile_form
from tsfc.spectral import _sum_factorisation_order


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
    """Check that a mapped tabulation is shared by both argument axes.

    Parameters
    ----------
    monkeypatch
        Pytest fixture used to disable the sharing pass for comparison.
    """
    mesh = Mesh(VectorElement("CG", triangle, 1))
    element = FiniteElement("Johnson-Mercier", triangle, 1)
    space = FunctionSpace(mesh, element)
    u = TrialFunction(space)
    v = TestFunction(space)
    form = (inner(u, v) + inner(div(u), div(v))) * dx

    optimized, = compile_form(form, parameters={"mode": "spectral"})
    monkeypatch.setattr(
        tsfc.spectral, "hoist_linear_index",
        lambda expression, indices: expression)
    baseline, = compile_form(form, parameters={"mode": "spectral"})

    optimized_shapes = [
        temporary.shape for temporary in
        optimized.ast.default_entrypoint.temporary_variables.values()]
    baseline_shapes = [
        temporary.shape for temporary in
        baseline.ast.default_entrypoint.temporary_variables.values()]

    assert optimized.flop_count < baseline.flop_count
    assert sum(not shape for shape in optimized_shapes) \
        < sum(not shape for shape in baseline_shapes)
    assert sum(shape == (15,) for shape in optimized_shapes) \
        > sum(shape == (15,) for shape in baseline_shapes)


def test_sum_factorisation_order() -> None:
    """Contract the quadrature direction with least argument support first."""
    i, j, q0, q1 = (gem.Index(extent=4) for _ in range(4))
    inner = gem.Indexed(gem.Variable("inner", (4, 4)), (i, q0))
    outer = gem.Indexed(
        gem.Variable("outer", (4, 4, 4)), (i, j, q1))
    monomial_sum = MonomialSum()
    monomial_sum.add((q0, q1), (inner * outer,), one)

    ordering = _sum_factorisation_order(
        (q1, q0), monomial_sum)

    assert ordering == (q0, q1)


def test_bernstein_candidate_selection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the least-cost algebraic plan for a Bernstein form."""
    mesh = Mesh(VectorElement("CG", triangle, 1))
    space = FunctionSpace(mesh, FiniteElement("Bernstein", triangle, 6))
    u = TrialFunction(space)
    v = TestFunction(space)
    form = inner(grad(u), grad(v)) * dx

    selected, = compile_form(form, parameters={"mode": "spectral"})

    def minimum_cost_plan(
            variable, candidates, quadrature_indices, index_replacer):
        return min(
            candidates,
            key=lambda candidate: tsfc.spectral._candidate_score(
                tsfc.spectral._optimise_candidate(
                    variable, candidate, quadrature_indices,
                    index_replacer)))

    monkeypatch.setattr(
        tsfc.spectral, "_select_factorisation_plan", minimum_cost_plan)
    minimum, = compile_form(form, parameters={"mode": "spectral"})

    assert selected.flop_count == minimum.flop_count


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
