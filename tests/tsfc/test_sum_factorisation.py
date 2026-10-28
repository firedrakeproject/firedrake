import numpy
import pytest

from ufl import (Mesh, FunctionSpace, TestFunction, TrialFunction,
                 TensorProductCell, dx, action, interval, triangle,
                 quadrilateral, hexahedron, tetrahedron, curl, dot, div,
                 grad, inner)
from finat.ufl import (FiniteElement, VectorElement, EnrichedElement,
                       TensorProductElement, HCurlElement, HDivElement)

import tsfc.spectral
from tsfc import compile_form


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


def count_storage(form):
    kernel, = compile_form(form, parameters=dict(mode='spectral'))
    temporaries = kernel.ast.default_entrypoint.temporary_variables
    return sum(numpy.prod(temporary.shape, dtype=int)
               for temporary in temporaries.values()
               if temporary.shape and not (temporary.read_only
                                           and temporary.initializer is not None))


def count_loops(form):
    kernel, = compile_form(form, parameters=dict(mode='spectral'))
    return len(kernel.ast.default_entrypoint.all_inames())


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
                         [(quadrilateral, 1),
                          (TensorProductCell(interval, interval), 1),
                          (TensorProductCell(triangle, interval), 1),
                          (TensorProductCell(quadrilateral, interval), 2)])
def test_contraction_storage_rate(cell, order):
    degrees = list(range(3, 8))
    storage = [count_storage(action(helmholtz(cell, degree)))
               for degree in degrees]
    rates = numpy.diff(numpy.log(storage)) / numpy.diff(numpy.log(degrees))
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


@pytest.mark.parametrize(('cell', 'equivalent_cell'),
                         [(quadrilateral, TensorProductCell(interval, interval)),
                          (hexahedron, TensorProductCell(quadrilateral, interval)),
                          (hexahedron, TensorProductCell(interval, interval, interval))])
@pytest.mark.parametrize('degree', [3, 5])
def test_equivalent_cells(cell, equivalent_cell, degree):
    """A form is compiled the same way on cells that hold the same space."""
    a = helmholtz(cell, degree)
    b = helmholtz(equivalent_cell, degree)
    assert count_flops(a) == count_flops(b)
    assert count_flops(action(a)) == count_flops(action(b))


@pytest.fixture
def expanded(monkeypatch):
    """Force the expanded representation, for comparison."""
    def force(monkeypatch=monkeypatch):
        collect_monomials = tsfc.spectral.collect_monomials
        monkeypatch.setattr(
            tsfc.spectral, "collect_monomials",
            lambda expressions, classifier, _: collect_monomials(
                expressions, classifier))
    return force


def piola_helmholtz(cell, degree):
    m = Mesh(VectorElement('CG', cell, 1))
    V = FunctionSpace(m, FiniteElement('RT', cell, degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    return (inner(u, v) + inner(div(u), div(v)))*dx


@pytest.mark.parametrize('cell', [triangle, tetrahedron],
                         ids=lambda cell: cell.cellname)
@pytest.mark.parametrize('degree', [1, 2, 3])
def test_piola_map_is_preserved(cell, degree, expanded):
    # Test and trial apply the same Piola map, so preserving it evaluates
    # the physical basis once instead of pushing the geometry through both
    # argument axes.
    form = piola_helmholtz(cell, degree)
    selected = count_flops(form)
    expanded()
    assert selected < count_flops(form)


@pytest.mark.parametrize('cell', [triangle, tetrahedron],
                         ids=lambda cell: cell.cellname)
@pytest.mark.parametrize('degree', [1, 3])
def test_preserving_a_map_is_never_worse(cell, degree, expanded):
    # Expanding a map exposes scalar factorisation of its entries, which at
    # some degrees beats sharing it.  Selection costs both, so neither
    # representation may regress the other.
    form = helmholtz(cell, degree)
    selected = count_flops(form)
    expanded()
    assert selected <= count_flops(form)


@pytest.mark.parametrize('cell', [triangle, tetrahedron],
                         ids=lambda cell: cell.cellname)
@pytest.mark.parametrize('degree', [1, 3])
def test_shared_map_is_tabulated_in_one_loop(cell, degree, expanded):
    # A map that both argument axes share is tabulated once, so it must be
    # tabulated in one loop.  An index per axis fissions the loop nest that
    # the expanded representation keeps whole, which costs more than the
    # flops it saves.
    form = piola_helmholtz(cell, degree)
    selected = count_loops(form)
    expanded()
    assert selected <= count_loops(form)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
