import numpy
import pytest

from ufl import (Mesh, FunctionSpace, TestFunction, TrialFunction,
                 TensorProductCell, dx, action, interval, triangle,
                 quadrilateral, tetrahedron, curl, dot, div,
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


def bernstein_mass(
        cell: object, degree: int, scheme: str = "collapsed") -> object:
    """Construct a Bernstein mass form.

    Parameters
    ----------
    cell
        Reference simplex.
    degree
        Polynomial degree.
    scheme
        Quadrature scheme.

    Returns
    -------
    object
        UFL bilinear form.
    """
    mesh = Mesh(VectorElement("CG", cell, 1))
    space = FunctionSpace(mesh, FiniteElement("Bernstein", cell, degree))
    u = TrialFunction(space)
    v = TestFunction(space)
    return inner(u, v) * dx(scheme=scheme)


def bernstein_laplacian(
        cell: object, degree: int, scheme: str = "collapsed") -> object:
    """Construct a Bernstein Laplacian form.

    Parameters
    ----------
    cell
        Reference simplex.
    degree
        Polynomial degree.
    scheme
        Quadrature scheme.

    Returns
    -------
    object
        UFL bilinear form.
    """
    mesh = Mesh(VectorElement("CG", cell, 1))
    space = FunctionSpace(mesh, FiniteElement("Bernstein", cell, degree))
    u = TrialFunction(space)
    v = TestFunction(space)
    return inner(grad(u), grad(v)) * dx(scheme=scheme)


@pytest.mark.parametrize(("cell", "order"), [(triangle, 3), (tetrahedron, 4)])
def test_bernstein_mass_action(cell: object, order: float) -> None:
    degrees = list(range(3, 9)) if cell is triangle else list(range(3, 8))
    flops = [
        count_flops(action(bernstein_mass(cell, degree)))
        for degree in degrees
    ]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(
    ("cell", "order"), [(triangle, 3), (tetrahedron, 4.4)])
def test_bernstein_laplacian_action(cell: object, order: float) -> None:
    degrees = list(range(3, 9)) if cell is triangle else list(range(3, 8))
    flops = [
        count_flops(action(bernstein_laplacian(cell, degree)))
        for degree in degrees
    ]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


def test_bernstein_laplacian_action_compact_literals() -> None:
    degree = 5
    form = action(bernstein_laplacian(tetrahedron, degree))
    kernel, = compile_form(form, parameters={"mode": "spectral"})
    temporaries = kernel.ast.default_entrypoint.temporary_variables
    literals = [
        numpy.asarray(temporary.initializer)
        for temporary in temporaries.values()
        if temporary.initializer is not None
    ]
    lattice_size = (degree + 1) ** 3
    assert max(literal.size for literal in literals) <= lattice_size
    assert sum(literal.size for literal in literals) < 10 * lattice_size


def test_bernstein_laplacian_bilinear_compact_codegen() -> None:
    import islpy as isl
    import loopy as lp

    degree = 10
    collapsed = bernstein_laplacian(
        tetrahedron, degree, scheme="collapsed")
    canonical = bernstein_laplacian(
        tetrahedron, degree, scheme="canonical")
    collapsed_kernel, = compile_form(
        collapsed, parameters={"mode": "spectral"})
    canonical_kernel, = compile_form(
        canonical, parameters={"mode": "spectral"})

    assert collapsed_kernel.flop_count < canonical_kernel.flop_count

    entrypoint = collapsed_kernel.ast.default_entrypoint
    collapsed_temporaries = tuple(entrypoint.temporary_variables.values())
    canonical_temporaries = tuple(
        canonical_kernel.ast.default_entrypoint.temporary_variables.values())
    collapsed_shapes = [
        temporary.shape
        for temporary in collapsed_temporaries
    ]
    assert max(map(len, collapsed_shapes)) <= 5
    assert sum(numpy.prod(temporary.shape)
               for temporary in collapsed_temporaries
               if temporary.base_storage is None) \
        < sum(numpy.prod(temporary.shape)
              for temporary in canonical_temporaries
              if temporary.base_storage is None)

    code = lp.generate_code_v2(collapsed_kernel.ast).device_code()
    assert sum(
        line.lstrip().startswith("for (")
        for line in code.splitlines()
    ) < 500

    # A tetrahedral lattice has a loop whose bound depends on two parents.
    assert max(
        domain.dim(isl.dim_type.param)
        for domain in entrypoint.domains
    ) >= 2


@pytest.mark.parametrize(("cell", "order"), [(triangle, 5), (tetrahedron, 7)])
def test_bernstein_mass_bilinear(cell: object, order: float) -> None:
    degrees = list(range(3, 9)) if cell is triangle else list(range(3, 8))
    flops = [
        count_flops(bernstein_mass(cell, degree))
        for degree in degrees
    ]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(("cell", "order"), [(triangle, 5), (tetrahedron, 7)])
def test_bernstein_laplacian_bilinear(
        cell: object, order: float) -> None:
    degrees = list(range(3, 9)) if cell is triangle else list(range(3, 8))
    flops = [
        count_flops(bernstein_laplacian(cell, degree))
        for degree in degrees
    ]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
