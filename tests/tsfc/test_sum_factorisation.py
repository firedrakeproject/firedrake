import numpy
import pytest

import gem
import tsfc.spectral
from gem.gem import one
from gem.refactorise import MonomialSum
from ufl import (Mesh, FunctionSpace, TestFunction, TrialFunction,
                 TensorProductCell, dx, action, interval, triangle,
                 tetrahedron, quadrilateral, curl, dot, div, grad, inner)
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

    optimized_source = str(optimized.ast)
    baseline_source = str(baseline.ast)
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
    assert optimized_source.count(" if ") < baseline_source.count(" if ")


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
    collapsed_shapes = [
        temporary.shape
        for temporary in entrypoint.temporary_variables.values()
    ]
    canonical_shapes = [
        temporary.shape
        for temporary in
        canonical_kernel.ast.default_entrypoint.temporary_variables.values()
    ]
    assert max(map(len, collapsed_shapes)) <= 5
    assert sum(map(numpy.prod, collapsed_shapes)) \
        < sum(map(numpy.prod, canonical_shapes))

    code = lp.generate_code_v2(collapsed_kernel.ast).device_code()
    assert sum(
        line.lstrip().startswith("for (")
        for line in code.splitlines()
    ) < 150

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
