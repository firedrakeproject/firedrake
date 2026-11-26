from functools import partial
import numpy
import pytest
import ufl

from ufl import (Mesh, MeshSequence, FunctionSpace,
                 interval, quadrilateral, hexahedron)
from finat.ufl import FiniteElement, VectorElement, TensorElement, MixedElement
from finat.element_factory import create_element

from tsfc import compile_form


@pytest.fixture(params=[interval, quadrilateral, hexahedron],
                ids=lambda x: x.cellname)
def mesh(request):
    return Mesh(VectorElement("P", request.param, 1))


@pytest.fixture(params=[FiniteElement, VectorElement, TensorElement],
                ids=lambda x: x.__name__)
def element(request, mesh):
    return partial(request.param, "DG", mesh.ufl_cell())


def interpolate_expression(domain, source, target, dual):
    Vsource = FunctionSpace(domain, source)
    Vtarget = FunctionSpace(domain, target)
    if dual:
        return ufl.Interpolate(
            ufl.Argument(Vsource, 0), ufl.Cofunction(Vtarget.dual())
        )
    return ufl.Interpolate(
        ufl.Coefficient(Vsource), ufl.Coargument(Vtarget.dual(), 0)
    )


def spectral_flop_count(form):
    kernel, = compile_form(form, parameters={"mode": "spectral"})
    return kernel.flop_count


def interpolate_flop_count(domain, source, target, dual):
    return spectral_flop_count(interpolate_expression(domain, source, target, dual))


def test_domainless_interpolate_in_form():
    mesh = Mesh(VectorElement("P", quadrilateral, 1))
    element = FiniteElement("Q", quadrilateral, 1)
    V = FunctionSpace(mesh, element)
    form = ufl.Interpolate(1, V) * ufl.dx

    compile_form(form, parameters={"mode": "spectral"})


@pytest.mark.parametrize("target_family", ["RT", "N1curl", "gradient"])
def test_interpolate_morley_argument_with_coupled_components(target_family):
    cell = ufl.triangle
    mesh = Mesh(VectorElement("P", cell, 1))
    source = FunctionSpace(mesh, VectorElement("Morley", cell, 2))
    trial = ufl.TrialFunction(source)
    if target_family == "gradient":
        operand = ufl.grad(trial)
        target_element = TensorElement("P", cell, 1)
    else:
        operand = trial
        target_element = FiniteElement(target_family, cell, 1)
    target = FunctionSpace(mesh, target_element)
    form = ufl.inner(ufl.Interpolate(operand, target), ufl.TestFunction(target)) * ufl.dx

    assert compile_form(form, parameters={"mode": "spectral"})


@pytest.mark.parametrize("dual", (False, True), ids=("primal", "dual"))
def test_sum_factorisation(mesh, element, dual):
    # Interpolation between sum factorisable elements should cost
    # O(p^{d+1})
    degrees = numpy.asarray([4, 8, 16])
    flops = []
    for lo, hi in zip(degrees - 1, degrees):
        flops.append(interpolate_flop_count(
            mesh, element(int(lo)), element(int(hi)), dual
        ))
    flops = numpy.asarray(flops)
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < mesh.topological_dimension + 1).all()


@pytest.mark.parametrize("vector", [
    pytest.param(partial(VectorElement, dim=2), id="VectorElement"),
    pytest.param(partial(TensorElement, shape=(2, 2)), id="TensorElement"),
])
@pytest.mark.parametrize("dual", (False, True), ids=("primal", "dual"))
def test_vector_flops(mesh, vector, dual):
    # Interpolation between vector-valued elements should cost one
    # scalar interpolation per component.
    scalar_source = FiniteElement("DG", mesh.ufl_cell(), 15)
    scalar_target = FiniteElement("DG", mesh.ufl_cell(), 16)
    vector_source = vector(scalar_source)
    vector_target = vector(scalar_target)

    scalar_flops = interpolate_flop_count(mesh, scalar_source, scalar_target, dual)
    flops = interpolate_flop_count(mesh, vector_source, vector_target, dual)
    assert flops == len(vector_target.sub_elements) * scalar_flops


@pytest.mark.parametrize("form", ("interp-primal", "interp-dual", "L2", "H1"))
def test_vector_zany_flops(form):
    # The components of a vector of zany elements share the geometric work,
    # so they cost less than one scalar interpolation each, and a form that
    # pairs a component only with itself shares all the work.
    cell = ufl.triangle
    mesh = Mesh(VectorElement("P", cell, 1))
    scalar_source = FiniteElement("Reduced-Hermite", cell, 3)
    scalar_target = FiniteElement("Rotated-Bernardi-Raugel", cell, 1)
    vector_source = VectorElement(scalar_source, dim=2)
    vector_target = VectorElement(scalar_target, dim=2)

    flops = []
    for source, target in [(scalar_source, scalar_target), (vector_source, vector_target)]:
        Vsource = FunctionSpace(mesh, source)
        Vtarget = FunctionSpace(mesh, target)
        if form == "interp-primal":
            expr = ufl.Interpolate(ufl.grad(ufl.Coefficient(Vsource)), ufl.Coargument(Vtarget.dual(), 0))
        elif form == "interp-dual":
            expr = ufl.Interpolate(ufl.grad(ufl.Argument(Vsource, 0)), ufl.Cofunction(Vtarget.dual()))
        else:
            u = ufl.Interpolate(ufl.grad(ufl.TrialFunction(Vsource)), Vtarget)
            v = ufl.Interpolate(ufl.grad(ufl.TestFunction(Vsource)), Vtarget)
            expr = ufl.inner(u, v) * ufl.dx
            if form == "H1":
                expr += ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        flops.append(spectral_flop_count(expr))

    scalar_flops, vector_flops = flops
    dim = len(vector_target.sub_elements)
    if form in ("L2", "H1"):
        # The element tensor is block diagonal with the scalar element
        # tensor in each block, so beyond the scalar work the components
        # only accumulate the entries of their own blocks.
        ndofs = create_element(scalar_source).space_dimension()
        assert vector_flops <= scalar_flops + dim * ndofs**2
    else:
        assert vector_flops < dim * scalar_flops


def q_rtce_elements(degree):
    return (FiniteElement("Q", quadrilateral, degree),
            FiniteElement("RTCE", quadrilateral, degree))


@pytest.mark.parametrize("dual", (False, True), ids=("primal", "dual"))
def test_sum_factorisation_mixed_q_rtce(dual):
    mesh = Mesh(VectorElement("Q", quadrilateral, 1))
    mixed_mesh = MeshSequence([mesh, mesh])
    degrees = numpy.asarray([4, 8, 16])
    mixed_flops = []
    component_flops = []
    for degree in degrees:
        source = q_rtce_elements(int(degree - 1))
        target = q_rtce_elements(int(degree))
        mixed_flops.append(interpolate_flop_count(
            mixed_mesh, MixedElement(*source), MixedElement(*target), dual
        ))
        component_flops.append(sum(
            interpolate_flop_count(mesh, source_element, target_element, dual)
            for source_element, target_element in zip(source, target, strict=True)
        ))

    numpy.testing.assert_equal(mixed_flops, component_flops)
    rates = numpy.diff(numpy.log(mixed_flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < quadrilateral.topological_dimension + 1).all()
