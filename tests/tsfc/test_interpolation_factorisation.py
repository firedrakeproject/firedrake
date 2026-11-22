from functools import partial
import numpy
import pytest
import ufl

from ufl import (Mesh, MeshSequence, FunctionSpace,
                 interval, quadrilateral, hexahedron)
from finat.ufl import FiniteElement, VectorElement, TensorElement, MixedElement

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


def interpolation_in_mass_form(mesh, source, target):
    source_space = FunctionSpace(mesh, source)
    target_space = FunctionSpace(mesh, target)
    trial = ufl.TrialFunction(source_space)
    test = ufl.TestFunction(target_space)
    return ufl.inner(ufl.Interpolate(trial, target_space), test) * ufl.dx


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


def test_vector_zany_flops():
    # The components of a vector Bernardi-Raugel element share the
    # geometric work, so the cost stays below a fixed budget.
    cell = ufl.triangle
    mesh = Mesh(VectorElement("P", cell, 1))
    element = VectorElement(FiniteElement("Bernardi-Raugel", cell, 1), dim=2)
    form = interpolation_in_mass_form(mesh, element, element)
    assert spectral_flop_count(form) <= 8392


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
