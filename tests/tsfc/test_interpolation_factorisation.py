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
    if mesh.ufl_cell() == interval:
        family = "DP"
    else:
        family = "DQ"
    return partial(request.param, family, mesh.ufl_cell())


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


def interpolate_flop_count(domain, source, target, dual):
    kernel, = compile_form(
        interpolate_expression(domain, source, target, dual),
        parameters={"mode": "spectral"},
    )
    return kernel.flop_count


def test_domainless_interpolate_in_form():
    mesh = Mesh(VectorElement("P", quadrilateral, 1))
    element = FiniteElement("Q", quadrilateral, 1)
    V = FunctionSpace(mesh, element)
    form = ufl.Interpolate(1, V) * ufl.dx

    compile_form(form, parameters={"mode": "spectral"})


def test_interpolate_wrapped_bernardi_raugel_argument_in_form():
    cell = ufl.triangle
    mesh = Mesh(VectorElement("P", cell, 1))
    base = FiniteElement("Bernardi-Raugel", cell, 1)
    space = FunctionSpace(mesh, VectorElement(base, dim=2))
    trial = ufl.TrialFunction(space)
    form = ufl.inner(ufl.Interpolate(trial, space), ufl.TestFunction(space)) * ufl.dx

    assert compile_form(form, parameters={"mode": "spectral"})


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


@pytest.mark.parametrize("dual", (False, True), ids=("primal", "dual"))
def test_sum_factorisation_scalar_tensor(mesh, element, dual):
    # Interpolation into tensor elements should cost value_shape
    # more than the equivalent scalar element.
    degree = 16
    source = element(degree - 1)
    target = element(degree)
    tensor_flops = interpolate_flop_count(mesh, source, target, dual)
    expect = FunctionSpace(mesh, target).value_size
    if isinstance(target, FiniteElement):
        scalar_flops = tensor_flops
    else:
        target = target.sub_elements[0]
        source = source.sub_elements[0]
        scalar_flops = interpolate_flop_count(mesh, source, target, dual)
    assert numpy.allclose(tensor_flops / scalar_flops, expect, rtol=1e-2)


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
