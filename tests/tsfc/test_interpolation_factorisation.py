from functools import partial
import numpy
import pytest
import ufl

from ufl import (Mesh, MeshSequence, FunctionSpace, Coefficient,
                 interval, quadrilateral, hexahedron)
from finat.ufl import FiniteElement, VectorElement, TensorElement, MixedElement

from tsfc import compile_expression_dual_evaluation, compile_form


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


def flop_count(mesh, source, target):
    Vtarget = FunctionSpace(mesh, target)
    Vsource = FunctionSpace(mesh, source)
    expr = Coefficient(Vsource)
    kernel = compile_expression_dual_evaluation(expr, Vtarget.ufl_element())
    return kernel.flop_count


def interpolate_flop_count(domain, source, target, dual=False):
    Vsource = FunctionSpace(domain, source)
    Vtarget = FunctionSpace(domain, target)
    if dual:
        expression = ufl.Interpolate(
            ufl.Argument(Vsource, 0), ufl.Cofunction(Vtarget.dual())
        )
    else:
        expression = ufl.Interpolate(
            ufl.Coefficient(Vsource), ufl.Coargument(Vtarget.dual(), 0)
        )
    kernel, = compile_form(expression, parameters={"mode": "spectral"})
    return kernel.flop_count


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
            mixed_mesh, MixedElement(*source), MixedElement(*target), dual=dual
        ))
        component_flops.append(sum(
            interpolate_flop_count(mesh, source_element, target_element, dual=dual)
            for source_element, target_element in zip(source, target, strict=True)
        ))

    numpy.testing.assert_equal(mixed_flops, component_flops)
    rates = numpy.diff(numpy.log(mixed_flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < quadrilateral.topological_dimension + 1).all()
def test_sum_factorisation(mesh, element):
    # Interpolation between sum factorisable elements should cost
    # O(p^{d+1})
    degrees = numpy.asarray([2**n - 1 for n in range(2, 9)])
    flops = []
    for lo, hi in zip(degrees - 1, degrees):
        flops.append(flop_count(mesh, element(int(lo)), element(int(hi))))
    flops = numpy.asarray(flops)
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < (mesh.topological_dimension+1)).all()


def test_sum_factorisation_scalar_tensor(mesh, element):
    # Interpolation into tensor elements should cost value_shape
    # more than the equivalent scalar element.
    degree = 2**7 - 1
    source = element(degree - 1)
    target = element(degree)
    tensor_flops = flop_count(mesh, source, target)
    expect = FunctionSpace(mesh, target).value_size
    if isinstance(target, FiniteElement):
        scalar_flops = tensor_flops
    else:
        target = target.sub_elements[0]
        source = source.sub_elements[0]
        scalar_flops = flop_count(mesh, source, target)
    assert numpy.allclose(tensor_flops / scalar_flops, expect, rtol=1e-2)
