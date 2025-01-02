from functools import partial
import numpy
import pytest

from ufl import (Mesh, FunctionSpace, Coefficient,
                 interval, quadrilateral, hexahedron)
from finat.ufl import FiniteElement, VectorElement, TensorElement

from tsfc import compile_expression_dual_evaluation
from finat.element_factory import create_element


@pytest.fixture(params=[interval, quadrilateral, hexahedron],
                ids=lambda x: x.cellname())
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
    to_element = create_element(Vtarget.ufl_element())
    expr = Coefficient(Vsource)
    kernel = compile_expression_dual_evaluation(expr, to_element, Vtarget.ufl_element())
    return kernel.flop_count


def test_sum_factorisation(mesh, element):
    # Interpolation between sum factorisable elements should cost
    # O(p^{d+1})
    degrees = numpy.asarray([2**n - 1 for n in range(2, 9)])
    flops = []
    for lo, hi in zip(degrees - 1, degrees):
        flops.append(flop_count(mesh, element(int(lo)), element(int(hi))))
    flops = numpy.asarray(flops)
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < (mesh.topological_dimension()+1)).all()


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
