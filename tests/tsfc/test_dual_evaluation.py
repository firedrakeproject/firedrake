import math

import pytest
import ufl
import finat.ufl
from finat.element_factory import create_element
from tsfc import compile_expression_dual_evaluation
from tsfc.kernel_args import OutputKernelArg


def test_ufl_only_simple():
    mesh = ufl.Mesh(finat.ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("P", ufl.triangle, 2))
    v = ufl.Coefficient(V)
    expr = ufl.inner(v, v)
    W = V
    kernel = compile_expression_dual_evaluation(expr, W.ufl_element())
    assert kernel.needs_external_coords is False


def test_ufl_only_nested_interpolate():
    mesh = ufl.Mesh(finat.ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.VectorElement("P", ufl.triangle, 2))
    W = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("RT", ufl.triangle, 1))
    X = ufl.FunctionSpace(mesh, finat.ufl.VectorElement("P", ufl.triangle, 2))
    v = ufl.Coefficient(V)
    expression = ufl.Interpolate(ufl.Interpolate(v, W), X)

    kernel = compile_expression_dual_evaluation(expression, X.ufl_element())

    assert kernel.needs_external_coords is True


def test_ufl_only_spatialcoordinate():
    mesh = ufl.Mesh(finat.ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("P", ufl.triangle, 2))
    x, y = ufl.SpatialCoordinate(mesh)
    expr = x*y - y**2 + x
    W = V
    kernel = compile_expression_dual_evaluation(expr, W.ufl_element())
    assert kernel.needs_external_coords is True


def test_ufl_only_from_contravariant_piola():
    mesh = ufl.Mesh(finat.ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("RT", ufl.triangle, 1))
    v = ufl.Coefficient(V)
    expr = ufl.inner(v, v)
    W = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("P", ufl.triangle, 2))
    kernel = compile_expression_dual_evaluation(expr, W.ufl_element())
    assert kernel.needs_external_coords is True


def test_ufl_only_to_contravariant_piola():
    mesh = ufl.Mesh(finat.ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("P", ufl.triangle, 2))
    v = ufl.Coefficient(V)
    expr = ufl.as_vector([v, v])
    W = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("RT", ufl.triangle, 1))
    kernel = compile_expression_dual_evaluation(expr, W.ufl_element())
    assert kernel.needs_external_coords is True


def test_ufl_only_shape_mismatch():
    mesh = ufl.Mesh(finat.ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement("RT", ufl.triangle, 1))
    v = ufl.Coefficient(V)
    expr = ufl.inner(v, v)
    assert expr.ufl_shape == ()
    W = V
    to_element = create_element(W.ufl_element())
    assert to_element.value_shape == (2,)
    with pytest.raises(ValueError):
        compile_expression_dual_evaluation(expr, W.ufl_element())


def dual_argument_kernel(cell, degree, restriction=None):
    """Compile a degree-lowering interpolation against a Cofunction dual argument.

    Parameters
    ----------
    cell
        The UFL cell to interpolate on.
    degree
        The polynomial degree to interpolate from.
    restriction
        An optional restriction, such as ``"facet"``, to apply to both elements.

    Returns
    -------
    Kernel
        The compiled dual evaluation kernel.
    """
    mesh = ufl.Mesh(finat.ufl.VectorElement("Q", cell, 1))
    source = finat.ufl.FiniteElement("Q", cell, degree)
    target = source.reconstruct(degree=degree - 1)
    if restriction:
        source, target = source[restriction], target[restriction]
    V = ufl.FunctionSpace(mesh, source)
    W = ufl.FunctionSpace(mesh, target)
    expr = ufl.Interpolate(ufl.Argument(V, 0), ufl.Cofunction(W.dual()))
    return compile_expression_dual_evaluation(expr, target)


def test_dual_argument_on_concatenated_dual_basis():
    """The dual basis of a facet-restricted element is a Concatenate.

    A Cofunction dual argument sums over the concatenated index, so the
    contraction has to be split along the Concatenate before it is formed.
    """
    element = finat.ufl.FiniteElement("Q", ufl.quadrilateral, 3)["facet"]
    kernel = dual_argument_kernel(ufl.quadrilateral, 3, restriction="facet")
    # The dual index is summed away, leaving the argument's own nodes.
    output, = (arg for arg in kernel.arguments if isinstance(arg, OutputKernelArg))
    assert output.loopy_arg.shape == (create_element(element).space_dimension(),)


@pytest.mark.parametrize("restriction", [None, "facet"])
@pytest.mark.parametrize("cell", [ufl.quadrilateral, ufl.hexahedron])
def test_dual_argument_is_sum_factorised(cell, restriction):
    """Contracting a Cofunction against the dual basis must sum factorise.

    Interpolation on a tensor product cell costs O(degree^(dim + 1)) flops when
    the contraction is sum factorised, and O(degree^(2 * dim)) when it is not.
    """
    # A facet restriction has one fewer dimension of nodes, and so costs one
    # power less.
    dim = cell.topological_dimension - (restriction is not None)
    degrees = [8, 16]
    flops = [dual_argument_kernel(cell, degree, restriction=restriction).flop_count
             for degree in degrees]
    rate = math.log(flops[1] / flops[0]) / math.log(degrees[1] / degrees[0])
    assert rate < dim + 1
