"""Kernel generation for interpolating UFL expressions onto finite
element function spaces with only point evaluation nodes, using the
existing TSFC infrastructure."""

from __future__ import absolute_import, print_function, division

from collections import namedtuple

import ufl
import coffee.base as ast

from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms import extract_arguments, extract_coefficients
from gem import gem, impero_utils
from tsfc import fem, ufl_utils
from tsfc.coffee import generate as generate_coffee
from tsfc.kernel_interface.firedrake import KernelBuilderBase, cell_orientations_coffee_arg


# Expression kernel description type
Kernel = namedtuple('Kernel', ['ast', 'oriented', 'coefficients'])


def compile_ufl_kernel(expression, to_pts, to_element, fs):

    # Imitate the compute_form_data processing pipeline
    #
    # Unfortunately, we cannot call compute_form_data here, since
    # we only have an expression, not a form
    expression = apply_algebra_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_function_pullbacks(expression)
    expression = apply_geometry_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_geometry_lowering(expression)
    expression = apply_derivatives(expression)

    # Replace coordinates (if any)
    if expression.ufl_domain():
        assert fs.mesh() == expression.ufl_domain()
        expression = ufl_utils.replace_coordinates(expression, fs.mesh().coordinates)

    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    builder = KernelBuilderBase()
    coefficient_split = {}

    coefficients = []
    args = []
    for i, coefficient in enumerate(extract_coefficients(expression)):
        if type(coefficient.ufl_element()) == ufl.MixedElement:
            coefficient_split[coefficient] = []
            for j, element in enumerate(coefficient.ufl_element().sub_elements()):
                subcoeff = ufl.Coefficient(ufl.FunctionSpace(coefficient.ufl_domain(), element))
                coefficient_split[coefficient].append(subcoeff)
                args.append(builder._coefficient(subcoeff, "w_%d_%d" % (i, j)))
            coefficients.extend(coefficient.split())
        else:
            args.append(builder._coefficient(coefficient, "w_%d" % (i,)))
            coefficients.append(coefficient)

    expression = ufl_utils.split_coefficients(expression, coefficient_split)

    point_index = gem.Index(name='p')
    ir = fem.compile_ufl(expression,
                         cell=fs.mesh().ufl_cell(),
                         points=to_pts,
                         point_index=point_index,
                         coefficient=builder.coefficient,
                         cell_orientation=builder.cell_orientation)
    assert len(ir) == 1

    # Deal with non-scalar expressions
    tensor_indices = ()
    if fs.shape:
        tensor_indices = tuple(gem.Index() for s in fs.shape)
        ir = [gem.Indexed(ir[0], tensor_indices)]

    # Build kernel body
    return_var = gem.Variable('A', (len(to_pts),) + fs.shape)
    return_expr = gem.Indexed(return_var, (point_index,) + tensor_indices)
    impero_c = impero_utils.compile_gem([return_expr], ir, (point_index,) + tensor_indices)
    body = generate_coffee(impero_c, index_names={point_index: 'p'})

    oriented = KernelBuilderBase.needs_cell_orientations(ir)
    if oriented:
        args.insert(0, cell_orientations_coffee_arg)

    # Build kernel
    args.insert(0, ast.Decl("double", ast.Symbol('A', rank=(len(to_pts),) + fs.shape)))
    kernel_code = builder.construct_kernel("expression_kernel", args, body)

    return Kernel(kernel_code, oriented, coefficients)
