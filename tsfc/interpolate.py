"""Kernel generation for interpolating UFL expressions onto finite
element function spaces with only point evaluation nodes, using the
existing TSFC infrastructure."""

from __future__ import absolute_import, print_function, division

from ufl.algorithms import extract_arguments, extract_coefficients

import coffee.base as ast

from gem import gem, impero_utils

from tsfc import fem, ufl_utils
from tsfc.coffee import generate as generate_coffee, SCALAR_TYPE
from tsfc.kernel_interface.firedrake import ExpressionKernelBuilder


def compile_expression_at_points(expression, points, coordinates):
    # No arguments, please!
    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression)

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    if domain:
        assert coordinates.ufl_domain() == domain
        expression = ufl_utils.replace_coordinates(expression, coordinates)

    # Initialise kernel builder
    builder = ExpressionKernelBuilder()
    builder.set_coefficients(extract_coefficients(expression))

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    point_index = gem.Index(name='p')
    ir, = fem.compile_ufl(expression,
                          cell=coordinates.ufl_domain().ufl_cell(),
                          points=points,
                          point_index=point_index,
                          coefficient=builder.coefficient,
                          cell_orientation=builder.cell_orientation)

    # Deal with non-scalar expressions
    tensor_indices = ()
    value_shape = ir.shape
    if value_shape:
        tensor_indices = tuple(gem.Index() for s in value_shape)
        ir = gem.Indexed(ir, tensor_indices)

    # Build kernel body
    return_shape = (len(points),) + value_shape
    return_indices = (point_index,) + tensor_indices
    return_var = gem.Variable('A', return_shape)
    return_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('A', rank=return_shape))
    return_expr = gem.Indexed(return_var, return_indices)
    impero_c = impero_utils.compile_gem([return_expr], [ir], return_indices)
    body = generate_coffee(impero_c, index_names={point_index: 'p'})

    # Handle cell orientations
    if builder.needs_cell_orientations([ir]):
        builder.require_cell_orientations()

    # Build kernel tuple
    return builder.construct_kernel(return_arg, body)
