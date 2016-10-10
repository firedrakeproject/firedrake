"""Kernel generation for interpolating UFL expressions onto finite
element function spaces with only point evaluation nodes, using the
existing TSFC infrastructure."""

from __future__ import absolute_import, print_function, division

from collections import namedtuple

import ufl
import coffee.base as ast

from ufl.algorithms import extract_arguments, extract_coefficients
from gem import gem, impero_utils
from tsfc import fem, ufl_utils
from tsfc.coffee import generate as generate_coffee
from tsfc.kernel_interface.firedrake import KernelBuilderBase, cell_orientations_coffee_arg


# Expression kernel description type
Kernel = namedtuple('Kernel', ['ast', 'oriented', 'coefficients'])


def compile_expression_at_points(expression, refpoints, coordinates):
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
                         cell=coordinates.ufl_domain().ufl_cell(),
                         points=refpoints,
                         point_index=point_index,
                         coefficient=builder.coefficient,
                         cell_orientation=builder.cell_orientation)
    assert len(ir) == 1

    # Deal with non-scalar expressions
    tensor_indices = ()
    fs_shape = ir[0].shape
    if fs_shape:
        tensor_indices = tuple(gem.Index() for s in fs_shape)
        ir = [gem.Indexed(ir[0], tensor_indices)]

    # Build kernel body
    return_var = gem.Variable('A', (len(refpoints),) + fs_shape)
    return_expr = gem.Indexed(return_var, (point_index,) + tensor_indices)
    impero_c = impero_utils.compile_gem([return_expr], ir, (point_index,) + tensor_indices)
    body = generate_coffee(impero_c, index_names={point_index: 'p'})

    oriented = KernelBuilderBase.needs_cell_orientations(ir)
    if oriented:
        args.insert(0, cell_orientations_coffee_arg)

    # Build kernel
    args.insert(0, ast.Decl("double", ast.Symbol('A', rank=(len(refpoints),) + fs_shape)))
    kernel_code = builder.construct_kernel("expression_kernel", args, body)

    return Kernel(kernel_code, oriented, coefficients)
