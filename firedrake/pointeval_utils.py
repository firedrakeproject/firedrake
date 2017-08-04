
from pyop2.datatypes import IntType, as_cstr

from coffee import base as ast

from ufl import MixedElement, TensorProductCell
from ufl.corealg.map_dag import map_expr_dags
from ufl.algorithms import extract_arguments, extract_coefficients

import gem

import tsfc
import tsfc.kernel_interface.firedrake as firedrake_interface
from tsfc.coffee import SCALAR_TYPE, generate as generate_coffee
from tsfc.parameters import default_parameters


def compile_element(expression, coordinates, parameters=None):
    """Generates C code for point evaluations.

    :arg expression: UFL expression
    :arg coordinates: coordinate field
    :arg parameters: form compiler parameters
    :returns: C code as string
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # No arguments, please!
    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    # Apply UFL preprocessing
    expression = tsfc.ufl_utils.preprocess_expression(expression)

    # Collect required coefficients
    coefficient, = extract_coefficients(expression)

    # Point evaluation of mixed coefficients not supported here
    if type(coefficient.ufl_element()) == MixedElement:
        raise NotImplementedError("Cannot point evaluate mixed elements yet!")

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    assert coordinates.ufl_domain() == domain
    expression = tsfc.ufl_utils.replace_coordinates(expression, coordinates)

    # Initialise kernel builder
    builder = firedrake_interface.KernelBuilderBase()
    x_arg = builder._coefficient(coordinates, "x")
    f_arg = builder._coefficient(coefficient, "f")

    # TODO: restore this for expression evaluation!
    # expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('X', rank=(dim,)))

    config = dict(interface=builder,
                  ufl_cell=coordinates.ufl_domain().ufl_cell(),
                  precision=parameters["precision"],
                  point_indices=(),
                  point_expr=point)
    # TODO: restore this for expression evaluation!
    # config["cellvolume"] = cellvolume_generator(coordinates.ufl_domain(), coordinates, config)
    context = tsfc.fem.GemPointContext(**config)

    # Abs-simplification
    expression = tsfc.ufl_utils.simplify_abs(expression)

    # Translate UFL -> GEM
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    tensor_indices = ()
    if expression.ufl_shape:
        tensor_indices = tuple(gem.Index() for s in expression.ufl_shape)
        return_variable = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
        result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('R', rank=expression.ufl_shape))
        result = gem.Indexed(result, tensor_indices)
    else:
        return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
        result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('R', rank=(1,)))

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> COFFEE
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)
    body = generate_coffee(impero_c, {}, parameters["precision"])

    # Build kernel tuple
    kernel_code = builder.construct_kernel("evaluate_kernel", [result_arg, point_arg, x_arg, f_arg], body)

    # Fill the code template
    extruded = isinstance(cell, TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "extruded_arg": ", %s nlayers" % as_cstr(IntType) if extruded else "",
        "nlayers": ", f->n_layers" if extruded else "",
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """static inline void wrap_evaluate(double *result, double *X, double *coords, %(IntType)s *coords_map, double *f, %(IntType)s *f_map%(extruded_arg)s, %(IntType)s cell);

int evaluate(struct Function *f, double *x, double *result)
{
    struct ReferenceCoords reference_coords;
    %(IntType)s cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &reference_coords);
    if (cell == -1) {
        return -1;
    }

    if (!result) {
        return 0;
    }

    wrap_evaluate(result, reference_coords.X, f->coords, f->coords_map, f->f, f->f_map%(nlayers)s, cell);
    return 0;
}
"""

    return (evaluate_template_c % code) + kernel_code.gencode()
