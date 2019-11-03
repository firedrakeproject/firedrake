
from pyop2.datatypes import IntType, as_cstr

from coffee import base as ast

from ufl import MixedElement, TensorProductCell
from ufl.corealg.map_dag import map_expr_dags
from ufl.algorithms import extract_arguments, extract_coefficients

import gem

import tsfc
import tsfc.kernel_interface.firedrake as firedrake_interface
from tsfc.coffee import generate as generate_coffee
from tsfc.parameters import default_parameters

from firedrake import utils


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

    # Initialise kernel builder
    builder = firedrake_interface.KernelBuilderBase(utils.ScalarType_c)
    builder.domain_coordinate[domain] = coordinates
    x_arg = builder._coefficient(coordinates, "x")
    f_arg = builder._coefficient(coefficient, "f")

    # TODO: restore this for expression evaluation!
    # expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = ast.Decl(utils.ScalarType_c, ast.Symbol('X', rank=(dim,)))

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
        result_arg = ast.Decl(utils.ScalarType_c, ast.Symbol('R', rank=expression.ufl_shape))
        result = gem.Indexed(result, tensor_indices)
    else:
        return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
        result_arg = ast.Decl(utils.ScalarType_c, ast.Symbol('R', rank=(1,)))

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> COFFEE
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)
    body = generate_coffee(impero_c, {}, parameters["precision"], utils.ScalarType_c)

    # Build kernel tuple
    kernel_code = builder.construct_kernel("evaluate_kernel", [result_arg, point_arg, x_arg, f_arg], body)

    # Fill the code template
    extruded = isinstance(cell, TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "layers_arg": ", int const *__restrict__ layers" if extruded else "",
        "layers": ", layers" if extruded else "",
        "IntType": as_cstr(IntType),
        "scalar_type": utils.ScalarType_c,
    }
    # if maps are the same, only need to pass one of them
    if coordinates.cell_node_map() == coefficient.cell_node_map():
        code["wrapper_map_args"] = "%(IntType)s const *__restrict__ coords_map" % code
        code["map_args"] = "f->coords_map"
    else:
        code["wrapper_map_args"] = "%(IntType)s const *__restrict__ coords_map, %(IntType)s const *__restrict__ f_map" % code
        code["map_args"] = "f->coords_map, f->f_map"

    evaluate_template_c = """
static inline void wrap_evaluate(%(scalar_type)s* const result, %(scalar_type)s* const X, int const start, int const end%(layers_arg)s,
    %(scalar_type)s const *__restrict__ coords, %(scalar_type)s const *__restrict__ f, %(wrapper_map_args)s);

int evaluate(struct Function *f, %(scalar_type)s *x, %(scalar_type)s *result)
{
    struct ReferenceCoords reference_coords;
    %(IntType)s cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &to_reference_coords_xtr, &reference_coords);
    if (cell == -1) {
        return -1;
    }

    if (!result) {
        return 0;
    }
    int layers[2] = {0, 0};
    if (f->extruded != 0) {
        int nlayers = f->n_layers;
        layers[1] = cell %% nlayers + 2;
        cell = cell / nlayers;
    }

    wrap_evaluate(result, reference_coords.X, cell, cell+1%(layers)s, f->coords, f->f, %(map_args)s);
    return 0;
}
"""

    return (evaluate_template_c % code) + kernel_code.gencode()
