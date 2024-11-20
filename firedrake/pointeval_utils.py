import loopy as lp
from firedrake.utils import IntType, as_cstr

from ufl import TensorProductCell
from finat.ufl import MixedElement
from ufl.corealg.map_dag import map_expr_dags
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.domain import extract_unique_domain

import gem

import tsfc
import tsfc.kernel_interface.firedrake_loopy as firedrake_interface
from tsfc.loopy import generate as generate_loopy
from tsfc.parameters import default_parameters

from firedrake import utils
from firedrake.petsc import PETSc


@PETSc.Log.EventDecorator()
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
    expression = tsfc.ufl_utils.preprocess_expression(expression, complex_mode=utils.complex_mode)

    # Collect required coefficients
    coefficient, = extract_coefficients(expression)

    # Point evaluation of mixed coefficients not supported here
    if type(coefficient.ufl_element()) == MixedElement:
        raise NotImplementedError("Cannot point evaluate mixed elements yet!")

    # Replace coordinates (if any)
    domain = extract_unique_domain(expression)
    assert extract_unique_domain(coordinates) == domain

    # Initialise kernel builder
    builder = firedrake_interface.KernelBuilderBase(utils.ScalarType)
    builder.domain_coordinate[domain] = coordinates
    builder._coefficient(coordinates, "x")
    x_arg = builder.generate_arg_from_expression(builder.coefficient_map[coordinates])
    builder._coefficient(coefficient, "f")
    f_arg = builder.generate_arg_from_expression(builder.coefficient_map[coefficient])

    # TODO: restore this for expression evaluation!
    # expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = lp.GlobalArg("X", dtype=utils.ScalarType, shape=(dim,))

    config = dict(interface=builder,
                  ufl_cell=extract_unique_domain(coordinates).ufl_cell(),
                  integral_type="cell",
                  point_indices=(),
                  point_expr=point,
                  scalar_type=utils.ScalarType)
    # TODO: restore this for expression evaluation!
    # config["cellvolume"] = cellvolume_generator(extract_unique_domain(coordinates), coordinates, config)
    context = tsfc.fem.GemPointContext(**config)

    # Abs-simplification
    expression = tsfc.ufl_utils.simplify_abs(expression, utils.complex_mode)

    # Translate UFL -> GEM
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    tensor_indices = ()
    if expression.ufl_shape:
        tensor_indices = tuple(gem.Index() for s in expression.ufl_shape)
        return_variable = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
        result_arg = lp.GlobalArg("R", dtype=utils.ScalarType, shape=expression.ufl_shape)
        result = gem.Indexed(result, tensor_indices)
    else:
        return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
        result_arg = lp.GlobalArg("R", dtype=utils.ScalarType, shape=(1,))

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> loopy
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)
    loopy_args = [result_arg, point_arg, x_arg, f_arg]
    loopy_kernel, _ = generate_loopy(
        impero_c, loopy_args, utils.ScalarType,
        kernel_name="evaluate_kernel", index_names={})
    kernel_code = lp.generate_code_v2(loopy_kernel).device_code()

    # Fill the code template
    extruded = isinstance(cell, TensorProductCell)

    code = {
        "geometric_dimension": domain.geometric_dimension(),
        "layers_arg": ", int const *__restrict__ layers" if extruded else "",
        "layers": ", layers" if extruded else "",
        "extruded_define": "1" if extruded else "0",
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
static inline void wrap_evaluate(%(scalar_type)s* const result, %(scalar_type)s* const X, %(IntType)s const start, %(IntType)s const end%(layers_arg)s,
    %(scalar_type)s const *__restrict__ coords, %(scalar_type)s const *__restrict__ f, %(wrapper_map_args)s);


int evaluate(struct Function *f, double *x, %(scalar_type)s *result)
{
    /* The type definitions and arguments used here are defined as statics in pointquery_utils.py */
    double found_ref_cell_dist_l1 = DBL_MAX;
    struct ReferenceCoords temp_reference_coords, found_reference_coords;
    int cells_ignore[1] = {-1};
    %(IntType)s cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &to_reference_coords_xtr, &temp_reference_coords, &found_reference_coords, &found_ref_cell_dist_l1, 1, cells_ignore);
    if (cell == -1) {
        return -1;
    }

    if (!result) {
        return 0;
    }
#if %(extruded_define)s
    int layers[2] = {0, 0};
    int nlayers = f->n_layers;
    layers[1] = cell %% nlayers + 2;
    cell = cell / nlayers;
#endif

    wrap_evaluate(result, found_reference_coords.X, cell, cell+1%(layers)s, f->coords, f->f, %(map_args)s);
    return 0;
}
"""

    return (evaluate_template_c % code) + kernel_code
