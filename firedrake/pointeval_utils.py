import loopy as lp
from firedrake.utils import IntType, as_cstr

from finat.element_factory import as_fiat_cell
from finat.point_set import UnknownPointSet
from finat.quadrature import QuadratureRule
from finat.ufl import MixedElement, FiniteElement, TensorElement

from ufl.corealg.map_dag import map_expr_dags
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.domain import extract_unique_domain

import gem
import ufl

import tsfc
import tsfc.kernel_interface.firedrake_loopy as firedrake_interface
from tsfc.loopy import generate as generate_loopy
from tsfc.parameters import default_parameters

from firedrake import utils
from firedrake.petsc import PETSc


def runtime_quadrature_space(domain, ufl_element, rt_var_name="rt_X"):
    """Construct a Quadrature FunctionSpace for interpolation onto a
    VertexOnlyMesh. The quadrature point is an UnknownPointSet of shape
    (1, tdim) where tdim is the topological dimension of domain.ufl_cell(). The
    weight is [1.0], since the single local dof in the VertexOnlyMesh function
    space corresponds to a point evaluation at the vertex.

    Parameters
    ----------
    domain : ufl.Domain
        The source domain.
    ufl_element : finat.ufl.FiniteElement
        The FInAT element to construct a QuadratureElement for.
    rt_var_name : str
        String beginning with 'rt_' which is used as the name of the
        gem.Variable used to represent the UnknownPointSet. The `rt_` prefix
        forces TSFC to do runtime tabulation.
    """
    assert rt_var_name.startswith("rt_")

    cell = domain.ufl_cell()
    point_expr = gem.Variable(rt_var_name, (1, cell.topological_dimension))
    point_set = UnknownPointSet(point_expr)
    rule = QuadratureRule(point_set, weights=[1.0], ref_el=as_fiat_cell(cell))

    shape = ufl_element.reference_value_shape
    ufl_element = FiniteElement("Quadrature", cell=cell, degree=0, quad_scheme=rule)
    if shape:
        ufl_element = TensorElement(ufl_element, shape=shape)
    return ufl.FunctionSpace(domain, ufl_element)


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
    builder._domain_integral_type_map = {domain: "cell"}
    builder._entity_ids = {domain: (0,)}
    builder.domain_coordinate[domain] = coordinates
    builder._coefficient(coordinates, "x")
    x_arg = builder.generate_arg_from_expression(builder.coefficient_map[coordinates])
    builder._coefficient(coefficient, "f")
    f_arg = builder.generate_arg_from_expression(builder.coefficient_map[coefficient])

    # TODO: restore this for expression evaluation!
    # expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension
    point = gem.Variable('X', (dim,))
    point_arg = lp.GlobalArg("X", dtype=utils.ScalarType, shape=(dim,))

    config = dict(interface=builder,
                  ufl_cell=extract_unique_domain(coordinates).ufl_cell(),
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
    extruded = isinstance(cell, ufl.TensorProductCell)

    code = {
        "geometric_dimension": domain.geometric_dimension,
        "layers_arg": f", {as_cstr(IntType)} const *__restrict__ layers" if extruded else "",
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
    %(IntType)s layers[2] = {0, 0};
    %(IntType)s nlayers = f->n_layers;
    layers[1] = cell %% nlayers + 2;
    cell = cell / nlayers;
#endif

    wrap_evaluate(result, found_reference_coords.X, cell, cell+1%(layers)s, f->coords, f->f, %(map_args)s);
    return 0;
}
"""

    return (evaluate_template_c % code) + kernel_code
