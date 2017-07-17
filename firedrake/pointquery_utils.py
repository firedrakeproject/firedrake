from __future__ import absolute_import, print_function, division
from six.moves import range

from os import path
import numpy
import sympy

from pyop2 import op2
from pyop2.datatypes import IntType, as_cstr
from pyop2.base import build_itspace
from pyop2.sequential import generate_cell_wrapper

import ufl
from ufl.corealg.map_dag import map_expr_dag

import gem
import gem.impero_utils as impero_utils

import tsfc
import tsfc.kernel_interface.firedrake as firedrake_interface
import tsfc.ufl_utils as ufl_utils

from coffee.base import ArrayInit


def make_args(function):
    arg = function.dat(op2.READ, function.cell_node_map())
    arg.position = 0
    return (arg,)


def make_wrapper(function, **kwargs):
    args = make_args(function)
    return generate_cell_wrapper(build_itspace(args, function.cell_set), args, **kwargs)


def src_locate_cell(mesh, tolerance=None):
    if tolerance is None:
        tolerance = 1e-14
    src = '#include <evaluate.h>\n'
    src += compile_coordinate_element(mesh.ufl_coordinate_element(), tolerance)
    src += make_wrapper(mesh.coordinates,
                        forward_args=["void*", "double*", "int*"],
                        kernel_name="to_reference_coords_kernel",
                        wrapper_name="wrap_to_reference_coords")

    with open(path.join(path.dirname(__file__), "locate.c")) as f:
        src += f.read()

    return src


def compile_coordinate_element(ufl_coordinate_element, contains_eps, parameters=None):
    """Generates C code for changing to reference coordinates.

    :arg ufl_coordinate_element: UFL element of the coordinates
    :returns: C code as string
    """
    if parameters is None:
        parameters = tsfc.default_parameters()
    else:
        _ = tsfc.default_parameters()
        _.update(parameters)
        parameters = _

    def dX_norm_square(topological_dimension):
        return " + ".join("dX[{0}]*dX[{0}]".format(i)
                          for i in range(topological_dimension))

    def X_isub_dX(topological_dimension):
        return "\n".join("\tX[{0}] -= dX[{0}];".format(i)
                         for i in range(topological_dimension))

    def is_affine(ufl_element):
        return ufl_element.cell().is_simplex() and ufl_element.degree() <= 1 and ufl_element.family() in ["Discontinuous Lagrange", "Lagrange"]

    def inside_check(fiat_cell):
        dim = fiat_cell.get_spatial_dimension()
        point = tuple(sympy.Symbol("X[%d]" % i) for i in range(dim))

        return " && ".join("(%s)" % arg for arg in fiat_cell.contains_point(point, epsilon=contains_eps).args)

    def init_X(fiat_cell):
        vertices = numpy.array(fiat_cell.get_vertices())
        X = numpy.average(vertices, axis=0)

        formatter = ArrayInit(X, precision=parameters["precision"])._formatter
        return "\n".join("%s = %s;" % ("X[%d]" % i, formatter(v)) for i, v in enumerate(X))

    def to_reference_coordinates(ufl_coordinate_element):
        # Set up UFL form
        cell = ufl_coordinate_element.cell()
        domain = ufl.Mesh(ufl_coordinate_element)
        K = ufl.JacobianInverse(domain)
        x = ufl.SpatialCoordinate(domain)
        x0_element = ufl.VectorElement("Real", cell, 0)
        x0 = ufl.Coefficient(ufl.FunctionSpace(domain, x0_element))
        expr = ufl.dot(K, x - x0)

        # Translation to GEM
        C = ufl_utils.coordinate_coefficient(domain)
        expr = ufl_utils.preprocess_expression(expr)
        expr = ufl_utils.replace_coordinates(expr, C)
        expr = ufl_utils.simplify_abs(expr)

        builder = firedrake_interface.KernelBuilderBase()
        builder._coefficient(C, "C")
        builder._coefficient(x0, "x0")

        dim = cell.topological_dimension()
        point = gem.Variable('X', (dim,))
        context = tsfc.fem.GemPointContext(
            interface=builder,
            ufl_cell=cell,
            precision=parameters["precision"],
            point_indices=(),
            point_expr=point,
        )
        translator = tsfc.fem.Translator(context)
        ir = map_expr_dag(translator, expr)

        # Unroll result
        ir = [gem.Indexed(ir, alpha) for alpha in numpy.ndindex(ir.shape)]

        # Unroll IndexSums
        max_extent = parameters["unroll_indexsum"]
        if max_extent:
            def predicate(index):
                return index.extent <= max_extent
        ir = gem.optimise.unroll_indexsum(ir, predicate=predicate)

        # Translate to COFFEE
        ir = impero_utils.preprocess_gem(ir)
        return_variable = gem.Variable('dX', (dim,))
        assignments = [(gem.Indexed(return_variable, (i,)), e)
                       for i, e in enumerate(ir)]
        impero_c = impero_utils.compile_gem(assignments, ())
        body = tsfc.coffee.generate(impero_c, {}, parameters["precision"])
        body.open_scope = False

        return body

    # Create FInAT element
    element = tsfc.finatinterface.create_element(ufl_coordinate_element)

    cell = ufl_coordinate_element.cell()
    extruded = isinstance(cell, ufl.TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "inside_predicate": inside_check(element.cell),
        "to_reference_coords": to_reference_coordinates(ufl_coordinate_element),
        "init_X": init_X(element.cell),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(cell.topological_dimension()),
        "X_isub_dX": X_isub_dX(cell.topological_dimension()),
        "extruded_arg": ", %s nlayers" % as_cstr(IntType) if extruded else "",
        "nlayers": ", f->n_layers" if extruded else "",
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """#include <math.h>
struct ReferenceCoords {
    double X[%(geometric_dimension)d];
};

static inline void to_reference_coords_kernel(void *result_, double *x0, int *return_value, double **C)
{
    struct ReferenceCoords *result = (struct ReferenceCoords *) result_;

    const int space_dim = %(geometric_dimension)d;

    /*
     * Mapping coordinates from physical to reference space
     */

    double *X = result->X;
%(init_X)s
    double x[space_dim];

    int converged = 0;
    for (int it = 0; !converged && it < %(max_iteration_count)d; it++) {
        double dX[%(topological_dimension)d] = { 0.0 };
%(to_reference_coords)s

        if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
            converged = 1;
        }

%(X_isub_dX)s
    }

    // Are we inside the reference element?
    *return_value = %(inside_predicate)s;
}

static inline void wrap_to_reference_coords(void *result_, double *x, int *return_value,
                                            double *coords, %(IntType)s *coords_map%(extruded_arg)s, %(IntType)s cell);

int to_reference_coords(void *result_, struct Function *f, int cell, double *x)
{
    int return_value;
    wrap_to_reference_coords(result_, x, &return_value, f->coords, f->coords_map%(nlayers)s, cell);
    return return_value;
}
"""

    return evaluate_template_c % code
