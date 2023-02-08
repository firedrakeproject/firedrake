from os import path
import numpy
import sympy
from sympy.printing.c import ccode

from pyop2 import op2
from pyop2.parloop import generate_single_cell_wrapper

from firedrake.petsc import PETSc
from firedrake.utils import IntType, as_cstr, ScalarType, ScalarType_c, complex_mode, RealType_c

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
    return (arg,)


def make_wrapper(function, **kwargs):
    args = make_args(function)
    return generate_single_cell_wrapper(function.cell_set, args, **kwargs)


def src_locate_cell(mesh, tolerance=None):
    if tolerance is None:
        tolerance = 1e-14
    src = ['#include <evaluate.h>']
    src.append(compile_coordinate_element(mesh.ufl_coordinate_element(), tolerance))
    src.append(make_wrapper(mesh.coordinates,
                            forward_args=["void*", "double*", RealType_c+"*"],
                            kernel_name="to_reference_coords_kernel",
                            wrapper_name="wrap_to_reference_coords"))
    with open(path.join(path.dirname(__file__), "locate.c")) as f:
        src.append(f.read())

    src = "\n".join(src)
    return src


def dX_norm_square(topological_dimension):
    return " + ".join("PetscRealPart(dX[{0}])*PetscRealPart(dX[{0}])".format(i)
                      for i in range(topological_dimension))


def X_isub_dX(topological_dimension):
    return "\n".join("\tX[{0}] -= dX[{0}];".format(i)
                     for i in range(topological_dimension))


def is_affine(ufl_element):
    return ufl_element.cell().is_simplex() and ufl_element.degree() <= 1 and ufl_element.family() in ["Discontinuous Lagrange", "Lagrange"]


def inside_check(fiat_cell, eps, X="X"):
    """Generate a C expression which is true if a point is inside a FIAT
    reference cell and false otherwise.

    Parameters
    ----------
    fiat_cell : FIAT Cell
        The FIAT cell with same geometric dimension as the coordinate X.

    eps : float
        The tolerance to use for the check. Usually some small number like
        1e-14.

    X : str
        The name of the input pointer variable to use in the generated C code:
        it should be a pointer to a type that is an acceptable input to the
        `PetscRealPart` function. Default is "X".

    celldist : str
        The name of the output variable.

    Returns
    -------
    str
        A C expression which is true if the point is inside the cell and false
        otherwise.
    """
    dim = fiat_cell.get_spatial_dimension()
    point = tuple(sympy.Symbol("PetscRealPart(%s[%d])" % (X, i)) for i in range(dim))
    return ccode(fiat_cell.contains_point(point, epsilon=eps))


def celldist_l1_c_expr(fiat_cell, X="X"):
    """Generate a C expression of type `PetscReal` to compute the L1 distance
    (aka 'manhatten', 'taxicab' or rectilinear distance) to a FIAT reference
    cell.

    Parameters
    ----------
    fiat_cell : FIAT cell
        The FIAT cell with same geometric dimension as the coordinate X.

    X : str
        The name of the input pointer variable to use.

    celldist : str
        The name of the output variable.

    Returns
    -------
    str
        A string of C code.
    """
    dim = fiat_cell.get_spatial_dimension()
    point = tuple(sympy.Symbol("PetscRealPart(%s[%d])" % (X, i)) for i in range(dim))
    return ccode(fiat_cell.distance_to_point_l1(point))


def init_X(fiat_cell, parameters):
    vertices = numpy.array(fiat_cell.get_vertices())
    X = numpy.average(vertices, axis=0)

    formatter = ArrayInit(X, precision=numpy.finfo(parameters["scalar_type"]).resolution)._formatter
    return "\n".join("%s = %s;" % ("X[%d]" % i, formatter(v)) for i, v in enumerate(X))


@PETSc.Log.EventDecorator()
def to_reference_coords_newton_step(ufl_coordinate_element, parameters):
    # Set up UFL form
    cell = ufl_coordinate_element.cell()
    domain = ufl.Mesh(ufl_coordinate_element)
    K = ufl.JacobianInverse(domain)
    x = ufl.SpatialCoordinate(domain)
    x0_element = ufl.VectorElement("Real", cell, 0)
    x0 = ufl.Coefficient(ufl.FunctionSpace(domain, x0_element))
    expr = ufl.dot(K, x - x0)

    # Translation to GEM
    C = ufl.Coefficient(ufl.FunctionSpace(domain, ufl_coordinate_element))
    expr = ufl_utils.preprocess_expression(expr, complex_mode=complex_mode)
    expr = ufl_utils.simplify_abs(expr, complex_mode)

    builder = firedrake_interface.KernelBuilderBase(ScalarType_c)
    builder.domain_coordinate[domain] = C
    builder._coefficient(C, "C")
    builder._coefficient(x0, "x0")

    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    context = tsfc.fem.GemPointContext(
        interface=builder,
        ufl_cell=cell,
        point_indices=(),
        point_expr=point,
        scalar_type=parameters["scalar_type"]
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
    body = tsfc.coffee.generate(impero_c, {}, ScalarType)
    body.open_scope = False

    return body


@PETSc.Log.EventDecorator()
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
    # Create FInAT element
    element = tsfc.finatinterface.create_element(ufl_coordinate_element)

    cell = ufl_coordinate_element.cell()
    extruded = isinstance(cell, ufl.TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "celldist_l1_c_expr": celldist_l1_c_expr(element.cell, "X"),
        "to_reference_coords_newton_step": to_reference_coords_newton_step(ufl_coordinate_element, parameters),
        "init_X": init_X(element.cell, parameters),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(cell.topological_dimension()),
        "X_isub_dX": X_isub_dX(cell.topological_dimension()),
        "extruded_arg": ", int const *__restrict__ layers" if extruded else "",
        "extr_comment_out": "//" if extruded else "",
        "non_extr_comment_out": "//" if not extruded else "",
        "IntType": as_cstr(IntType),
        "ScalarType": ScalarType_c,
        "RealType": RealType_c,
        "tolerance": contains_eps,
    }

    evaluate_template_c = """#include <math.h>
struct ReferenceCoords {
    %(ScalarType)s X[%(geometric_dimension)d];
};

static %(RealType)s tolerance = %(tolerance)s; /* used in locate_cell */

static inline void to_reference_coords_kernel(void *result_, double *x0, %(RealType)s *cell_dist_l1, %(ScalarType)s *C)
{
    struct ReferenceCoords *result = (struct ReferenceCoords *) result_;

    /*
     * Mapping coordinates from physical to reference space
     */

    %(ScalarType)s *X = result->X;
    %(init_X)s

    int converged = 0;
    for (int it = 0; !converged && it < %(max_iteration_count)d; it++) {
        %(ScalarType)s dX[%(topological_dimension)d] = { 0.0 };
%(to_reference_coords_newton_step)s

        if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
            converged = 1;
        }

%(X_isub_dX)s
    }

    *cell_dist_l1 = %(celldist_l1_c_expr)s;
}

static inline void wrap_to_reference_coords(
    void* const result_, double* const x, %(RealType)s* const cell_dist_l1, %(IntType)s const start, %(IntType)s const end%(extruded_arg)s,
    %(ScalarType)s const *__restrict__ coords, %(IntType)s const *__restrict__ coords_map);

%(RealType)s to_reference_coords(void *result_, struct Function *f, int cell, double *x)
{
    %(RealType)s cell_dist_l1 = 0.0;
    %(extr_comment_out)swrap_to_reference_coords(result_, x, &cell_dist_l1, cell, cell+1, f->coords, f->coords_map);
    return cell_dist_l1;
}

%(RealType)s to_reference_coords_xtr(void *result_, struct Function *f, int cell, int layer, double *x)
{
    %(RealType)s cell_dist_l1 = 0.0;
    %(non_extr_comment_out)sint layers[2] = {0, layer+2};  // +2 because the layer loop goes to layers[1]-1, which is nlayers-1
    %(non_extr_comment_out)swrap_to_reference_coords(result_, x, &cell_dist_l1, cell, cell+1, layers, f->coords, f->coords_map);
    return cell_dist_l1;
}

"""

    return evaluate_template_c % code
