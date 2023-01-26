from os import path
import numpy
import sympy
from sympy.printing.c import ccode

from pyop2 import op2
from pyop2.parloop import generate_single_cell_wrapper

from firedrake.petsc import PETSc
from firedrake.utils import IntType, as_cstr, ScalarType, ScalarType_c, complex_mode

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
                            forward_args=["void*", "double*", "int*"],
                            kernel_name="to_reference_coords_kernel",
                            wrapper_name="wrap_to_reference_coords"))
    src.append(compute_distance_to_cell(mesh.ufl_cell()))
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
    dim = fiat_cell.get_spatial_dimension()
    point = tuple(sympy.Symbol("PetscRealPart(%s[%d])" % (X, i)) for i in range(dim))
    return ccode(fiat_cell.contains_point(point, epsilon=eps))


def compute_celldist(fiat_cell, X="X", celldist="celldist"):
    dim = fiat_cell.get_spatial_dimension()
    s = """
    %(celldist)s = PetscRealPart(%(X)s[0]);
    for (int celldistdim = 1; celldistdim < %(dim)s; celldistdim++) {
        if (%(celldist)s > PetscRealPart(%(X)s[celldistdim])) {
            %(celldist)s = PetscRealPart(%(X)s[celldistdim]);
        }
    }
    %(celldist)s *= -1;
    """ % {"celldist": celldist,
           "dim": dim,
           "X": X}

    return s


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
        "inside_predicate": inside_check(element.cell, eps=contains_eps),
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
        "tolerance": contains_eps,
    }

    evaluate_template_c = """#include <math.h>
struct ReferenceCoords {
    %(ScalarType)s X[%(geometric_dimension)d];
};

static double tolerance = %(tolerance)s; /* used in locate_cell */

static inline void to_reference_coords_kernel(void *result_, double *x0, int *return_value, %(ScalarType)s *C)
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

    // Are we inside the reference element?
    *return_value = %(inside_predicate)s;
}

static inline void wrap_to_reference_coords(
    void* const result_, double* const x, int* const return_value, %(IntType)s const start, %(IntType)s const end%(extruded_arg)s,
    %(ScalarType)s const *__restrict__ coords, %(IntType)s const *__restrict__ coords_map);

int to_reference_coords(void *result_, struct Function *f, int cell, double *x)
{
    int return_value = 0;
    %(extr_comment_out)swrap_to_reference_coords(result_, x, &return_value, cell, cell+1, f->coords, f->coords_map);
    return return_value;
}

int to_reference_coords_xtr(void *result_, struct Function *f, int cell, int layer, double *x)
{
    int return_value = 0;
    %(non_extr_comment_out)sint layers[2] = {0, layer+2};  // +2 because the layer loop goes to layers[1]-1, which is nlayers-1
    %(non_extr_comment_out)swrap_to_reference_coords(result_, x, &return_value, cell, cell+1, layers, f->coords, f->coords_map);
    return return_value;
}

"""

    return evaluate_template_c % code


def compute_distance_to_cell(ufl_cell):
    """Generate C code for computing the approximate
    (within an order of magnitude) distance to a cell.

    If the return value of the generated code is negative,
    the point is guaranteed to be inside the cell.

    Parameters
    ----------
    ufl_cell : ufl.Cell
        The cell to compute the distance to.

    Returns
    -------
    code : str
        The C code for the distance computation.
    """
    # Append code for compute_distance_to_cell which will be used by locate_cell.c
    # if it is defined.
    if ufl_cell == ufl.vertex:
        # todo
        return ""
    elif ufl_cell == ufl.interval:
        return src_compute_distance_to_cell_interval()
    elif ufl_cell == ufl.triangle:
        return src_compute_distance_to_cell_triangle()
    elif ufl_cell == ufl.tetrahedron:
        # todo
        return ""
    elif ufl_cell == ufl.prism:
        # todo
        return ""
    elif ufl_cell == ufl.pyramid:
        # todo
        return ""
    elif ufl_cell == ufl.quadrilateral:
        # todo
        return ""
    elif ufl_cell == ufl.hexahedron:
        # todo
        return ""
    else:
        # Something unexpected like TensorProductCell
        return ""


def src_compute_distance_to_cell_interval():
    """Generate C code for computing the approximate distance to a
    reference interval as at FIAT.reference_element.UFCTriangle.

    If the point is inside the interval, the distance is negative.
    Note that in this case the distance outputted is exact.

    Returns
    -------
    code : str
        The C code for the distance computation.
    """
    return """
#define COMPUTE_DISTANCE_TO_CELL /* Opens necessary code paths in locate.c */
#include <assert.h>
double compute_distance_to_cell(double *X, int dim)
{
    assert(dim == 1);
    /* We use barycentric coordinates to determine if the point is inside
       the reference cell. We two vertices which make the reference interval,
       P0 = 0 and P1 = 1. Barycentric coordinates are defined as
       X[0] = alpha * P0 + beta * P1 where alpha + beta = 1.0. The solution is
       alpha = 1 - X[0] and beta = X[0]. If both alpha and beta are positive,
       the point is inside the reference cell.

       ---regionA---P0=0------P1=1---regionB---

       If we are in regionA, alpha is negative and -alpha = X[0] - 1.0 is the
       (positive) distance from P0.
       If we are in regionB, beta is negative and -beta = -X[0] is the
       (positive) distance from P1.
       If we are in the interval we can just return -X[0] since we don't care
       about how close to either vertex we are. */

    /* Is alpha negative? */
    if (X[0] > 1.0) {
        return X[0] - 1.0;
    } else {
        /* Either beta is negative or we are in the interval */
        return -X[0];
    }
}
"""


def src_compute_distance_to_cell_triangle():
    """Generate C code for computing the aproximate distance to a
    reference triangle as at FIAT.reference_element.UFCTriangle.

    If the point is inside the triangle, the distance is negative.

    Returns
    -------
    code : str
        The C code for the distance computation.
    """
    return """
#define COMPUTE_DISTANCE_TO_CELL /* Opens necessary code paths in locate.c */
#include <assert.h>
double compute_distance_to_cell(double *X, int dim)
{
    /* We use barycentric coordinates to determine if the point is inside
       the reference cell. We have three vertices which make the reference
       triangle, P0 = (1, 0), P1 = (0, 1) and P2 = (0, 0). Below is a
       diagram of the cell:

                y-axis
                |
                |
          (1,0) P1
                | \
                |  \
                |   \
                |    \
                |  T  \
                |      \
                |       \
                |        \
            ---P2--------P0--- x-axis
          (0,0) |         (0,1)

    Barycentric coordinates are defined as
      X[0] = alpha * P0 + beta * P1 + gamma * P2 where
      alpha + beta + gamma = 1.0.
    The solution is
      alpha = X[0]
      beta = X[1] and
      gamma = 1 - X[0] - X[1].
    If all three are positive, the point is inside the reference cell.
    If any is negative, we are outside it. The negative barycentric coordinate
    which is closest to 0.0 is a reasonable approximation of the closest point
    to the triangle.

    */
    assert(dim == 2);
    double alpha = X[0];
    double beta = X[1];
    double gamma = 1.0 - X[0] - X[1];
    if (alpha > 0.0 && beta > 0.0 && gamma > 0.0) {
        /* We are inside the triangle */
        return -alpha;
    } else {
        /* We are outside the triangle */
        /* Find the negative alpha, beta or gamma closest to 0.0 */
        if (alpha < 0.0 && beta > 0.0 && gamma > 0.0) {
            return -alpha;
        } else if (alpha > 0.0 && beta < 0.0 && gamma > 0.0) {
            return -beta;
        } else if (alpha > 0.0 && beta > 0.0 && gamma < 0.0) {
            return -gamma;
        } else if (alpha < 0.0 && beta < 0.0 && gamma > 0.0) {
            return alpha > beta ? -alpha : -beta;
        } else if (alpha < 0.0 && beta > 0.0 && gamma < 0.0) {
            return alpha > gamma ? -alpha : -gamma;
        } else if (alpha > 0.0 && beta < 0.0 && gamma < 0.0) {
            return beta > gamma ? -beta : -gamma;
        } else {
            /* All are negative */
            return alpha > beta ? (alpha > gamma ? -alpha : -gamma) : (beta > gamma ? -beta : -gamma);
        }
    }
}
"""
