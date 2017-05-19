from __future__ import absolute_import, print_function, division
from six.moves import range

from os import path
import numpy
import sympy

from pyop2 import op2
from pyop2.datatypes import IntType, as_cstr
from pyop2.base import build_itspace
from pyop2.sequential import generate_cell_wrapper

from ufl import TensorProductCell


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


format = {
    "assign": lambda v, w: "%s = %s;" % (v, str(w)),
}


def set_float_formatting(precision):
    "Set floating point formatting based on precision."

    # Options for float formatting
    f1 = "%%.%dg" % precision
    f2 = "%%.%dg" % precision
    f_int = "%%.%df" % 1

    eps = eval("1e-%s" % precision)

    # Regular float formatting
    def floating_point_regular(v):
        if abs(v - round(v, 1)) < eps:
            return f_int % v
        elif abs(v) < 100.0:
            return f1 % v
        else:
            return f2 % v

    # Special float formatting on Windows (remove extra leading zero)
    def floating_point_windows(v):
        return floating_point_regular(v).replace("e-0", "e-").replace("e+0", "e+")

    # Set float formatting
    import platform
    if platform.system() == "Windows":
        format["float"] = floating_point_windows
    else:
        format["float"] = floating_point_regular

    # FIXME: KBO: Remove once we agree on the format of 'f1'
    format["floating point"] = format["float"]

    # Set machine precision
    format["epsilon"] = 10.0*eval("1e-%s" % precision)


def compile_coordinate_element(ufl_coordinate_element, contains_eps):
    """Generates C code for changing to reference coordinates.

    :arg ufl_coordinate_element: UFL element of the coordinates
    :returns: C code as string
    """
    from tsfc import default_parameters
    from tsfc.finatinterface import create_element

    # Set code generation parameters
    set_float_formatting(default_parameters()["precision"])

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
        f_float = format["floating point"]
        f_assign = format["assign"]

        vertices = numpy.array(fiat_cell.get_vertices())
        X = numpy.average(vertices, axis=0)
        return "\n".join(f_assign("X[%d]" % i, f_float(v)) for i, v in enumerate(X))

    def to_reference_coordinates(ufl_cell, finat_element):
        pass

    # Create FInAT element
    element = create_element(ufl_coordinate_element)

    cell = ufl_coordinate_element.cell()
    extruded = isinstance(cell, TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "inside_predicate": inside_check(element.cell),
        "to_reference_coords": to_reference_coordinates(cell, element),
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
    double J[%(geometric_dimension)d * %(topological_dimension)d];
    double K[%(topological_dimension)d * %(geometric_dimension)d];
    double detJ;
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
    double *J = result->J;
    double *K = result->K;
    double detJ;

    double dX[%(topological_dimension)d];
    int converged = 0;

    for (int it = 0; !converged && it < %(max_iteration_count)d; it++) {
%(to_reference_coords)s

         if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
             converged = 1;
         }

%(X_isub_dX)s
    }

    result->detJ = detJ;

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
