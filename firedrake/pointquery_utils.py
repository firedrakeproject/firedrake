from __future__ import absolute_import

from os import path
import numpy

from pyop2 import op2
from pyop2.base import build_itspace
from pyop2.sequential import generate_cell_wrapper

from ufl import Cell, OuterProductCell


def make_args(function):
    arg = function.dat(op2.READ, function.cell_node_map())
    arg.position = 0
    return (arg,)


def make_wrapper(function, **kwargs):
    args = make_args(function)
    return generate_cell_wrapper(build_itspace(args, function.cell_set), args, **kwargs)


def src_locate_cell(mesh):
    src = '#include <evaluate.h>\n'
    src += compile_coordinate_element(mesh.ufl_coordinate_element())
    src += make_wrapper(mesh.coordinates,
                        forward_args=["void*", "double*", "int*"],
                        kernel_name="to_reference_coords_kernel",
                        wrapper_name="wrap_to_reference_coords")

    with open(path.join(path.dirname(__file__), "locate.cpp")) as f:
        src += f.read()

    return src


# Code snippets for computing Jacobian inverses.  Only uses Jacobian entries, so don't
# need a separate set for interior facets.

_compute_jacobian_inverse_interval_1d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[1];
double detJ%(restriction)s;
compute_jacobian_inverse_interval_1d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_interval_2d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[2];
double detJ%(restriction)s;
compute_jacobian_inverse_interval_2d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_interval_3d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[3];
double detJ%(restriction)s;
compute_jacobian_inverse_interval_3d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_triangle_2d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[4];
double detJ%(restriction)s;
compute_jacobian_inverse_triangle_2d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_triangle_3d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[6];
double detJ%(restriction)s;
compute_jacobian_inverse_triangle_3d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_tetrahedron_3d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[9];
double detJ%(restriction)s;
compute_jacobian_inverse_tetrahedron_3d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_quad_2d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[4];
double detJ%(restriction)s;
compute_jacobian_inverse_quad_2d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_quad_3d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[6];
double detJ%(restriction)s;
compute_jacobian_inverse_quad_3d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_prism_3d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[9];
double detJ%(restriction)s;
compute_jacobian_inverse_prism_3d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

_compute_jacobian_inverse_hex_3d = """\
// Compute Jacobian inverse and determinant
double K%(restriction)s[9];
double detJ%(restriction)s;
compute_jacobian_inverse_hex_3d(K%(restriction)s, detJ%(restriction)s, J%(restriction)s);
"""

compute_jacobian_inverse = {
    Cell("interval"): _compute_jacobian_inverse_interval_1d,
    Cell("interval", 2): _compute_jacobian_inverse_interval_2d,
    Cell("interval", 3): _compute_jacobian_inverse_interval_3d,
    Cell("triangle"): _compute_jacobian_inverse_triangle_2d,
    Cell("triangle", 3): _compute_jacobian_inverse_triangle_3d,
    Cell("tetrahedron"): _compute_jacobian_inverse_tetrahedron_3d,
    Cell("quadrilateral"): _compute_jacobian_inverse_quad_2d,
    Cell("quadrilateral", 3): _compute_jacobian_inverse_quad_3d,
    OuterProductCell(Cell("interval"), Cell("interval")): _compute_jacobian_inverse_quad_2d,
    OuterProductCell(Cell("interval"), Cell("interval"), gdim=3): _compute_jacobian_inverse_quad_3d,
    OuterProductCell(Cell("triangle"), Cell("interval")): _compute_jacobian_inverse_prism_3d,
    OuterProductCell(Cell("quadrilateral"), Cell("interval")): _compute_jacobian_inverse_hex_3d,
}


def _declaration(type, name, value=None):
    if value is None:
        return "%s %s;" % (type, name)
    return "%s %s = %s;" % (type, name, str(value))


def _component(var, k):
    if not isinstance(k, (list, tuple)):
        k = [k]
    return "%s" % var + "".join("[%s]" % str(i) for i in k)


def _tabulate_tensor(vals):
    "Tabulate a multidimensional tensor. (Replace tabulate_matrix and tabulate_vector)."

    # Prefetch formats to speed up code generation
    f_block = format["block"]
    f_list_sep = format["list separator"]
    f_block_sep = format["block separator"]
    # FIXME: KBO: Change this to "float" once issue in set_float_formatting is fixed.
    f_float = format["floating point"]
    f_epsilon = format["epsilon"]

    # Create numpy array and get shape.
    tensor = numpy.array(vals)
    shape = numpy.shape(tensor)
    if len(shape) == 1:
        # Create zeros if value is smaller than tolerance.
        values = []
        for v in tensor:
            if not isinstance(v, (float, int)):
                values.append(str(v))
            elif abs(v) < f_epsilon:
                values.append(f_float(0.0))
            else:
                values.append(f_float(v))
        # Format values.
        return f_block(f_list_sep.join(values))
    elif len(shape) > 1:
        return f_block(f_block_sep.join([_tabulate_tensor(tensor[i]) for i in range(shape[0])]))
    else:
        raise ValueError("Not an N-dimensional array:\n%s" % tensor)


format = {
    "assign": lambda v, w: "%s = %s;" % (v, str(w)),
    "declaration": _declaration,
    "float declaration": "double",
    "compute_jacobian_inverse": lambda cell: compute_jacobian_inverse[cell] % {"restriction": ""},
    "block": lambda v: "{%s}" % v,
    "list separator": ", ",
    "block separator": ",\n",
    "new line": "\\\n",
    "component": _component,
    "tabulate tensor": _tabulate_tensor,
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


def compile_coordinate_element(ufl_coordinate_element):
    """Generates C code for changing to reference coordinates.

    :arg ufl_coordinate_element: UFL element of the coordinates
    :returns: C code as string
    """
    from tsfc.constants import PRECISION
    from tsfc.fiatinterface import create_element
    from firedrake.pointeval_utils import ssa_arrays, c_print
    from FIAT.reference_element import two_product_cell
    import sympy as sp
    import numpy as np

    # Set code generation parameters
    set_float_formatting(PRECISION)

    def dX_norm_square(topological_dimension):
        return " + ".join("dX[{0}]*dX[{0}]".format(i)
                          for i in xrange(topological_dimension))

    def X_isub_dX(topological_dimension):
        return "\n".join("\tX[{0}] -= dX[{0}];".format(i)
                         for i in xrange(topological_dimension))

    def is_affine(ufl_element):
        return ufl_element.cell().is_simplex() and ufl_element.degree() <= 1 and ufl_element.family() in ["Discontinuous Lagrange", "Lagrange"]

    def inside_check(ufl_cell, fiat_cell):
        dim = ufl_cell.topological_dimension()
        point = tuple(sp.Symbol("X[%d]" % i) for i in xrange(dim))

        return " && ".join("(%s)" % arg for arg in fiat_cell.contains_point(point, epsilon=1e-14).args)

    def init_X(fiat_element):
        f_float = format["floating point"]
        f_assign = format["assign"]

        fiat_cell = fiat_element.get_reference_element()
        vertices = np.array(fiat_cell.get_vertices())
        X = np.average(vertices, axis=0)
        return "\n".join(f_assign("X[%d]" % i, f_float(v)) for i, v in enumerate(X))

    def to_reference_coordinates(ufl_cell, fiat_element):
        f_decl = format["declaration"]
        f_float_decl = format["float declaration"]

        # Get the element cell name and geometric dimension.
        cell = ufl_cell
        gdim = cell.geometric_dimension()
        tdim = cell.topological_dimension()

        code = []

        # Symbolic tabulation
        tabs = fiat_element.tabulate(1, np.array([[sp.Symbol("X[%d]" % i) for i in xrange(tdim)]]))
        tabs = sorted((d, value.reshape(value.shape[:-1])) for d, value in tabs.iteritems())

        # Generate code for intermediate values
        s_code, d_phis = ssa_arrays(map(lambda (k, v): v, tabs), prefix="t")
        phi = d_phis.pop(0)

        for name, value in s_code:
            code += [f_decl(f_float_decl, name, c_print(value))]

        # Cell coordinate data
        C = np.array([[sp.Symbol("C[%d][%d]" % (i, j)) for j in range(gdim)]
                      for i in range(fiat_element.space_dimension())])

        # Generate physical coordinates
        x = phi.dot(C)
        for i, e in enumerate(x):
            code += ["\tx[%d] = %s;" % (i, e)]

        # Generate Jacobian
        grad_phi = np.vstack(reversed(d_phis))
        J = np.transpose(grad_phi.dot(C))
        for i, row in enumerate(J):
            for j, e in enumerate(row):
                code += ["\tJ[%d * %d + %d] = %s;" % (i, tdim, j, e)]

        # Get code snippets for Jacobian, inverse of Jacobian and mapping of
        # coordinates from physical element to the FIAT reference element.
        code_ = [format["compute_jacobian_inverse"](cell)]
        # FIXME: use cell orientations!
        # if needs_orientation:
        #     code_ += [format["orientation"]["ufc"](tdim, gdim)]
        # FIXME: ugly hack
        code_ = "\n".join(code_).split("\n")
        code_ = filter(lambda line: not line.startswith(("double J", "double K", "double detJ")), code_)
        code += code_

        x = np.array([sp.Symbol("x[%d]" % i) for i in xrange(gdim)])
        x0 = np.array([sp.Symbol("x0[%d]" % i) for i in xrange(gdim)])
        K = np.array([[sp.Symbol("K[%d]" % (i*gdim + j)) for j in range(gdim)]
                      for i in range(tdim)])

        dX = K.dot(x - x0)
        for i, e in enumerate(dX):
            code += ["\tdX[%d] = %s;" % (i, e)]

        return "\n".join(code)

    # Create FIAT element
    element = create_element(ufl_coordinate_element, vector_is_mixed=False)
    cell = ufl_coordinate_element.cell()

    # calculate_basisvalues, vdim = calculate_basisvalues(cell, element)
    extruded = isinstance(element.get_reference_element(), two_product_cell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "inside_predicate": inside_check(cell, element.get_reference_element()),
        "to_reference_coords": to_reference_coordinates(cell, element),
        "init_X": init_X(element),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(cell.topological_dimension()),
        "X_isub_dX": X_isub_dX(cell.topological_dimension()),
        "extruded_arg": ", int nlayers" if extruded else "",
        "nlayers": ", f->n_layers" if extruded else "",
    }

    evaluate_template_c = """#include <math.h>
#include <firedrake_geometry.h>

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
                                            double *coords, int *coords_map%(extruded_arg)s, int cell);

int to_reference_coords(void *result_, struct Function *f, int cell, double *x)
{
    int return_value;
    wrap_to_reference_coords(result_, x, &return_value, f->coords, f->coords_map%(nlayers)s, cell);
    return return_value;
}
"""

    return evaluate_template_c % code
