from os import path
import numpy
import textwrap
import sympy
from sympy.printing.c import ccode
import loopy as lp

import pyop3 as op3
from pyop2 import op2
from pyop2.parloop import generate_single_cell_wrapper

from firedrake.mesh import MeshGeometry
from firedrake.petsc import PETSc
from firedrake.utils import IntType, as_cstr, ScalarType, ScalarType_c, complex_mode, RealType_c, RealType, IntType_c

import ufl
import finat.ufl
from ufl.corealg.map_dag import map_expr_dag

import gem
import gem.impero_utils as impero_utils

import tsfc
import tsfc.kernel_interface.firedrake_loopy as firedrake_interface
import tsfc.ufl_utils as ufl_utils


def src_locate_cell(mesh, tolerance=None):
    src = ['#include <evaluate.h>']
    src.append(compile_coordinate_element(mesh, tolerance))

    shape = numpy.prod(mesh.coordinates.function_space().finat_element.index_shape, dtype=int)
    gdim = mesh.geometric_dimension()

    wrapper_src = textwrap.dedent(f"""\
        #include <complex.h>
        #include <math.h>
        #include <petsc.h>
        #include <stdint.h>
        #include <stdbool.h>

        void wrap_to_reference_coords(void* const farg0, double* const farg1, {RealType_c}* const farg2, int32_t const start, int32_t const end, {ScalarType_c} const *__restrict__ dat0, {IntType_c} const *__restrict__ map0)
        {{
          {ScalarType_c} t0[{shape}*{gdim}];

          for (int32_t i = 0; i < {shape}; ++i)
            for (int32_t j = 0; j < {gdim}; ++j)
              t0[{gdim} * i + j] = dat0[{gdim} * map0[i + {shape} * start] + j];
          to_reference_coords_kernel(farg0, farg1, farg2, &(t0[0]));
        }}"""
    )
    src.append(wrapper_src)

    with open(path.join(path.dirname(__file__), "locate.c")) as f:
        src.append(f.read())

    return "\n".join(src)


def dX_norm_square(topological_dimension):
    return " + ".join("PetscRealPart(dX[{0}])*PetscRealPart(dX[{0}])".format(i)
                      for i in range(topological_dimension))


def X_isub_dX(topological_dimension):
    return "\n".join("\tX[{0}] -= dX[{0}];".format(i)
                     for i in range(topological_dimension))


def is_affine(ufl_element):
    return ufl_element.cell.is_simplex() and ufl_element.degree() <= 1 and ufl_element.family() in ["Discontinuous Lagrange", "Lagrange"]


def inside_check(fiat_cell, eps, X="X"):
    """Generate a C expression which is true if a point is inside a FIAT
    reference cell and false otherwise.

    Parameters
    ----------
    fiat_cell : FIAT.finite_element.FiniteElement
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
    (aka 'manhattan', 'taxicab' or rectilinear distance) to a FIAT reference
    cell.

    Parameters
    ----------
    fiat_cell : FIAT.finite_element.FiniteElement
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
    return "\n".join(f"X[{i}] = {v};" for i, v in enumerate(X))


@PETSc.Log.EventDecorator()
def to_reference_coords_newton_step(ufl_coordinate_element, parameters, x0_dtype="double", dX_dtype=ScalarType):
    # Set up UFL form
    cell = ufl_coordinate_element.cell
    domain = ufl.Mesh(ufl_coordinate_element)
    gdim = domain.geometric_dimension()
    K = ufl.JacobianInverse(domain)
    x = ufl.SpatialCoordinate(domain)
    x0_element = finat.ufl.VectorElement("Real", cell, 0, dim=gdim)
    x0 = ufl.Coefficient(ufl.FunctionSpace(domain, x0_element))
    expr = ufl.dot(K, x - x0)

    # Translation to GEM
    C = ufl.Coefficient(ufl.FunctionSpace(domain, ufl_coordinate_element))
    expr = ufl_utils.preprocess_expression(expr, complex_mode=complex_mode)
    expr = ufl_utils.simplify_abs(expr, complex_mode)

    builder = firedrake_interface.KernelBuilderBase(ScalarType)
    builder.domain_coordinate[domain] = C

    Cexpr = builder._coefficient(C, "C")
    x0_expr = builder._coefficient(x0, "x0")
    loopy_args = [
        lp.GlobalArg(
            "C", dtype=ScalarType, shape=(numpy.prod(Cexpr.shape, dtype=int),)
        ),
        lp.GlobalArg(
            "x0", dtype=x0_dtype, shape=(numpy.prod(x0_expr.shape, dtype=int),)
        ),
    ]

    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    loopy_args.append(lp.GlobalArg("X", dtype=ScalarType, shape=(dim,)))
    context = tsfc.fem.GemPointContext(
        interface=builder,
        ufl_cell=cell,
        integral_type="cell",
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

    # Translate to loopy
    ir = impero_utils.preprocess_gem(ir)
    return_variable = gem.Variable('dX', (dim,))
    loopy_args.append(lp.GlobalArg("dX", dtype=dX_dtype, shape=(dim,)))
    assignments = [(gem.Indexed(return_variable, (i,)), e)
                   for i, e in enumerate(ir)]
    impero_c = impero_utils.compile_gem(assignments, ())
    kernel, _ = tsfc.loopy.generate(
        impero_c, loopy_args, ScalarType,
        kernel_name="to_reference_coords_newton_step")
    return lp.generate_code_v2(kernel).device_code()


@PETSc.Log.EventDecorator()
def compile_coordinate_element(mesh: MeshGeometry, contains_eps: float, parameters: dict | None = None):
    """Generates C code for changing to reference coordinates.

    Parameters
    ----------
    mesh :
        The mesh.

    contains_eps :
        The tolerance used to verify that a point is contained by a cell.
    parameters :
        Form compiler parameters, defaults to whatever TSFC defaults to.

    Returns
    -------
    str
        A string of C code.
    """
    if parameters is None:
        parameters = tsfc.default_parameters()
    else:
        _ = tsfc.default_parameters()
        _.update(parameters)
        parameters = _

    ufl_coordinate_element = mesh.ufl_coordinate_element()
    # Create FInAT element
    element = finat.element_factory.create_element(ufl_coordinate_element)

    code = {
        "geometric_dimension": mesh.geometric_dimension(),
        "topological_dimension": mesh.topological_dimension(),
        "celldist_l1_c_expr": celldist_l1_c_expr(element.cell, "X"),
        "to_reference_coords_newton_step": to_reference_coords_newton_step(ufl_coordinate_element, parameters),
        "init_X": init_X(element.cell, parameters),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(mesh.topological_dimension()),
        "X_isub_dX": X_isub_dX(mesh.topological_dimension()),
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

%(to_reference_coords_newton_step)s

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
        to_reference_coords_newton_step(C, x0, X, dX);

        if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
            converged = 1;
        }

%(X_isub_dX)s
    }

    *cell_dist_l1 = %(celldist_l1_c_expr)s;
}

static inline void wrap_to_reference_coords(
    void* const result_, double* const x, %(RealType)s* const cell_dist_l1, %(IntType)s const start, %(IntType)s const end,
    %(ScalarType)s const *__restrict__ coords, %(IntType)s const *__restrict__ coords_map);

%(RealType)s to_reference_coords(void *result_, struct Function *f, int cell, double *x)
{
    %(RealType)s cell_dist_l1 = 0.0;
    wrap_to_reference_coords(result_, x, &cell_dist_l1, cell, cell+1, f->coords, f->coords_map);
    return cell_dist_l1;
}
"""

    return evaluate_template_c % code
