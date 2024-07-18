import numpy
import string
from fractions import Fraction
from pyop2 import op2
from firedrake.utils import IntType, as_cstr, complex_mode, ScalarType
from firedrake.functionspacedata import entity_dofs_key
from firedrake.functionspaceimpl import FiredrakeDualSpace
import firedrake
from firedrake.mg import utils

from ufl.algorithms.analysis import extract_arguments, extract_coefficients
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.corealg.map_dag import map_expr_dags
from ufl.domain import extract_unique_domain

import loopy as lp
import pymbolic as pym

import gem
import gem.impero_utils as impero_utils

import ufl
import finat.ufl
import tsfc

import tsfc.kernel_interface.firedrake_loopy as firedrake_interface

from tsfc.loopy import generate as generate_loopy
from tsfc import fem, ufl_utils, spectral
from tsfc.driver import TSFCIntegralDataInfo
from tsfc.kernel_interface.common import lower_integral_type
from tsfc.parameters import default_parameters
from tsfc.finatinterface import create_element
from finat.quadrature import make_quadrature
from firedrake.pointquery_utils import dX_norm_square, X_isub_dX, init_X, inside_check, is_affine, celldist_l1_c_expr
from firedrake.pointquery_utils import to_reference_coords_newton_step as to_reference_coords_newton_step_body


def to_reference_coordinates(ufl_coordinate_element, parameters=None):
    if parameters is None:
        parameters = tsfc.default_parameters()
    else:
        _ = tsfc.default_parameters()
        _.update(parameters)
        parameters = _

    # Create FInAT element
    element = tsfc.finatinterface.create_element(ufl_coordinate_element)

    cell = ufl_coordinate_element.cell

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "to_reference_coords_newton_step": to_reference_coords_newton_step_body(ufl_coordinate_element, parameters, x0_dtype=ScalarType, dX_dtype="double"),
        "init_X": init_X(element.cell, parameters),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(cell.topological_dimension()),
        "X_isub_dX": X_isub_dX(cell.topological_dimension()),
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """#include <math.h>
#include <stdio.h>
#include <petsc.h>

%(to_reference_coords_newton_step)s

static inline void to_reference_coords_kernel(PetscScalar *X, const PetscScalar *x0, const PetscScalar *C)
{
    const int space_dim = %(geometric_dimension)d;

    /*
     * Mapping coordinates from physical to reference space
     */

%(init_X)s

    int converged = 0;
    for (int it = 0; !converged && it < %(max_iteration_count)d; it++) {
        double dX[%(topological_dimension)d] = { 0.0 };
        to_reference_coords_newton_step(C, x0, X, dX);

        if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
            converged = 1;
        }

%(X_isub_dX)s
    }
}"""

    return evaluate_template_c % code


def compile_element(expression, dual_space=None, parameters=None,
                    name="evaluate"):
    """Generate code for point evaluations.

    :arg expression: A UFL expression (may contain up to one coefficient, or one argument)
    :arg dual_space: if the expression has an argument, should we also distribute residual data?
    :returns: The generated code (:class:`loopy.TranslationUnit`)
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    expression = tsfc.ufl_utils.preprocess_expression(expression, complex_mode=complex_mode)

    # # Collect required coefficients

    try:
        arg, = extract_coefficients(expression)
        argument_multiindices = ()
        coefficient = True
        if expression.ufl_shape:
            tensor_indices = tuple(gem.Index() for s in expression.ufl_shape)
        else:
            tensor_indices = ()
    except ValueError:
        arg, = extract_arguments(expression)
        finat_elem = create_element(arg.ufl_element())
        argument_multiindices = (finat_elem.get_indices(), )
        argument_multiindex, = argument_multiindices
        value_shape = finat_elem.value_shape
        if value_shape:
            tensor_indices = argument_multiindex[-len(value_shape):]
        else:
            tensor_indices = ()
        coefficient = False

    # Replace coordinates (if any)
    builder = firedrake_interface.KernelBuilderBase(scalar_type=ScalarType)
    domain = extract_unique_domain(expression)
    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = lp.GlobalArg("X", dtype=ScalarType, shape=(dim,))

    config = dict(interface=builder,
                  ufl_cell=cell,
                  point_indices=(),
                  point_expr=point,
                  argument_multiindices=argument_multiindices,
                  scalar_type=parameters["scalar_type"])
    context = tsfc.fem.GemPointContext(**config)

    # Abs-simplification
    expression = tsfc.ufl_utils.simplify_abs(expression, complex_mode)

    # Translate UFL -> GEM
    if coefficient:
        assert dual_space is None
        builder._coefficient(arg, "f")
        f_arg = [builder.generate_arg_from_expression(builder.coefficient_map[arg])]
    else:
        f_arg = []
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    b_arg = []
    if coefficient:
        if expression.ufl_shape:
            return_variable = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
            result_arg = lp.GlobalArg("R", dtype=ScalarType, shape=expression.ufl_shape)
            result = gem.Indexed(result, tensor_indices)
        else:
            return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
            result_arg = lp.GlobalArg("R", dtype=ScalarType, shape=(1,))

    else:
        return_variable = gem.Indexed(gem.Variable('R', finat_elem.index_shape), argument_multiindex)
        result = gem.Indexed(result, tensor_indices)
        if dual_space:
            elem = create_element(dual_space.ufl_element())
            if elem.value_shape:
                var = gem.Indexed(gem.Variable("b", elem.value_shape),
                                  tensor_indices)
                b_arg = [lp.GlobalArg("b", dtype=ScalarType, shape=elem.value_shape)]
            else:
                var = gem.Indexed(gem.Variable("b", (1, )), (0, ))
                b_arg = [lp.GlobalArg("b", dtype=ScalarType, shape=(1,))]
            result = gem.Product(result, var)

        result_arg = lp.GlobalArg("R", dtype=ScalarType, shape=finat_elem.index_shape)

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> loopy
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)

    loopy_args = [result_arg] + b_arg + f_arg + [point_arg]
    kernel_code, _ = generate_loopy(
        impero_c, loopy_args, ScalarType,
        kernel_name="pyop2_kernel_"+name, index_names={})

    return lp.generate_code_v2(kernel_code).device_code()


def prolong_kernel(expression):
    meshc = extract_unique_domain(expression)
    hierarchy, level = utils.get_level(extract_unique_domain(expression))
    levelf = level + Fraction(1, hierarchy.refinements_per_level)
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = extract_unique_domain(expression).coordinates
    if meshc.cell_set._extruded:
        idx = levelf * hierarchy.refinements_per_level
        assert idx == int(idx)
        assert hierarchy._meshes[int(idx)].cell_set._extruded
    V = expression.function_space()
    key = (("prolong",)
           + expression.ufl_element().value_shape
           + entity_dofs_key(V.finat_element.complex.get_topology())
           + entity_dofs_key(V.finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        mesh = extract_unique_domain(coordinates)
        eval_code = compile_element(expression)
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())
        element = create_element(expression.ufl_element())
        coords_element = create_element(coordinates.ufl_element())

        my_kernel = """#include <petsc.h>
        %(to_reference)s
        %(evaluate)s
        __attribute__((noinline)) /* Clang bug */
        static void pyop2_kernel_prolong(PetscScalar *R, PetscScalar *f, const PetscScalar *X, const PetscScalar *Xc)
        {
            PetscScalar Xref[%(tdim)d];
            int cell = -1;
            int bestcell = -1;
            double bestdist = 1e10;
            for (int i = 0; i < %(ncandidate)d; i++) {
                const PetscScalar *Xci = Xc + i*%(Xc_cell_inc)d;
                double celldist = 2*bestdist;
                to_reference_coords_kernel(Xref, X, Xci);
                if (%(inside_cell)s) {
                    cell = i;
                    break;
                }

                celldist = %(celldist_l1_c_expr)s;
                if (celldist < bestdist) {
                    bestdist = celldist;
                    bestcell = i;
                }

            }
            if (cell == -1) {
                /* We didn't find a cell that contained this point exactly.
                   Did we find one that was close enough? */
                if (bestdist < 10) {
                    cell = bestcell;
                } else {
                    fprintf(stderr, "Could not identify cell in transfer operator. Point: ");
                    for (int coord = 0; coord < %(spacedim)s; coord++) {
                      fprintf(stderr, "%%.14e ", X[coord]);
                    }
                    fprintf(stderr, "\\n");
                    fprintf(stderr, "Number of candidates: %%d. Best distance located: %%14e", %(ncandidate)d, bestdist);
                    abort();
                }
            }
            const PetscScalar *coarsei = f + cell*%(coarse_cell_inc)d;
            for ( int i = 0; i < %(Rdim)d; i++ ) {
                R[i] = 0;
            }
            pyop2_kernel_evaluate(R, coarsei, Xref);
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": eval_code,
               "spacedim": element.cell.get_spatial_dimension(),
               "ncandidate": hierarchy.fine_to_coarse_cells[levelf].shape[1],
               "Rdim": numpy.prod(element.value_shape),
               "inside_cell": inside_check(element.cell, eps=1e-8, X="Xref"),
               "celldist_l1_c_expr": celldist_l1_c_expr(element.cell, X="Xref"),
               "Xc_cell_inc": coords_element.space_dimension(),
               "coarse_cell_inc": element.space_dimension(),
               "tdim": mesh.topological_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="pyop2_kernel_prolong"))


def restrict_kernel(Vf, Vc):
    hierarchy, level = utils.get_level(Vc.mesh())
    levelf = level + Fraction(1, hierarchy.refinements_per_level)
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = Vc.mesh().coordinates
    if Vf.extruded:
        assert Vc.extruded
    key = (("restrict",)
           + Vf.ufl_element().value_shape
           + entity_dofs_key(Vf.finat_element.complex.get_topology())
           + entity_dofs_key(Vc.finat_element.complex.get_topology())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        assert isinstance(Vc, FiredrakeDualSpace) and isinstance(Vf, FiredrakeDualSpace)
        mesh = extract_unique_domain(coordinates)
        evaluate_code = compile_element(firedrake.TestFunction(Vc.dual()), Vf.dual())
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())
        coords_element = create_element(coordinates.ufl_element())
        element = create_element(Vc.ufl_element())

        my_kernel = """#include <petsc.h>
        %(to_reference)s
        %(evaluate)s

        __attribute__((noinline)) /* Clang bug */
        static void pyop2_kernel_restrict(PetscScalar *R, PetscScalar *b, const PetscScalar *X, const PetscScalar *Xc)
        {
            PetscScalar Xref[%(tdim)d];
            int cell = -1;
            int bestcell = -1;
            double bestdist = 1e10;
            for (int i = 0; i < %(ncandidate)d; i++) {
                const PetscScalar *Xci = Xc + i*%(Xc_cell_inc)d;
                double celldist = 2*bestdist;
                to_reference_coords_kernel(Xref, X, Xci);
                if (%(inside_cell)s) {
                    cell = i;
                    break;
                }

                celldist = %(celldist_l1_c_expr)s;
                /* fprintf(stderr, "cell %%d celldist: %%.14e\\n", i, celldist);
                fprintf(stderr, "Xref: %%.14e %%.14e %%.14e\\n", Xref[0], Xref[1], Xref[2]); */
                if (celldist < bestdist) {
                    bestdist = celldist;
                    bestcell = i;
                }
            }
            if (cell == -1) {
                /* We didn't find a cell that contained this point exactly.
                   Did we find one that was close enough? */
                if (bestdist < 10) {
                    cell = bestcell;
                } else {
                    fprintf(stderr, "Could not identify cell in transfer operator. Point: ");
                    for (int coord = 0; coord < %(spacedim)s; coord++) {
                      fprintf(stderr, "%%.14e ", X[coord]);
                    }
                    fprintf(stderr, "\\n");
                    fprintf(stderr, "Number of candidates: %%d. Best distance located: %%14e", %(ncandidate)d, bestdist);
                    abort();
                }
            }

            {
            const PetscScalar *Ri = R + cell*%(coarse_cell_inc)d;
            pyop2_kernel_evaluate(Ri, b, Xref);
            }
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": evaluate_code,
               "ncandidate": hierarchy.fine_to_coarse_cells[levelf].shape[1],
               "inside_cell": inside_check(element.cell, eps=1e-8, X="Xref"),
               "celldist_l1_c_expr": celldist_l1_c_expr(element.cell, X="Xref"),
               "Xc_cell_inc": coords_element.space_dimension(),
               "coarse_cell_inc": element.space_dimension(),
               "spacedim": element.cell.get_spatial_dimension(),
               "tdim": mesh.topological_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="pyop2_kernel_restrict"))


def inject_kernel(Vf, Vc):
    hierarchy, level = utils.get_level(Vc.mesh())
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = Vf.mesh().coordinates
    if Vf.extruded:
        assert Vc.extruded
        level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
    else:
        level_ratio = 1
    key = (("inject", level_ratio)
           + Vf.ufl_element().value_shape
           + entity_dofs_key(Vc.finat_element.complex.get_topology())
           + entity_dofs_key(Vf.finat_element.complex.get_topology())
           + entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + entity_dofs_key(Vc.mesh().coordinates.function_space().finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        ncandidate = hierarchy.coarse_to_fine_cells[level].shape[1] * level_ratio
        if Vc.finat_element.entity_dofs() == Vc.finat_element.entity_closure_dofs():
            return cache.setdefault(key, (dg_injection_kernel(Vf, Vc, ncandidate), True))

        coordinates = Vf.mesh().coordinates
        evaluate_code = compile_element(ufl.Coefficient(Vf))
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())

        coords_element = create_element(coordinates.ufl_element())
        Vf_element = create_element(Vf.ufl_element())
        kernel = """
        %(to_reference)s
        %(evaluate)s

        __attribute__((noinline)) /* Clang bug */
        static void pyop2_kernel_inject(PetscScalar *R, const PetscScalar *X, const PetscScalar *f, const PetscScalar *Xf)
        {
            PetscScalar Xref[%(tdim)d];
            int cell = -1;
            int bestcell = -1;
            double bestdist = 1e10;
            for (int i = 0; i < %(ncandidate)d; i++) {
                const PetscScalar *Xfi = Xf + i*%(Xf_cell_inc)d;
                double celldist = 2*bestdist;
                to_reference_coords_kernel(Xref, X, Xfi);
                if (%(inside_cell)s) {
                    cell = i;
                    break;
                }

                celldist = %(celldist_l1_c_expr)s;
                if (celldist < bestdist) {
                    bestdist = celldist;
                    bestcell = i;
                }
            }
            if (cell == -1) {
                /* We didn't find a cell that contained this point exactly.
                   Did we find one that was close enough? */
                if (bestdist < 10) {
                    cell = bestcell;
                } else {
                    fprintf(stderr, "Could not identify cell in transfer operator. Point: ");
                    for (int coord = 0; coord < %(spacedim)s; coord++) {
                      fprintf(stderr, "%%.14e ", X[coord]);
                    }
                    fprintf(stderr, "\\n");
                    fprintf(stderr, "Number of candidates: %%d. Best distance located: %%14e", %(ncandidate)d, bestdist);
                    abort();
                }
            }
            const PetscScalar *fi = f + cell*%(f_cell_inc)d;
            for ( int i = 0; i < %(Rdim)d; i++ ) {
                R[i] = 0;
            }
            pyop2_kernel_evaluate(R, fi, Xref);
        }
        """ % {
            "to_reference": str(to_reference_kernel),
            "evaluate": evaluate_code,
            "inside_cell": inside_check(Vc.finat_element.cell, eps=1e-8, X="Xref"),
            "spacedim": Vc.finat_element.cell.get_spatial_dimension(),
            "celldist_l1_c_expr": celldist_l1_c_expr(Vc.finat_element.cell, X="Xref"),
            "tdim": Vc.mesh().topological_dimension(),
            "ncandidate": ncandidate,
            "Rdim": numpy.prod(Vf_element.value_shape),
            "Xf_cell_inc": coords_element.space_dimension(),
            "f_cell_inc": Vf_element.space_dimension()
        }
        return cache.setdefault(key, (op2.Kernel(kernel, name="pyop2_kernel_inject"), False))


class MacroKernelBuilder(firedrake_interface.KernelBuilderBase):
    """Kernel builder for integration on a macro-cell."""

    oriented = False

    def __init__(self, scalar_type, num_entities):
        """:arg num_entities: the number of micro-entities to integrate over."""
        super().__init__(scalar_type)
        self.indices = (gem.Index("entity", extent=num_entities), )
        self.shape = tuple(i.extent for i in self.indices)
        self.unsummed_coefficient_indices = frozenset(self.indices)

    def set_coefficients(self, coefficients):
        self.coefficients = []
        self.coefficient_split = {}
        self.kernel_args = []
        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) == finat.ufl.MixedElement:
                raise NotImplementedError("Sorry, not for mixed.")
            self.coefficients.append(coefficient)
            self.kernel_args.append(self._coefficient(coefficient, "macro_w_%d" % (i, )))

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.AbstractDomain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self._coefficient(f, "macro_coords")

    def _coefficient(self, coefficient, name):
        element = create_element(coefficient.ufl_element())
        shape = self.shape + element.index_shape
        size = numpy.prod(shape, dtype=int)
        funarg = lp.GlobalArg(name, dtype=ScalarType, shape=(size,))
        expression = gem.reshape(gem.Variable(name, (size, )), shape)
        expression = gem.partial_indexed(expression, self.indices)
        self.coefficient_map[coefficient] = expression
        return funarg


def dg_injection_kernel(Vf, Vc, ncell):
    from firedrake import Tensor, AssembledVector, TestFunction, TrialFunction
    from firedrake.slate.slac import compile_expression
    if complex_mode:
        raise NotImplementedError("In complex mode we are waiting for Slate")
    macro_builder = MacroKernelBuilder(ScalarType, ncell)
    f = ufl.Coefficient(Vf)
    macro_builder.set_coefficients([f])
    macro_builder.set_coordinates(Vf.mesh())

    Vfe = create_element(Vf.ufl_element())
    ref_complex = Vfe.complex
    variant = Vf.ufl_element().variant() or "default"
    if "alfeld" in variant.lower():
        from FIAT import macro
        ref_complex = macro.PowellSabinSplit(Vfe.cell)

    macro_quadrature_rule = make_quadrature(ref_complex, estimate_total_polynomial_degree(ufl.inner(f, f)))
    index_cache = {}
    parameters = default_parameters()
    integration_dim, entity_ids = lower_integral_type(Vfe.cell, "cell")
    macro_cfg = dict(interface=macro_builder,
                     ufl_cell=Vf.ufl_cell(),
                     integration_dim=integration_dim,
                     entity_ids=entity_ids,
                     index_cache=index_cache,
                     quadrature_rule=macro_quadrature_rule,
                     scalar_type=parameters["scalar_type"])

    macro_context = fem.PointSetContext(**macro_cfg)
    fexpr, = fem.compile_ufl(f, macro_context)
    X = ufl.SpatialCoordinate(Vf.mesh())
    C_a, = fem.compile_ufl(X, macro_context)
    detJ = ufl_utils.preprocess_expression(abs(ufl.JacobianDeterminant(extract_unique_domain(f))),
                                           complex_mode=complex_mode)
    macro_detJ, = fem.compile_ufl(detJ, macro_context)

    Vce = create_element(Vc.ufl_element())

    info = TSFCIntegralDataInfo(domain=Vc.mesh(),
                                integral_type="cell",
                                subdomain_id=("otherwise",),
                                domain_number=0,
                                arguments=(ufl.TestFunction(Vc), ),
                                coefficients=(),
                                coefficient_numbers=())

    coarse_builder = firedrake_interface.KernelBuilder(info, parameters["scalar_type"])
    coarse_builder.set_coordinates(Vc.mesh())
    argument_multiindices = coarse_builder.argument_multiindices
    argument_multiindex, = argument_multiindices
    return_variable, = coarse_builder.return_variables

    integration_dim, entity_ids = lower_integral_type(Vce.cell, "cell")
    # Midpoint quadrature for jacobian on coarse cell.
    quadrature_rule = make_quadrature(Vce.cell, 0)

    coarse_cfg = dict(interface=coarse_builder,
                      ufl_cell=Vc.ufl_cell(),
                      integration_dim=integration_dim,
                      entity_ids=entity_ids,
                      index_cache=index_cache,
                      quadrature_rule=quadrature_rule,
                      scalar_type=parameters["scalar_type"])

    X = ufl.SpatialCoordinate(Vc.mesh())
    K = ufl_utils.preprocess_expression(ufl.JacobianInverse(Vc.mesh()),
                                        complex_mode=complex_mode)
    coarse_context = fem.PointSetContext(**coarse_cfg)
    C_0, = fem.compile_ufl(X, coarse_context)
    K, = fem.compile_ufl(K, coarse_context)

    i = gem.Index()
    j = gem.Index()

    C_0 = gem.Indexed(C_0, (j, ))
    C_0 = gem.index_sum(C_0, quadrature_rule.point_set.indices)
    C_a = gem.Indexed(C_a, (j, ))
    X_a = gem.Sum(C_0, gem.Product(gem.Literal(-1), C_a))

    K_ij = gem.Indexed(K, (i, j))
    K_ij = gem.index_sum(K_ij, quadrature_rule.point_set.indices)
    X_a = gem.index_sum(gem.Product(K_ij, X_a), (j, ))
    C_0, = quadrature_rule.point_set.points
    C_0 = gem.Indexed(gem.Literal(C_0), (i, ))
    # fine quad points in coarse reference space.
    X_a = gem.Sum(C_0, gem.Product(gem.Literal(-1), X_a))
    X_a = gem.ComponentTensor(X_a, (i, ))

    # Coarse basis function evaluated at fine quadrature points
    phi_c = fem.fiat_to_ufl(Vce.point_evaluation(0, X_a, (Vce.cell.get_dimension(), 0)), 0)

    tensor_indices = tuple(gem.Index(extent=d) for d in f.ufl_shape)

    phi_c = gem.Indexed(phi_c, argument_multiindex + tensor_indices)
    fexpr = gem.Indexed(fexpr, tensor_indices)
    quadrature_weight = macro_quadrature_rule.weight_expression
    expr = gem.Product(gem.IndexSum(gem.Product(phi_c, fexpr), tensor_indices),
                       gem.Product(macro_detJ, quadrature_weight))

    quadrature_indices = macro_builder.indices + macro_quadrature_rule.point_set.indices

    reps = spectral.Integrals([expr], quadrature_indices, argument_multiindices, parameters)
    assignments = spectral.flatten([(return_variable, reps)], index_cache)
    return_variables, expressions = zip(*assignments)
    expressions = impero_utils.preprocess_gem(expressions, **spectral.finalise_options)
    assignments = list(zip(return_variables, expressions))
    impero_c = impero_utils.compile_gem(assignments, quadrature_indices + argument_multiindex,
                                        remove_zeros=True)

    index_names = []

    def name_index(index, name):
        index_names.append((index, name))
        if index in index_cache:
            for multiindex, suffix in zip(index_cache[index],
                                          string.ascii_lowercase):
                name_multiindex(multiindex, name + suffix)

    def name_multiindex(multiindex, name):
        if len(multiindex) == 1:
            name_index(multiindex[0], name)
        else:
            for i, index in enumerate(multiindex):
                name_index(index, name + str(i))

    name_multiindex(quadrature_indices, 'ip')
    for multiindex, name in zip(argument_multiindices, ['j', 'k']):
        name_multiindex(multiindex, name)

    index_names.extend(zip(macro_builder.indices, ["entity"]))

    # now construct the outermost kernel
    domains = []
    instructions = []
    kernel_data = []
    subkernels = []
    depends_on = frozenset()
    local_tensor = coarse_builder.generate_arg_from_expression(coarse_builder.return_variables)

    # 1. Zero the local tensor
    iname = "i0"
    domains.append(f"{{ [{iname}]: 0 <= {iname} < {Vce.space_dimension()} }}")
    instructions.append(
        lp.Assignment(
            pym.subscript(pym.var(local_tensor.name), (pym.var(iname),)), 0,
            within_inames=frozenset({iname}), id="zero", depends_on=depends_on))
    kernel_data.append(
        lp.TemporaryVariable(local_tensor.name, shape=local_tensor.shape, dtype=local_tensor.dtype))
    depends_on |= {"zero"}

    # 2. Fill the local tensor
    macro_coordinates_arg = macro_builder.generate_arg_from_expression(
        macro_builder.coefficient_map[macro_builder.domain_coordinate[Vf.mesh()]])
    coarse_coordinates_arg = coarse_builder.generate_arg_from_expression(
        coarse_builder.coefficient_map[coarse_builder.domain_coordinate[Vc.mesh()]])
    eval_args = [
        lp.GlobalArg(
            local_tensor.name, dtype=local_tensor.dtype, shape=local_tensor.shape,
            is_input=True, is_output=True),
        *macro_builder.kernel_args,
        macro_coordinates_arg,
        coarse_coordinates_arg,
    ]
    eval_kernel, _ = generate_loopy(
        impero_c, eval_args,
        ScalarType, kernel_name="pyop2_kernel_evaluate", index_names=index_names)
    subkernels.append(eval_kernel)

    fill_insn, extra_domains = _generate_call_insn(
        "pyop2_kernel_evaluate", eval_args, iname_prefix="fill", id="fill",
        depends_on=depends_on, within_inames_is_final=True)
    instructions.append(fill_insn)
    domains.extend(extra_domains)
    depends_on |= {fill_insn.id}

    # 3. Now we have the kernel that computes <f, phi_c>dx_c.
    # So now we need to hit it with the inverse mass matrix on dx_c
    retarg = lp.GlobalArg(
        "R", dtype=ScalarType, shape=local_tensor.shape, is_output=True)

    kernel_data = [
        retarg, *macro_builder.kernel_args, macro_coordinates_arg,
        coarse_coordinates_arg, *kernel_data]

    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    expr = Tensor(ufl.inner(u, v)*ufl.dx).inv * AssembledVector(ufl.Coefficient(Vc))
    Ainv, = compile_expression(expr)
    Ainv = Ainv.kinfo.kernel
    subkernels.append(Ainv.code)

    eval_args = [retarg, coarse_coordinates_arg, local_tensor]
    inv_insn, extra_domains = _generate_call_insn(
        Ainv.name, eval_args, iname_prefix="inv", id="inv",
        depends_on=depends_on, within_inames_is_final=True)
    instructions.append(inv_insn)
    domains.extend(extra_domains)
    depends_on |= {inv_insn.id}

    kernel_name = "pyop2_kernel_injection_dg"
    kernel = lp.make_kernel(
        domains, instructions, kernel_data, name=kernel_name,
        target=tsfc.parameters.target, lang_version=(2018, 2))
    kernel = lp.merge([kernel, *subkernels])
    return op2.Kernel(
        kernel, name=kernel_name, include_dirs=Ainv.include_dirs,
        headers=Ainv.headers, events=Ainv.events)


def _generate_call_insn(name, args, *, iname_prefix=None, **kwargs):
    """Create an appropriate loopy call instruction from its arguments.

    This function is useful because :class:`loopy.CallInstruction` are a
    faff to build since each argument needs to be wrapped in a
    :class:`loopy.symbolic.SubArrayRef` with an associated iname and domain.

    Parameters
    ----------
    name : str
        The name of the kernel to be called.
    args : iterable of loopy.ArrayArg
        The arguments used to construct the callee kernel. These must have
        vector shape.
    iname_prefix : str, optional
        Prefix to the autogenerated inames, defaults to ``name``.
    kwargs
        All other keyword arguments are passed to the
        :class:`loopy.CallInstruction` constructor.

    Returns
    -------
    insn : loopy.CallInstruction
        The generated call instruction.
    extra_domains
        Iterable of extra loop domains that must be added to the caller kernel.

    """
    if not iname_prefix:
        iname_prefix = name

    domains = []
    assignees = []
    parameters = []
    swept_iname_counter = 0
    for arg in args:
        try:
            shape, = arg.shape
        except ValueError:
            raise NotImplementedError("Expecting vector-shaped arguments")

        swept_iname = f"{iname_prefix}_i{swept_iname_counter}"
        swept_iname_counter += 1
        domains.append(f"{{ [{swept_iname}]: 0 <= {swept_iname} < {shape} }}")
        swept_index = (pym.var(swept_iname),)
        param = lp.symbolic.SubArrayRef(
            swept_index, pym.subscript(pym.var(arg.name), swept_index))
        parameters.append(param)
        if arg.is_output:
            assignees.append(param)
    assignees = tuple(assignees)
    parameters = tuple(parameters)
    expression = pym.primitives.Call(pym.var(name), parameters)
    return lp.CallInstruction(assignees, expression, **kwargs), domains
