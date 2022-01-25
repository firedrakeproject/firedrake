import numpy
import string
from fractions import Fraction
from pyop2 import op2
from firedrake.utils import IntType, as_cstr, complex_mode, ScalarType_c, ScalarType
from firedrake.functionspacedata import entity_dofs_key
import firedrake
from firedrake.mg import utils

from ufl.algorithms.analysis import extract_arguments, extract_coefficients
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.corealg.map_dag import map_expr_dags

import coffee.base as ast

import gem
import gem.impero_utils as impero_utils

import ufl
import tsfc

import tsfc.kernel_interface.firedrake as firedrake_interface

from tsfc.coffee import generate as generate_coffee
from tsfc import fem, ufl_utils, spectral
from tsfc.driver import lower_integral_type
from tsfc.parameters import default_parameters
from tsfc.finatinterface import create_element
from finat.quadrature import make_quadrature
from firedrake.pointquery_utils import dX_norm_square, X_isub_dX, init_X, inside_check, is_affine, compute_celldist
from firedrake.pointquery_utils import to_reference_coordinates as to_reference_coordinates_body


def to_reference_coordinates(ufl_coordinate_element, parameters=None):
    if parameters is None:
        parameters = tsfc.default_parameters()
    else:
        _ = tsfc.default_parameters()
        _.update(parameters)
        parameters = _

    # Create FInAT element
    element = tsfc.finatinterface.create_element(ufl_coordinate_element)

    cell = ufl_coordinate_element.cell()

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "to_reference_coords": to_reference_coordinates_body(ufl_coordinate_element, parameters),
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

static inline void to_reference_coords_kernel(PetscScalar *X, const PetscScalar *x0, const PetscScalar *C)
{
    const int space_dim = %(geometric_dimension)d;

    /*
     * Mapping coordinates from physical to reference space
     */

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
}"""

    return evaluate_template_c % code


def compile_element(expression, dual_space=None, parameters=None,
                    name="evaluate"):
    """Generate code for point evaluations.

    :arg expression: A UFL expression (may contain up to one coefficient, or one argument)
    :arg dual_space: if the expression has an argument, should we also distribute residual data?
    :returns: Some coffee AST
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
    builder = firedrake_interface.KernelBuilderBase(scalar_type=ScalarType_c)
    domain = expression.ufl_domain()
    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = ast.Decl(ScalarType_c, ast.Symbol('X', rank=(dim,)))

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
        f_arg = [builder._coefficient(arg, "f")]
    else:
        f_arg = []
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    b_arg = []
    if coefficient:
        if expression.ufl_shape:
            return_variable = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
            result_arg = ast.Decl(ScalarType_c, ast.Symbol('R', rank=expression.ufl_shape))
            result = gem.Indexed(result, tensor_indices)
        else:
            return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
            result_arg = ast.Decl(ScalarType_c, ast.Symbol('R', rank=(1,)))

    else:
        return_variable = gem.Indexed(gem.Variable('R', finat_elem.index_shape), argument_multiindex)
        result = gem.Indexed(result, tensor_indices)
        if dual_space:
            elem = create_element(dual_space.ufl_element())
            if elem.value_shape:
                var = gem.Indexed(gem.Variable("b", elem.value_shape),
                                  tensor_indices)
                b_arg = [ast.Decl(ScalarType_c, ast.Symbol("b", rank=elem.value_shape))]
            else:
                var = gem.Indexed(gem.Variable("b", (1, )), (0, ))
                b_arg = [ast.Decl(ScalarType_c, ast.Symbol("b", rank=(1, )))]
            result = gem.Product(result, var)

        result_arg = ast.Decl(ScalarType_c, ast.Symbol('R', rank=finat_elem.index_shape))

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> COFFEE
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)
    body = generate_coffee(impero_c, {}, ScalarType)

    # Build kernel tuple
    kernel_code = builder.construct_kernel("pyop2_kernel_" + name, [result_arg] + b_arg + f_arg + [point_arg], body)

    return kernel_code


def prolong_kernel(expression):
    meshc = expression.ufl_domain()
    hierarchy, level = utils.get_level(expression.ufl_domain())
    levelf = level + Fraction(1 / hierarchy.refinements_per_level)
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = expression.ufl_domain().coordinates
    if meshc.cell_set._extruded:
        idx = levelf * hierarchy.refinements_per_level
        assert idx == int(idx)
        level_ratio = (hierarchy._meshes[int(idx)].layers - 1) // (meshc.layers - 1)
    else:
        level_ratio = 1
    key = (("prolong", level_ratio)
           + expression.ufl_element().value_shape()
           + entity_dofs_key(expression.function_space().finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        mesh = coordinates.ufl_domain()
        evaluate_kernel = compile_element(expression)
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())
        element = create_element(expression.ufl_element())
        eval_args = evaluate_kernel.args[:-1]
        coords_element = create_element(coordinates.ufl_element())

        args = eval_args[-1].gencode(not_scope=True)
        R, coarse = (a.sym.symbol for a in eval_args)
        my_kernel = """#include <petsc.h>
        %(to_reference)s
        %(evaluate)s
        __attribute__((noinline)) /* Clang bug */
        static void pyop2_kernel_prolong(PetscScalar *R, %(args)s, const PetscScalar *X, const PetscScalar *Xc)
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

                %(compute_celldist)s
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
            const PetscScalar *coarsei = %(coarse)s + cell*%(coarse_cell_inc)d;
            for ( int i = 0; i < %(Rdim)d; i++ ) {
                %(R)s[i] = 0;
            }
            pyop2_kernel_evaluate(%(R)s, coarsei, Xref);
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": str(evaluate_kernel),
               "args": args,
               "R": R,
               "spacedim": element.cell.get_spatial_dimension(),
               "coarse": coarse,
               "ncandidate": hierarchy.fine_to_coarse_cells[levelf].shape[1] * level_ratio,
               "Rdim": numpy.prod(element.value_shape),
               "inside_cell": inside_check(element.cell, eps=1e-8, X="Xref"),
               "compute_celldist": compute_celldist(element.cell, X="Xref", celldist="celldist"),
               "Xc_cell_inc": coords_element.space_dimension(),
               "coarse_cell_inc": element.space_dimension(),
               "tdim": mesh.topological_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="pyop2_kernel_prolong"))


def restrict_kernel(Vf, Vc):
    hierarchy, level = utils.get_level(Vc.ufl_domain())
    levelf = level + Fraction(1 / hierarchy.refinements_per_level)
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = Vc.ufl_domain().coordinates
    if Vf.extruded:
        assert Vc.extruded
        level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
    else:
        level_ratio = 1
    key = (("restrict", level_ratio)
           + Vf.ufl_element().value_shape()
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        mesh = coordinates.ufl_domain()
        evaluate_kernel = compile_element(firedrake.TestFunction(Vc), Vf)
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())
        coords_element = create_element(coordinates.ufl_element())
        element = create_element(Vc.ufl_element())
        eval_args = evaluate_kernel.args[:-1]
        args = eval_args[-1].gencode(not_scope=True)
        R, fine = (a.sym.symbol for a in eval_args)
        my_kernel = """#include <petsc.h>
        %(to_reference)s
        %(evaluate)s

        __attribute__((noinline)) /* Clang bug */
        static void pyop2_kernel_restrict(PetscScalar *R, %(args)s, const PetscScalar *X, const PetscScalar *Xc)
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

                %(compute_celldist)s
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
            const PetscScalar *Ri = %(R)s + cell*%(coarse_cell_inc)d;
            pyop2_kernel_evaluate(Ri, %(fine)s, Xref);
            }
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": str(evaluate_kernel),
               "ncandidate": hierarchy.fine_to_coarse_cells[levelf].shape[1]*level_ratio,
               "inside_cell": inside_check(element.cell, eps=1e-8, X="Xref"),
               "compute_celldist": compute_celldist(element.cell, X="Xref", celldist="celldist"),
               "Xc_cell_inc": coords_element.space_dimension(),
               "coarse_cell_inc": element.space_dimension(),
               "args": args,
               "spacedim": element.cell.get_spatial_dimension(),
               "R": R,
               "fine": fine,
               "tdim": mesh.topological_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="pyop2_kernel_restrict"))


def inject_kernel(Vf, Vc):
    hierarchy, level = utils.get_level(Vc.ufl_domain())
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = Vf.ufl_domain().coordinates
    if Vf.extruded:
        assert Vc.extruded
        level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
    else:
        level_ratio = 1
    key = (("inject", level_ratio)
           + Vf.ufl_element().value_shape()
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

        coordinates = Vf.ufl_domain().coordinates
        evaluate_kernel = compile_element(ufl.Coefficient(Vf))
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

                %(compute_celldist)s
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
            "evaluate": str(evaluate_kernel),
            "inside_cell": inside_check(Vc.finat_element.cell, eps=1e-8, X="Xref"),
            "spacedim": Vc.finat_element.cell.get_spatial_dimension(),
            "compute_celldist": compute_celldist(Vc.finat_element.cell, X="Xref", celldist="celldist"),
            "tdim": Vc.ufl_domain().topological_dimension(),
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
            if type(coefficient.ufl_element()) == ufl.MixedElement:
                raise NotImplementedError("Sorry, not for mixed.")
            self.coefficients.append(coefficient)
            self.kernel_args.append(self._coefficient(coefficient, "macro_w_%d" % (i, )))

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self.coordinates_arg = self._coefficient(f, "macro_coords")

    def _coefficient(self, coefficient, name):
        element = create_element(coefficient.ufl_element())
        shape = self.shape + element.index_shape
        size = numpy.prod(shape, dtype=int)
        funarg = ast.Decl(ScalarType_c, ast.Symbol(name), pointers=[("restrict", )],
                          qualifiers=["const"])
        expression = gem.reshape(gem.Variable(name, (size, )), shape)
        expression = gem.partial_indexed(expression, self.indices)
        self.coefficient_map[coefficient] = expression
        return funarg


def dg_injection_kernel(Vf, Vc, ncell):
    from firedrake import Tensor, AssembledVector, TestFunction, TrialFunction
    from firedrake.slate.slac import compile_expression
    if complex_mode:
        raise NotImplementedError("In complex mode we are waiting for Slate")
    macro_builder = MacroKernelBuilder(ScalarType_c, ncell)
    f = ufl.Coefficient(Vf)
    macro_builder.set_coefficients([f])
    macro_builder.set_coordinates(Vf.mesh())

    Vfe = create_element(Vf.ufl_element())
    macro_quadrature_rule = make_quadrature(Vfe.cell, estimate_total_polynomial_degree(ufl.inner(f, f)))
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
    detJ = ufl_utils.preprocess_expression(abs(ufl.JacobianDeterminant(f.ufl_domain())),
                                           complex_mode=complex_mode)
    macro_detJ, = fem.compile_ufl(detJ, macro_context)

    Vce = create_element(Vc.ufl_element())

    coarse_builder = firedrake_interface.KernelBuilder("cell", "otherwise", 0, ScalarType_c)
    coarse_builder.set_coordinates(Vc.mesh())
    argument_multiindices = (Vce.get_indices(), )
    argument_multiindex, = argument_multiindices
    return_variable, = coarse_builder.set_arguments((ufl.TestFunction(Vc), ), argument_multiindices)

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
    body = generate_coffee(impero_c, index_names, ScalarType)

    retarg = ast.Decl(ScalarType_c, ast.Symbol("R", rank=(Vce.space_dimension(), )))
    local_tensor = coarse_builder.output_arg.coffee_arg
    local_tensor.init = ast.ArrayInit(numpy.zeros(Vce.space_dimension(), dtype=ScalarType))
    body.children.insert(0, local_tensor)
    args = [retarg] + macro_builder.kernel_args + [macro_builder.coordinates_arg,
                                                   coarse_builder.coordinates_arg.coffee_arg]

    # Now we have the kernel that computes <f, phi_c>dx_c
    # So now we need to hit it with the inverse mass matrix on dx_c

    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    expr = Tensor(ufl.inner(u, v)*ufl.dx).inv * AssembledVector(ufl.Coefficient(Vc))
    Ainv, = compile_expression(expr, coffee=True)
    Ainv = Ainv.kinfo.kernel
    A = ast.Symbol(local_tensor.sym.symbol)
    R = ast.Symbol("R")
    body.children.append(ast.FunCall(Ainv.name, R, coarse_builder.coordinates_arg.coffee_arg.sym, A))
    from coffee.base import Node
    assert isinstance(Ainv.code, Node)
    return op2.Kernel(ast.Node([Ainv.code,
                                ast.FunDecl("void", "pyop2_kernel_injection_dg", args, body,
                                            pred=["static", "inline"])]),
                      name="pyop2_kernel_injection_dg",
                      cpp=True,
                      include_dirs=Ainv.include_dirs,
                      headers=Ainv.headers)
