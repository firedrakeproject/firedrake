import numpy
import string
from pyop2 import op2
from pyop2.utils import as_tuple
from firedrake.utils import IntType, as_cstr, complex_mode, ScalarType
from firedrake.functionspacedata import entity_dofs_key
from firedrake.functionspaceimpl import FiredrakeDualSpace
from firedrake.pointeval_utils import runtime_quadrature_space
from firedrake.mg import utils

from ufl.algorithms import estimate_total_polynomial_degree
from ufl.domain import extract_unique_domain

import loopy as lp
import pymbolic as pym

import gem
import gem.impero_utils as impero_utils

import ufl
import tsfc

import tsfc.kernel_interface.firedrake_loopy as firedrake_interface

from tsfc.loopy import generate as generate_loopy
from tsfc import fem, ufl_utils, spectral
from tsfc.driver import TSFCIntegralDataInfo, compile_expression_dual_evaluation
from tsfc.kernel_interface.common import lower_integral_type
from tsfc.parameters import default_parameters

from finat.ufl import MixedElement
from finat.element_factory import create_element
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
    element = create_element(ufl_coordinate_element)
    gdim, = ufl_coordinate_element.reference_value_shape
    cell = ufl_coordinate_element.cell

    code = {
        "geometric_dimension": gdim,
        "topological_dimension": cell.topological_dimension,
        "to_reference_coords_newton_step": to_reference_coords_newton_step_body(ufl_coordinate_element, parameters, x0_dtype=ScalarType, dX_dtype="double"),
        "init_X": init_X(element.cell, parameters),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(cell.topological_dimension),
        "X_isub_dX": X_isub_dX(cell.topological_dimension),
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


def compile_element(operand, dual_arg, parameters=None,
                    name="evaluate"):
    """Generate code for point evaluations.

    Parameters
    ----------
    operand: ufl.Expr
        A primal expression
    dual_arg: ufl.Coargument | ufl.Cofunction
        A dual argument or coefficient

    Returns
    -------
    str
        The generated code
    """
    source_mesh = extract_unique_domain(operand)
    target_space = dual_arg.arguments()[0].ufl_function_space()

    # Reconstruct the target space as a runtime Quadrature space
    target_space = runtime_quadrature_space(source_mesh, target_space.ufl_element())
    ufl_element = target_space.ufl_element()

    # Reconstruct the dual argument in the runtime Quadrature space
    if isinstance(dual_arg, ufl.Cofunction):
        dual_arg = ufl.Cofunction(target_space.dual())
    else:
        dual_arg = ufl.Coargument(target_space.dual(), number=dual_arg.number())
    expression = ufl.Interpolate(operand, dual_arg)

    # Create a runtime Quadrature element
    to_element = create_element(ufl_element)

    kernel = compile_expression_dual_evaluation(expression,
                                                to_element, ufl_element,
                                                parameters=parameters,
                                                name="pyop2_kernel_"+name)
    return lp.generate_code_v2(kernel.ast).device_code()


def _make_kernel_args(element, *args):
    """Returns a string of argument names to call the kernel.
       Discards coordinate arguments if they do not appear in the kernel."""
    # NOTE: TSFC will sometimes drop run-time arguments in generated
    # kernels if they are deemed not-necessary.
    # For further information, see the same note in interpolation.py.
    mask = [True] * len(args)
    # Drop the source coordinates if the element is affinely-mapped.
    needs_coordinates = element.mapping != "affine"
    mask[1] = needs_coordinates
    # Drop the target location if the element is constant.
    is_constant = sum(as_tuple(element.degree)) == 0 and element.space_dimension() == numpy.prod(element.value_shape)
    mask[-1] = not is_constant
    kernel_args = ", ".join(arg for arg, include in zip(args, mask) if include)
    return kernel_args


def _make_element_key(element):
    """Returns a cache key for a finat element."""
    return entity_dofs_key(element.complex.get_topology()) + entity_dofs_key(element.entity_dofs())


def prolong_kernel(expression, Vf):
    Vc = expression.ufl_function_space()
    hierarchy, levelf = utils.get_level(Vf.mesh())
    hierarchy, levelc = utils.get_level(Vc.mesh())
    if Vc.mesh().extruded:
        assert Vf.mesh().extruded
        level_ratio = (Vc.mesh().layers - 1) // (Vf.mesh().layers - 1)
    else:
        level_ratio = 1
    if levelf > levelc:
        # prolong
        ncandidate = hierarchy.fine_to_coarse_cells[levelf].shape[1]
    else:
        # inject
        ncandidate = hierarchy.coarse_to_fine_cells[levelf].shape[1]
        ncandidate *= level_ratio
    coordinates = Vc.mesh().coordinates
    key = (("prolong", ncandidate)
           + (Vf.block_size,)
           + _make_element_key(Vf.finat_element)
           + _make_element_key(Vc.finat_element)
           + _make_element_key(coordinates.function_space().finat_element))
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    try:
        return cache[key]
    except KeyError:
        evaluate_code = compile_element(expression, ufl.TestFunction(Vf.dual()))
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())
        coords_element = create_element(coordinates.ufl_element())
        element = create_element(expression.ufl_element())

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
                    for (int coord = 0; coord < %(tdim)s; coord++) {
                      fprintf(stderr, "%%.14e ", X[coord]);
                    }
                    fprintf(stderr, "\\n");
                    fprintf(stderr, "Number of candidates: %%d. Best distance located: %%14e", %(ncandidate)d, bestdist);
                    abort();
                }
            }
            const PetscScalar *fi = f + cell*%(coarse_cell_inc)d;
            const PetscScalar *Xci = Xc + cell*%(Xc_cell_inc)d;
            for ( int i = 0; i < %(Rdim)d; i++ ) {
                R[i] = 0;
            }
            pyop2_kernel_evaluate(%(kernel_args)s);
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": evaluate_code,
               "kernel_args": _make_kernel_args(element, "R", "Xci", "fi", "Xref"),
               "ncandidate": ncandidate,
               "Rdim": Vf.block_size,
               "inside_cell": inside_check(element.cell, eps=1e-8, X="Xref"),
               "celldist_l1_c_expr": celldist_l1_c_expr(element.cell, X="Xref"),
               "Xc_cell_inc": coords_element.space_dimension(),
               "coarse_cell_inc": element.space_dimension(),
               "tdim": element.cell.get_spatial_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="pyop2_kernel_prolong"))


def restrict_kernel(Vf, Vc):
    hierarchy, levelf = utils.get_level(Vf.mesh())
    if Vf.mesh().extruded:
        assert Vc.mesh().extruded
    ncandidate = hierarchy.fine_to_coarse_cells[levelf].shape[1]
    coordinates = Vc.mesh().coordinates
    key = (("restrict", ncandidate)
           + (Vf.block_size,)
           + _make_element_key(Vf.finat_element)
           + _make_element_key(Vc.finat_element)
           + _make_element_key(coordinates.function_space().finat_element))
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    try:
        return cache[key]
    except KeyError:
        assert isinstance(Vc, FiredrakeDualSpace) and isinstance(Vf, FiredrakeDualSpace)
        evaluate_code = compile_element(ufl.TestFunction(Vc.dual()), ufl.Cofunction(Vf))
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
                    for (int coord = 0; coord < %(tdim)s; coord++) {
                      fprintf(stderr, "%%.14e ", X[coord]);
                    }
                    fprintf(stderr, "\\n");
                    fprintf(stderr, "Number of candidates: %%d. Best distance located: %%14e", %(ncandidate)d, bestdist);
                    abort();
                }
            }

            {
            const PetscScalar *Ri = R + cell*%(coarse_cell_inc)d;
            pyop2_kernel_evaluate(%(kernel_args)s);
            }
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": evaluate_code,
               "kernel_args": _make_kernel_args(element, "Ri", "Xc", "b", "Xref"),
               "ncandidate": ncandidate,
               "inside_cell": inside_check(element.cell, eps=1e-8, X="Xref"),
               "celldist_l1_c_expr": celldist_l1_c_expr(element.cell, X="Xref"),
               "Xc_cell_inc": coords_element.space_dimension(),
               "coarse_cell_inc": element.space_dimension(),
               "tdim": element.cell.get_spatial_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="pyop2_kernel_restrict"))


def inject_kernel(Vf, Vc):
    if Vc.finat_element.is_dg():
        hierarchy, level = utils.get_level(Vc.mesh())
        if Vf.extruded:
            assert Vc.extruded
            level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
        else:
            level_ratio = 1
        key = (("inject", level_ratio)
               + (Vf.block_size,)
               + entity_dofs_key(Vc.finat_element.complex.get_topology())
               + entity_dofs_key(Vf.finat_element.complex.get_topology())
               + entity_dofs_key(Vc.finat_element.entity_dofs())
               + entity_dofs_key(Vf.finat_element.entity_dofs())
               + entity_dofs_key(Vc.mesh().coordinates.function_space().finat_element.entity_dofs())
               + entity_dofs_key(Vf.mesh().coordinates.function_space().finat_element.entity_dofs()))
        cache = hierarchy._shared_data_cache["transfer_kernels"]
        try:
            return cache[key]
        except KeyError:
            ncandidate = hierarchy.coarse_to_fine_cells[level].shape[1] * level_ratio
            return cache.setdefault(key, (dg_injection_kernel(Vf, Vc, ncandidate), True))
    else:
        expression = ufl.Coefficient(Vf)
        return (prolong_kernel(expression, Vc), False)


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
        self.kernel_args = []
        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) is MixedElement:
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
    macro_builder._domain_integral_type_map = {Vf.mesh(): "cell"}
    macro_builder._entity_ids = {Vf.mesh(): (0,)}
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
    integration_dim, _ = lower_integral_type(Vfe.cell, "cell")
    macro_cfg = dict(interface=macro_builder,
                     ufl_cell=Vf.ufl_cell(),
                     integration_dim=integration_dim,
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
                                domain_integral_type_map={Vc.mesh(): "cell"},
                                arguments=(ufl.TestFunction(Vc), ),
                                coefficients=(),
                                coefficient_split={},
                                coefficient_numbers=())

    coarse_builder = firedrake_interface.KernelBuilder(info, parameters["scalar_type"])
    coarse_builder.set_coordinates([Vc.mesh()])
    coarse_builder.set_entity_numbers([Vc.mesh()])
    argument_multiindices = coarse_builder.argument_multiindices
    argument_multiindex, = argument_multiindices
    return_variable, = coarse_builder.return_variables

    integration_dim, _ = lower_integral_type(Vce.cell, "cell")
    # Midpoint quadrature for jacobian on coarse cell.
    quadrature_rule = make_quadrature(Vce.cell, 0)

    coarse_cfg = dict(interface=coarse_builder,
                      ufl_cell=Vc.ufl_cell(),
                      integration_dim=integration_dim,
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

    index_shape = f.ufl_element().reference_value_shape
    tensor_indices = tuple(gem.Index(extent=d) for d in index_shape)

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
    kernel = lp.merge([kernel, *subkernels]).with_entrypoints({kernel_name})
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
