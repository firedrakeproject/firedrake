import numpy
import string
from fractions import Fraction
from pyop2 import op2
from pyop2.datatypes import IntType, as_cstr
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
from firedrake.utils import ScalarType_c
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
import loopy as lp
import numpy as np


def to_reference_coordinates_loopy(element, ufl_coordinate_element, parameters, cell, dim):
    vertices = numpy.array(element.cell.get_vertices())
    X = numpy.average(vertices, axis=0)

    init_vals = np.array([v for i, v in enumerate(X)])

    point_kernel = to_reference_coordinates_body(ufl_coordinate_element, parameters, coffee=False)
    knl = lp.make_kernel(
        "{[it, i, ii, s, ss, j, k, m, num]: 0 <= it < max_iteration_count and 0 <= i,j,ii, k,m, s, ss < topological_dimension and 0 <= num < 2}",
        """
        for m
            X[m] = tmp[m]
        end
        converged = 0
        for it
            <> ok = converged
            if ok == 0
                for i
                    dX[i] = 0.0
                end
                loopy_kernel_point([ii]: dX[ii], [num]: C[num], [s]: x0[s], [ss]: X[ss])
                <> sum = 0
                for j
                    sum = sum + dX[j] * dX[j]
                end
                if sum < convergence_epsilon*convergence_epsilon
                    converged = 1
                end
                for k
                    X[k] = X[k] - dX[k]
                end
             end
        end
        """,
        [
            lp.GlobalArg("X, x0", dtype=np.double, shape=("topological_dimension",), is_output_only=False),
            lp.GlobalArg("C", np.double, shape=("coords_space_dim",)),
            lp.TemporaryVariable("converged",
                                 dtype=np.int8,
                                 shape=(),
                                 ),
            lp.TemporaryVariable("x",
                                 dtype=np.double,
                                 shape=("geometric_dimension",),
                                 initializer=None,
                                 ),
            lp.TemporaryVariable("dX",
                                 dtype=np.double,
                                 shape=("topological_dimension",),
                                 ),
            lp.TemporaryVariable("tmp",
                                 initializer=init_vals,
                                 shape=lp.auto,
                                 address_space=lp.AddressSpace.PRIVATE,
                                 read_only=True)

        ],
        name='to_reference_coords_kernel',
        seq_dependencies=True,
        target=lp.CudaTarget())
    max_iteration_count = 1 if is_affine(ufl_coordinate_element) else 16
    knl = lp.fix_parameters(knl,
                            max_iteration_count=max_iteration_count,
                            convergence_epsilon=1e-12,
                            geometric_dimension=cell.geometric_dimension(),
                            topological_dimension=cell.topological_dimension(),
                            coords_space_dim=dim)
    knl = lp.register_callable_kernel(knl, point_kernel)
    knl = lp.inline_callable_kernel(knl, point_kernel.name)
    knl = knl.root_kernel
    return knl


def to_reference_coordinates(ufl_coordinate_element, parameters=None, coffee=True, dim=0):
    if parameters is None:
        parameters = tsfc.default_parameters()
    else:
        _ = tsfc.default_parameters()
        _.update(parameters)
        parameters = _

    # Create FInAT element
    element = tsfc.finatinterface.create_element(ufl_coordinate_element)
    cell = ufl_coordinate_element.cell()

    return to_reference_coordinates_loopy(element, ufl_coordinate_element, parameters, cell, dim)


def compile_element(expression, dual_space=None, parameters=None, coffee=True,
                    name="evaluate"):
    """Generate code for point evaluations.

    :arg expression: A UFL expression (may contain up to one coefficient, or one argument)
    :arg dual_space: if the expression has an argument, should we also distribute residual data?
    :returns: Some coffee AST or loopy kernel
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    expression = tsfc.ufl_utils.preprocess_expression(expression)

    # # Collect required coefficients

    ## Choose builder
    if coffee:
        import tsfc.kernel_interface.firedrake as firedrake_interface_coffee
        interface = firedrake_interface_coffee.KernelBuilderBase
    else:
        # Delayed import, loopy is a runtime dependency
        import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
        interface = firedrake_interface_loopy.KernelBuilderBase

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
    builder = interface(scalar_type=ScalarType_c)
    domain = expression.ufl_domain()
    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = ast.Decl(ScalarType_c, ast.Symbol('X', rank=(dim,)))

    if not coffee:
        poin_arg = lp.GlobalArg("X", dtype=parameters["scalar_type"], shape=(dim,))

    config = dict(interface=builder,
                  ufl_cell=cell,
                  precision=parameters["precision"],
                  point_indices=(),
                  point_expr=point,
                  argument_multiindices=argument_multiindices)
    context = tsfc.fem.GemPointContext(**config)

    # Abs-simplification
    expression = tsfc.ufl_utils.simplify_abs(expression)

    # Translate UFL -> GEM
    if coefficient:
        assert dual_space is None
        #if coffee:
        f_arg = [builder._coefficient(arg, "f")]
        #else:
        #fun_arg = [builder._coefficient(arg, "f")]
        fun_arg = lp.GlobalArg("f", dtype=parameters["scalar_type"], shape=lp.auto)
    else:
        #if coffee:
        f_arg = []
        #else:
        fun_arg = None
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    b_arg = []
    bs_arg = []
    if coefficient:
        if expression.ufl_shape:
            return_variable = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
            result_arg = ast.Decl(ScalarType_c, ast.Symbol('R', rank=expression.ufl_shape))
            result = gem.Indexed(result, tensor_indices)
            if not coffee:
                return_arg = lp.GlobalArg("R", dtype=parameters["scalar_type"], shape=expression.ufl_shape, is_output_only=False)
        else:
            return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
            result_arg = ast.Decl(ScalarType_c, ast.Symbol('R', rank=(1,)))
            if not coffee:
                return_arg = lp.GlobalArg("R", dtype=parameters["scalar_type"], shape=(1,), is_output_only=False)

    else:
        return_variable = gem.Indexed(gem.Variable('R', finat_elem.index_shape), argument_multiindex)
        result = gem.Indexed(result, tensor_indices)
        if dual_space:
            elem = create_element(dual_space.ufl_element())
            if elem.value_shape:
                var = gem.Indexed(gem.Variable("b", elem.value_shape),
                                  tensor_indices)
                b_arg = [ast.Decl(ScalarType_c, ast.Symbol("b", rank=elem.value_shape))]
                if not coffee:
                    bs_arg = lp.GlobalArg("b", dtype=parameters["scalar_type"], shape=elem.value_shape)
            else:
                var = gem.Indexed(gem.Variable("b", (1, )), (0, ))
                b_arg = [ast.Decl(ScalarType_c, ast.Symbol("b", rank=(1, )))]
                if not coffee:
                    bs_arg = lp.GlobalArg("b", dtype=parameters["scalar_type"], shape=(1,))
            result = gem.Product(result, var)

        result_arg = ast.Decl(ScalarType_c, ast.Symbol('R', rank=finat_elem.index_shape))
        if not coffee:
            return_arg = lp.GlobalArg("R", dtype=parameters["scalar_type"], shape=finat_elem.index_shape, is_output_only=False)

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> COFFEE
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)

    if coffee:
        body = generate_coffee(impero_c, {}, parameters["precision"], ScalarType_c)
        kernel_code = builder.construct_kernel("pyop2_kernel_" + name, [result_arg] + b_arg + f_arg + [point_arg], body)
        return kernel_code

    if not coffee:
        #builder.register_requirements([result])
        from tsfc.loopy import generate as generate_loopy
        # Build kernel tuple
        #builder.set_coefficients(builder.coefficient_map)
        #kernel = builder.construct_kernel("pyop2_kernel_" + name, [result_arg] + b_arg + f_arg + [point_arg], body)

                                     # base_storage="base"#kernel_code_ret = generate_loopy(impero_c, [return_arg, fun_arg, poin_arg],  parameters["precision"], ScalarType_c)
        args = [return_arg]
        if fun_arg:
            args += [fun_arg]
        args += [poin_arg]
        if bs_arg:
            args += [bs_arg]
        kernel_code_ret = builder.construct_kernel(name, args, impero_c,  parameters["precision"])

        return kernel_code_ret


def construct_common_kernel(initial=None, func=None, evaluate_args=None, copy_loop=None):
    if initial is None:
        initial = "coarsei"
        func = "f"
        evaluate_args = "[jj]: R[jj], [c]: coarsei[c], [p]: Xref[p]"
        copy_loop = ""

    kernel = """
        for ri
            Xref[ri] = 0
        end
        cell = -1
        error = 0
        bestcell = -1
        bestdist = 1e10
        <> stop = 0
        for i
            <> tmp = stop
            if tmp == 0
                <> tm = i * coords_space_dim
                for k
                    Xci[k] = Xc[k + tm]
                end
                celldist = 2 * bestdist
                to_reference_coords_kernel([p]: Xref[p], [c]: X[c], [q]: Xci[q])
                <> check = 1
                for l
                    check = check and (Xref[l] + constant >= 0) and (Xref[l] - constant <= 1)
                end
                if check
                    cell = i
                    stop = 1
                end
            end
            tmp = stop
            if tmp == 0
                celldist = Xref[0]
                for celldistdim
                    <> temp = celldist
                    if temp > Xref[celldistdim]
                        celldist = Xref[celldistdim]
                    end
                end
                celldist = celldist * (-1)
                <> temp1 = bestdist
                if celldist < temp1
                    bestdist = celldist
                    bestcell = i
                end
            end
        end
        <> tmp2 = cell
        if tmp2 == -1
            if bestdist < 10
                cell = bestcell
            else
                error = 1
            end
        end

        for ci
            %(initial)s[ci] = %(func)s[ci + cell * coarse_cell_inc]
        end

        loopy_kernel_evaluate(%(evaluate_args)s)
        %(copy_loop)s
        """ % {
        "initial": initial,
        "func": func,
        "evaluate_args": evaluate_args,
        "copy_loop": copy_loop
    }
    return kernel


def common_kernel_args():
    return [
        lp.TemporaryVariable("cell",
                             dtype=np.int32,
                             shape=()),
        lp.TemporaryVariable("bestcell",
                             dtype=np.int32,
                             shape=()),
        lp.TemporaryVariable("error",
                             dtype=np.int8,
                             shape=()),
        lp.TemporaryVariable("bestdist",
                             dtype=np.double,
                             shape=()),
        lp.TemporaryVariable("celldist",
                             dtype=np.double,
                             shape=()),
        lp.TemporaryVariable("Xref",
                             dtype=np.double,
                             shape=("tdim",)),
        lp.TemporaryVariable("Xci",
                             dtype=np.double,
                             shape=("coords_space_dim",))
    ]


def prolong_kernel_loopy(expression, coordinates, hierarchy, levelf, cache, key):
    mesh = coordinates.ufl_domain()
    evaluate_kernel = compile_element(expression, coffee=False)
    element = create_element(expression.ufl_element())
    coords_element = create_element(coordinates.ufl_element())
    to_reference_kernel = to_reference_coordinates(coordinates.ufl_element(), coffee=False,
                                                   dim=coords_element.space_dimension())

    global_list = [ lp.GlobalArg("R", np.double, shape=("Rdim",), is_output_only=True),
                    lp.GlobalArg("f", np.double, shape=("coords_space_dim",)),
                    lp.GlobalArg("X", np.double, shape=("tdim",)),
                    lp.GlobalArg("Xc", np.double, shape=("coords_space_dim",))]
    var_list = global_list + common_kernel_args() + \
               [lp.TemporaryVariable("coarsei", dtype=np.double, shape=("coarse_cell_inc",))]

    parent_knl = lp.make_kernel(
        {"{[i, j, jj, c, celldistdim, k, p, q, ci, l, ri]: 0 <= i < ncandidate and 0 <= j, jj < Rdim and "
         "0 <= celldistdim < tdim and 0 <= k, q < coords_space_dim and 0 <= p, l, ri < tdim and "
         "0<= c < tdim and 0 <=  ci < coarse_cell_inc}"},
        construct_common_kernel(),
        var_list,
        name='loopy_kernel_prolong',
        seq_dependencies=True,
        target=lp.CudaTarget())

    parent_knl = lp.fix_parameters(parent_knl,
                                   ncandidate=hierarchy.fine_to_coarse_cells[levelf].shape[1],
                                   tdim=mesh.topological_dimension(),
                                   coords_space_dim=coords_element.space_dimension(),
                                   constant=1e-8,
                                   Rdim=int(numpy.prod(element.value_shape)),
                                   coarse_cell_inc=element.space_dimension())

    from loopy.transform.callable import _match_caller_callee_argument_dimension_
    knl = lp.register_callable_kernel(parent_knl, evaluate_kernel)
    knl = _match_caller_callee_argument_dimension_(knl, evaluate_kernel.name)
    knl = lp.inline_callable_kernel(knl, evaluate_kernel.name)

    knl = lp.register_callable_kernel(knl, to_reference_kernel)
    knl = _match_caller_callee_argument_dimension_(knl, to_reference_kernel.name)
    knl = lp.inline_callable_kernel(knl, to_reference_kernel.name)
    knl = knl.root_kernel
    # smth = lp.generate_code(knl)
    return cache.setdefault(key, op2.Kernel(knl, name="loopy_kernel_prolong"))


def prolong_kernel(expression):
    hierarchy, level = utils.get_level(expression.ufl_domain())
    levelf = level + Fraction(1 / hierarchy.refinements_per_level)
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = expression.ufl_domain().coordinates
    key = (("prolong", )
           + expression.ufl_element().value_shape()
           + entity_dofs_key(expression.function_space().finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        return prolong_kernel_loopy(expression, coordinates, hierarchy, levelf, cache, key)


def restrict_kernel_loopy(Vc, Vf, coordinates, hierarchy, levelf, cache, key):
    mesh = coordinates.ufl_domain()
    evaluate_kernel = compile_element(firedrake.TestFunction(Vc), Vf, coffee=False)
    coords_element = create_element(coordinates.ufl_element())
    to_reference_kernel = to_reference_coordinates(coordinates.ufl_element(), coffee=False,
                                                   dim=coords_element.space_dimension())
    element = create_element(Vc.ufl_element())
    eval_args = evaluate_kernel.args[:-1]

    R, fine = (a for a in eval_args)
    fine = np.array(fine)
    Rdim = np.prod(R.shape)

    global_list = [lp.GlobalArg("R", np.double, shape=("Rdim",), is_output_only=True),
                   lp.GlobalArg("b", np.double, shape=("finedim",)),
                   lp.GlobalArg("X", np.double, shape=("tdim",)),
                   lp.GlobalArg("Xc", np.double, shape=("coords_space_dim",))]
    var_list = global_list + common_kernel_args() + [lp.TemporaryVariable("Ri", dtype=np.double,shape=("Rdim",))]
    kern = construct_common_kernel("Ri", "R", "[jj]: Ri[jj], [p]: Xref[p], [c]: b[c]", """	
             for cci
                 R[cci+ cell * coarse_cell_inc] = Ri[cci]
             end """)

    parent_knl = lp.make_kernel(
        {"{[i, j, jj, c, cci, celldistdim, k, p, q, ci, l, ri]: 0 <= i < ncandidate and 0 <= j, jj < Rdim and "
         "0 <= celldistdim < tdim and 0 <= k, q < coords_space_dim and 0 <= p, l, ri < tdim and "
         "0 <= ci, cci < Rdim and 0 <= c < finedim}"},
        kern,
        var_list,
        name='loopy_kernel_restrict',
        seq_dependencies=True,
        target=lp.CudaTarget())

    parent_knl = lp.fix_parameters(parent_knl,
                                   ncandidate=hierarchy.fine_to_coarse_cells[levelf].shape[1],
                                   tdim=mesh.topological_dimension(),
                                   coords_space_dim=coords_element.space_dimension(),
                                   coarse_cell_inc=element.space_dimension(),
                                   constant=1e-8,
                                   Rdim=Rdim,
                                   finedim=fine.size)

    from loopy.transform.callable import _match_caller_callee_argument_dimension_
    knl = lp.register_callable_kernel(parent_knl, evaluate_kernel)
    knl = _match_caller_callee_argument_dimension_(knl, evaluate_kernel.name)
    knl = lp.inline_callable_kernel(knl, evaluate_kernel.name)

    knl = lp.register_callable_kernel(knl, to_reference_kernel)
    knl = _match_caller_callee_argument_dimension_(knl, to_reference_kernel.name)
    knl = lp.inline_callable_kernel(knl, to_reference_kernel.name)
    knl = knl.root_kernel
    # smth = lp.generate_code(knl)
    return cache.setdefault(key, op2.Kernel(knl, name="loopy_kernel_restrict"))


def restrict_kernel(Vf, Vc):
    hierarchy, level = utils.get_level(Vc.ufl_domain())
    levelf = level + Fraction(1 / hierarchy.refinements_per_level)
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = Vc.ufl_domain().coordinates
    key = (("restrict", )
           + Vf.ufl_element().value_shape()
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        return restrict_kernel_loopy(Vc, Vf, coordinates, hierarchy, levelf, cache, key)


def inject_kernel_loopy(hierarchy, level, Vc, Vf, key, cache):
    ncandidate = hierarchy.coarse_to_fine_cells[level].shape[1]
    if Vc.finat_element.entity_dofs() == Vc.finat_element.entity_closure_dofs():
        return cache.setdefault(key, (dg_injection_kernel(Vf, Vc, ncandidate), True))

    coordinates = Vf.ufl_domain().coordinates
    evaluate_kernel = compile_element(ufl.Coefficient(Vf), coffee=False)
    coords_element = create_element(coordinates.ufl_element())
    to_reference_kernel = to_reference_coordinates(coordinates.ufl_element(), coffee=False, dim=coords_element.space_dimension())
    Vf_element = create_element(Vf.ufl_element())

    global_list = [lp.GlobalArg("R", np.double, shape=("Rdim",), is_output_only=True),
                   lp.GlobalArg("X", np.double, shape=("tdim",)),
                   lp.GlobalArg("f", np.double, shape=("coords_space_dim",)),
                   lp.GlobalArg("Xc", np.double, shape=("coords_space_dim",))]
    var_list = global_list + common_kernel_args() + \
               [lp.TemporaryVariable("coarsei", dtype=np.double, shape=("coarse_cell_inc",))]

    parent_knl = lp.make_kernel(
        {"{[i, j, jj, c, celldistdim, k, p, q, ci, l, ri]: 0 <= i < ncandidate and 0 <= j, jj < Rdim and "
         "0 <= celldistdim < tdim and 0 <= k, q < coords_space_dim and 0 <= p, l, ri < tdim and "
         "0<= c < tdim and 0 <=  ci < coarse_cell_inc}"},
        construct_common_kernel(),
        var_list,
        name='loopy_kernel_inject',
        seq_dependencies=True,
        target=lp.CudaTarget())

    parent_knl = lp.fix_parameters(parent_knl,
                                   ncandidate=ncandidate,
                                   tdim=Vc.ufl_domain().topological_dimension(),
                                   coords_space_dim=coords_element.space_dimension(),
                                   constant=1e-8,
                                   Rdim=int(numpy.prod(Vf_element.value_shape)),
                                   coarse_cell_inc=Vf_element.space_dimension())

    from loopy.transform.callable import _match_caller_callee_argument_dimension_
    knl = lp.register_callable_kernel(parent_knl, evaluate_kernel)
    knl = _match_caller_callee_argument_dimension_(knl, evaluate_kernel.name)
    knl = lp.inline_callable_kernel(knl, evaluate_kernel.name)

    knl = lp.register_callable_kernel(knl, to_reference_kernel)
    knl = _match_caller_callee_argument_dimension_(knl, to_reference_kernel.name)
    knl = lp.inline_callable_kernel(knl, to_reference_kernel.name)
    knl = knl.root_kernel
    # smth = lp.generate_code(knl)
    return cache.setdefault(key, (op2.Kernel(knl, name="loopy_kernel_inject"), False))


def inject_kernel(Vf, Vc):
    hierarchy, level = utils.get_level(Vc.ufl_domain())
    cache = hierarchy._shared_data_cache["transfer_kernels"]
    coordinates = Vf.ufl_domain().coordinates
    key = (("inject", )
           + Vf.ufl_element().value_shape()
           + entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + entity_dofs_key(Vc.mesh().coordinates.function_space().finat_element.entity_dofs())
           + entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        return inject_kernel_loopy(hierarchy, level, Vc, Vf, key, cache)


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
    raise NotImplementedError('Not yet implemented')
    from firedrake import Tensor, AssembledVector, TestFunction, TrialFunction
    from firedrake.slate.slac import compile_expression
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
                     precision=parameters["precision"],
                     integration_dim=integration_dim,
                     entity_ids=entity_ids,
                     index_cache=index_cache,
                     quadrature_rule=macro_quadrature_rule)

    fexpr, = fem.compile_ufl(f, **macro_cfg)
    X = ufl.SpatialCoordinate(Vf.mesh())
    C_a, = fem.compile_ufl(X, **macro_cfg)
    detJ = ufl_utils.preprocess_expression(abs(ufl.JacobianDeterminant(f.ufl_domain())))
    macro_detJ, = fem.compile_ufl(detJ, **macro_cfg)

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
                      precision=parameters["precision"],
                      integration_dim=integration_dim,
                      entity_ids=entity_ids,
                      index_cache=index_cache,
                      quadrature_rule=quadrature_rule)

    X = ufl.SpatialCoordinate(Vc.mesh())
    K = ufl_utils.preprocess_expression(ufl.JacobianInverse(Vc.mesh()))
    C_0, = fem.compile_ufl(X, **coarse_cfg)
    K, = fem.compile_ufl(K, **coarse_cfg)

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
    body = generate_coffee(impero_c, index_names, parameters["precision"], ScalarType_c)

    retarg = ast.Decl(ScalarType_c, ast.Symbol("R", rank=(Vce.space_dimension(), )))
    local_tensor = coarse_builder.local_tensor
    local_tensor.init = ast.ArrayInit(numpy.zeros(Vce.space_dimension(), dtype=ScalarType_c))
    body.children.insert(0, local_tensor)
    args = [retarg] + macro_builder.kernel_args + [macro_builder.coordinates_arg,
                                                   coarse_builder.coordinates_arg]

    # Now we have the kernel that computes <f, phi_c>dx_c
    # So now we need to hit it with the inverse mass matrix on dx_c

    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    expr = Tensor(ufl.inner(u, v)*ufl.dx).inv * AssembledVector(ufl.Coefficient(Vc))
    Ainv, = compile_expression(expr)
    Ainv = Ainv.kinfo.kernel
    A = ast.Symbol(local_tensor.sym.symbol)
    R = ast.Symbol("R")
    body.children.append(ast.FunCall(Ainv.name, R, coarse_builder.coordinates_arg.sym, A))
    from coffee.base import Node
    assert isinstance(Ainv._code, Node)
    return op2.Kernel(ast.Node([Ainv._code,
                                ast.FunDecl("void", "pyop2_kernel_injection_dg", args, body,
                                            pred=["static", "inline"])]),
                      name="pyop2_kernel_injection_dg",
                      cpp=True,
                      include_dirs=Ainv._include_dirs,
                      headers=Ainv._headers)
