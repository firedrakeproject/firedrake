"""This is Slate's Linear Algebra Compiler. This module is
responsible for generating C++ kernel functions representing
symbolic linear algebra expressions written in Slate.

This linear algebra compiler uses both Firedrake's form compiler,
the Two-Stage Form Compiler (TSFC) and COFFEE's kernel abstract
syntax tree (AST) optimizer. TSFC provides this compiler with
appropriate kernel functions (in C) for evaluating integral
expressions (finite element variational forms written in UFL).
COFFEE's AST base helps with the construction of code blocks
throughout the kernel returned by: `compile_expression`.

The Eigen C++ library (http://eigen.tuxfamily.org/) is required, as
all low-level numerical linear algebra operations are performed using
this templated function library.
"""
from coffee import base as ast

import time
from hashlib import md5

from firedrake_citations import Citations
from firedrake.tsfc_interface import SplitKernel, KernelInfo, TSFCKernel

from firedrake.slate.slac.kernel_builder import LocalLoopyKernelBuilder, LocalKernelBuilder
from firedrake.slate.slac.utils import topological_sort, slate_to_gem, merge_loopy
from firedrake.slate.slac.optimise import optimise

from firedrake import op2, tsfc_interface
from firedrake.logging import logger
from firedrake.parameters import parameters
from firedrake.petsc import get_petsc_variables
from firedrake.utils import complex_mode, ScalarType_c, as_cstr
from ufl.log import GREEN
from gem.utils import groupby
from gem import impero_utils
from itertools import chain

from pyop2.utils import get_petsc_dir, as_tuple
from pyop2.mpi import COMM_WORLD
from pyop2.codegen.rep2loopy import SolveCallable, INVCallable

import firedrake.slate.slate as slate
import numpy as np
import loopy
import gem
from gem import indices as make_indices
from tsfc.kernel_args import OutputKernelArg
from tsfc.loopy import generate as generate_loopy
import copy

__all__ = ['compile_expression']


try:
    PETSC_DIR, PETSC_ARCH = get_petsc_dir()
except ValueError:
    PETSC_DIR, = get_petsc_dir()
    PETSC_ARCH = None

EIGEN_INCLUDE_DIR = None
BLASLAPACK_LIB = None
BLASLAPACK_INCLUDE = None
if not complex_mode:
    if COMM_WORLD.rank == 0:
        petsc_variables = get_petsc_variables()
        EIGEN_INCLUDE_DIR = petsc_variables.get("EIGEN_INCLUDE")
        if EIGEN_INCLUDE_DIR is None:
            raise ValueError("""Could not find Eigen configuration in %s. Did you build PETSc with Eigen?""" % PETSC_ARCH or PETSC_DIR)
        EIGEN_INCLUDE_DIR = EIGEN_INCLUDE_DIR.lstrip('-I')
        EIGEN_INCLUDE_DIR = COMM_WORLD.bcast(EIGEN_INCLUDE_DIR, root=0)

        BLASLAPACK_LIB = petsc_variables.get("BLASLAPACK_LIB", "")
        BLASLAPACK_LIB = COMM_WORLD.bcast(BLASLAPACK_LIB, root=0)
        BLASLAPACK_INCLUDE = petsc_variables.get("BLASLAPACK_INCLUDE", "")
        BLASLAPACK_INCLUDE = COMM_WORLD.bcast(BLASLAPACK_INCLUDE, root=0)
    else:
        EIGEN_INCLUDE_DIR = COMM_WORLD.bcast(None, root=0)
        BLASLAPACK_LIB = COMM_WORLD.bcast(None, root=0)
        BLASLAPACK_INCLUDE = COMM_WORLD.bcast(None, root=0)

cell_to_facets_dtype = np.dtype(np.int8)


class SlateKernel(TSFCKernel):
    @classmethod
    def _cache_key(cls, expr, compiler_parameters, coffee):
        return md5((expr.expression_hash
                    + str(sorted(compiler_parameters.items()))
                    + str(coffee)).encode()).hexdigest(), expr.ufl_domains()[0].comm

    def __init__(self, expr, compiler_parameters, coffee=False):
        if self._initialized:
            return
        if coffee:
            self.split_kernel = generate_kernel(expr, compiler_parameters)
        else:
            self.split_kernel = generate_loopy_kernel(expr, compiler_parameters)
        self._initialized = True


def compile_expression(slate_expr, compiler_parameters=None, coffee=False):
    """Takes a Slate expression `slate_expr` and returns the appropriate
    :class:`firedrake.op2.Kernel` object representing the Slate expression.

    :arg slate_expr: a :class:'TensorBase' expression.
    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
        be passed to TSFC during the compilation of ufl forms.

    Returns: A `tuple` containing a `SplitKernel(idx, kinfo)`
    """
    if complex_mode:
        raise NotImplementedError("SLATE doesn't work in complex mode yet")
    if not isinstance(slate_expr, slate.TensorBase):
        raise ValueError("Expecting a `TensorBase` object, not %s" % type(slate_expr))

    # Update default parameters with passed parameters
    # The deepcopy is needed because parameters is a nested dict
    params = copy.deepcopy(parameters)
    if compiler_parameters and "slate_compiler" in compiler_parameters.keys():
        params["slate_compiler"].update(compiler_parameters.pop("slate_compiler"))
    if compiler_parameters:
        params["form_compiler"].update(compiler_parameters)

    # If the expression has already been symbolically compiled, then
    # simply reuse the produced kernel.
    cache = slate_expr._metakernel_cache
    key = str(sorted(params.items()))
    try:
        return cache[key]
    except KeyError:
        kernel = SlateKernel(slate_expr, params, coffee).split_kernel
        return cache.setdefault(key, kernel)


def get_temp_info(loopy_kernel):
    """Get information about temporaries in loopy kernel.

    Returns memory in bytes and number of temporaries.
    """
    mems = [temp.nbytes for temp in loopy_kernel.temporary_variables.values()]
    mem_total = sum(mems)
    num_temps = len(loopy_kernel.temporary_variables)

    # Get number of temporaries of different shapes
    shapes = {}
    for temp in loopy_kernel.temporary_variables.values():
        shape = temp.shape
        if temp.storage_shape is not None:
            shape = temp.storage_shape

        shapes[len(shape)] = shapes.get(len(shape), 0) + 1
    return mem_total, num_temps, mems, shapes


def generate_loopy_kernel(slate_expr, compiler_parameters=None):
    cpu_time = time.time()
    if len(slate_expr.ufl_domains()) > 1:
        raise NotImplementedError("Multiple domains not implemented.")

    Citations().register("Gibson2018")

    orig_expr = slate_expr
    # Optimise slate expr, e.g. push blocks as far inward as possible
    if compiler_parameters["slate_compiler"]["optimise"]:
        slate_expr = optimise(slate_expr, compiler_parameters["slate_compiler"])

    # Create a loopy builder for the Slate expression,
    # e.g. contains the loopy kernels coming from TSFC
    gem_expr, var2terminal = slate_to_gem(slate_expr, compiler_parameters["slate_compiler"])

    scalar_type = compiler_parameters["form_compiler"]["scalar_type"]
    slate_loopy, output_arg = gem_to_loopy(gem_expr, var2terminal, scalar_type)

    builder = LocalLoopyKernelBuilder(expression=slate_expr,
                                      tsfc_parameters=compiler_parameters["form_compiler"])

    name = "slate_wrapper"
    loopy_merged, arguments = merge_loopy(slate_loopy, output_arg, builder, var2terminal, name)
    loopy_merged = loopy.register_callable(loopy_merged, INVCallable.name, INVCallable())
    loopy_merged = loopy.register_callable(loopy_merged, SolveCallable.name, SolveCallable())

    loopykernel = tsfc_interface.as_pyop2_local_kernel(loopy_merged, name, len(arguments),
                                                       include_dirs=BLASLAPACK_INCLUDE.split(),
                                                       ldargs=BLASLAPACK_LIB.split())

    # map the coefficients in the order that PyOP2 needs
    new_coeffs = slate_expr.coefficients()
    orig_coeffs = orig_expr.coefficients()
    get_index = lambda n: orig_coeffs.index(new_coeffs[n]) if new_coeffs[n] in orig_coeffs else n
    coeff_map = tuple((get_index(n), split_map) for (n, split_map) in slate_expr.coeff_map)

    kinfo = KernelInfo(kernel=loopykernel,
                       integral_type="cell",  # slate can only do things as contributions to the cell integrals
                       oriented=builder.bag.needs_cell_orientations,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=coeff_map,
                       needs_cell_facets=builder.bag.needs_cell_facets,
                       pass_layer_arg=builder.bag.needs_mesh_layers,
                       needs_cell_sizes=builder.bag.needs_cell_sizes,
                       arguments=arguments)

    # Cache the resulting kernel
    # Slate kernels are never split, so indicate that with None in the index slot.
    idx = tuple([None]*slate_expr.rank)
    logger.info(GREEN % "compile_slate_expression finished in %g seconds.", time.time() - cpu_time)
    return (SplitKernel(idx, kinfo),)


def generate_kernel(slate_expr, compiler_parameters=None):
    cpu_time = time.time()

    if len(slate_expr.ufl_domains()) > 1:
        raise NotImplementedError("Multiple domains not implemented.")

    Citations().register("Gibson2018")
    # Create a builder for the Slate expression
    builder = LocalKernelBuilder(expression=slate_expr,
                                 tsfc_parameters=compiler_parameters["form_compiler"])

    # Keep track of declared temporaries
    declared_temps = {}
    statements = []

    # Declare terminal tensor temporaries
    terminal_declarations = terminal_temporaries(builder, declared_temps)
    statements.extend(terminal_declarations)

    # Generate assembly calls for tensor assembly
    subkernel_calls = tensor_assembly_calls(builder)
    statements.extend(subkernel_calls)

    # Create coefficient temporaries if necessary
    if builder.coefficient_vecs:
        coefficient_temps = coefficient_temporaries(builder, declared_temps)
        statements.extend(coefficient_temps)

    # Create auxiliary temporaries/expressions (if necessary)
    statements.extend(auxiliary_expressions(builder, declared_temps))

    # Generate the kernel information with complete AST
    kinfo = generate_kernel_ast(builder, statements, declared_temps)

    # Cache the resulting kernel
    # Slate kernels are never split, so indicate that with None in the index slot.
    idx = tuple([None]*slate_expr.rank)
    logger.info(GREEN % "compile_slate_expression finished in %g seconds.", time.time() - cpu_time)
    return (SplitKernel(idx, kinfo),)


def generate_kernel_ast(builder, statements, declared_temps):
    """Glues together the complete AST for the Slate expression
    contained in the :class:`LocalKernelBuilder`.

    :arg builder: The :class:`LocalKernelBuilder` containing
        all relevant expression information.
    :arg statements: A list of COFFEE objects containing all
        assembly calls and temporary declarations.
    :arg declared_temps: A `dict` containing all previously
        declared temporaries.

    Return: A `KernelInfo` object describing the complete AST.
    """
    slate_expr = builder.expression
    if slate_expr.rank == 0:
        # Scalars are treated as 1x1 MatrixBase objects
        shape = (1,)
    else:
        shape = slate_expr.shape

    # Now we create the result statement by declaring its eigen type and
    # using Eigen::Map to move between Eigen and C data structs.
    statements.append(ast.FlatBlock("/* Map eigen tensor into C struct */\n"))
    result_sym = ast.Symbol("T%d" % len(declared_temps))
    result_data_sym = ast.Symbol("A%d" % len(declared_temps))
    result_type = "Eigen::Map<%s >" % eigen_matrixbase_type(shape)
    result = ast.Decl(ScalarType_c, ast.Symbol(result_data_sym), pointers=[("restrict",)])
    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" % (result_type,
                                                             result_sym,
                                                             ScalarType_c,
                                                             result_data_sym))
    statements.append(result_statement)

    # Generate the complete c++ string performing the linear algebra operations
    # on Eigen matrices/vectors
    statements.append(ast.FlatBlock("/* Linear algebra expression */\n"))
    cpp_string = ast.FlatBlock(slate_to_cpp(slate_expr, declared_temps))
    statements.append(ast.Incr(result_sym, cpp_string))

    # Generate arguments for the macro kernel
    args = [result, ast.Decl(ScalarType_c, builder.coord_sym,
                             pointers=[("restrict",)],
                             qualifiers=["const"])]

    # Orientation information
    if builder.oriented:
        args.append(ast.Decl("int", builder.cell_orientations_sym,
                             pointers=[("restrict",)],
                             qualifiers=["const"]))

    # Coefficient information
    expr_coeffs = slate_expr.coefficients()
    for c in expr_coeffs:
        args.extend([ast.Decl(ScalarType_c, csym,
                              pointers=[("restrict",)],
                              qualifiers=["const"]) for csym in builder.coefficient(c)])

    # Facet information
    if builder.needs_cell_facets:
        f_sym = builder.cell_facet_sym
        f_arg = ast.Symbol("arg_cell_facets")
        f_dtype = as_cstr(cell_to_facets_dtype)

        # cell_facets is locally a flattened 2-D array. We typecast here so we
        # can access its entries using standard array notation.
        cast = "%s (*%s)[2] = (%s (*)[2])%s;\n" % (f_dtype, f_sym, f_dtype, f_arg)
        statements.insert(0, ast.FlatBlock(cast))
        args.append(ast.Decl(f_dtype, f_arg,
                             pointers=[("restrict",)],
                             qualifiers=["const"]))

    # NOTE: We need to be careful about the ordering here. Mesh layers are
    # added as the final argument to the kernel
    # and the amount of layers before that.
    if builder.needs_mesh_layers:
        args.append(ast.Decl("int", builder.mesh_layer_count_sym,
                             pointers=[("restrict",)],
                             qualifiers=["const"]))
        args.append(ast.Decl("int", builder.mesh_layer_sym))

    # Cell size information
    if builder.needs_cell_sizes:
        args.append(ast.Decl(ScalarType_c, builder.cell_size_sym,
                             pointers=[("restrict",)],
                             qualifiers=["const"]))

    # Macro kernel
    macro_kernel_name = "pyop2_kernel_compile_slate"
    stmts = ast.Block(statements)
    macro_kernel = ast.FunDecl("void", macro_kernel_name, args,
                               stmts, pred=["static", "inline"])

    # Construct the final ast
    kernel_ast = ast.Node(builder.templated_subkernels + [macro_kernel])

    # Now we wrap up the kernel ast as a PyOP2 kernel and include the
    # Eigen header files
    include_dirs = list(builder.include_dirs)
    include_dirs.append(EIGEN_INCLUDE_DIR)
    flop_count = builder.expression_flops + builder.terminal_flops
    op2kernel = op2.Kernel(kernel_ast,
                           macro_kernel_name,
                           cpp=True,
                           include_dirs=include_dirs,
                           headers=['#include <Eigen/Dense>',
                                    '#define restrict __restrict'],
                           flop_count=flop_count)

    # Send back a "TSFC-like" SplitKernel object with an
    # index and KernelInfo
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type=builder.integral_type,
                       oriented=builder.oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=slate_expr.coeff_map,
                       needs_cell_facets=builder.needs_cell_facets,
                       pass_layer_arg=builder.needs_mesh_layers,
                       needs_cell_sizes=builder.needs_cell_sizes,
                       arguments=None)

    return kinfo


def auxiliary_expressions(builder, declared_temps):
    """Generates statements for assigning auxiliary temporaries
    and declaring factorizations for local matrix inverses
    (if the matrix is larger than 4 x 4).

    :arg builder: The :class:`LocalKernelBuilder` containing
        all relevant expression information.
    :arg declared_temps: A `dict` containing all previously
        declared temporaries. This dictionary is updated as
        auxiliary expressions are assigned temporaries.
    """

    # These are either already declared terminals or expressions
    # which do not require an extra temporary/expression
    terminals = (slate.Tensor, slate.AssembledVector,
                 slate.Negative, slate.Transpose)
    statements = []

    sorted_exprs = [exp for exp in topological_sort(builder.expression_dag)
                    if ((builder.ref_counter[exp] > 1 and not isinstance(exp, terminals))
                        or isinstance(exp, slate.Factorization))]

    for exp in sorted_exprs:
        if exp not in declared_temps:
            if isinstance(exp, slate.Factorization):
                t = ast.Symbol("dec%d" % len(declared_temps))
                operand, = exp.operands
                expr = slate_to_cpp(operand, declared_temps)
                tensor_type = eigen_matrixbase_type(shape=exp.shape)
                stmt = "Eigen::%s<%s > %s(%s);\n" % (exp.decomposition,
                                                     tensor_type, t, expr)
                statements.append(stmt)
            else:
                t = ast.Symbol("auxT%d" % len(declared_temps))
                result = slate_to_cpp(exp, declared_temps)
                tensor_type = eigen_matrixbase_type(shape=exp.shape)
                stmt = ast.Decl(tensor_type, t)
                assignment = ast.Assign(t, result)
                statements.extend([stmt, assignment])

            declared_temps[exp] = t

    return statements


def coefficient_temporaries(builder, declared_temps):
    """Generates coefficient temporary statements for assigning
    coefficients to vector temporaries.

    :arg builder: The :class:`LocalKernelBuilder` containing
        all relevant expression information.
    :arg declared_temps: A `dict` keeping track of all declared
        temporaries. This dictionary is updated as coefficients
        are assigned temporaries.

    'AssembledVector's require creating coefficient temporaries to
    store data. The temporaries are created by inspecting the function
    space of the coefficient to compute node and dof extents. The
    coefficient is then assigned values by looping over both the node
    extent and dof extent (double FOR-loop). A double FOR-loop is needed
    for each function space (if the function space is mixed, then a loop
    will be constructed for each component space). The general structure
    of each coefficient loop will be:

         FOR (i1=0; i1<node_extent; i1++):
             FOR (j1=0; j1<dof_extent; j1++):
                 VT0[offset + (dof_extent * i1) + j1] = w_0_0[i1][j1]
                 VT1[offset + (dof_extent * i1) + j1] = w_1_0[i1][j1]
                 .
                 .
                 .

    where wT0, wT1, ... are temporaries for coefficients sharing the
    same node and dof extents. The offset is computed based on whether
    the function space is mixed. The offset is always 0 for non-mixed
    coefficients. If the coefficient is mixed, then the offset is
    incremented by the total number of nodal unknowns associated with
    the component spaces of the mixed space.
    """
    statements = [ast.FlatBlock("/* Coefficient temporaries */\n")]
    j = ast.Symbol("j1")
    loops = [ast.FlatBlock("/* Loops for coefficient temps */\n")]
    for dofs, cinfo_list in builder.coefficient_vecs.items():
        # Collect all coefficients which share the same node/dof extent
        assignments = []
        for cinfo in cinfo_list:
            fs_i = cinfo.space_index
            offset = cinfo.offset_index
            c_shape = cinfo.shape
            vector = cinfo.vector
            function = vector._function
            t = cinfo.local_temp

            if vector not in declared_temps:
                # Declare and initialize coefficient temporary
                c_type = eigen_matrixbase_type(shape=c_shape)
                statements.append(ast.Decl(c_type, t))
                declared_temps[vector] = t

            # Assigning coefficient values into temporary
            coeff_sym = ast.Symbol(builder.coefficient(function)[fs_i],
                                   rank=(j, ))
            index = ast.Sum(offset, j)
            coeff_temp = ast.Symbol(t, rank=(index, ))
            assignments.append(ast.Assign(coeff_temp, coeff_sym))

        # loop over dofs
        loop = ast.For(ast.Decl("unsigned int", j, init=0),
                       ast.Less(j, dofs),
                       ast.Incr(j, 1),
                       assignments)

        loops.append(loop)

    statements.extend(loops)

    return statements


def tensor_assembly_calls(builder):
    """Generates a block of statements for assembling the local
    finite element tensors.

    :arg builder: The :class:`LocalKernelBuilder` containing
        all relevant expression information and assembly calls.
    """
    assembly_calls = builder.assembly_calls
    statements = [ast.FlatBlock("/* Assemble local tensors */\n")]

    # Cell integrals are straightforward. Just splat them out.
    statements.extend(assembly_calls["cell"])

    if builder.needs_cell_facets:
        # The for-loop will have the general structure:
        #
        #    FOR (facet=0; facet<num_facets; facet++):
        #        IF (facet is interior):
        #            *interior calls
        #        ELSE IF (facet is exterior):
        #            *exterior calls
        #
        # If only interior (exterior) facets are present,
        # then only a single IF-statement checking for interior
        # (exterior) facets will be present within the loop. The
        # cell facets are labelled `1` for interior, and `0` for
        # exterior.
        statements.append(ast.FlatBlock("/* Loop over cell facets */\n"))
        int_calls = list(chain(*[assembly_calls[it_type]
                                 for it_type in ("interior_facet",
                                                 "interior_facet_vert")]))
        ext_calls = list(chain(*[assembly_calls[it_type]
                                 for it_type in ("exterior_facet",
                                                 "exterior_facet_vert")]))

        # Generate logical statements for handling exterior/interior facet
        # integrals on subdomains.
        # Currently only facet integrals are supported.
        for sd_type in ("subdomains_exterior_facet", "subdomains_interior_facet"):
            stmts = []
            for sd, sd_calls in groupby(assembly_calls[sd_type], lambda x: x[0]):
                _, calls = zip(*sd_calls)
                if_sd = ast.Eq(ast.Symbol(builder.cell_facet_sym, rank=(builder.it_sym, 1)), sd)
                stmts.append(ast.If(if_sd, (ast.Block(calls, open_scope=True),)))

            if sd_type == "subdomains_exterior_facet":
                ext_calls.extend(stmts)
            if sd_type == "subdomains_interior_facet":
                int_calls.extend(stmts)

        # Compute the number of facets to loop over
        domain = builder.expression.ufl_domain()
        if domain.cell_set._extruded:
            num_facets = domain.ufl_cell()._cells[0].num_facets()
        else:
            num_facets = domain.ufl_cell().num_facets()

        if_ext = ast.Eq(ast.Symbol(builder.cell_facet_sym,
                                   rank=(builder.it_sym, 0)), 0)
        if_int = ast.Eq(ast.Symbol(builder.cell_facet_sym,
                                   rank=(builder.it_sym, 0)), 1)
        body = []
        if ext_calls:
            body.append(ast.If(if_ext, (ast.Block(ext_calls, open_scope=True),)))
        if int_calls:
            body.append(ast.If(if_int, (ast.Block(int_calls, open_scope=True),)))

        statements.append(ast.For(ast.Decl("unsigned int", builder.it_sym, init=0),
                                  ast.Less(builder.it_sym, num_facets),
                                  ast.Incr(builder.it_sym, 1), body))

    if builder.needs_mesh_layers:
        # In the presence of interior horizontal facet calls, an
        # IF-ELIF-ELSE block is generated using the mesh levels
        # as conditions for which calls are needed:
        #
        #    IF (layer == bottom_layer):
        #        *bottom calls
        #    ELSE IF (layer == top_layer):
        #        *top calls
        #    ELSE:
        #        *top calls
        #        *bottom calls
        #
        # Any extruded top or bottom calls for extruded facets are
        # included within the appropriate mesh-level IF-blocks. If
        # no interior horizontal facet calls are present, then
        # standard IF-blocks are generated for exterior top/bottom
        # facet calls when appropriate:
        #
        #    IF (layer == bottom_layer):
        #        *bottom calls
        #
        #    IF (layer == top_layer):
        #        *top calls
        #
        # The mesh level is an integer provided as a macro kernel
        # argument.

        # FIXME: No variable layers assumption
        statements.append(ast.FlatBlock("/* Mesh levels: */\n"))
        num_layers = ast.Symbol(builder.mesh_layer_count_sym, rank=(0,))
        layer = builder.mesh_layer_sym
        types = ["interior_facet_horiz_top",
                 "interior_facet_horiz_bottom",
                 "exterior_facet_top",
                 "exterior_facet_bottom"]
        decide = [ast.Less(layer, num_layers),
                  ast.Greater(layer, 0),
                  ast.Eq(layer, num_layers),
                  ast.Eq(layer, 0)]
        for (integral_type, which) in zip(types, decide):
            statements.append(ast.If(which, (ast.Block(assembly_calls[integral_type], open_scope=True),)))

    return statements


def terminal_temporaries(builder, declared_temps):
    """Generates statements for assigning auxiliary temporaries
    for nodes in an expression with "high" reference count.
    Expressions which require additional temporaries are provided
    by the :class:`LocalKernelBuilder`.

    :arg builder: The :class:`LocalKernelBuilder` containing
                  all relevant expression information.
    :arg declared_temps: A `dict` keeping track of all declared
                         temporaries. This dictionary is updated
                         as terminal tensors are assigned temporaries.
    """
    statements = [ast.FlatBlock("/* Declare and initialize */\n")]
    for exp in builder.temps:
        t = builder.temps[exp]
        statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))
        declared_temps[exp] = t

    return statements


def parenthesize(arg, prec=None, parent=None):
    """Parenthesizes an expression."""
    if prec is None or parent is None or prec >= parent:
        return arg
    return "(%s)" % arg


def gem_to_loopy(gem_expr, var2terminal, scalar_type):
    """ Method encapsulating stage 2.
    Converts the gem expression dag into imperoc first, and then further into loopy.
    :return slate_loopy: 2-tuple of loopy kernel for slate operations
        and loopy GlobalArg for the output variable.
    """
    # Creation of return variables for outer loopy
    shape = gem_expr.shape if len(gem_expr.shape) != 0 else (1,)
    idx = make_indices(len(shape))
    indexed_gem_expr = gem.Indexed(gem_expr, idx)

    output_loopy_arg = loopy.GlobalArg("output", shape=shape,
                                       dtype=scalar_type,
                                       is_input=True,
                                       is_output=True)
    args = [output_loopy_arg] + [loopy.GlobalArg(var.name, shape=var.shape, dtype=scalar_type)
                                 for var in var2terminal.keys()]
    ret_vars = [gem.Indexed(gem.Variable("output", shape), idx)]

    preprocessed_gem_expr = impero_utils.preprocess_gem([indexed_gem_expr])

    # glue assignments to return variable
    assignments = list(zip(ret_vars, preprocessed_gem_expr))

    # Part A: slate to impero_c
    impero_c = impero_utils.compile_gem(assignments, (), remove_zeros=False)

    # Part B: impero_c to loopy
    output_arg = OutputKernelArg(output_loopy_arg)
    return generate_loopy(impero_c, args, scalar_type, "slate_loopy", []), output_arg


def slate_to_cpp(expr, temps, prec=None):
    """Translates a Slate expression into its equivalent representation in
    the Eigen C++ syntax.

    :arg expr: a :class:`slate.TensorBase` expression.
    :arg temps: a `dict` of temporaries which map a given expression to its
        corresponding representation as a `coffee.Symbol` object.
    :arg prec: an argument dictating the order of precedence in the linear
        algebra operations. This ensures that parentheticals are placed
        appropriately and the order in which linear algebra operations
        are performed are correct.

    Returns:
        a `string` which represents the C/C++ code representation of the
        `slate.TensorBase` expr.
    """
    # If the tensor is terminal, it has already been declared.
    # Coefficients defined as AssembledVectors will have been declared
    # by now, as well as any other nodes with high reference count or
    # matrix factorizations.
    if expr in temps:
        return temps[expr].gencode()

    elif isinstance(expr, slate.Transpose):
        tensor, = expr.operands
        return "(%s).transpose()" % slate_to_cpp(tensor, temps)

    elif isinstance(expr, slate.Inverse):
        tensor, = expr.operands
        return "(%s).inverse()" % slate_to_cpp(tensor, temps)

    elif isinstance(expr, slate.Negative):
        tensor, = expr.operands
        result = "-%s" % slate_to_cpp(tensor, temps, expr.prec)
        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, (slate.Add, slate.Mul)):
        op = {slate.Add: '+',
              slate.Mul: '*'}[type(expr)]
        A, B = expr.operands
        result = "%s %s %s" % (slate_to_cpp(A, temps, expr.prec),
                               op,
                               slate_to_cpp(B, temps, expr.prec))

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, slate.Block):
        tensor, = expr.operands
        indices = expr._indices
        try:
            ridx, cidx = indices
        except ValueError:
            ridx, = indices
            cidx = 0
        rids = as_tuple(ridx)
        cids = as_tuple(cidx)

        # Check if indices are non-contiguous
        if not all(all(ids[i] + 1 == ids[i + 1] for i in range(len(ids) - 1))
                   for ids in (rids, cids)):
            raise NotImplementedError("Non-contiguous blocks not implemented")

        rshape = expr.shape[0]
        rstart = sum(tensor.shapes[0][:min(rids)])
        if expr.rank == 1:
            cshape = 1
            cstart = 0
        else:
            cshape = expr.shape[1]
            cstart = sum(tensor.shapes[1][:min(cids)])

        result = "(%s).block<%d, %d>(%d, %d)" % (slate_to_cpp(tensor,
                                                              temps,
                                                              expr.prec),
                                                 rshape, cshape,
                                                 rstart, cstart)

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, slate.Solve):
        A, B = expr.operands
        result = "%s.solve(%s)" % (slate_to_cpp(A, temps, expr.prec),
                                   slate_to_cpp(B, temps, expr.prec))

        return parenthesize(result, expr.prec, prec)

    else:
        raise NotImplementedError("Type %s not supported.", type(expr))


def eigen_matrixbase_type(shape):
    """Returns the Eigen::Matrix declaration of the tensor.

    :arg shape: a tuple of integers the denote the shape of the
        :class:`slate.TensorBase` object.

    Returns:
        a string indicating the appropriate declaration of the
        `slate.TensorBase` object in the appropriate Eigen C++ template
        library syntax.
    """
    if len(shape) == 0:
        rows = 1
        cols = 1
    elif len(shape) == 1:
        rows = shape[0]
        cols = 1
    else:
        if not len(shape) == 2:
            raise NotImplementedError(
                "%d-rank tensors are not supported." % len(shape)
            )
        rows = shape[0]
        cols = shape[1]
    if cols != 1:
        order = ", Eigen::RowMajor"
    else:
        order = ""

    return "Eigen::Matrix<double, %d, %d%s>" % (rows, cols, order)
