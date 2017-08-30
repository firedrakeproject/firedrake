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

from collections import OrderedDict

from firedrake.constant import Constant
from firedrake.tsfc_interface import SplitKernel, KernelInfo
from firedrake.slate.slate import (TensorBase, Transpose, Inverse,
                                   Negative, Add, Sub, Mul,
                                   Action)
from firedrake.slate.slac.kernel_builder import KernelBuilderBase
from firedrake.slate.slac.utils import Transformer
from firedrake.utils import cached_property
from firedrake import op2

from itertools import chain

from pyop2.utils import get_petsc_dir
from pyop2.datatypes import as_cstr

from tsfc.parameters import SCALAR_TYPE

import numpy as np


__all__ = ['compile_expression']


PETSC_DIR = get_petsc_dir()

cell_to_facets_dtype = np.dtype(np.int8)

# These are globally constant symbols
coord_sym = ast.Symbol("coords")
cell_orientations_sym = ast.Symbol("cell_orientations")
cell_facet_sym = ast.Symbol("cell_facets")
it_sym = ast.Symbol("i0")
mesh_layer_sym = ast.Symbol("layer")

# Supported integral types
supported_integral_types = [
    "cell",
    "interior_facet",
    "exterior_facet",
    # The "interior_facet_horiz" measure is separated into two parts:
    # "top" and "bottom"
    "interior_facet_horiz_top",
    "interior_facet_horiz_bottom",
    "interior_facet_vert",
    "exterior_facet_top",
    "exterior_facet_bottom",
    "exterior_facet_vert"
]


class LocalKernelBuilder(KernelBuilderBase):
    """The primary helper class for constructing cell-local linear
    algebra kernels from Slate expressions.

    This class provides access to all temporaries and subkernels associated
    with a Slate expression. If the Slate expression contains nodes that
    require operations on already assembled data (such as the action of a
    slate tensor on a `ufl.Coefficient`), this class provides access to the
    expression which needs special handling.

    Instructions for assembling the full kernel AST of a Slate expression is
    provided by the method `construct_ast`.
    """
    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the LocalKernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
                              TSFC when constructing subkernels associated
                              with the expression.
        """
        super(LocalKernelBuilder, self).__init__(expression=expression,
                                                 tsfc_parameters=tsfc_parameters)
        transformer = Transformer()
        include_dirs = []
        templated_subkernels = []
        assembly_calls = OrderedDict([(it, []) for it in supported_integral_types])
        coords = None
        oriented = False
        for cxt_kernel in self.context_kernels:
            local_coefficients = cxt_kernel.coefficients
            it_type = cxt_kernel.original_integral_type
            exp = cxt_kernel.tensor

            if it_type not in supported_integral_types:
                raise ValueError("Integral type '%s' not recognized" % it_type)

            # Explicit checking of coordinates
            coordinates = cxt_kernel.tensor.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords, "Mismatching coordinates!"
            else:
                coords = coordinates

            for split_kernel in cxt_kernel.tsfc_kernels:
                indices = split_kernel.indices
                kinfo = split_kernel.kinfo

                # TODO: Implement subdomains for Slate tensors
                if kinfo.subdomain_id != "otherwise":
                    raise NotImplementedError("Subdomains not implemented.")

                args = [c for i in kinfo.coefficient_map
                        for c in self.coefficient(local_coefficients[i])]

                if kinfo.oriented:
                    args.insert(0, cell_orientations_sym)

                if kinfo.integral_type in ["interior_facet",
                                           "exterior_facet",
                                           "interior_facet_vert",
                                           "exterior_facet_vert"]:
                    args.append(ast.FlatBlock("&%s" % it_sym))

                # Assembly calls within the macro kernel
                tensor = eigen_tensor(exp, self.temps[exp], indices)
                call = ast.FunCall(kinfo.kernel.name,
                                   tensor,
                                   coord_sym,
                                   *args)
                assembly_calls[it_type].append(call)

                # Subkernels for local assembly (Eigen templated functions)
                kast = transformer.visit(kinfo.kernel._ast)
                templated_subkernels.append(kast)
                include_dirs.extend(kinfo.kernel._include_dirs)
                oriented = oriented or kinfo.oriented

        self.assembly_calls = assembly_calls
        self.templated_subkernels = templated_subkernels
        self.include_dirs = list(set(include_dirs))
        self.oriented = oriented

    @property
    def integral_type(self):
        """Returns the integral type associated with a Slate kernel."""
        return "cell"

    @cached_property
    def needs_cell_facets(self):
        """Searches for any embedded forms (by inspecting the ContextKernels)
        which require looping over cell facets. If any are found, this function
        returns `True` and `False` otherwise.
        """
        cell_facet_types = ["interior_facet",
                            "exterior_facet",
                            "interior_facet_vert",
                            "exterior_facet_vert"]
        return any(cxt_k.original_integral_type in cell_facet_types
                   for cxt_k in self.context_kernels)

    @cached_property
    def needs_mesh_layers(self):
        """Searches for any embedded forms (by inspecting the ContextKernels)
        which require mesh level information (extrusion measures). If any are
        found, this function returns `True` and `False` otherwise.
        """
        mesh_layer_types = ["interior_facet_horiz_top",
                            "interior_facet_horiz_bottom",
                            "exterior_facet_bottom",
                            "exterior_facet_top"]
        return any(cxt_k.original_integral_type in mesh_layer_types
                   for cxt_k in self.context_kernels)


def compile_expression(slate_expr, tsfc_parameters=None):
    """Takes a Slate expression `slate_expr` and returns the appropriate
    :class:`firedrake.op2.Kernel` object representing the Slate expression.

    :arg slate_expr: a :class:'TensorBase' expression.
    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
                          be passed onto TSFC during the compilation of
                          ufl forms.

    Returns: A `tuple` containing a `SplitKernel(idx, kinfo)`
    """
    if not isinstance(slate_expr, TensorBase):
        raise ValueError("Expecting a `TensorBase` object, not %s" % type(slate_expr))

    # TODO: Get PyOP2 to write into mixed dats
    if slate_expr.is_mixed:
        raise NotImplementedError("Compiling mixed slate expressions")

    if len(slate_expr.ufl_domains()) > 1:
        raise NotImplementedError("Multiple domains not implemented.")

    # If the expression has already been symbolically compiled, then
    # simply reuse the produced kernel.
    if slate_expr._metakernel_cache is not None:
        return slate_expr._metakernel_cache

    if slate_expr.rank == 0:
        # Scalars are treated as 1x1 MatrixBase objects
        shape = (1,)
    else:
        shape = slate_expr.shape

    # Create a builder for the Slate expression
    builder = LocalKernelBuilder(expression=slate_expr,
                                 tsfc_parameters=tsfc_parameters)

    declared_temps = {}
    statements = [ast.FlatBlock("/* Declare and initialize */\n")]
    for exp in builder.temps:
        # Declare and initialize terminal temporaries
        t = builder.temps[exp]
        statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))
        declared_temps[exp] = t

    statements.append(ast.FlatBlock("/* Assemble local tensors */\n"))

    # Cell integrals are straightforward. Just splat them out.
    statements.extend(builder.assembly_calls["cell"])

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
        int_calls = list(chain(*[builder.assembly_calls[it_type]
                                 for it_type in ("interior_facet",
                                                 "interior_facet_vert")]))
        ext_calls = list(chain(*[builder.assembly_calls[it_type]
                                 for it_type in ("exterior_facet",
                                                 "exterior_facet_vert")]))

        # Compute the number of facets to loop over
        domain = slate_expr.ufl_domain()
        if domain.cell_set._extruded:
            num_facets = domain.ufl_cell()._cells[0].num_facets()
        else:
            num_facets = domain.ufl_cell().num_facets()

        if_ext = ast.Eq(ast.Symbol(cell_facet_sym, rank=(it_sym,)), 0)
        if_int = ast.Eq(ast.Symbol(cell_facet_sym, rank=(it_sym,)), 1)
        if int_calls and ext_calls:
            else_if = ast.If(if_ext, (ast.Block(ext_calls, open_scope=True),))
            body = ast.If(if_int, (ast.Block(int_calls, open_scope=True),
                                   else_if))
        elif int_calls:
            body = ast.If(if_int, (ast.Block(int_calls, open_scope=True),))
        elif ext_calls:
            body = ast.If(if_ext, (ast.Block(ext_calls, open_scope=True),))
        else:
            raise RuntimeError("Cell facets are requested, but no facet calls found.")

        statements.append(ast.For(ast.Decl("unsigned int", it_sym, init=0),
                                  ast.Less(it_sym, num_facets),
                                  ast.Incr(it_sym, 1), body))

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
        num_layers = slate_expr.ufl_domain().topological.layers - 1
        int_top = builder.assembly_calls["interior_facet_horiz_top"]
        int_btm = builder.assembly_calls["interior_facet_horiz_bottom"]
        ext_top = builder.assembly_calls["exterior_facet_top"]
        ext_btm = builder.assembly_calls["exterior_facet_bottom"]

        if int_top + int_btm:
            statements.append(ast.FlatBlock("/* Interior and top/bottom calls */\n"))
            # NOTE: The "top" cell has a interior horizontal facet
            # which is locally on the "bottom." And vice versa.
            top_calls = int_btm + ext_top
            btm_calls = int_top + ext_btm

            block_else = ast.Block(int_top + int_btm, open_scope=True)
            block_top = ast.Block(top_calls, open_scope=True)
            block_btm = ast.Block(btm_calls, open_scope=True)

            elif_block = ast.If(ast.Eq(mesh_layer_sym, num_layers - 1),
                                (block_top, block_else))

            statements.append(ast.If(ast.Eq(mesh_layer_sym, 0),
                                     (block_btm, elif_block)))
        else:
            if ext_btm:
                statements.append(ast.FlatBlock("/* Bottom calls */\n"))
                layer = 0
                block_btm = ast.Block(ext_btm, open_scope=True)
                statements.append(ast.If(ast.Eq(mesh_layer_sym, layer),
                                         (block_btm,)))

            if ext_top:
                statements.append(ast.FlatBlock("/* Top calls */\n"))
                layer = num_layers - 1
                block_top = ast.Block(ext_top, open_scope=True)
                statements.append(ast.If(ast.Eq(mesh_layer_sym, layer),
                                         (block_top,)))

        if not ext_btm + ext_top + int_btm + int_top:
            raise RuntimeError("Mesh levels requested, but no level-dependent calls found.")

    # Populate any coefficient temporaries for actions
    if builder.action_coefficients:
        # Action computations require creating coefficient temporaries to
        # compute the matrix-vector product. The temporaries are created by
        # inspecting the function space of the coefficient to compute node
        # and dof extents. The coefficient is then assigned values by looping
        # over both the node extent and dof extent (double FOR-loop). A double
        # FOR-loop is needed for each function space (if the function space is
        # mixed, then a loop will be constructed for each component space).
        # The general structure of each coefficient loop will be:
        #
        #    FOR (i1=0; i1<node_extent; i1++):
        #        FOR (j1=0; j1<dof_extent; j1++):
        #            wT0[offset + (dof_extent * i1) + j1] = w_0_0[i1][j1]
        #            wT1[offset + (dof_extent * i1) + j1] = w_1_0[i1][j1]
        #            .
        #            .
        #            .
        #
        # where wT0, wT1, ... are temporaries for coefficients sharing the
        # same function space. The offset is computed based on whether
        # the function space is mixed. The offset is always 0 for non-mixed
        # coefficients. If the coefficient is mixed, then the offset is
        # incremented by the total number of nodal unknowns associated with
        # the component spaces of the mixed space.

        statements.append(ast.FlatBlock("/* Coefficient temporaries */\n"))
        i_sym = ast.Symbol("i1")
        j_sym = ast.Symbol("j1")
        loops = [ast.FlatBlock("/* Loops for coefficient temps */\n")]

        # clist is a list of tuples (i, shp, c) where i is the function space
        # index, shp is the shape of the coefficient temp, and c is the
        # coefficient
        for (nodes, dofs), clist in builder.action_coefficients.items():
            # Collect all coefficients which share the same node/dof extent
            assignments = []
            for (fs, offset, shp_info, actee) in clist:
                if actee not in declared_temps:
                    # Declare and initialize coefficient temporary
                    c_type = eigen_matrixbase_type(shape=(shp_info,))
                    t = ast.Symbol("wT%d" % len(declared_temps))
                    statements.append(ast.Decl(c_type, t))
                    statements.append(ast.FlatBlock("%s.setZero();\n" % t))

                    # Assigning coefficient values into temporary
                    coeff_sym = ast.Symbol(builder.coefficient(actee)[fs],
                                           rank=(i_sym, j_sym))
                    index = ast.Sum(offset,
                                    ast.Sum(ast.Prod(dofs, i_sym), j_sym))
                    coeff_temp = ast.Symbol(t, rank=(index,))
                    assignments.append(ast.Assign(coeff_temp, coeff_sym))
                    declared_temps[actee] = t

            # Inner-loop running over dof extent
            inner_loop = ast.For(ast.Decl("unsigned int", j_sym, init=0),
                                 ast.Less(j_sym, dofs),
                                 ast.Incr(j_sym, 1),
                                 assignments)

            # Outer-loop running over node extent
            loop = ast.For(ast.Decl("unsigned int", i_sym, init=0),
                           ast.Less(i_sym, nodes),
                           ast.Incr(i_sym, 1),
                           inner_loop)

            loops.append(loop)

        statements.extend(loops)

    # Now we handle any terms that require auxiliary temporaries
    if builder.aux_exprs:
        statements.append(ast.FlatBlock("/* Auxiliary temporaries */\n"))
        results = [ast.FlatBlock("/* Assign auxiliary temps */\n")]
        for exp in builder.aux_exprs:
            if exp not in declared_temps:
                t = ast.Symbol("auxT%d" % len(declared_temps))
                result = metaphrase_slate_to_cpp(exp, declared_temps)
                tensor_type = eigen_matrixbase_type(shape=exp.shape)
                statements.append(ast.Decl(tensor_type, t))
                statements.append(ast.FlatBlock("%s.setZero();\n" % t))
                results.append(ast.Assign(t, result))
                declared_temps[exp] = t

        statements.extend(results)

    # Now we create the result statement by declaring its eigen type and
    # using Eigen::Map to move between Eigen and C data structs.
    statements.append(ast.FlatBlock("/* Map eigen tensor into C struct */\n"))
    result_sym = ast.Symbol("T%d" % len(declared_temps))
    result_data_sym = ast.Symbol("A%d" % len(declared_temps))
    result_type = "Eigen::Map<%s >" % eigen_matrixbase_type(shape)
    result = ast.Decl(SCALAR_TYPE, ast.Symbol(result_data_sym, shape))
    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" % (result_type,
                                                             result_sym,
                                                             SCALAR_TYPE,
                                                             result_data_sym))
    statements.append(result_statement)

    # Generate the complete c++ string performing the linear algebra operations
    # on Eigen matrices/vectors
    statements.append(ast.FlatBlock("/* Linear algebra expression */\n"))
    cpp_string = ast.FlatBlock(metaphrase_slate_to_cpp(slate_expr,
                                                       declared_temps))
    statements.append(ast.Incr(result_sym, cpp_string))

    # Generate arguments for the macro kernel
    args = [result, ast.Decl("%s **" % SCALAR_TYPE, coord_sym)]

    # Orientation information
    if builder.oriented:
        args.append(ast.Decl("int **", cell_orientations_sym))

    # Coefficient information
    expr_coeffs = slate_expr.coefficients()
    for c in expr_coeffs:
        if isinstance(c, Constant):
            ctype = "%s *" % SCALAR_TYPE
        else:
            ctype = "%s **" % SCALAR_TYPE
        args.extend([ast.Decl(ctype, csym) for csym in builder.coefficient(c)])

    # Facet information
    if builder.needs_cell_facets:
        args.append(ast.Decl("%s *" % as_cstr(cell_to_facets_dtype),
                             cell_facet_sym))

    # NOTE: We need to be careful about the ordering here. Mesh layers are
    # added as the final argument to the kernel.
    if builder.needs_mesh_layers:
        args.append(ast.Decl("int", mesh_layer_sym))

    # Macro kernel
    macro_kernel_name = "compile_slate"
    stmts = ast.Block(statements)
    macro_kernel = ast.FunDecl("void", macro_kernel_name, args,
                               stmts, pred=["static", "inline"])

    # Construct the final ast
    kernel_ast = ast.Node(builder.templated_subkernels + [macro_kernel])

    # Now we wrap up the kernel ast as a PyOP2 kernel and include the
    # Eigen header files
    include_dirs = builder.include_dirs
    include_dirs.extend(["%s/include/eigen3/" % d for d in PETSC_DIR])
    op2kernel = op2.Kernel(kernel_ast,
                           macro_kernel_name,
                           cpp=True,
                           include_dirs=include_dirs,
                           headers=['#include <Eigen/Dense>',
                                    '#define restrict __restrict'])

    # Send back a "TSFC-like" SplitKernel object with an
    # index and KernelInfo
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type=builder.integral_type,
                       oriented=builder.oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=tuple(range(len(expr_coeffs))),
                       needs_cell_facets=builder.needs_cell_facets,
                       pass_layer_arg=builder.needs_mesh_layers)

    # Cache the resulting kernel
    idx = tuple([0]*slate_expr.rank)
    kernel = (SplitKernel(idx, kinfo),)
    slate_expr._metakernel_cache = kernel

    return kernel


def parenthesize(arg, prec=None, parent=None):
    """Parenthesizes an expression."""
    if prec is None or parent is None or prec >= parent:
        return arg
    return "(%s)" % arg


def metaphrase_slate_to_cpp(expr, temps, prec=None):
    """Translates a Slate expression into its equivalent representation in
    the Eigen C++ syntax.

    :arg expr: a :class:`slate.TensorBase` expression.
    :arg temps: a `dict` of temporaries which map a given expression to its
                corresponding representation as a `coffee.Symbol` object.
    :arg prec: an argument dictating the order of precedence in the linear
               algebra operations. This ensures that parentheticals are placed
               appropriately and the order in which linear algebra operations
               are performed are correct.

    Returns
        This function returns a `string` which represents the C/C++ code
        representation of the `slate.TensorBase` expr.
    """
    # If the tensor is terminal, it has already been declared.
    # Coefficients in action expressions will have been declared by now,
    # as well as any other nodes with high reference count.
    if expr in temps:
        return temps[expr].gencode()

    elif isinstance(expr, Transpose):
        tensor, = expr.operands
        return "(%s).transpose()" % metaphrase_slate_to_cpp(tensor, temps)

    elif isinstance(expr, Inverse):
        tensor, = expr.operands
        return "(%s).inverse()" % metaphrase_slate_to_cpp(tensor, temps)

    elif isinstance(expr, Negative):
        tensor, = expr.operands
        result = "-%s" % metaphrase_slate_to_cpp(tensor, temps, expr.prec)
        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, (Add, Sub, Mul)):
        op = {Add: '+',
              Sub: '-',
              Mul: '*'}[type(expr)]
        A, B = expr.operands
        result = "%s %s %s" % (metaphrase_slate_to_cpp(A, temps, expr.prec),
                               op,
                               metaphrase_slate_to_cpp(B, temps, expr.prec))

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, Action):
        tensor, = expr.operands
        c, = expr.actee
        result = "(%s) * %s" % (metaphrase_slate_to_cpp(tensor,
                                                        temps,
                                                        expr.prec), temps[c])
        return parenthesize(result, expr.prec, prec)

    else:
        raise NotImplementedError("Type %s not supported.", type(expr))


def eigen_matrixbase_type(shape):
    """Returns the Eigen::Matrix declaration of the tensor.

    :arg shape: a tuple of integers the denote the shape of the
                :class:`slate.TensorBase` object.

    Returns: Returns a string indicating the appropriate declaration of the
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


def eigen_tensor(expr, temporary, index):
    """Returns an appropriate assignment statement for populating a particular
    `Eigen::MatrixBase` tensor. If the tensor is mixed, then access to the
    :meth:`block` of the eigen tensor is provided. Otherwise, no block
    information is needed and the tensor is returned as is.

    :arg expr: a `slate.Tensor` node.
    :arg temporary: the associated temporary of the expr argument.
    :arg index: a tuple of integers used to determine row and column
                information. This is provided by the SplitKernel
                associated with the expr.
    """
    if expr.is_mixed:
        try:
            row, col = index
        except ValueError:
            row = index[0]
            col = 0
        rshape = expr.shapes[0][row]
        rstart = sum(expr.shapes[0][:row])
        try:
            cshape = expr.shapes[1][col]
            cstart = sum(expr.shapes[1][:col])
        except KeyError:
            cshape = 1
            cstart = 0

        tensor = ast.FlatBlock("%s.block<%d, %d>(%d, %d)" % (temporary,
                                                             rshape, cshape,
                                                             rstart, cstart))
    else:
        tensor = temporary

    return tensor
