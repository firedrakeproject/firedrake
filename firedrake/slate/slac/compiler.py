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

supported_integral_types = [
    "cell",
    "interior_facet",
    "exterior_facet",
    "interior_facet_horiz_top",
    "interior_facet_horiz_bottom",
    "interior_facet_vert",
    "exterior_facet_top",
    "exterior_facet_bottom",
    "exterior_facet_vert"
]


class KernelBuilder(KernelBuilderBase):
    """
    """

    def __init__(self, expression, tsfc_parameters=None):
        """
        """
        super(KernelBuilder, self).__init__(expression=expression,
                                            tsfc_parameters=tsfc_parameters)
        self._organize_assembly_calls()

    def _organize_assembly_calls(self):
        """
        """
        transformer = Transformer()
        include_dirs = []
        templated_subkernels = []
        assembly_calls = OrderedDict()
        for k in supported_integral_types:
            assembly_calls[k] = []
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

                # TSFC No-op kernels do not contain a kernel body. If this is
                # the case, we don't create a function call.
                if kinfo.kernel:
                    # TODO: Implement subdomains for Slate tensors
                    if kinfo.subdomain_id != "otherwise":
                        raise NotImplementedError("Subdomains not implemented.")

                    args = [c for i in kinfo.coefficient_map
                            for c in self.coefficient(local_coefficients[i])]

                    oriented = oriented or kinfo.oriented
                    if oriented:
                        args.insert(0, cell_orientations_sym)

                    if kinfo.integral_type in ["interior_facet",
                                               "exterior_facet",
                                               "interior_facet_vert",
                                               "exterior_facet_vert"]:
                        args.append(ast.FlatBlock("&%s" % it_sym))

                    # Assembly calls within the macro kernel
                    tensor = eigen_tensor(exp, self.temps[exp], indices)
                    include_dirs.extend(kinfo.kernel._include_dirs)
                    call = ast.FunCall(kinfo.kernel.name,
                                       tensor,
                                       coord_sym,
                                       *args)
                    assembly_calls[it_type].append(call)

                    # Subkernels for local assembly (Eigen templated functions)
                    kast = transformer.visit(kinfo.kernel._ast)
                    templated_subkernels.append(kast)

        self.assembly_calls = assembly_calls
        self.templated_subkernels = templated_subkernels
        self.include_dirs = list(set(include_dirs))
        self.oriented = oriented

    @cached_property
    def needs_cell_facets(self):
        """
        """
        cell_facet_types = ["interior_facet",
                            "exterior_facet",
                            "interior_facet_vert",
                            "exterior_facet_vert"]
        return any(self.assembly_calls[it] for it in cell_facet_types)

    @cached_property
    def needs_mesh_layers(self):
        """
        """
        mesh_layer_types = ["interior_facet_horiz_top",
                            "interior_facet_horiz_bottom",
                            "exterior_facet_bottom",
                            "exterior_facet_top"]
        return any(self.assembly_calls[it] for it in mesh_layer_types)

    def construct_ast(self, macro_kernels):
        """Constructs the final kernel AST.

        :arg macro_kernels: A `list` of macro kernel functions, which
                            call subkernels and perform elemental
                            linear algebra.

        Returns: The complete kernel AST as a COFFEE `ast.Node`
        """
        assert isinstance(macro_kernels, list), (
            "Please wrap all macro kernel functions in a list"
        )
        kernel_ast = self.templated_subkernels
        kernel_ast.extend(macro_kernels)

        return ast.Node(kernel_ast)


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
        raise ValueError(
            "Expecting a `TensorBase` expression, not %s" % type(slate_expr)
        )

    # TODO: Get PyOP2 to write into mixed dats
    if slate_expr.is_mixed:
        raise NotImplementedError("Compiling mixed slate expressions")

    # If the expression has already been symbolically compiled, then
    # simply reuse the produced kernel.
    if slate_expr._metakernel_cache is not None:
        return slate_expr._metakernel_cache

    # Initialize coefficients, shape and statements list
    expr_coeffs = slate_expr.coefficients()

    # We treat scalars as 1x1 MatrixBase objects, so we give
    # the right shape to do so and everything just falls out.
    # This bit here ensures the return result has the right
    # shape
    if slate_expr.rank == 0:
        shape = (1,)
    else:
        shape = slate_expr.shape

    statements = []

    # Create a builder for the Slate expression
    builder = KernelBuilder(expression=slate_expr,
                            tsfc_parameters=tsfc_parameters)

    # We keep track of temporaries that have been declared
    declared_temps = {}
    for exp in builder.temps:
        t = builder.temps[exp]

        if exp not in declared_temps:
            # Declare and initialize the temporary
            statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
            statements.append(ast.FlatBlock("%s.setZero();\n" % t))
            declared_temps[exp] = t

    statements.extend(builder.assembly_calls["cell"])

    if builder.needs_cell_facets:
        chker = {"interior_facet": 1,
                 "exterior_facet": 0,
                 "interior_facet_vert": 1,
                 "exterior_facet_vert": 0}
        int_calls = []
        ext_calls = []
        for it_type in chker:
            if chker[it_type] == 1:
                int_calls.extend(builder.assembly_calls[it_type])
            else:
                ext_calls.extend(builder.assembly_calls[it_type])

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
            raise RuntimeError("Cell facets are needed, but no facet calls are found.")

        statements.append(ast.For(ast.Decl("unsigned int", it_sym, init=0),
                                  ast.Less(it_sym, num_facets),
                                  ast.Incr(it_sym, 1), body))

    if builder.needs_mesh_layers:
        # FIXME: No variable layers assumption
        num_layers = slate_expr.ufl_domain().topological.layers - 1

        # In the presence of interior horizontal facets, we can use the
        # if(top)---elif(bottom)---else(top and bottom) structure. Any
        # extruded top or bottom calls for extruded facets are included
        # within the appropriate mesh-level if-blocks (if present).
        int_top = builder.assembly_calls["interior_facet_horiz_top"]
        int_btm = builder.assembly_calls["interior_facet_horiz_bottom"]
        ext_top = builder.assembly_calls["exterior_facet_top"]
        ext_btm = builder.assembly_calls["exterior_facet_bottom"]
        if int_top + int_btm:
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

        # No interior horizontal facets. Just standard if-blocks.
        else:
            if ext_btm:
                layer = 0
                block_btm = ast.Block(ext_btm, open_scope=True)
                statements.append(ast.If(ast.Eq(mesh_layer_sym, layer),
                                         (block_btm,)))

            if ext_top:
                layer = num_layers - 1
                block_top = ast.Block(ext_top, open_scope=True)
                statements.append(ast.If(ast.Eq(mesh_layer_sym, layer),
                                         (block_top,)))

    # Now we handle any terms that require auxiliary temporaries
    if builder.aux_exprs:
        aux_statements = auxiliary_temporaries(builder, declared_temps)
        statements.extend(aux_statements)

    # Now we create the result statement by declaring its eigen type and
    # using Eigen::Map to move between Eigen and C data structs.
    result_sym = ast.Symbol("T%d" % len(builder.temps))
    result_data_sym = ast.Symbol("A%d" % len(builder.temps))
    result_type = "Eigen::Map<%s >" % eigen_matrixbase_type(shape)
    result = ast.Decl(SCALAR_TYPE, ast.Symbol(result_data_sym, shape))
    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" % (result_type,
                                                             result_sym,
                                                             SCALAR_TYPE,
                                                             result_data_sym))
    statements.append(result_statement)

    # Generate the complete c++ string performing the linear algebra operations
    # on Eigen matrices/vectors
    cpp_string = ast.FlatBlock(metaphrase_slate_to_cpp(slate_expr,
                                                       declared_temps))
    statements.append(ast.Incr(result_sym, cpp_string))

    # Generate arguments for the macro kernel
    args = [result, ast.Decl("%s **" % SCALAR_TYPE, coord_sym)]

    # Orientation information
    if builder.oriented:
        args.append(ast.Decl("int **", cell_orientations_sym))

    # Coefficient information
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

    # NOTE: In the future we may want to have more than one "macro_kernel"
    macro_kernel_name = "compile_slate"
    stmt = ast.Block(statements)
    macro_kernel = builder.construct_macro_kernel(name=macro_kernel_name,
                                                  args=args,
                                                  statements=stmt)

    # Tell the builder to construct the final ast
    kernel_ast = builder.construct_ast([macro_kernel])

    # Now we wrap up the kernel ast as a PyOP2 kernel.
    # Include the Eigen header files
    include_dirs = builder.include_dirs
    include_dirs.extend(["%s/include/eigen3/" % d for d in PETSC_DIR])
    op2kernel = op2.Kernel(kernel_ast,
                           macro_kernel_name,
                           cpp=True,
                           include_dirs=include_dirs,
                           headers=['#include <Eigen/Dense>',
                                    '#define restrict __restrict'])

    assert len(slate_expr.ufl_domains()) == 1, (
        "No support for multiple domains yet!"
    )

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

    idx = tuple([0]*slate_expr.rank)

    kernels = (SplitKernel(idx, kinfo),)

    # Store the resulting kernel for reuse
    slate_expr._metakernel_cache = kernels

    return kernels


def auxiliary_temporaries(builder, declared_temps):
    """This function generates auxiliary information regarding special
    handling of expressions that require creating additional temporaries.

    :arg builder: a :class:`KernelBuilder` object that contains all the
                  necessary temporary and expression information.
    :arg declared_temps: a `dict` of temporaries that have already been
                         declared and assigned values. This will be
                         updated in this method and referenced later
                         in the compiler.
    Returns: a list of auxiliary statements are returned that contain temporary
             declarations and any code-blocks needed to evaluate the
             expression.
    """
    aux_statements = []
    for exp in builder.aux_exprs:
        if isinstance(exp, Action):
            # Action computations are relatively inexpensive, so
            # we don't waste memory space on creating temps for
            # these expressions. However, we must create a temporary
            # for the actee coefficient (if we haven't already).
            actee, = exp.actee
            if actee not in declared_temps:
                # Declare a temporary for the coefficient
                V = actee.function_space()
                shape_array = [(Vi.finat_element.space_dimension(),
                                np.prod(Vi.shape))
                               for Vi in V.split()]
                ctemp = ast.Symbol("auxT%d" % len(declared_temps))
                shape = sum(n * d for (n, d) in shape_array)
                typ = eigen_matrixbase_type(shape=(shape,))
                aux_statements.append(ast.Decl(typ, ctemp))
                aux_statements.append(ast.FlatBlock("%s.setZero();\n" % ctemp))

                # Now we populate the temporary with the coefficient
                # information and insert in the right place.
                offset = 0
                for i, shp in enumerate(shape_array):
                    node_extent, dof_extent = shp
                    # Now we unpack the function and insert its entries into a
                    # 1D vector temporary
                    isym = ast.Symbol("i1")
                    jsym = ast.Symbol("j1")
                    tensor_index = ast.Sum(offset,
                                           ast.Sum(ast.Prod(dof_extent,
                                                            isym), jsym))

                    # Inner-loop running over dof_extent
                    coeff_sym = ast.Symbol(builder.coefficient(actee)[i],
                                           rank=(isym, jsym))
                    coeff_temp = ast.Symbol(ctemp, rank=(tensor_index,))
                    inner_loop = ast.For(ast.Decl("unsigned int", jsym,
                                                  init=0),
                                         ast.Less(jsym, dof_extent),
                                         ast.Incr(jsym, 1),
                                         ast.Assign(coeff_temp, coeff_sym))
                    # Outer-loop running over node_extent
                    loop = ast.For(ast.Decl("unsigned int", isym, init=0),
                                   ast.Less(isym, node_extent),
                                   ast.Incr(isym, 1),
                                   inner_loop)

                    aux_statements.append(loop)
                    offset += node_extent * dof_extent

                # Update declared temporaries with the coefficient
                declared_temps[actee] = ctemp

        elif isinstance(exp, (Inverse, Transpose, Negative, Add, Sub, Mul)):
            if exp not in declared_temps:
                # Get the temporary for the particular expression
                result = metaphrase_slate_to_cpp(exp, declared_temps)

                # Now we use the generated result and assign the value to the
                # corresponding temporary.
                temp = ast.Symbol("auxT%d" % len(declared_temps))
                shape = exp.shape
                aux_statements.append(ast.Decl(eigen_matrixbase_type(shape), temp))
                aux_statements.append(ast.FlatBlock("%s.setZero();\n" % temp))
                aux_statements.append(ast.Assign(temp, result))

                # Update declared temporaries
                declared_temps[exp] = temp

        else:
            raise NotImplementedError(
                "Auxiliary expr type %s not currently implemented." % type(exp)
            )

    return aux_statements


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
