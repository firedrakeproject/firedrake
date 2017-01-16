"""This is SLATE's Linear Algebra Compiler. This module is
responsible for generating C++ kernel functions representing
symbolic linear algebra expressions written in SLATE.

This linear algebra compiler uses both Firedrake's form compiler,
the Two-Stage Form Compiler (TSFC) and COFFEE's kernel abstract
syntax tree (AST) optimizer. TSFC provides this compiler with
appropriate kernel functions (in C) for evaluating integral
expressions (finite element variational forms written in UFL).
COFFEE's AST optimizing framework produces the resulting kernel AST
returned by: `compile_slate_expression`.

The Eigen C++ library (http://eigen.tuxfamily.org/) is required, as
all low-level numerical linear algebra operations are performed using
this templated function library.
"""
from __future__ import absolute_import, print_function, division
from six.moves import range

from coffee import base as ast

from firedrake.constant import Constant
from firedrake.tsfc_interface import SplitKernel, KernelInfo
from firedrake.slate.slate import (TensorBase, Tensor,
                                   Transpose, Inverse, Negative,
                                   Add, Sub, Mul, Action)
from firedrake.slate.slac.kernel_builder import KernelBuilder
from firedrake import op2

from pyop2.utils import get_petsc_dir

from tsfc.parameters import SCALAR_TYPE

from ufl import Coefficient, TensorProductCell

import numpy as np


__all__ = ['compile_expression']


PETSC_DIR = get_petsc_dir()

supported_integral_types = [
    "cell",
    "interior_facet",
    "exterior_facet",
    "interior_facet_horiz",
    "interior_facet_vert",
    "exterior_facet_top",
    "exterior_facet_bottom",
    "exterior_facet_vert"
]


def compile_expression(slate_expr, tsfc_parameters=None):
    """Takes a SLATE expression `slate_expr` and returns the appropriate
    :class:`firedrake.op2.Kernel` object representing the SLATE expression.

    :arg slate_expr: a :class:'TensorBase' expression.
    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
                          be passed onto TSFC during the compilation of ufl forms.
    """
    if not isinstance(slate_expr, TensorBase):
        raise ValueError("Expecting a `slate.TensorBase` expression, not a %r" % slate_expr)

    # TODO: Get PyOP2 to write into mixed dats
    if any(len(a.function_space()) > 1 for a in slate_expr.arguments()):
        raise NotImplementedError("Compiling mixed slate expressions")

    # Initialize shape and statements list
    shape = slate_expr.shape
    statements = []

    # Create a builder for the SLATE expression
    builder = KernelBuilder(expression=slate_expr, tsfc_parameters=tsfc_parameters)

    # Initialize coordinate and facet symbols
    coordsym = ast.Symbol("coords")
    coords = None
    cellfacetsym = ast.Symbol("cell_facets")
    inc = []

    # Special kernels dict for handling horizontal extruded case
    # TODO: Avoid creating this variable?
    extruded_horiz_kernels = {}
    # Now we construct the list of statements to provide to the builder
    context_temps = builder.temps.copy()
    for exp, t in context_temps.items():
        statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))

        for splitkernel in builder.kernel_exprs[exp]:
            clist = []
            index = splitkernel.indices
            kinfo = splitkernel.kinfo
            integral_type = kinfo.integral_type

            if integral_type not in supported_integral_types:
                raise NotImplementedError("Integral type %s not currently supported." % integral_type)

            coordinates = exp.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords
            else:
                coords = coordinates

            for cindex in kinfo.coefficient_map:
                c = exp.coefficients()[cindex]
                # Handles both mixed and non-mixed coefficient cases
                clist.extend(builder.extract_coefficient(c))

            inc.extend(kinfo.kernel._include_dirs)

            tensor = eigen_tensor(exp, t, index)

            # Most facets are treated the same way: by looping over facet index.
            if integral_type in ["interior_facet",
                                 "exterior_facet",
                                 "interior_facet_vert",
                                 "exterior_facet_vert"]:
                # Require cell facets
                builder.require_cell_facets()
                ufl_cell = exp.ufl_domain().ufl_cell()
                loop = facet_integral_loop(kinfo, tensor, ufl_cell, coordsym, cellfacetsym)
                statements.append(loop)

            # The horizontal facets in the extruded case require special treatment.
            # Instead of looping over facets, we loop over levels of the cell. This will
            # determine when we call the "exterior_facet_top" and "exterior_facet_bottom"
            # kernels.
            elif integral_type in ["interior_facet_horiz",
                                   "exterior_facet_bottom",
                                   "exterior_facet_top"]:
                extruded_horiz_kernels[exp] = builder.kernel_exprs[exp]
            else:
                statements.append(ast.FunCall(kinfo.kernel.name, tensor, coordsym, *clist))

    # Now we handle the extruded horizontal facets
    if bool(extruded_horiz_kernels):
        for exp, klist in extruded_horiz_kernels.items():
            assert isinstance(exp.ufl_domain().ufl_cell(), TensorProductCell)
            assert len(klist) == 2, "Must have a top and bottom kernel"
            levels = 2
            lsym = ast.Symbol("level")

            top_kernel = [splitkernel for splitkernel in klist
                          if splitkernel.kinfo.integral_type == "exterior_facet_top"][0]
            bottom_kernel = [splitkernel for splitkernel in klist
                             if splitkernel.kinfo.integral_type == "exterior_facet_bottom"][0]

            # Create dummy variable
            # E = ast.Symbol("E")
            iftop = [ast.If(ast.Eq(lsym, 1),
                            [ast.Block([ast.FunCall(top_kernel.kinfo.kernel.name,
                                                    eigen_tensor(exp, context_temps[exp],
                                                                 top_kernel.indices),
                                                    coordsym)])])]
            ifbottom = [ast.If(ast.Eq(lsym, 0),
                               [ast.Block([ast.FunCall(bottom_kernel.kinfo.kernel.name,
                                                       eigen_tensor(exp, context_temps[exp],
                                                                    bottom_kernel.indices),
                                                       coordsym)])])]
            body = ifbottom + iftop  # + [ast.Incr(context_temps[exp], E)]
            # statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), E))
            loop = ast.For(ast.Decl("unsigned int", lsym, init=0), ast.Less(lsym, levels),
                           ast.Incr(lsym, 1), body)
            statements.append(loop)

    # Now we handle any terms that require auxiliary data (if any)
    if bool(builder.aux_exprs):
        aux_temps, aux_statements = auxiliary_information(builder)
        context_temps.update(aux_temps)
        statements.extend(aux_statements)

    result_sym = ast.Symbol("T%d" % len(builder.temps))
    result_data_sym = ast.Symbol("A%d" % len(builder.temps))
    result_type = "Eigen::Map<%s >" % eigen_matrixbase_type(shape)
    result = ast.Decl(SCALAR_TYPE, ast.Symbol(result_data_sym, shape))
    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" % (result_type,
                                                             result_sym,
                                                             SCALAR_TYPE,
                                                             result_data_sym))
    statements.append(result_statement)

    cpp_string = ast.FlatBlock(metaphrase_slate_to_cpp(slate_expr, context_temps))
    statements.append(ast.Assign(result_sym, cpp_string))

    # Generate arguments for the macro kernel
    args = [result, ast.Decl("%s **" % SCALAR_TYPE, coordsym)]
    for c in slate_expr.coefficients():
        if isinstance(c, Constant):
            ctype = "%s *" % SCALAR_TYPE
        else:
            ctype = "%s **" % SCALAR_TYPE
        args.extend([ast.Decl(ctype, sym_c) for sym_c in builder.extract_coefficient(c)])

    if builder.needs_cell_facets:
        args.append(ast.Decl("char *", cellfacetsym))

    macro_kernel_name = "compile_slate"
    kernel_ast, oriented = builder.construct_ast(name=macro_kernel_name,
                                                 args=args,
                                                 statements=ast.Block(statements))

    inc.extend(["%s/include/eigen3/" % d for d in PETSC_DIR])
    op2kernel = op2.Kernel(kernel_ast, macro_kernel_name, cpp=True, include_dirs=inc,
                           headers=['#include <Eigen/Dense>', '#define restrict __restrict'])

    assert len(slate_expr.ufl_domains()) == 1
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type="cell",
                       oriented=oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=tuple(range(len(slate_expr.coefficients()))),
                       needs_cell_facets=builder.needs_cell_facets)
    idx = tuple([0]*slate_expr.rank)

    return (SplitKernel(idx, kinfo),)


def facet_integral_loop(kinfo, tensor, ufl_cell, coordsym, cellfacetsym):
    """Generates a integral facet loop corresponding to a particular facet integral
    type as a COFFEE AST node.

    :arg kinfo: a `SplitKernel` object containing information about the compiled form.
    :arg tensor: a string describing the `Eigen::MatrixBase` declaration of the tensor.
    :arg ufl_cell: a `ufl.Cell` object that the integral form is defined on. This parameter
                   is used to help determine the appropriate number of facets located on
                   a particular cell.
    :arg clist: a `list` of coefficient information described as a COFFEE AST node.
    :arg coffee_symbols: a `tuple` of `coffee.Symbol` objects representing the following:
                         (1) the coordinates provided to the kernel function, (2) the
                         iteration symbol for the facet loops and (3) the cell facet index.

    Returns: a `coffee.For` object describing the loop needed to locally evaluate the facet
             integral.
    """
    integral_type = kinfo.integral_type
    itsym = ast.Symbol("i0")

    # Compute the correct number of facets for a particular facet measure
    if integral_type in ["interior_facet", "exterior_facet"]:
        # Non-extruded case
        nfacet = ufl_cell.num_facets()

    elif integral_type in ["interior_facet_vert", "exterior_facet_vert"]:
        assert isinstance(ufl_cell, TensorProductCell)
        # Extrusion case
        A, _ = ufl_cell._cells
        nfacet = A.num_facets()

    else:
        raise ValueError("Integral type %s not currently supported." % integral_type)

    if integral_type in ["exterior_facet", "exterior_facet_vert"]:
        checker = 1
    else:
        checker = 0

    loop_body = ast.If(ast.Eq(ast.Symbol(cellfacetsym, rank=(itsym,)), checker),
                       [ast.Block([ast.FunCall(kinfo.kernel.name,
                                               tensor,
                                               coordsym,
                                               ast.FlatBlock("&%s" % itsym))],
                                  open_scope=True)])
    loop = ast.For(ast.Decl("unsigned int", itsym, init=0), ast.Less(itsym, nfacet),
                   ast.Incr(itsym, 1), loop_body)

    return loop


def auxiliary_information(builder):
    """This function generates any auxiliary information regarding special
    handling of expressions that do not have any integral forms or subkernels
    associated with it.

    :arg builder: a :class:`SlateKernelBuilder` object that contains all the
                  necessary temporary and expression information.

    Returns: a mapping of the form ``{aux_node: aux_temp}``, where `aux_node`
             is an already assembled data-object provided as a `ufl.Coefficient`
             and `aux_temp` is the corresponding temporary.

             a list of auxiliary statements are returned that contain temporary
             declarations and any code-blocks needed to evaluate the expression.
    """
    aux_temps = {}
    aux_statements = []
    for i, exp in enumerate(builder.aux_exprs):
        if isinstance(exp, Action):
            acting_coefficient = exp.operands[1]
            assert isinstance(acting_coefficient, Coefficient)

            temp = ast.Symbol("C%d" % i)
            V = acting_coefficient.function_space()
            node_extent = V.fiat_element.space_dimension()
            dof_extent = np.prod(V.ufl_element().value_shape())
            aux_statements.append(ast.Decl(eigen_matrixbase_type(shape=(dof_extent * node_extent,)), temp))
            aux_statements.append(ast.FlatBlock("%s.setZero();\n" % temp))

            # Now we unpack the coefficient and insert its entries into a 1D vector temporary
            isym = ast.Symbol("i1")
            jsym = ast.Symbol("j1")
            tensor_index = ast.Sum(ast.Prod(dof_extent, isym), jsym)
            # Inner-loop running over dof_extent
            inner_loop = ast.For(ast.Decl("unsigned int", jsym, init=0),
                                 ast.Less(jsym, dof_extent),
                                 ast.Incr(jsym, 1),
                                 ast.Assign(ast.Symbol(temp, rank=(tensor_index,)),
                                            ast.Symbol(builder.coefficient_map[acting_coefficient],
                                                       rank=(isym, jsym))))
            # Outer-loop running over node_extent
            loop = ast.For(ast.Decl("unsigned int", isym, init=0),
                           ast.Less(isym, node_extent),
                           ast.Incr(isym, 1),
                           inner_loop)

            aux_statements.append(loop)
            aux_temps[acting_coefficient] = temp
        else:
            raise NotImplementedError("Auxiliary expression type %s not currently implemented." % type(exp))

    return aux_temps, aux_statements


def parenthesize(arg, prec=None, parent=None):
    """Parenthesizes an expression."""
    if prec is None or prec >= parent:
        return arg
    return "(%s)" % arg


def metaphrase_slate_to_cpp(expr, temps, prec=None):
    """Translates a SLATE expression into its equivalent representation in
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
    if isinstance(expr, Tensor):
        return temps[expr].gencode()

    elif isinstance(expr, Transpose):
        tensor = expr.operands[0]
        return "(%s).transpose()" % metaphrase_slate_to_cpp(tensor, temps)

    elif isinstance(expr, Inverse):
        tensor = expr.operands[0]
        return "(%s).inverse()" % metaphrase_slate_to_cpp(tensor, temps)

    elif isinstance(expr, Negative):
        tensor = expr.operands[0]
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
        tensor, c = expr.operands
        # Extra check
        assert isinstance(c, Coefficient)
        result = "(%s) * %s" % (metaphrase_slate_to_cpp(tensor, temps, expr.prec), temps[c])

        return parenthesize(result, expr.prec, prec)
    else:
        # If expression is not recognized, throw a NotImplementedError.
        raise NotImplementedError("Expression of type %s not supported.", type(expr).__name__)


def eigen_matrixbase_type(shape):
    """Returns the Eigen::Matrix declaration of the tensor.

    :arg shape: a tuple of integers the denote the shape of the
                :class:`slate.TensorBase` object.

    Returns: Returns a string indicating the appropriate declaration of the
             `slate.TensorBase` object in the appropriate Eigen C++ template
             library syntax.
    """
    if len(shape) == 0:
        raise NotImplementedError("Scalar-valued expressions cannot be declared as an Eigen::MatrixBase object.")
    elif len(shape) == 1:
        rows = shape[0]
        cols = 1
    else:
        if not len(shape) == 2:
            raise NotImplementedError("%d-rank tensors are not currently supported." % len(shape))
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
    :meth:`block` of the eigen tensor is provided. Otherwise, no block information
    is needed and the tensor is returned as is.

    :arg expr: a `slate.Tensor` node.
    :arg temporary: the associated temporary of the expr argument.
    :arg index: a tuple of integers used to determine row and column information.
                This is provided by the SplitKernel associated with the expr.
    """
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

    # Create sub-block if tensor is mixed
    if (rshape, cshape) != expr.shape:
        tensor = ast.FlatBlock("%s.block<%d, %d>(%d, %d)" % (temporary,
                                                             rshape, cshape,
                                                             rstart, cstart))
    else:
        tensor = temporary

    return tensor
