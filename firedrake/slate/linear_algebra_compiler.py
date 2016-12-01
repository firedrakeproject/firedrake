"""This is SLATE's Linear Algebra Compiler. This module is responsible for generating
C++ kernel functions representing symbolic linear algebra expressions written in SLATE.

This linear algebra compiler uses both Firedrake's form compiler, the Two-Stage
Form Compiler (TSFC) and COFFEE's kernel abstract syntax tree (AST) optimizer. TSFC
provides this compiler with appropriate kernel functions (in C) for evaluating integral
expressions (finite element variational forms written in UFL). COFFEE's AST optimizing
framework produces the resulting kernel AST returned by: `compile_slate_expression`.

The Eigen C++ library (http://eigen.tuxfamily.org/) is required, as all low-level numerical
linear algebra operations are performed using the `Eigen::Matrix` methods built into Eigen.
"""
from __future__ import absolute_import, print_function, division

import petsc

from coffee import base as ast

from firedrake.constant import Constant
from firedrake.tsfc_interface import SplitKernel, KernelInfo
from firedrake.slate.slate import (TensorBase, Tensor, Transpose, Inverse, Negative,
                                   BinaryOp, TensorAdd, TensorSub, TensorMul, TensorAction)
from firedrake.slate.kernel_builder import SlateKernelBuilder
from firedrake import op2

from tsfc.parameters import SCALAR_TYPE


__all__ = ['compile_slate_expression']


def compile_slate_expression(slate_expr, tsfc_parameters=None):
    """Takes a SLATE expression `slate_expr` and returns the appropriate
    :class:`firedrake.op2.Kernel` object representing the SLATE expression.

    :arg slate_expr: a :class:'TensorBase' expression.

    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
                          be passed onto TSFC during the compilation of ufl forms.
    """
    # Only SLATE expressions are allowed as inputs
    if not isinstance(slate_expr, TensorBase):
        raise ValueError("Expecting a `slate.TensorBase` expression, not a %r" % slate_expr)

    # SLATE currently does not support arguments in mixed function spaces
    # TODO: Get PyOP2 to write into mixed dats
    if any(len(a.function_space()) > 1 for a in slate_expr.arguments()):
        raise NotImplementedError("Compiling mixed slate expressions")

    # Initialize variables: dtype and dictionary objects.
    dtype = SCALAR_TYPE
    shape = slate_expr.shape
    statements = []

    builder = SlateKernelBuilder(expression=slate_expr, tsfc_parameters=tsfc_parameters)

    # initialize coordinate and facet symbols
    coordsym = ast.Symbol("coords")
    coords = None
    cellfacetsym = ast.Symbol("cell_facets")
    inc = []

    for exp, t in builder.temps.items():
        statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))

        for splitkernel in builder.kernel_exprs[exp]:
            clist = []
            index = splitkernel.indices
            kinfo = splitkernel.kinfo
            integral_type = kinfo.integral_type

            if integral_type not in ["cell", "interior_facet", "exterior_facet"]:
                raise NotImplementedError("Integral type %s not currently supported." % integral_type)

            if integral_type in ["interior_facet", "exterior_facet"]:
                builder.require_cell_facets()

            coordinates = exp.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords
            else:
                coords = coordinates

            for cindex in kinfo.coefficient_map:
                c = exp.coefficients()[cindex]
                clist.append(builder.coefficient_map()[c])

            inc.extend(kinfo.kernel._include_dirs)

            tensor = eigen_tensor(exp, t, index)

            if builder.needs_cell_facets:
                itsym = ast.Symbol("i0")
                clist.append(ast.FlatBlock("&%s" % itsym))
                loop_body = []
                nfacet = exp.ufl_domain().ufl_cell().num_facets()

                if kinfo.integral_type == "exterior_facet":
                    checker = 1
                else:
                    checker = 0
                loop_body.append(ast.If(ast.Eq(ast.Symbol(cellfacetsym, rank=(itsym,)), checker),
                                        [ast.Block([ast.FunCall(kinfo.kernel.name, tensor, coordsym, *clist)],
                                                   open_scope=True)]))
                loop = ast.For(ast.Decl("unsigned int", itsym, init=0), ast.Less(itsym, nfacet),
                               ast.Incr(itsym, 1), loop_body)
                statements.append(loop)

            elif isinstance(exp, TensorAction):
                # Implement something that generates a loop for applying action
                pass
            else:
                statements.append(ast.FunCall(kinfo.kernel.name, tensor, coordsym, *clist))

    result_sym = ast.Symbol("T%d" % len(builder.temps))
    result_data_sym = ast.Symbol("A%d" % len(builder.temps))
    result_type = "Eigen::Map<%s >" % eigen_matrixbase_type(shape)
    result = ast.Decl(dtype, ast.Symbol(result_data_sym, shape))
    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" % (result_type, result_sym, dtype, result_data_sym))
    statements.append(result_statement)

    cpp_string = ast.FlatBlock(metaphrase_slate_to_cpp(slate_expr, builder.temps))
    statements.append(ast.Assign(result_sym, cpp_string))

    args = [result, ast.Decl("%s **" % dtype, coordsym)]
    for c in slate_expr.coefficients():
        if isinstance(c, Constant):
            ctype = "%s *" % dtype
        else:
            ctype = "%s **" % dtype
        args.append(ast.Decl(ctype, builder.coefficient_map()[c]))

    if builder.needs_cell_facets:
        args.append(ast.Decl("char *", cellfacetsym))

    macro_kernel_name = "compile_slate"
    kernel_ast, oriented = builder.construct_ast(name=macro_kernel_name,
                                                 args=args,
                                                 statements=ast.Block(statements))

    inc.append(petsc.get_petsc_dir() + '/include/eigen3/')
    op2kernel = op2.Kernel(kernel_ast, macro_kernel_name, cpp=True, include_dirs=inc,
                           headers=['#include <Eigen/Dense>', '#define restrict __restrict'])

    assert len(slate_expr.ufl_domains()) == 1
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type="cell",
                       oriented=oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=range(len(slate_expr.coefficients())),
                       needs_cell_facets=builder.needs_cell_facets)
    idx = tuple([0]*slate_expr.rank)

    return (SplitKernel(idx, kinfo),)


def metaphrase_slate_to_cpp(expr, temps, prec=None):
    """Translates a SLATE expression into its equivalent representation in the Eigen C++ syntax.

    :arg expr: a :class:`slate.TensorBase` expression.

    :arg temps: a `dict` of temporaries which map a given `slate.TensorBase` object to its
                corresponding representation as a `coffee.Symbol` object.

    :arg prec: an argument dictating the order of precedence in the linear algebra operations.
               This ensures that parentheticals are placed appropriately and the order in which
               linear algebra operations are performed are correct.

    Returns
        This function returns a `string` which represents the C/C++ code representation
        of the `slate.TensorBase` expr.
    """
    if isinstance(expr, (Tensor, TensorAction)):
        return temps[expr].gencode()

    elif isinstance(expr, Transpose):
        return "(%s).transpose()" % metaphrase_slate_to_cpp(expr.tensor, temps)

    elif isinstance(expr, Inverse):
        return "(%s).inverse()" % metaphrase_slate_to_cpp(expr.tensor, temps)

    elif isinstance(expr, Negative):
        result = "-%s" % metaphrase_slate_to_cpp(expr.tensor, temps, expr.prec)

        # Make sure we parenthesize correctly
        if expr.prec is None or prec >= expr.prec:
            return result
        else:
            return "(%s)" % result

    elif isinstance(expr, BinaryOp):
        op = {TensorAdd: '+',
              TensorSub: '-',
              TensorMul: '*'}[type(expr)]
        result = "%s %s %s" % (metaphrase_slate_to_cpp(expr.operands[0], temps, expr.prec),
                               op,
                               metaphrase_slate_to_cpp(expr.operands[1], temps, expr.prec))

        # Make sure we parenthesize correctly
        if expr.prec is None or prec >= expr.prec:
            return result
        else:
            return "(%s)" % result
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


def eigen_tensor(expr, temp, index):
    """
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
        tensor = ast.FlatBlock("%s.block<%d, %d>(%d, %d)" % (temp, rshape, cshape, rstart, cstart))
    else:
        tensor = temp

    return tensor
