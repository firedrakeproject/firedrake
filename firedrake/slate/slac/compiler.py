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
from __future__ import absolute_import, print_function, division
from six.moves import range

from coffee import base as ast

from collections import OrderedDict

from firedrake.constant import Constant
from firedrake.tsfc_interface import SplitKernel, KernelInfo
from firedrake.slate.slate import (TensorBase, Tensor,
                                   Transpose, Inverse, Negative,
                                   Add, Sub, Mul, Action)
from firedrake.slate.slac.kernel_builder import KernelBuilder
from firedrake import op2

from pyop2.utils import get_petsc_dir

from tsfc.parameters import SCALAR_TYPE

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
    if any(len(a.function_space()) > 1 for a in slate_expr.arguments()):
        raise NotImplementedError("Compiling mixed slate expressions")

    # If the expression has already been symbolically compiled, then
    # simply reuse the produced kernel.
    if slate_expr._kernels is not None:
        return slate_expr._kernels

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

    # Initialize coordinate and facet/layer symbols
    coordsym = ast.Symbol("coords")
    coords = None
    cellfacetsym = ast.Symbol("cell_facets")
    mesh_layer_sym = ast.Symbol("layer")
    inc = []

    declared_temps = {}
    for cxt_kernel in builder.context_kernels:
        exp = cxt_kernel.tensor
        t = builder.get_temporary(exp)

        if exp not in declared_temps:
            # Declare and initialize the temporary
            statements.append(ast.Decl(eigen_matrixbase_type(exp.shape), t))
            statements.append(ast.FlatBlock("%s.setZero();\n" % t))
            declared_temps[exp] = t

        it_type = cxt_kernel.original_integral_type

        if it_type not in supported_integral_types:
            raise NotImplementedError("Type %s not supported." % it_type)

        # Explicit checking of coordinates
        coordinates = exp.ufl_domain().coordinates
        if coords is not None:
            assert coordinates == coords
        else:
            coords = coordinates

        if it_type == "cell":
            # Nothing difficult about cellwise integrals. Just need
            # to get coefficient info, include_dirs and append
            # function calls to the appropriate subkernels.

            # If tensor is mixed, there will be more than one SplitKernel
            incl = []
            for splitkernel in cxt_kernel.tsfc_kernels:
                index = splitkernel.indices
                kinfo = splitkernel.kinfo

                # Generate an iterable of coefficients to pass to the subkernel
                # if any are required
                clist = [c for ci in kinfo.coefficient_map
                         for c in builder.coefficient(exp.coefficients()[ci])]

                incl.extend(kinfo.kernel._include_dirs)
                tensor = eigen_tensor(exp, t, index)
                statements.append(ast.FunCall(kinfo.kernel.name,
                                              tensor, coordsym,
                                              *clist))

        elif it_type in ["interior_facet", "exterior_facet",
                         "interior_facet_vert", "exterior_facet_vert"]:
            # These integral types will require accessing local facet
            # information and looping over facet indices.
            builder.require_cell_facets()
            loop_stmt, incl = facet_integral_loop(cxt_kernel, builder,
                                                  coordsym, cellfacetsym)
            statements.append(loop_stmt)

        elif it_type == "interior_facet_horiz":
            # The infamous interior horizontal facet
            # will have two SplitKernels: one top,
            # one bottom. The mesh layer will determine
            # which kernels we call.
            builder.require_mesh_layers()
            top_sks = [k for k in cxt_kernel.tsfc_kernels
                       if k.kinfo.integral_type == "exterior_facet_top"]
            bottom_sks = [k for k in cxt_kernel.tsfc_kernels
                          if k.kinfo.integral_type == "exterior_facet_bottom"]
            assert len(top_sks) == len(bottom_sks), (
                "Number of top and bottom kernels should be equal"
            )
            # Top and bottom kernels need to be sorted by kinfo.indices
            # if the space is mixed to ensure indices match.
            top_sks = sorted(top_sks, key=lambda x: x.indices)
            bottom_sks = sorted(bottom_sks, key=lambda x: x.indices)
            stmt, incl = extruded_int_horiz_facet(exp,
                                                  builder,
                                                  top_sks,
                                                  bottom_sks,
                                                  coordsym,
                                                  mesh_layer_sym)
            statements.append(stmt)

        elif it_type in ["exterior_facet_bottom", "exterior_facet_top"]:
            # These kernels will only be called if we are on
            # the top or bottom layers of the extruded mesh.
            builder.require_mesh_layers()
            stmt, incl = extruded_top_bottom_facet(cxt_kernel, builder,
                                                   coordsym, mesh_layer_sym)
            statements.append(stmt)

        else:
            raise ValueError("Kernel type not recognized: %s" % it_type)

        # Don't duplicate include lines
        inc_dir = list(set(incl) - set(inc))
        inc.extend(inc_dir)

    # Now we handle any terms that require auxiliary data (if any)
    # and update our temporaries and code statements.
    # In particular, if we need to apply the action of a
    # Slate tensor on an already assembled coefficient, or
    # compute matrix inverses/transposes.
    if bool(builder.aux_temps):
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
    args = [result, ast.Decl("%s **" % SCALAR_TYPE, coordsym)]
    for c in expr_coeffs:
        if isinstance(c, Constant):
            ctype = "%s *" % SCALAR_TYPE
        else:
            ctype = "%s **" % SCALAR_TYPE
        args.extend([ast.Decl(ctype, csym) for csym in builder.coefficient(c)])

    if builder.needs_cell_facets:
        args.append(ast.Decl("char *", cellfacetsym))

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
    inc.extend(["%s/include/eigen3/" % d for d in PETSC_DIR])
    op2kernel = op2.Kernel(kernel_ast,
                           macro_kernel_name,
                           cpp=True,
                           include_dirs=inc,
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
    slate_expr._kernels = kernels

    return kernels


def extruded_int_horiz_facet(exp, builder, top_sks, bottom_sks,
                             coordsym, mesh_layer_sym):
    """Generates a code statement for evaluating interior horizontal
    facet integrals.

    :arg exp: A :class:`TensorBase` expression.
    :arg builder: A :class:`KernelBuilder` containing the expression context.
    :arg top_sks: An iterable of index ordered TSFC kernels for the top
                  kernels.
    :arg bottom_sks: An iterable of index ordered TSFC kernels for the bottom
                     kernels.
    :arg coordsym: An `ast.Symbol` object representing coordinate arguments
                   for the kernel.
    :arg mesh_layer_sym: An `ast.Symbol` representing the mesh layer.

    Returns: A COFFEE code statement and updated include_dirs
    """
    t = builder.get_temporary(exp)
    nlayers = exp.ufl_domain().topological.layers - 1

    incl = []
    top_calls = []
    bottom_calls = []
    for top, btm in zip(top_sks, bottom_sks):
        assert top.indices == btm.indices, (
            "Top and bottom kernels must have the same indices"
        )
        index = top.indices

        # Generate an iterable of coefficients to pass to the subkernel
        # if any are required
        c_set = top.kinfo.coefficient_map + btm.kinfo.coefficient_map
        coefficient_map = tuple(OrderedDict.fromkeys(c_set))
        clist = [c for ci in coefficient_map
                 for c in builder.coefficient(exp.coefficients()[ci])]

        dirs = top.kinfo.kernel._include_dirs + btm.kinfo.kernel._include_dirs
        incl.extend(tuple(OrderedDict.fromkeys(dirs)))

        tensor = eigen_tensor(exp, t, index)
        top_calls.append(ast.FunCall(top.kinfo.kernel.name,
                                     tensor, coordsym, *clist))
        bottom_calls.append(ast.FunCall(btm.kinfo.kernel.name,
                                        tensor, coordsym, *clist))

    else_stmt = ast.Block(top_calls + bottom_calls, open_scope=True)
    inter_stmt = ast.If(ast.Eq(mesh_layer_sym, nlayers - 1),
                        (ast.Block(bottom_calls, open_scope=True),
                         else_stmt))
    stmt = ast.If(ast.Eq(mesh_layer_sym, 0),
                  (ast.Block(top_calls, open_scope=True),
                   inter_stmt))
    return stmt, incl


def extruded_top_bottom_facet(cxt_kernel, builder, coordsym, mesh_layer_sym):
    """Generates a code statement for evaluating exterior top/bottom
    facet integrals.

    :arg cxt_kernel: A :namedtuple:`ContextKernel` containing all relevant
                     integral types and TSFC kernels associated with the
                     form nested in the expression.
    :arg builder: A :class:`KernelBuilder` containing the expression context.
    :arg coordsym: An `ast.Symbol` object representing coordinate arguments
                   for the kernel.
    :arg mesh_layer_sym: An `ast.Symbol` representing the mesh layer.

    Returns: A COFFEE code statement and updated include_dirs
    """
    exp = cxt_kernel.tensor
    t = builder.get_temporary(exp)
    nlayers = exp.ufl_domain().topological.layers - 1

    incl = []
    body = []
    for splitkernel in cxt_kernel.tsfc_kernels:
        index = splitkernel.indices
        kinfo = splitkernel.kinfo

        # Generate an iterable of coefficients to pass to the subkernel
        # if any are required
        clist = [c for ci in kinfo.coefficient_map
                 for c in builder.coefficient(exp.coefficients()[ci])]

        incl.extend(kinfo.kernel._include_dirs)
        tensor = eigen_tensor(exp, t, index)
        body.append(ast.FunCall(kinfo.kernel.name,
                                tensor, coordsym, *clist))

    if cxt_kernel.original_integral_type == "exterior_facet_bottom":
        layer = 0
    else:
        layer = nlayers - 1

    stmt = ast.If(ast.Eq(mesh_layer_sym, layer),
                  [ast.Block(body, open_scope=True)])

    return stmt, incl


def facet_integral_loop(cxt_kernel, builder, coordsym, cellfacetsym):
    """Generates a code statement for evaluating exterior/interior facet
    integrals.

    :arg cxt_kernel: A :namedtuple:`ContextKernel` containing all relevant
                     integral types and TSFC kernels associated with the
                     form nested in the expression.
    :arg builder: A :class:`KernelBuilder` containing the expression context.
    :arg coordsym: An `ast.Symbol` object representing coordinate arguments
                   for the kernel.
    :arg cellfacetsym: An `ast.Symbol` representing the cell facets.

    Returns: A COFFEE code statement and updated include_dirs
    """
    exp = cxt_kernel.tensor
    t = builder.get_temporary(exp)
    it_type = cxt_kernel.original_integral_type
    itsym = ast.Symbol("i0")

    chker = {"interior_facet": 1,
             "interior_facet_vert": 1,
             "exterior_facet": 0,
             "exterior_facet_vert": 0}

    # Compute the correct number of facets for a particular facet measure
    if it_type in ["interior_facet", "exterior_facet"]:
        # Non-extruded case
        nfacet = exp.ufl_domain().ufl_cell().num_facets()

    elif it_type in ["interior_facet_vert", "exterior_facet_vert"]:
        # Extrusion case
        base_cell = exp.ufl_domain().ufl_cell()._cells[0]
        nfacet = base_cell.num_facets()

    else:
        raise ValueError(
            "Integral type %s not supported." % it_type
        )

    incl = []
    funcalls = []
    checker = chker[it_type]
    for splitkernel in cxt_kernel.tsfc_kernels:
        index = splitkernel.indices
        kinfo = splitkernel.kinfo

        # Generate an iterable of coefficients to pass to the subkernel
        # if any are required
        clist = [c for ci in kinfo.coefficient_map
                 for c in builder.coefficient(exp.coefficients()[ci])]

        incl.extend(kinfo.kernel._include_dirs)
        tensor = eigen_tensor(exp, t, index)

        clist.append(ast.FlatBlock("&%s" % itsym))
        funcalls.append(ast.FunCall(kinfo.kernel.name,
                                    tensor,
                                    coordsym,
                                    *clist))

    loop_body = ast.If(ast.Eq(ast.Symbol(cellfacetsym, rank=(itsym,)),
                              checker), [ast.Block(funcalls, open_scope=True)])

    loop_stmt = ast.For(ast.Decl("unsigned int", itsym, init=0),
                        ast.Less(itsym, nfacet),
                        ast.Incr(itsym, 1), loop_body)

    return loop_stmt, incl


def auxiliary_temporaries(builder, declared_temps):
    """This function generates any auxiliary information regarding special
    handling of expressions that require creating additional temporaries.

    :arg builder: a :class:`KernelBuilder` object that contains all the
                  necessary temporary and expression information.
    :arg declared_temps: a `dict` of temporaries that have already been
                         declared and assigned values.
    Returns: a list of auxiliary statements are returned that contain temporary
             declarations and any code-blocks needed to evaluate the
             expression.
    """
    aux_statements = []
    for exp in builder.aux_temps:
        if isinstance(exp, Transpose):
            # Get the temporary for the particular transpose
            # expression
            tensor, = exp.operands
            result = "(%s).transpose()" % (
                metaphrase_slate_to_cpp(tensor, declared_temps)
            )

        elif isinstance(exp, Inverse):
            # Get the temporary for the particular inverse
            # expression
            tensor, = exp.operands
            result = "(%s).inverse()" % (
                metaphrase_slate_to_cpp(tensor, declared_temps)
            )

        elif isinstance(exp, Action):
            actee, = exp.actee
            if actee not in declared_temps:
                # Declare a temporary for the coefficient
                ctemp = ast.Symbol("C%d" % len(declared_temps))
                V = actee.function_space()
                node_extent = V.fiat_element.space_dimension()
                dof_extent = np.prod(V.ufl_element().value_shape())
                typ = eigen_matrixbase_type(shape=(dof_extent * node_extent,))
                aux_statements.append(ast.Decl(typ, ctemp))
                aux_statements.append(ast.FlatBlock("%s.setZero();\n" % ctemp))

                # Now we unpack the function and insert its entries into a
                # 1D vector temporary
                isym = ast.Symbol("i1")
                jsym = ast.Symbol("j1")
                tensor_index = ast.Sum(ast.Prod(dof_extent, isym), jsym)

                # Inner-loop running over dof_extent
                coeff_sym = ast.Symbol(builder.coefficient_map[actee],
                                       rank=(isym, jsym))
                coeff_temp = ast.Symbol(ctemp, rank=(tensor_index,))
                inner_loop = ast.For(ast.Decl("unsigned int", jsym, init=0),
                                     ast.Less(jsym, dof_extent),
                                     ast.Incr(jsym, 1),
                                     ast.Assign(coeff_temp, coeff_sym))
                # Outer-loop running over node_extent
                loop = ast.For(ast.Decl("unsigned int", isym, init=0),
                               ast.Less(isym, node_extent),
                               ast.Incr(isym, 1),
                               inner_loop)

                aux_statements.append(loop)
                declared_temps[actee] = ctemp

            tensor, = exp.operands
            result = "(%s) * %s" % (metaphrase_slate_to_cpp(tensor,
                                                            declared_temps,
                                                            exp.prec),
                                    declared_temps[actee])

        else:
            raise NotImplementedError(
                "Auxiliary expr type %s not currently implemented." % type(exp)
            )

        # Now we use the generated result and assign the value to the
        # corresponding temporary.
        temp = builder.aux_temps[exp]
        aux_statements.append(ast.Decl(eigen_matrixbase_type(exp.shape),
                                       temp))
        aux_statements.append(ast.FlatBlock("%s.setZero();\n" % temp))
        aux_statements.append(ast.Assign(temp, result))

        # Update declared temps
        declared_temps[exp] = temp

    return aux_statements


def parenthesize(arg, prec=None, parent=None):
    """Parenthesizes an expression."""
    if prec is None or prec >= parent:
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
    # For transposes/inverses/actions, these are recursively declared
    # in order of expression complexity. If an expression contains an
    # one of these objects, they will be declared as well.
    # This minimizes the times an inverse/tranpose/action is computed.
    if expr in temps:
        return temps[expr].gencode()

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
    if expr.rank == 0:
        tensor = temporary
    else:
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
                                                                 rshape,
                                                                 cshape,
                                                                 rstart,
                                                                 cstart))
        else:
            tensor = temporary

    return tensor
