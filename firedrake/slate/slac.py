"""This is SLATE's Linear Algebra Compiler (SLAC). This module is responsible for
generating C++ kernel functions representing symbolic linear algebra expressions
written in SLATE.

This linear algebra compiler uses both Firedrake's form compiler, the Two-Stage
Form Compiler (TSFC) and COFFEE's kernel abstract syntax tree (AST) optimizer. TSFC
provides SLAC with appropriate kernel functions (in C) for evaluating integral expressions
(finite element variational forms written in UFL). COFFEE's AST optimizing framework
produces the resulting kernel AST returned by: `compile_slate_expression`.

The Eigen C++ library (http://eigen.tuxfamily.org/) is required, as all low-level numerical
linear algebra operations are performed using the `Eigen::Matrix` methods built into Eigen.
"""

import sys
import firedrake
import tsfc
import operator

from coffee import base as ast
from coffee.visitor import Visitor

from firedrake.tsfc_interface import SplitKernel, KernelInfo

from slate import *

from functools import partial

from ufl.algorithms.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.form import Form


__all__ = ['compile_slate_expression']


class RemoveRestrictions(MultiFunction):
    """UFL MultiFunction for removing any restrictions on the integrals of forms."""

    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class Transformer(Visitor):
    """Replaces all out-put tensor references with a specified
    name of :type: `Eigen::Matrix` with appropriate shape.

    The default name of :data:`"A"` is assigned, otherwise a
    specified name may be passed as the :data:`name` keyword
    argument when calling the visitor."""

    def visit_object(self, o, *args, **kwargs):
        """Visits an object and returns it.

        e.g. string ---> string
        """
        return o

    def visit_list(self, o, *args, **kwargs):
        """Visits an input of COFFEE objects and returns
        the complete list of said objects."""

        newlist = [self.visit(e, *args, **kwargs) for e in o]
        if all(newo is e for newo, e in zip(newlist, o)):
            return o

        return newlist

    visit_Node = Visitor.maybe_reconstruct

    def visit_FunDecl(self, o, *args, **kwargs):
        """Visits a COFFEE FunDecl object and reconstructs
        the FunDecl body and header to generate
        Eigen::MatrixBase C++ template functions.

        Creates a template function for each subkernel form
        template <typename Derived>:

        template <typename Derived>
        static inline void foo(Eigen::MatrixBase<Derived> const & A, ...)
        {
          [Body...]
        }
        """

        name = kwargs.get("name", "A")
        new = self.visit_Node(o, *args, **kwargs)
        ops, okwargs = new.operands()
        if all(new is old for new, old in zip(ops, o.operands()[0])):
            return o

        template = "template <typename Derived>"
        body = ops[3]
        args, _ = body.operands()
        nargs = [ast.FlatBlock("Eigen::MatrixBase<Derived> & %s = const_cast<Eigen::MatrixBase<Derived> &>(%s_);\n" % (name, name))] + args
        ops[3] = nargs
        ops[6] = template

        return o.reconstruct(*ops, **okwargs)

    def visit_Decl(self, o, *args, **kwargs):
        """Visits a declared tensor and changes its type to
        :template: result `Eigen::MatrixBase<Derived>`.

        i.e. double A[n][m] ---> const Eigen::MatrixBase<Derived> &A_
        """

        name = kwargs.get("name", "A")
        if o.sym.symbol != name:
            return o
        newtype = "const Eigen::MatrixBase<Derived> &"

        return o.reconstruct(newtype, ast.Symbol("%s_" % name))

    def visit_Symbol(self, o, *args, **kwargs):
        """Visits a COFFEE symbol and redefines it as a FunCall object.

        i.e. A[j][k] ---> A(j, k)
        """

        name = kwargs.get("name", "A")
        if o.symbol != name:
            return o
        shape = o.rank

        return ast.FunCall(ast.Symbol(name), *shape)


def compile_slate_expression(slate_expr, tsfc_parameters=None):
    """Takes a SLATE expression `slate_expr` and returns the
    appropriate :class:`firedrake.op2.Kernel` object representing
    the SLATE expression.

    :arg slate_expr: a SLATE expression.

    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
                          be passed onto TSFC during the compilation of
                          integral forms.
    """
    # Only SLATE expressions are allowed as inputs
    if not isinstance(slate_expr, Tensor):
        raise ValueError("Expecting a `slate.Tensor` expression, not a %r" % slate_expr)

    # SLATE currently does not support arguments in mixed function spaces
    # TODO: Get PyOP2 to write into mixed dats
    if any(len(a.function_space()) > 1 for a in slate_expr.arguments()):
        raise NotImplementedError("Compiling mixed slate expressions")

    # Initialize variables: dtype and dictionary objects.
    dtype = tsfc.parameters.SCALAR_TYPE
    shape = slate_expr.shape
    temps = {}
    kernel_exprs = {}
    statements = []

    # Extract all coefficents in the SLATE expression and construct the
    # appropriate coefficent mapping
    coeffs = slate_expr.coefficients()
    coeffmap = dict((c, ast.Symbol("w%d" % i)) for i, c in enumerate(coeffs))

    # By default, we don't assume that we need to perform facet integral loops.
    # If an expression contains facet integrals within the expression, we activate
    # this to indicate that we need to loop over cell facets later on.
    need_cell_facets = False

    # initialize coordinate and facet symbols
    coordsym = ast.Symbol("coords")
    coords = None
    cellfacetsym = ast.Symbol("cell_facets")
    inc = []

    # Provide the SLATE expression as input to extract kernel functions
    get_kernel_expr(slate_expr, tsfc_parameters, temps, kernel_exprs, statements)

    # Now we construct the body of the macro kernel
    for exp, t in temps.items():
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))

        for klist in kernel_exprs[exp]:
            clist = []
            index = klist[0]
            kinfo = klist[1]
            kernel = kinfo.kernel
            integral_type = kinfo[1]

            if integral_type not in ['cell', 'interior_facet', 'exterior_facet']:
                raise NotImplementedError("Integral type %s not currently supported." % integral_type)

            # Checking for facet integrals
            if integral_type in ['interior_facet', 'exterior_facet']:
                need_cell_facets = True

            # Checking coordinates
            coordinates = exp.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords
            else:
                coords = coordinates

            # Extracting coefficients
            for cindex in list(kinfo[5]):
                coeff = exp.coefficients()[cindex]
                clist.append(coeffmap[coeff])

            # Defining tensor matrices of appropriate size
            inc.extend(kernel._include_dirs)
            try:
                row, col = index
            except ValueError:
                row = index[0]
                col = 0
            rshape = exp.shapes[0][row]
            rstart = sum(exp.shapes[0][:row])
            try:
                cshape = exp.shapes[1][col]
                cstart = sum(exp.shapes[1][:col])
            except KeyError:
                cshape = 1
                cstart = 0

            # Creating sub-block if tensor is mixed.
            if (rshape, cshape) != exp.shape:
                tensor = ast.FlatBlock("%s.block<%d,%d>(%d, %d)" %
                                       (t, rshape, cshape,
                                        rstart, cstart))
            else:
                tensor = t

            # Facet integral loop
            if integral_type in ['interior_facet', 'exterior_facet']:
                itsym = ast.Symbol("i0")
                block = []
                nfacet = exp.ufl_domain().ufl_cell().num_facets()
                clist.append(ast.FlatBlock("&%s" % itsym))

                # Check if facet is exterior (they are the only allowed
                # facet integral type in SLATE)
                if integral_type == 'exterior_facet':
                    # Perform facet integral loop
                    checker = 1
                else:
                    checker = 0

                # Construct the body of the facet loop
                block.append(ast.If(ast.Eq(ast.Symbol(cellfacetsym,
                                                      rank=(itsym, )),
                                           checker),
                                    [ast.Block([ast.FunCall(kernel.name,
                                                            tensor,
                                                            coordsym,
                                                            *clist)],
                                               open_scope=True)]))
                # Assemble loop
                loop = ast.For(ast.Decl("unsigned int", itsym, init=0),
                               ast.Less(itsym, nfacet),
                               ast.Incr(itsym, 1), block)
                statements.append(loop)
            else:
                statements.append(ast.FunCall(kernel.name,
                                              tensor,
                                              coordsym,
                                              *clist))

    result_type = map_type(mat_type(shape))
    result_sym = ast.Symbol("T%d" % len(temps))
    result_data_sym = ast.Symbol("A%d" % len(temps))
    result = ast.Decl(dtype, ast.Symbol(result_data_sym, shape))

    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" %
                                     (result_type, result_sym,
                                      dtype, result_data_sym))

    statements.append(result_statement)

    # Call the get_c_str method to return the full C/C++ statement
    c_string = ast.FlatBlock(get_c_str(slate_expr, temps))
    statements.append(ast.Assign(result_sym, c_string))

    # Generating function arguments for macro kernel function
    arglist = [result, ast.Decl("%s **" % dtype, coordsym)]
    for c in coeffs:
        ctype = "%s **" % dtype
        if isinstance(c, firedrake.Constant):
            ctype = "%s *" % dtype
        arglist.append(ast.Decl(ctype, coeffmap[c]))

    if need_cell_facets:
        arglist.append(ast.Decl("char *", cellfacetsym))

    kernel = ast.FunDecl("void", "slac_compile_slate", arglist,
                         ast.Block(statements),
                         pred=["static", "inline"])

    # Finally we transform the kernel into a set of C++ template functions.
    # This produces the final kernel AST
    klist = []
    transformkernel = Transformer()
    oriented = False
    for v in kernel_exprs.values():
        for ks in v:
            oriented = oriented or ks.kinfo.oriented
            # TODO: Think about this. Is this true for SLATE?
            assert ks.kinfo.subdomain_id == "otherwise"
            kast = transformkernel.visit(ks.kinfo.kernel._ast)
            klist.append(kast)

    klist.append(kernel)
    kernelast = ast.Node(klist)
    inc.append('%s/lib/python2.7/site-packages/petsc/include/eigen3/' % sys.prefix)

    # Produce the op2 kernel object for assembly
    op2kernel = firedrake.op2.Kernel(kernelast,
                                     "slac_compile_slate",
                                     cpp=True,
                                     include_dirs=inc,
                                     headers=['#include <Eigen/Dense>',
                                              '#define restrict'])

    # TODO: What happens for multiple ufl domains?
    assert len(slate_expr.ufl_domains()) == 1
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type="cell",
                       oriented=oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=range(len(coeffs)),
                       needs_cell_facets=need_cell_facets)
    idx = tuple([0]*len(slate_expr.arguments()))

    return (SplitKernel(idx, kinfo), )


def get_c_str(expr, temps, prec=None):
    """Translates a SLATE expression into its equivalent
    representation in C. This is done by using COFFEE's
    code generation functionality and we construct an
    appropriate C/C++ representation of the `slate.Tensor`
    expression.

    :arg expr: a :class:`slate.Tensor` expression.

    :arg temps: a `dict` of temporaries which map a given
                `slate.Tensor` object to its corresponding
                representation as a `coffee.Symbol` object.

    :arg prec: an argument dictating the order of precedence
               in the linear algebra operations. This ensures
               that parentheticals are placed appropriately
               and the order in which linear algebra operations
               are performed are correct.

    Returns
        This function returns a `string` which represents the
        C/C++ code representation of the `slate.Tensor` expr.
    """
    if isinstance(expr, (Scalar, Vector, Matrix)):
        return temps[expr].gencode()

    elif isinstance(expr, UnaryOp):
        op = {operator.neg: '-',
              operator.pos: '+'}[expr.op]
        prec = expr.prec
        result = "%s%s" % (op, get_c_str(expr.children,
                                         temps,
                                         prec))

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, BinaryOp):
        op = {operator.add: '+',
              operator.sub: '-',
              operator.mul: '*'}[expr.op]
        prec = expr.prec
        result = "%s %s %s" % (get_c_str(expr.children[0], temps,
                                         prec),
                               op,
                               get_c_str(expr.children[1], temps,
                                         prec))

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, Inverse):
        return "(%s).inverse()" % get_c_str(expr.children, temps)

    elif isinstance(expr, Transpose):
        return "(%s).transpose()" % get_c_str(expr.children, temps)

    else:
        # If expression is not recognized, throw a NotImplementedError.
        raise NotImplementedError("Expression of type %s not supported.",
                                  type(expr).__name__)


def get_kernel_expr(expr, parameters=None, temps=None, kernel_exprs=None, statements=None):
    """Retrieves the TSFC kernel function for a slate expression.
    Compiled integral forms use the TSFC form compiler. This function
    walks through the nodes of a given :class:`slate.Tensor` which is
    typically comprised of few or several :class:`slate.BinaryOp` or
    :class:`slate.UnaryOp` objects.

    :arg expr: a :class:`slate.Tensor` expression.

    :arg parameters: optional `dict` of parameters to pass to the form compiler.

    :arg temps: an :obj:`dict` whose entries are :class:`coffee.Symbol` objects
    which represent a terminal `slate.Tensor` object (the keys are the corresponding
    `slate.Tensor` objects). The `temps` argument is initialized as an empty `dict`
    and gets populated in the body of the function.

    :arg kernel_exprs: an :obj:`dict` whose entries are TSFC compiled forms. The keys
    correspond to the associated `slate.Tensor` object where the kernels are used to
    dictate how to populate the tensor. The `kernel_exprs` argument is initialized as
    an empty `dict` and gets populated in the body of the function.

    :arg statements: a :obj:`list` of :class:`coffee.FlatBlock` statements which
    provide the relevant information for building the kernel functions in
    :meth:`compile_slate_expression`. This object is initialized as an empty list.

    Returns:
        temps: returned with all assigned temporaries.
        kernel_exprs: returned with all relevant kernel functions
                      provided by TSFC.
        statemenets: returned with appropriate declarations and
                     `coffee.base` objects for building the full
                     kernel.
    """
    temps = {} or temps
    kernel_exprs = {} or kernel_exprs
    statements = [] or statements

    if isinstance(expr, (Scalar, Vector, Matrix)):
        # No need to store identical variables more than once
        if expr in temps.keys():
            return temps, kernel_exprs, statements
        else:
            sym = "T%d" % len(temps)
            temp = ast.Symbol(sym)
            temp_type = mat_type(expr.shape)
            temps[expr] = temp
            statements.append(ast.Decl(temp_type, temp))

            # Compile integrals using TSFC
            integrals = expr.form.integrals()
            mapper = RemoveRestrictions()
            integrals = map(partial(map_integrand_dags, mapper), integrals)
            prefix = "subkernel%d_" % len(kernel_exprs)

            int_fac_integrals = filter(lambda x: x.integral_type() == "interior_facet",
                                       integrals)
            # Reconstruct all interior_facet integrals to be of type: exterior_facet
            # This is because locally over each element, we view them as being "exterior"
            # with respect to that element.
            int_fac_integrals = [it.reconstruct(integral_type="exterior_facet")
                                 for it in int_fac_integrals]
            other_integrals = filter(lambda x: x.integral_type() != "interior_facet",
                                     integrals)

            compiled_forms = []
            if len(int_fac_integrals) != 0:
                intf_form = Form(int_fac_integrals)
                compiled_forms.extend(firedrake.tsfc_interface.compile_form(intf_form,
                                                                            prefix+"int_ext_conv",
                                                                            parameters=parameters))
            other_form = Form(other_integrals)
            compiled_forms.extend(firedrake.tsfc_interface.compile_form(other_form,
                                                                        prefix,
                                                                        parameters=parameters))
            kernel_exprs[expr] = tuple(compiled_forms)
            return temps, kernel_exprs, statements

    elif isinstance(expr, (UnaryOp, BinaryOp, Transpose, Inverse)):
        # Map the children to :meth:`get_kernel_expr` recursively
        map(lambda x: get_kernel_expr(x, parameters, temps, kernel_exprs, statements),
            expr.operands)
        return
    else:
        # If none of the expressions are recognized, raise NotImplementedError
        raise NotImplementedError("Expression of type %s not supported.",
                                  type(expr).__name__)


def mat_type(shape):
    """Returns the Eigen::Matrix declaration of the tensor.

    :arg shape: a tuple of integers the denote the shape of
                the :class:`slate.Tensor` object.

    Returns: Returns a string indicating the appropriate declaration
             of the `slate.Tensor` in the appropriate Eigen C++ template
             library syntax.
    """

    if len(shape) == 1:
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


def map_type(matrix):
    """Returns an eigen map to output the resulting matrix
    in an appropriate data array.

    :arg matrix: a string returned from :meth:`mat_type`; a
                 string denoting the appropriate declaration
                 of an `Eigen::Matrix` object.
    """

    return "Eigen::Map<%s >" % matrix


def parenthesize(expr, prec=None, parent=None):
    """Parenthezises a slate expression and returns a string. This function is
    fairly self-explanatory."""
    if prec is None or parent >= prec:
        return expr
    return "(%s)" % expr
