"""This is The Hybridized Form Compiler (HFC). This module is responsible
for interpreting SLATE expressions and generating C++ kernel functions
for assembly in Firedrake.

The HFC uses both Firedrake's form compiler, the Two-Step Form Compiler (TSFC)
and COFFEE's kernel abstract syntax tree (AST) optimizer. TSFC provides HFC with
appropriate kernel functions (in C) for evaluating integlral expressions. COFFEE's
AST framework helps procude the resulting kernel AST returned by:
`compile_slate_expression`

In addition, the Eigen C++ library (http://eigen.tuxfamily.org/) is required,
as all low-level numerical linear algebra operations are performed using
the `Eigen::Matrix` methods built into Eigen.

Written by: Thomas Gibson (t.gibson15@imperial.ac.uk)
"""

import sys
import firedrake
import operator

from coffee import base as ast
from coffee.visitor import Visitor

from firedrake.tsfc_interface import SplitKernel, KernelInfo

from slate import *
from slate_assertions import *

from singledispatch import singledispatch

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.form import Form


__all__ = ['compile_slate_expression']


class Transformer(Visitor):
    """Replaces all out-put tensor references with a specified
    name of :type: `Eigen::Matrix` with appropriate shape.

    The default name of :data:`"A"` is assigned, otherwise a
    specified name may be passed as the :data:`name` keyword
    argument when calling the visitor."""

    def visit_object(self, o, *args, **kwargs):
        """Visits a a COFFEE object and returns it.

        i.e. string ---> string
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

        pred = ["template <typename Derived>\nstatic", "inline"]
        body = ops[3]
        args, _ = body.operands()
        nargs = [ast.FlatBlock("Eigen::MatrixBase<Derived> & %s = const_cast<Eigen::MatrixBase<Derived> &>(%s_);\n" % (name, name))] + args
        ops[3] = nargs
        ops[4] = pred

        return o.reconstruct(*ops, **okwargs)

    def visit_Decl(self, o, *args, **kwargs):
        """Visits a declared tensor and changes its type to
        :template: result `Eigen::MatrixBase<Derived>`.

        i.e. double A[n][m] ---> Eigen::MatrixBase<Derived> const &A_
        """

        name = kwargs.get("name", "A")
        if o.sym.symbol != name:
            return o
        newtype = "Eigen::MatrixBase<Derived> const &"

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


def compile_slate_expression(slate_expr, testing=False):
    """Takes a SLATE expression `slate_expr` and returns the
    appropriate :class:`firedrake.op2.Kernel` object representing
    the SLATE expression.

    :arg slate_expr: a SLATE expression.
    :arg testing: an optional argument that changes the output of
    the function to return coordinates, coefficents, facet_flag and
    the resulting op2.Kernel. This argument is for testing purposes.
    """

    if not isinstance(slate_expr, Tensor):
        expecting_slate_expr(slate_expr)

    if any(len(a.function_space()) > 1 for a in slate_expr.arguments()):
        raise NotImplementedError("Compiling mixed slate expressions")
    # Initialize variables: dtype and associated data structures.
    dtype = "double"
    shape = slate_expr.shape
    temps = {}
    kernel_exprs = {}
    coeffs = slate_expr.coefficients()
    coeffmap = dict((c, ast.Symbol("w%d" % i)) for i, c in enumerate(coeffs))
    statements = []
    need_cell_facets = False

    # Auxillary functions for constructing matrix types.
    def mat_type(shape):
        """Returns the Eigen::Matrix declaration of the tensor."""

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
        in an appropriate data array."""

        return "Eigen::Map<%s >" % matrix

    # Compile integrals associated with a temporary using TSFC
    @singledispatch
    def get_kernel_expr(expr):
        """Retrieves the TSFC kernel function for a specific tensor."""

        raise NotImplementedError("Expression of type %s not supported.",
                                  type(expr).__name__)

    @get_kernel_expr.register(Action)
    @get_kernel_expr.register(Scalar)
    @get_kernel_expr.register(Vector)
    @get_kernel_expr.register(Matrix)
    def get_kernel_expr_tensor(expr):
        """Compile integral forms using TSFC form compiler."""

        if expr not in temps.keys():
            sym = "T%d" % len(temps)
            temp = ast.Symbol(sym)
            temp_type = mat_type(expr.shape)
            temps[expr] = temp
            statements.append(ast.Decl(temp_type, temp))

            # Compile integrals using TSFC
            integrals = expr.get_ufl_integrals()
            kernel_exprs[expr] = []
            mapper = RemoveRestrictions()
            for i, it in enumerate(integrals):
                typ = it.integral_type()
                form = Form([it])
                prefix = "subkernel%d_%d_%s_" % (len(kernel_exprs), i, typ)

                if typ == "interior_facet":
                    # Reconstruct "interior_facet" integrals to be
                    # of type "exterior_facet"
                    newit = map_integrand_dags(mapper, it)
                    newit = newit.reconstruct(integral_type="exterior_facet")
                    form = Form([newit])

                tsfc_compiled_form = firedrake.tsfc_interface.compile_form(form, prefix)
                kernel_exprs[expr].append((typ, tsfc_compiled_form))
        return

#    @get_kernel_expr.register(Action)
#    def get_kernel_expr_action(expr):
#        """Gets the input coefficient and acting tensor."""
#
#        tensor = expr.tensor
#        get_kernel_expr(tensor)

    @get_kernel_expr.register(UnaryOp)
    @get_kernel_expr.register(BinaryOp)
    @get_kernel_expr.register(Transpose)
    @get_kernel_expr.register(Inverse)
    def get_kernel_expr_ops(expr):
        """Map operands of expressions into
        `get_kernel_expr` recursively."""

        map(get_kernel_expr, expr.operands)
        return

    # initialize coordinate and facet symbols
    coordsym = ast.Symbol("coords")
    coords = None
    cellfacetsym = ast.Symbol("cell_facets")
    inc = []

    get_kernel_expr(slate_expr)

    # Now we construct the body of the macro kernel
    for exp, t in temps.items():
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))
        for (typ, klist) in kernel_exprs[exp]:
            for ks in klist:
                clist = []
                kinfo = ks[1]
                kernel = kinfo.kernel
                if typ not in ['cell', 'interior_facet', 'exterior_facet']:
                    raise NotImplementedError("Integral type %s not currently supported." % typ)

                # Checking for facet integrals
                if typ in ['interior_facet', 'exterior_facet']:
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
                    row, col = ks[0]
                except:
                    row = ks[0][0]
                    col = 0
                rshape = exp.shapes[0][row]
                rstart = sum(exp.shapes[0][:row])
                try:
                    cshape = exp.shapes[1][col]
                    cstart = sum(exp.shapes[1][:col])
                except:
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
                if typ in ['interior_facet', 'exterior_facet']:
                    itsym = ast.Symbol("i0")
                    block = []
                    mesh = coords.function_space().mesh()
                    nfacet = mesh._plex.getConeSize(mesh._plex.getHeightStratum(0)[0])
                    clist.append(ast.FlatBlock("&%s" % itsym))

                    # Check if facet is interior or exterior
                    if typ == 'exterior_facet':
                        isinterior = 0
                    else:
                        isinterior = 1

                    # Construct the body of the facet loop
                    block.append(ast.If(ast.Eq(ast.Symbol(cellfacetsym,
                                                          rank=(itsym, )),
                                               isinterior),
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

    def pars(expr, prec=None, parent=None):
        """Parses a slate expression and returns a string representation."""

        if prec is None or parent >= prec:
            return expr
        return "(%s)" % expr

    @singledispatch
    def get_c_str(expr, temps, prec=None):
        """Translates a SLATE expression into its equivalent
        representation in C."""
        raise NotImplementedError("Expression of type %s not supported.",
                                  type(expr).__name__)

    @get_c_str.register(Action)
    @get_c_str.register(Scalar)
    @get_c_str.register(Vector)
    @get_c_str.register(Matrix)
    def get_c_str_tensors(expr, temps, prec=None):
        """Generates code representation of the SLATE tensor
        via COFFEE gencode method."""

        return temps[expr].gencode()

    @get_c_str.register(UnaryOp)
    def get_c_str_uop(expr, temps, prec=None):
        """Generates code representation of the unary
        SLATE operators."""

        op = {operator.neg: '-',
              operator.pos: '+'}[expr.op]
        prec = expr.prec
        result = "%s%s" % (op, get_c_str(expr.children,
                                         temps,
                                         prec))

        return pars(result, expr.prec, prec)

    @get_c_str.register(BinaryOp)
    def get_c_str_bop(expr, temps, prec=None):
        """Generates code representation of the binary
        SLATE operations."""

        op = {operator.add: '+',
              operator.sub: '-',
              operator.mul: '*'}[expr.op]
        prec = expr.prec
        result = "%s %s %s" % (get_c_str(expr.children[0], temps,
                                         prec),
                               op,
                               get_c_str(expr.children[1], temps,
                                         prec))

        return pars(result, expr.prec, prec)

    @get_c_str.register(Inverse)
    def get_c_str_inv(expr, temps, prec=None):
        """Generates code representation of the
        inverse operation."""

        return "(%s).inverse()" % get_c_str(expr.children, temps)

    @get_c_str.register(Transpose)
    def get_c_str_t(expr, temps, prec=None):
        """Generates code representation of the
        transpose operation."""

        return "(%s).transpose()" % get_c_str(expr.children, temps)

    # Creating c statement for the resulting linear algebra expression
    result_type = map_type(mat_type(shape))
    result_sym = ast.Symbol("T%d" % len(temps))
    result_data_sym = ast.Symbol("A%d" % len(temps))
    result = ast.Decl(dtype, ast.Symbol(result_data_sym, shape))

    result_statement = ast.FlatBlock("%s %s((%s *)%s);\n" %
                                     (result_type, result_sym,
                                      dtype, result_data_sym))

    statements.append(result_statement)
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

    kernel = ast.FunDecl("void", "hfc_compile_slate", arglist,
                         ast.Block(statements),
                         pred=["static", "inline"])

    # Finally we transform the kernel into a set of C++ template functions.
    # This produces the final kernel AST
    klist = []
    transformkernel = Transformer()
    oriented = False
    for v in kernel_exprs.values():
        for (_, ks) in v:
            for k in ks:
                oriented = oriented or k.kinfo.oriented
                # TODO: Think about this. Is this true for SLATE?
                assert k.kinfo.subdomain_id == "otherwise"
                kast = transformkernel.visit(k.kinfo.kernel._ast)
                klist.append(kast)

    klist.append(kernel)
    kernelast = ast.Node(klist)
    inc.append('%s/include/eigen3' % sys.prefix)

    # Produce the op2 kernel object for assembly
    op2kernel = firedrake.op2.Kernel(kernelast,
                                     "hfc_compile_slate",
                                     cpp=True,
                                     include_dirs=inc,
                                     headers=['#include <Eigen/Dense>',
                                              '#define restrict'])

    # TODO: What happens for multiple ufl domains?
    assert len(slate_expr.ufl_domains()) == 1
    kinfo = KernelInfo(kernel=op2kernel,
                       integral_type=slate_expr.integrals()[0].integral_type(),
                       oriented=oriented,
                       subdomain_id="otherwise",
                       domain_number=0,
                       coefficient_map=range(len(coeffs)),
                       needs_cell_facets=need_cell_facets)
    idx = tuple([0]*len(slate_expr.arguments()))

    if testing:
        return coords, coeffs, need_cell_facets, op2kernel
    return (SplitKernel(idx, kinfo), )
