"""This is a system for linear algebra operations on algebraic tensor expressions.

Written by: Thomas Gibson
"""

from __future__ import absolute_import

import numpy as np
from singledispatch import singledispatch
import operator
import itertools
import functools

import firedrake
from ufl.form import Form
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction

from coffee import base as ast


class CheckRestrictions(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def negative_restricted(self, o):
        raise ValueError("Cell-wise integrals may only contain positive restrictions.")


class RemoveRestrictions(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class Tensor(Form):
    """An abstract class for Tensor SLATE nodes.
    This tensor class also inherits directly from
    ufl.form for composability purposes.
    """

    children = ()
    id_num = 0

    def __init__(self, arguments, coefficients, integrals):
        self.id = Tensor.id_num
        self._arguments = tuple(arguments)
        self._coefficients = tuple(coefficients)
        self._integrals = tuple(integrals)
        self._hash = None
        self._ufl_domain = self._arguments[0].ufl_domain()
        shape = []
        shapes = {}
        for i, arg in enumerate(self._arguments):
            V = arg.function_space()
            shapeList = []
            for funcspace in V:
                shapeList.append(funcspace.fiat_element.space_dimension() *
                                 funcspace.dim)
            shapes[i] = tuple(shapeList)
            shape.append(sum(shapeList))
        self.shapes = shapes
        self.shape = tuple(shape)

    def arguments(self):
        return self._arguments

    def coefficients(self):
        return self._coefficients

    def __add__(self, other):
        return TensorAdd(self, other)

    def __sub__(self, other):
        return TensorSub(self, other)

    def __mul__(self, other):
        return TensorMul(self, other)

    def __neg__(self):
        return Negative(self)

    def __pos__(self):
        return Positive(self)

    @property
    def inv(self):
        return Inverse(self)

    @property
    def T(self):
        return Transpose(self)

    @property
    def operands(self):
        """Returns the objects which this object
        operates on.
        """
        return ()

    @classmethod
    def check_integrals(cls, integrals):
        mapper = CheckRestrictions()
        for it in integrals:
            map_integrand_dags(mapper, it)

    def _bop(self, other, op):
        ops = {operator.add: TensorAdd,
               operator.sub: TensorSub,
               operator.mul: TensorMul}
        assert isinstance(other, Tensor)
        return ops[op](self, other)

    def ufl_domain(self):
        """Returns the ufl_domain of the tensor
        object."""
        return self._ufl_domain

    def tensor_integrals(self):
        """Return a sequence of all integrals
        associated with this tensor object."""
        return self._integrals

    def tensor_integrals_by_type(self, integral_type):
        """Return a sequence of integrals of the
        associated form of the tensor with all
        particular domain types."""
        return tuple(integral for integral in self.tensor_integrals()
                     if integral.integral_type() == integral_type)

    def __hash__(self):
        """Hash code for use in dictionary objects."""
        if self._hash is None:
            self._hash = hash((type(self), )
                              + tuple(hash(i) for i in self.tensor_integrals()))
        return self._hash


class Scalar(Tensor):
    """An abstract class for scalar-valued SLATE nodes."""

    __slots__ = ('rank', 'form')
    __front__ = ('rank', 'form')

    def __init__(self, form):
        r = len(form.arguments())
        assert r == 0
        self.rank = r
        self.check_integrals(form.integrals())
        self.form = form
        Tensor.id_num += 1
        super(Scalar, self).__init__(arguments=(),
                                     coefficients=form.coefficients(),
                                     integrals=())

    def __str__(self):
        return "S_%d" % self.id

    __repr__ = __str__


class Vector(Tensor):
    """An abstract class for vector-valued SLATE nodes."""

    __slots__ = ('rank', 'form')
    __front__ = ('rank', 'form')

    def __init__(self, form):
        r = len(form.arguments())
        assert r == 1
        self.rank = r
        self.check_integrals(form.integrals())
        self.form = form
        Tensor.id_num += 1
        super(Vector, self).__init__(arguments=form.arguments(),
                                     coefficients=form.coefficients(),
                                     integrals=form.integrals())

    def __str__(self):
        return "V_%d" % self.id

    __repr__ = __str__


class Matrix(Tensor):
    """An abstract class for matrix-valued SLATE nodes."""

    __slots__ = ('rank', 'form')
    __front__ = ('rank', 'form')

    def __init__(self, form):
        r = len(form.arguments())
        assert r == 2
        self.rank = r
        self.check_integrals(form.integrals())
        self.form = form
        Tensor.id_num += 1
        super(Matrix, self).__init__(arguments=form.arguments(),
                                     coefficients=form.coefficients(),
                                     integrals=form.integrals())

    def __str__(self):
        return "M_%d" % self.id

    __repr__ = __str__


class Inverse(Tensor):
    """An abstract class for the tensor inverse SLATE node."""

    __slots__ = ('children', 'rank')

    def __init__(self, tensor):
        if (tensor.shape[0] != tensor.shape[1]):
            raise ValueError("Cannot take the inverse of a non-square tensor.")
        self.children = tensor
        reversedargs = tensor.arguments()[::-1]
        Tensor.id_num += 1
        super(Inverse, self).__init__(arguments=reversedargs,
                                      coefficients=tensor.coefficients(),
                                      integrals=tensor.tensor_integrals())

    def __str__(self):
        return "%s.inv" % self.children

    def __repr__(self):
        return "Inverse(%s)" % self.children

    @property
    def operands(self):
        return (self.children, )


class Transpose(Tensor):
    """An abstract class for the tensor transpose SLATE node."""

    __slots__ = ('children', 'rank')

    def __init__(self, tensor):
        self.children = tensor
        reversedargs = tensor.arguments()[::-1]
        Tensor.id_num += 1
        super(Transpose, self).__init__(arguments=reversedargs,
                                        coefficients=tensor.coefficients(),
                                        integrals=tensor.tensor_integrals())

    def __str__(self):
        return "%s.T" % self.children

    def __repr__(self):
        return "Transpose(%s)" % self.children

    @property
    def operands(self):
        return (self.children, )


class UnaryOp(Tensor):
    """An abstract SLATE class for unary operations on tensors.
    Such operations take only one operand, ie a single input.
    An example is the negation operator: ('Negative(A)' = -A).
    """

    __slots__ = ('children', 'rank')

    def __init__(self, tensor):
        self.children = tensor
        args = get_arguments(tensor)
        coeffs = get_coefficients(tensor)
        ints = get_integrals(tensor)
        Tensor.id_num += 1
        super(UnaryOp, self).__init__(arguments=args,
                                      coefficients=coeffs,
                                      integrals=ints)

    def __str__(self, order_of_operation=None):
        ops = {operator.neg: '-',
               operator.pos: '+'}
        if (order_of_operation is None) or (self.order_of_operation >= order_of_operation):
            pars = lambda X: X
        else:
            pars = lambda X: "(%s)" % X

        return pars("%s%s" % (ops[self.operation], self.children__str__()))

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.children)

    @property
    def operands(self):
        return (self.children, )


class Negative(UnaryOp):
    """Class for the negation of a tensor object."""

    # Class variables for the negation operator
    order_of_operation = 1
    operation = operator.neg


class Positive(UnaryOp):
    """Class for the positive operation on a tensor."""

    # Class variables
    order_of_operation = 1
    operation = operator.pos


class BinaryOp(Tensor):
    """An abstract SLATE class for binary operations on tensors.
    Such operations take two operands, and returns a tensor-valued
    expression.
    """

    __slots__ = ('children', 'rank')

    def __init__(self, A, B):
        self.children = A, B
        args = self.assemble_arguments(A, B)
        coeffs = self.assemble_coefficients(A, B)
        integs = self.assemble_integrals(A, B)
        Tensor.id_num += 1
        super(BinaryOp, self).__init__(arguments=args,
                                       coefficients=coeffs,
                                       integrals=integs)

    @classmethod
    def assemble_arguments(cls, A, B):
        pass

    @classmethod
    def assemble_coefficients(cls, A, B):
        clist = []
        A_coeffs = get_coefficients(A)
        uniquelistforA = set(A_coeffs)
        for c in get_coefficients(B):
            if c not in uniquelistforA:
                clist.append(c)
        return tuple(list(A_coeffs) + clist)

    @classmethod
    def assemble_integrals(cls, A, B):
        pass

    @property
    def operands(self):
        return (self.children)

    def __str__(self, order_of_operation=None):
        ops = {operator.add: '+',
               operator.sub: '-',
               operator.mul: '*'}
        if (order_of_operation is None) or (self.order_of_operation >= order_of_operation):
            pars = lambda X: X
        else:
            pars = lambda X: "(%s)" % X
        operand1 = self.children[0].__str__()
        operand2 = self.children[1].__str__()
        result = "%s %s %s" % (operand1, ops[self.operation],
                               operand2)
        return pars(result)

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__,
                               self.children[0],
                               self.children[1])


class TensorAdd(BinaryOp):
    """This class represents the binary operation of
    addition on tensor objects.
    """

    # Class variables for tensor addition
    order_of_operation = 1
    operation = operator.add

    @classmethod
    def assemble_arguments(cls, A, B):
        if isinstance(A, Scalar):
            return get_arguments(B)
        elif isinstance(B, Scalar):
            return get_arguments(A)
        assert A.shape == B.shape
        return get_arguments(A)

    @classmethod
    def assemble_integrals(cls, A, B):
        return get_integrals(A) + get_integrals(B)


class TensorSub(BinaryOp):
    """This class represents the binary operation of
    subtraction on tensor objects.
    """

    # Class variables for tensor subtraction
    order_of_operation = 1
    operation = operator.sub

    @classmethod
    def assemble_arguments(cls, A, B):
        if isinstance(A, Scalar):
            return get_arguments(B)
        elif isinstance(B, Scalar):
            return get_arguments(A)
        assert A.shape == B.shape
        return get_arguments(A)

    @classmethod
    def assemble_integrals(cls, A, B):
        return get_integrals(A) + get_integrals(B)


class TensorMul(BinaryOp):
    """This class represents the binary operation of
    multiplication on tensor objects.
    """

    # Class variables for tensor product
    order_of_operation = 2
    operation = operator.mul

    @classmethod
    def assemble_arguments(cls, A, B):
        if isinstance(A, Scalar):
            return get_arguments(B)
        elif isinstance(B, Scalar):
            return get_arguments(A)
        argsA = get_arguments(A)
        argsB = get_arguments(B)
        assert (argsA[-1].function_space() ==
                argsB[0].function_space())
        return argsA[:-1] + argsB[1:]

    @classmethod
    def assemble_integrals(cls, A, B):
        intsA = get_integrals(A)
        intsB = get_integrals(B)
        return intsA[:-1] + intsB[1:]


@singledispatch
def get_arguments(expr):
    raise ValueError("Tensors of type %s are not supported." % type(expr))


@get_arguments.register(Scalar)
@get_arguments.register(Vector)
@get_arguments.register(Matrix)
def get_tensor_arguments(tensor):
    return tensor.arguments()


@get_arguments.register(Inverse)
@get_arguments.register(Transpose)
@get_arguments.register(UnaryOp)
def get_uop_arguments(expr):
    if isinstance(expr, (Inverse, Transpose)):
        return get_arguments(expr.children)[::-1]
    return get_arguments(expr.children)


@get_arguments.register(TensorAdd)
@get_arguments.register(TensorSub)
def get_bop_arguments(expr):
    A = expr.children[0]
    B = expr.children[1]
    # Scalars distribute over sums/diffs
    if isinstance(A, Scalar):
        return get_arguments(B)
    elif isinstance(B, Scalar):
        return get_arguments(A)
    assert A.shape == B.shape
    return get_arguments(A)


@get_arguments.register(TensorMul)
def get_product_arguments(expr):
    A = expr.children[0]
    B = expr.children[1]
    # Scalars distribute over sums
    if isinstance(A, Scalar):
        return get_arguments(B)
    elif isinstance(B, Scalar):
        return get_arguments(A)
    # Check for function space type to perform contraction
    # over middle indices
    assert (get_arguments(A)[-1].function_space() ==
            get_arguments(B)[0].function_space())
    return get_arguments(A)[:-1] + get_arguments(B)[1:]


@singledispatch
def get_coefficients(expr):
    raise ValueError("Tensors of type %s is not supported." % type(expr))


@get_coefficients.register(Scalar)
@get_coefficients.register(Vector)
@get_coefficients.register(Matrix)
def get_tensor_coefficients(tensor):
    return tensor.coefficients()


@get_coefficients.register(Inverse)
@get_coefficients.register(Transpose)
@get_coefficients.register(UnaryOp)
def get_uop_coefficients(expr):
    return get_coefficients(expr.children)


@get_coefficients.register(BinaryOp)
def get_bop_coefficients(expr):
    A = expr.children[0]
    B = expr.children[1]
    # Remove duplicate coefficients in forms
    coeffs = []
    A_uniquecoeffs = set(get_coefficients(A))
    for c in get_coefficients(B):
        if c not in A_uniquecoeffs:
            coeffs.append(c)
    return tuple(list(get_coefficients(A)) + coeffs)


@singledispatch
def get_integrals(expr):
    raise ValueError("Tensors of type %s is not supported." % type(expr))


@get_integrals.register(Scalar)
@get_integrals.register(Vector)
@get_integrals.register(Matrix)
def get_tensor_integrals(tensor):
    return tensor.tensor_integrals()


@get_integrals.register(Inverse)
@get_integrals.register(Transpose)
@get_integrals.register(UnaryOp)
def get_uop_integrals(expr):
    return get_integrals(expr.children)


@get_integrals.register(TensorAdd)
@get_integrals.register(TensorSub)
def get_bop_integrals(expr):
    A = expr.children[0]
    B = expr.children[1]
    return get_integrals(A) + get_integrals(B)


@get_integrals.register(TensorMul)
def get_product_integrals(expr):
    A = expr.children[0]
    B = expr.children[1]
    return get_integrals(A)[:-1] + get_integrals(B)[1:]


def compile_slate_expression(slate_expr):
    """Takes a SLATE expression `slate_expr` and returns the appropriate
    coordinates `coords` of the mesh its defined on, the `coefficients` in
    the SLATE expression and the :class:`firedrake.op2.Kernel` object
    representing the SLATE expression.

    :arg slate_expr: A SLATE expression to unpack.

    """

    dtype = "double"
    shape = slate_expr.shape
    temps = {}
    kernel_exprs = {}
    coeffs = slate_expr.coefficients()
    statements = []
    need_cell_facets = False

    def mat_type(shape):
        if len(shape) == 1:
            rows = shape[0]
            cols = 1
        else:
            if not len(shape) == 2:
                raise ValueError(
                    "%d-rank tensors are not currently supported" % len(shape))
            rows = shape[0]
            cols = shape[1]
        if cols != 1:
            order = ", Eigen::RowMajor"
        else:
            order = ""
        return "Eigen::Matrix<double, %d, %d%s>" % (rows, cols, order)

    def map_type(matrix):
        return "Eigen::Map<%s >" % matrix

    # Compile forms associated with a temporary
    @singledispatch
    def get_kernel_expr(expr):
        raise NotImplementedError("Expression of type %s not supported",
                                  type(expr).__name__)

    @get_kernel_expr.register(Scalar)
    @get_kernel_expr.register(Vector)
    @get_kernel_expr.register(Matrix)
    def get_kernel_expr_tensor(expr):
        if expr not in temps.keys():
            sym = "T%d" % len(temps)
            temp = ast.Symbol(sym)
            temp_type = mat_type(expr.shape)
            temps[expr] = temp
            statements.append(ast.Decl(temp_type, temp))

            integrands = expr.integrals()
            kernel_exprs[expr] = []
            mapper = RemoveRestrictions()
            for i, integ in enumerate(integrands):
                typ = integ.integral_type()
                form = Form([integ])
                prefix = "subkernel%d_%d_%s_" % (len(kernel_exprs), i, typ)
                if typ == "interior_facet":
                    newinteg = map_integrand_dags(mapper, integ)
                    newinteg = newinteg.reconstruct(integral_type="exterior_facet")
                    form = Form([newinteg])

                compiled_form = firedrake.tsfc_interface.compile_form(form, prefix)
                kernel_exprs[expr].append((typ, compiled_form))
        return

    @get_kernel_expr.register(UnaryOp)
    @get_kernel_expr.register(BinaryOp)
    @get_kernel_expr.register(Transpose)
    @get_kernel_expr.register(Inverse)
    def get_kernel_expr_ops(expr):
        map(get_kernel_expr, expr.operands)
        return

    coeffmap = dict((c, ast.Symbol("w%d" % i)) for i, c in enumerate(coeffs))
    coordsym = ast.Symbol("coords")
    coords = None
    cellfacetsym = ast.Symbol("cell_facets")
    get_kernel_expr(slate_expr)
    inc = []
    for exp, t in temps.items():
        statements.append(ast.FlatBlock("%s.setZero();\n" % t))
        for (typ, klist) in kernel_exprs[exp]:
            for ks in klist:
                clist = []
                kinfo = ks[1]
                kernel = kinfo.kernel
                if typ not in ["cell", "interior_facet", "exterior_facet"]:
                    raise NotImplementedError("Integral type '%s' not supported." % typ)

                # Checking for facet integral
                if typ in ["interior_facet", "exterior_facet"]:
                    need_cell_facets = True

                # Extracting coordinates
                if coords is not None:
                    assert exp.ufl_domain().coordinates == coords
                else:
                    coords = exp.ufl_domain().coordinates

                # Extracting coefficients
                for cindex in list(kinfo[4]):
                    coeff = exp.coefficients()[cindex]
                    clist.append(coeffmap[coeff])

                # Defining tensor matrices of appropriate size
                inc.extend(kernel._include_dirs)
                row, col = ks[0]
                rshape = exp.shapes[0][row]
                rstart = sum(exp.shapes[0][:row])
                try:
                    cshape = exp.shapes[1][col]
                    cstart = sum(exp.shapes[1][:col])
                except:
                    cshape = 1
                    cstart = 0

                # Creating sub-block if tensor is mixed
                if (rshape, cshape) != exp.shape:
                    tensor = ast.FlatBlock("%s.block<%d,%d>(%d, %d)" %
                                           (t, rshape, cshape,
                                            rstart, cstart))
                else:
                    tensor = t

                # Facet integral loop
                if typ in ["exterior_facet", "interior_facet"]:
                    integsym = ast.Symbol("i0")
                    block = []
                    mesh = coords.function_space().mesh()
                    nfacet = mesh._plex.getConeSize(mesh._plex.getHeightStratum(0)[0])
                    clist.append(ast.FlatBlock("&%s" % integsym))
                    if typ == "exterior_facet":
                        check = 0
                    else:
                        check = 1
                    block.append(
                        ast.If(ast.Eq(ast.Symbol(cellfacetsym,
                                                 rank=(integsym, )),
                                      check), [ast.Block([ast.FunCall(kernel.name,
                                                                      tensor,
                                                                      coordsym,
                                                                      *clist)],
                                                         open_scope=True)]))
                    loop = ast.For(ast.Decl("unsigned int", integsym, init=0),
                                   ast.Less(integsym, nfacet),
                                   ast.Incr(integsym, 1),
                                   block)
                    statements.append(loop)
                else:
                    statements.append(ast.FunCall(kernel.name,
                                                  tensor,
                                                  coordsym,
                                                  *clist))

    def pars(expr, order_of_op=None, parent=None):
        if order_of_op is None or parent >= order_of_op:
            return expr
        return "(%s)" % expr

    @singledispatch
    def get_c_str(expr, temps, order_of_op=None):
        raise NotImplementedError("Expression of type %s not supported",
                                  type(expr).__name__)

    @get_c_str.register(Scalar)
    @get_c_str.register(Vector)
    @get_c_str.register(Matrix)
    def get_c_str_tensors(expr, temps, order_of_op=None):
        return temps[expr].gencode()

    @get_c_str.register(UnaryOp)
    def get_c_str_uop(expr, temps, order_of_op=None):
        op = {operator.neg: '-',
              operator.pos: '+'}[expr.operation]
        order_of_op = expr.order_of_operation
        result = "%s%s" % (op, get_c_str(expr.children,
                                         temps,
                                         order_of_op))
        return pars(result, expr.order_of_operation, order_of_op)

    @get_c_str.register(BinaryOp)
    def get_c_str_bop(expr, temps, order_of_op=None):
        op = {operator.add: '+',
              operator.sub: '_',
              operator.mul: '*'}[expr.operation]
        order_of_op = expr.order_of_operation
        result = "%s %s %s" % (get_c_str(expr.children[0], temps,
                                         order_of_op),
                               op,
                               get_c_str(expr.children[1], temps,
                                         order_of_op))
        return pars(result, expr.order_of_operation, order_of_op)

    @get_c_str.register(Inverse)
    def get_c_str_inv(expr, temps, order_of_op=None):
        return "(%s).inverse()" % get_c_str(expr.children, temps)

    @get_c_str.register(Transpose)
    def get_c_str_t(expr, temps, order_of_op=None):
        return "(%s).transpose()" % get_c_str(expr.children, temps)

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

    arglist = [result, ast.Decl("%s **" % dtype, coordsym)]
    for c in coeffs:
        ctype = "%s **" % dtype
        if isinstance(c, firedrake.Constant):
            ctype = "%s *" % dtype
        arglist.append(ast.Decl(ctype, coeffmap[c]))

    if need_cell_facets:
        arglist.append(ast.Decl("char *", cellfacetsym))

    kernel = ast.FunDecl("void", "compile_slate_expression", arglist,
                         ast.Block(statements),
                         pred=["static", "inline"])

    klist = []
    for v in kernel_exprs.values():
        for (_, ks) in v:
            for k in ks:
                kast = k.kinfo.kernel._ast
                klist.append(ast.FlatBlock(kast.gencode()))

    klist.append(kernel)
    kernelast = ast.Node(klist)
    op2kernel = firedrake.op2.Kernel(kernelast,
                                     "compile_slate_expression",
                                     cpp=True,
                                     include_dirs=inc,
                                     headers=["#include <eigen3/Eigen/Dense>"])

    return coords, coeffs, need_cell_facets, op2kernel


def slate_assemble(expr, bcs=None):
    """Assemble the SLATE expression `expr` and return a Firedrake object
    representing the result. This will be a :class:`float` for rank-0
    tensors, a :class:`.Function` for rank-1 tensors and a :class:`.Matrix`
    for rank-2 tensors. The result will be returned as a `tensor` of
    :class:`firedrake.Function` for rank-0 and rank-1 SLATE expressions and
    :class:`firedrake.op2.Mat` for rank-2 SLATE expressions.

    :arg expr: A SLATE object to assemble.
    :arg bcs: A tuple of :class:`.DirichletBC`\s to be applied.

    """

    arguments = expr.arguments()
    integrals = expr.integrals()
    rank = len(arguments)

    # If the expression is a rank-2 tensor: matrix
    if rank == 2:
        test_function, trial_function = arguments
        maps = tuple((test_function.cell_node_map(), trial_function.cell_node_map()))
        sparsity = firedrake.op2.Sparsity((test_function.function_space().dof_dset,
                                           trial_function.function_space().dof_dset),
                                          maps)
        tensor = firedrake.op2.Mat(sparsity, np.float64)
        tensor_arg = tensor(firedrake.op2.INC, (test_function.cell_node_map()[firedrake.op2.i[0]],
                                                trial_function.cell_node_map()[firedrake.op2.i[0]]),
                            flatten=True)

    # If the expression is a rank-1 tensor: vector
    elif rank == 1:
        test_function = arguments[0]
        tensor = firedrake.Function(test_function.function_space())
        tensor_arg = tensor.dat(firedrake.op2.INC, test_function.cell_node_map()[firedrake.op2.i[0]],
                                flatten=True)

    # if the expression is a rank-0 tensor: scalar
    elif rank == 0:
        tensor = firedrake.op2.Global(1, [0.0])
        tensor_arg = tensor(firedrake.op2.INC)
    else:
        raise NotImplementedError("Not implemented for rank-%d tensors.", rank)

    coords, coefficients, need_cell_facets, klist, kernel = compile_slate_expression(expr)
    mesh = coords.function_space().mesh()
    args = [kernel, mesh.cell_set, tensor_arg, coords.dat(firedrake.op2.READ,
                                                          coords.cell_node_map(),
                                                          flatten=True)]
    for c in coefficients:
        args.append(c.dat(firedrake.op2.READ, c.cell_node_map(), flatten=True))

    if need_cell_facets:
        for it in integrals:
            if it.integral_type() == "cell":
                pass
            elif it.integral_type() == "exterior_facet":
                args.append(mesh.exterior_facets.local_facet_dat(firedrake.op2.READ))
            elif it.integral_type() == "interior_facet":
                args.append(mesh.interior_facets.local_facet_dat(firedrake.op2.READ))
            else:
                raise NotImplementedError("Integrals of type %s not implemeneted." % it.integral_type())

    firedrake.op2.par_loop(*args)
    if bcs is not None and rank == 2:
        for bc in bcs:
            tensor.set_local_diagonal_entries(bc.nodes)

    return tensor
