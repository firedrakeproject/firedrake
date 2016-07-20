"""This is a symbolic language for algebraic tensor expressions.
This work is based on a template written by Lawrence Mitchell.

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
from coffee import base as ast
from tsfc import compile_form as tsfc_compile_form


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
        self._integrals = integrals
        self._hash = None
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
        self.rank = tensor.rank
        self.form = tensor.form
        Tensor.id_num += 1
        super(Inverse, self).__init__(arguments=reversed(tensor.arguments()),
                                      coefficients=tensor.coefficients(),
                                      integrals=tensor.tensor_integrals())

    def __str__(self):
        return "%s.inverse()" % self.children

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
        self.rank = tensor.rank
        self.form = tensor.form
        Tensor.id_num += 1
        super(Transpose, self).__init__(arguments=reversed(tensor.arguments()),
                                        coefficients=tensor.coefficients(),
                                        integrals=tensor.tensor_integrals())

    def __str__(self):
        return "%s.transpose()" % self.children

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

    __slots__ = ('children', )

    def __init__(self, tensor):
        self.children = tensor
        Tensor.id_num += 1
        super(UnaryOp, self).__init__(arguments=tensor.arguments(),
                                      coefficients=tensor.coefficients(),
                                      integrals=tensor.tensor_integrals())

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

    __slots__ = ('children', )

    def __init__(self, A, B):
        self.children = A, B
        args = self.get_arguments(A, B)
        coeffs = self.get_coefficients(A, B)
        integs = tuple(list(A.tensor_integrals()) +
                          list(B.tensor_integrals()))
        Tensor.id_num += 1
        super(BinaryOp, self).__init__(arguments=args,
                                       coefficients=coeffs,
                                       integrals=integs)

    @classmethod
    def get_arguments(cls, A, B):
        pass

    @classmethod
    def get_coefficients(cls, A, B):
        # Remove duplicate coefficients in forms
        coeffs = []
        A_uniquecoeffs = set(A.coefficients())
        for c in B.coefficients():
            if c not in A_uniquecoeffs:
                coeffs.append(c)
        return tuple(list(A.coefficients()) + coeffs)

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
    def get_arguments(cls, A, B):
        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        assert A.shape == B.shape
        return A.arguments()


class TensorSub(BinaryOp):
    """This class represents the binary operation of
    subtraction on tensor objects.
    """

    # Class variables for tensor subtraction
    order_of_operation = 1
    operation = operator.sub

    @classmethod
    def get_arguments(cls, A, B):
        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        assert A.shape == B.shape
        return A.arguments()


class TensorMul(BinaryOp):
    """This class represents the binary operation of
    multiplication on tensor objects.
    """

    # Class variables for tensor product
    order_of_operation = 2
    operation = operator.mul

    @classmethod
    def get_arguments(cls, A, B):
        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        # Check for function space type to perform contraction
        # over middle indices
        assert (A.arguments()[-1].function_space() ==
                B.arguments()[0].function_space())
        return A.arguments()[:-1] + B.arguments()[1:]


def macro_kernel(slate_expr):
    dtype = "double"
    shape = slate_expr.shape
    temps = {}
    kernel_exprs = {}
    templist = []
    coeffs = slate_expr.coefficients()

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
    @get_kernel_expr.register(Transpose)
    @get_kernel_expr.register(Inverse)
    def get_kernel_expr_tensor(expr):
        temp = ast.Symbol("t%d" % len(temps))
        temp_type = mat_type(expr.shape)
        temps[expr] = temp
        print len(temps)
        compiled_form = tsfc_compile_form(expr.form)
        coefflist = [c for c in coeffs if c in expr.coefficients()]
        print coefflist
        templist.append(((temp.symbol, temp_type, expr), coefflist, compiled_form))
        kernel_exprs[expr] = ((temp.symbol, temp_type), coefflist, compiled_form)
        return

    @get_kernel_expr.register(UnaryOp)
    @get_kernel_expr.register(BinaryOp)
    def get_kernel_expr_ops(expr):
        map(get_kernel_expr, expr.operands)
        return

    get_kernel_expr(slate_expr)

    # Declare temporary variables here
    return kernel_exprs, templist, temps
