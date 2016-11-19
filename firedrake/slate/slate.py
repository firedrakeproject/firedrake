"""SLATE is a symbolic language defining a framework for
performing linear algebra operations on finite element
tensors. It is similar in principle to most linear algebra
libraries in notation.

The design of SLATE was heavily influenced by UFL, and
utilizes much of UFL's functionality for FEM-specific form
manipulation.

Unlike UFL, however, once forms are assembled into SLATE
`Tensor` objects, one can utilize the operations defined
in SLATE to express complicated linear algebra operations.
(Such as the Schur-complement reduction of a block-matrix
system)

All SLATE expressions are handled by a specialized linear
algebra compiler, which interprets SLATE expressions and
produces C++ kernel functions to be executed within the
Firedrake architecture.
"""
from __future__ import absolute_import, print_function, division

from firedrake.utils import cached_property

from ufl.form import Form
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction
from ufl.domain import join_domains, sort_domains


__all__ = ['TensorBase', 'Tensor', 'Inverse', 'Transpose',
           'UnaryOp', 'Negative', 'BinaryOp', 'TensorAdd',
           'TensorSub', 'TensorMul']


class TensorBase(object):
    """An abstract SLATE node class.

    .. warning::

       Do not instantiate this class on its own. This is an abstract
       node class; is not meant to be worked with directly. Only use
       the appropriate subclasses.
    """

    id = 0

    def __init__(self):
        """Constructor for the TensorBase abstract class."""
        self.id = TensorBase.id
        TensorBase.id += 1
        self._integral_domains = None
        self._subdomain_data = None
        self._hash = None

    @cached_property
    def shape(self):
        """Computes the shape information of the local tensor."""
        shape = []
        for i, arg in enumerate(self.arguments()):
            V = arg.function_space()
            shapelist = []
            for fs in V:
                shapelist.append(fs.fiat_element.space_dimension() * fs.dim)
            shape.append(sum(shapelist))
        return tuple(shape)

    @cached_property
    def shapes(self):
        """Computers the internal shape information of its components.
        This is particularly useful to know if the tensor comes from a
        mixed form.
        """
        shapes = {}
        for i, arg in enumerate(self.arguments()):
            V = arg.function_space()
            shapelist = []
            for fs in V:
                shapelist.append(fs.fiat_element.space_dimension() * fs.dim)
            shapes[i] = tuple(shapelist)
        return shapes

    @cached_property
    def rank(self):
        """Returns the rank information of the tensor object."""
        return len(self.arguments())

    @property
    def inv(self):
        return Inverse(self)

    @property
    def T(self):
        return Transpose(self)

    def __add__(self, other):
        return TensorAdd(self, other)

    def __radd__(self, other):
        """Ordering of tensor addition does not matter."""
        return self.__add__(other)

    def __sub__(self, other):
        return TensorSub(self, other)

    def __rsub__(self, other):
        """Ordering of tensor subtraction does not matter."""
        return self.__sub__(other)

    def __mul__(self, other):
        return TensorMul(self, other)

    def __rmul__(self, other):
        """Tensor multiplication is not commutative in general."""
        return TensorMul(other, self)

    def __neg__(self):
        return Negative(self)

    def __pos__(self):
        return self

    def check_integrals(self, integrals):
        """Checks for positive restrictions on integrals."""
        mapper = CheckRestrictions()
        for it in integrals:
            map_integrand_dags(mapper, it)

    def generate_integral_info(self, integrals):
        """This function generates all relevant information for
        assembly that relies on :class:`ufl.Integral` objects.
        This function will generate the following information:

        (1) ufl_domains, which come from the integrals themselves;
        (2) and subdomain_data, which is a mapping on the tensor that maps
            integral_type to subdomain_id.

        :arg integrals: `ufl.Integral` objects that come from a `ufl.Form`
        """
        # Compute integration domains
        if self._integral_domains is None:
            integral_domains = join_domains([it.ufl_domain() for it in integrals])
            self._integral_domains = sort_domains(integral_domains)

        # Generate subdomain data
        if self._subdomain_data is None:
            # Scalar case
            if self.rank == 0:
                # subdomain_data should be None
                return
            else:
                # Initializing subdomain_data
                subdomain_data = {}
                for domain in self._integral_domains:
                    subdomain_data[domain] = {}

                    for integral in integrals:
                        domain = integral.ufl_domain()
                        it_type = integral.integral_type()
                        subdata = integral.subdomain_data()

                        data = subdomain_data[domain].get(it_type)
                        if data is None:
                            subdomain_data[domain][it_type] = subdata
                        elif subdata is not None:
                            assert data.ufl_id() == subdata.ufl_id(), "Integrals in the tensor must have the same subdomain_data objects."
                self._subdomain_data = subdomain_data

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        return self._integral_domains

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found.
        """
        domains = self.ufl_domains()
        assert all(domain == domains[0] for domain in domains), "All integrals must share the same domain of integration."

        return domains[0]

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`.
        """
        return self._subdomain_data

    def __hash__(self):
        """Returns a hash code for use in dictionary objects."""
        if self._hash is None:
            self._hash = hash((type(self), )
                              + tuple(hash(a) for a in self.arguments())
                              + tuple(hash(c) for c in self.coefficients()))
        return self._hash


class Tensor(TensorBase):
    """This class is a symbolic representation of a finite element tensor
    derived from a bilinear or linear form. This class implements all
    supported ranks of general tensor (rank-0, rank-1 and rank-2 tensor
    objects). This class is the primary user-facing class that the SLATE
    symbolic algebra supports.

    :arg form: a :class:`ufl.Form` object.

    A :class:`ufl.Form` is currently the only supported input of creating
    a `slate.Tensor` object:

        (1) If the form is a bilinear form, namely a form with two
            :class:`ufl.Argument` objects, then the SLATE Tensor will be
            a rank-2 Matrix.
        (2) If the form has one `ufl.Argument` as in the case of a typical
            linear form, then this will create a rank-1 Vector.
        (3) A zero-form will create a rank-0 Scalar.

    These are all under the same type `slate.Tensor`. The attribute `self.rank`
    is used to determine what kind of tensor object is being handled.
    """

    def __init__(self, form):
        """Constructor for the Tensor class."""
        if not isinstance(form, Form):
            raise NotImplementedError("Only UFL forms are currently supported for creating SLATE tensors.")
        r = len(form.arguments())
        if r not in (0, 1, 2):
            raise NotImplementedError("Currently don't support tensors of rank %d." % r)
        self.check_integrals(form.integrals())
        self.form = form
        super(Tensor, self).__init__()

        # Generate ufl.Integral data from input form
        self.generate_integral_info(form.integrals())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.form.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.form.coefficients()

    def __str__(self, prec=None):
        """String representation of a tensor object in SLATE."""
        if self.rank == 0:
            return "S_%d" % self.id
        elif self.rank == 1:
            return "V_%d" % self.id
        elif self.rank == 2:
            return "M_%d" % self.id

    def __repr__(self):
        """SLATE representation of the tensor object."""
        if self.rank == 0:
            return "Scalar(%r)" % self.form
        elif self.rank == 1:
            return "Vector(%r)" % self.form
        elif self.rank == 2:
            return "Matrix(%r)" % self.form


class UnaryOp(TensorBase):
    """An abstract SLATE class for representing unary operations on a
    Tensor object.

    The currently supported unary operations on a SLATE :class:`TensorBase`
    expression are the following:

        (1) the inverse of a tensor, `A.inv`, implemented in the subclass
            `Inverse`;
        (2) the tranpose of a tensor, `A.T`, implemented in the subclass
            `Transpose`;
        (3) and the negative operation, `-A` (subclass `Negative`).

    :arg A: a :class:`TensorBase` object. This can be a terminal tensor object
            (:class:`Tensor`) or any derived expression resulting from any number
            of linear algebra operations on `Tensor` objects. For example,
            another instance of a `UnaryOp` object is an acceptable input, or
            a `BinaryOp` object.
    """

    def __init__(self, A):
        """Constructor for the UnaryOp class."""
        if not isinstance(A, TensorBase):
            raise Exception("Expecting a `slate.TensorBase` object, not %r" % A)
        self.tensor = A
        self.check_operand_inversion(A)
        super(UnaryOp, self).__init__()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.get_uop_arguments(self.tensor)

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.tensor.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        return self.tensor.ufl_domains()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`.
        """
        return self.tensor.subdomain_data()

    @classmethod
    def check_operand_inversion(cls, A):
        pass

    @classmethod
    def get_uop_arguments(cls, A):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.

        Implemented in subclass.
        """
        pass

    @property
    def operands(self):
        """Returns an iterable of the operands of this operation."""
        return (self.tensor,)

    def __str__(self, prec=None):
        """String representation of a resulting tensor after a unary
        operation is performed.
        """
        if self.op == "Inverse":
            return "(%s).inv" % self.tensor

        elif self.op == "Transpose":
            return "(%s).T" % self.tensor

        elif self.op == "Negative":
            if prec is None or self.prec >= prec:
                par = lambda x: x
            else:
                par = lambda x: "(%s)" % x

            return par("-%s" % self.tensor.__str__(prec=self.prec))

    def __repr__(self):
        """SLATE representation of the resulting tensor."""
        return "%s(%r)" % (type(self).__name__, self.tensor)


class Inverse(UnaryOp):
    """An abstract SLATE class representing the tranpose of a tensor."""

    prec = None
    op = "Inverse"

    @classmethod
    def check_operand_inversion(cls, A):
        """Ensures that the tensor satisfies the necessary conditions for taking
        inverses.
        """
        assert A.rank == 2, "The tensor must be rank 2."
        assert A.shape[0] == A.shape[1], "The inverse can only be computed on square tensors."

    @classmethod
    def get_uop_arguments(cls, A):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        return A.arguments()[::-1]


class Transpose(UnaryOp):
    """An abstract SLATE class representing the tranpose of a tensor."""

    prec = None
    op = "Transpose"

    @classmethod
    def get_uop_arguments(cls, A):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        return A.arguments()[::-1]


class Negative(UnaryOp):
    """Abstract SLATE class representing the negation of a tensor object."""

    prec = 1
    op = "Negative"

    @classmethod
    def get_uop_arguments(cls, A):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        return A.arguments()


class BinaryOp(TensorBase):
    """An abstract SLATE class representing binary operations on tensors.
    Such operations take two operands and returns a tensor-valued expression.

    The currently supported binary operations include the following:

        (1) The addition of two identical-rank :class:`TensorBase` objects. That is,
            the addition of two scalars, vectors or matrices. This operation is
            implemented in the subclass `TensorAdd`.
        (2) The subtraction of two scalar, vector or matrix expressions. See `TensorSub`.
        (3) The multiplication of two matrices, the action of a matrix on a vector or
            the action of a scalar on any other rank tensor. All cases are handled by the
            `TensorMul` subclass.

    :arg A: a :class:`TensorBase` object. This can be a terminal tensor object
            (:class:`Tensor`) or any derived expression resulting from any number
            of linear algebra operations on `Tensor` objects. For example,
            another instance of a `BinaryOp` object is an acceptable input, or
            a `UnaryOp` object.
    :arg B: a :class:`TensorBase` object.
    """

    def __init__(self, A, B):
        """Constructor for the BinaryOp class."""
        assert isinstance(A, TensorBase) and isinstance(B, TensorBase), ("Both operands must be slate.TensorBase objects. The operands given are of type (%s, %s)" % (type(A), type(B)))
        self.check_dimensions(A, B)
        self.tensors = A, B
        super(BinaryOp, self).__init__()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        A, B = self.tensors
        return self.get_bop_arguments(A, B)

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        A, B = self.tensors
        return self.get_bop_coefficients(A, B)

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        A, B = self.tensors
        return join_domains(A.ufl_domains() + B.ufl_domains())

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`.
        """
        # TODO: We need to have a discussion on how we need to handle subdomain_data
        A, B = self.tensors
        if A.rank == 0:
            return B.subdomain_data()
        elif B.rank == 0:
            return A.subdomain_data()
        return A.subdomain_data()

    @classmethod
    def check_dimensions(cls, A, B):
        """Performs a check on the shapes of the tensors before
        performing the binary operation.

        Implemented in subclass.
        """
        pass

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the expected arguments of the resulting tensor of
        performing a specific binary operation on two tensors.

        Implemented in subclass."""
        pass

    @classmethod
    def get_bop_coefficients(cls, A, B):
        """Returns the expected coefficients of the resulting tensor
        of performing a binary operation on two tensors. Note that
        the coefficients are handled the same way for all binary operations.
        """
        clist = []
        A_coeffs = A.coefficients()
        uniqueAcoeffs = set(A_coeffs)
        for c in B.coefficients():
            if c not in uniqueAcoeffs:
                clist.append(c)

        return tuple(list(A_coeffs) + clist)

    @property
    def operands(self):
        """Returns an iterable of the operands of the binary operation."""
        return self.tensors

    def __str__(self, prec=None):
        """String representation of the binary operation."""

        ops = {"Addition": '+',
               "Subtraction": '-',
               "Multiplication": '*'}
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x
        operand1 = self.operands[0].__str__(prec=self.prec)
        operand2 = self.operands[1].__str__(prec=self.prec)
        result = "%s %s %s" % (operand1, ops[self.op], operand2)

        return par(result)

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__,
                               self.operands[0],
                               self.operands[1])


class TensorAdd(BinaryOp):
    """Abstract SLATE class representing tensor addition."""

    prec = 1
    op = "Addition"

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor addition.
        """
        if A.shape != B.shape:
            raise Exception("Cannot perform the operation of addition on %s-tensor with a %s-tensor." % (A.shape, B.shape))

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the appropriate arguments of the resulting
        tensor via tensor addition.
        """
        # Scalars distribute over sums
        if A.rank == 0:
            return B.arguments()
        elif B.rank == 0:
            return A.arguments()
        return A.arguments()


class TensorSub(BinaryOp):
    """Abstract SLATE class representing tensor subtraction."""

    prec = 1
    op = "Subtraction"

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor subtraction."""

        if A.shape != B.shape:
            raise Exception("Cannot perform the operation of substraction on %s-tensor with a %s-tensor." % (A.shape, B.shape))

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the appropriate arguments of the resulting
        tensor via tensor subtraction.
        """
        # Scalars distribute over sums
        if A.rank == 0:
            return B.arguments()
        elif B.rank == 0:
            return A.arguments()
        return A.arguments()


class TensorMul(BinaryOp):
    """Abstract SLATE class representing standard tensor multiplication.

    This includes Matrix-Matrix and Matrix-Vector multiplication.
    """

    prec = 2
    op = "Multiplication"

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor multiplication.
        """
        if A.shape[1] != B.shape[0]:
            raise Exception("Cannot perform the operation of multiplication on %s-tensor with a %s-tensor." % (A.shape, B.shape))

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the arguments of a tensor resulting
        from multiplying two tensors A and B.
        """
        # Scalar case
        if A.rank == 0:
            return B.arguments()
        elif B.rank == 0:
            return A.arguments()
        argsA = A.arguments()
        argsB = B.arguments()
        assert argsA[-1].function_space() == argsB[0].function_space(), ("Cannot perform the contraction over middle arguments. They need to be in the space function space.")

        return argsA[:-1] + argsB[1:]


class CheckRestrictions(MultiFunction):
    """UFL MultiFunction for enforcing cell-wise integrals to contain
    only positive restrictions.
    """
    expr = MultiFunction.reuse_if_untouched

    def negative_restrictions(self, o):
        raise ValueError("Cell-wise integrals must contain only positive restrictions.")
