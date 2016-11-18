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

import operator

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction
from ufl.domain import join_domains, sort_domains


__all__ = ['Tensor', 'Scalar', 'Vector', 'Matrix',
           'Inverse', 'Transpose', 'UnaryOp', 'Negative', 'Positive',
           'BinaryOp', 'TensorAdd', 'TensorSub', 'TensorMul']


class CheckRestrictions(MultiFunction):
    """UFL MultiFunction for enforcing cell-wise integrals to contain
    only positive restrictions."""

    expr = MultiFunction.reuse_if_untouched

    def negative_restrictions(self, o):
        raise ValueError("Cell-wise integrals must contain only positive restrictions.")


class Tensor(object):
    """An abstract representation of a finite element tensor in SLATE.
    All derived tensors, `Scalar`, `Vector`, and `Matrix`, as well as all
    `BinaryOp` and `UnaryOp` objects derive from this abstract base class.
    """
    # Initializing tensor id class variable for output purposes
    id = 0

    def __init__(self):
        """Constructor for the Tensor class."""
        self.id = Tensor.id
        Tensor.id += 1

        # Compute Tensor shape information
        shape = []
        shapes = {}

        for i, arg in enumerate(self.arguments()):
            V = arg.function_space()
            shapelist = []

            for fs in V:
                shapelist.append(fs.fiat_element.space_dimension() * fs.dim)

            shapes[i] = tuple(shapelist)
            shape.append(sum(shapelist))

        self.shapes = shapes
        self.shape = tuple(shape)
        self.rank = len(self.arguments())

        self._integral_domains = None
        self._subdomain_data = None
        self._hash = None

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
        return Positive(self)

    @property
    def inv(self):
        return Inverse(self)

    @property
    def T(self):
        return Transpose(self)

    @property
    def operands(self):
        """Returns the tensor objects which this tensor operates on."""
        return ()

    @classmethod
    def check_integrals(cls, integrals):
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
        the tensor."""
        return self._integral_domains

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found."""
        domains = self.ufl_domains()
        assert all(domain == domains[0] for domain in domains), "All integrals must share the same domain of integration."

        return domains[0]

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`."""
        return self._subdomain_data

    def __hash__(self):
        """Returns a hash code for use in dictionary objects."""
        if self._hash is None:
            self._hash = hash((type(self), )
                              + tuple(hash(a) for a in self.arguments())
                              + tuple(hash(c) for c in self.coefficients()))

        return self._hash


class Scalar(Tensor):
    """A scalar representation of a 0-form. This class wraps the
    relevant information provided from a :class:`ufl.Form` object
    for use in symbolic linear algebra computations.

    Note that this class will complain if you provide an incompatible
    form object as input. The `Scalar` class expects a form of rank 0.

    :arg form: a :class:`ufl.Form` object representing a 0-form.
    """

    def __init__(self, form):
        """Constructor for the Scalar class."""
        r = len(form.arguments())
        if r != 0:
            raise Exception("Cannot create a `slate.Scalar` from a form with rank %d" % r)
        self.form = form
        self.check_integrals(form.integrals())
        super(Scalar, self).__init__()
        # Generate integral data after tensor has been initialized
        self.generate_integral_info(form.integrals())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.form.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.form.coefficients()

    def __str__(self, prec=None):
        """String representation of a SLATE Scalar object."""
        return "S_%d" % self.id

    __repr__ = __str__


class Vector(Tensor):
    """A vector representation of a 1-form. This class wraps the
    relevant information provided from a :class:`ufl.Form` object
    for use in symbolic linear algebra computations.

    Note that this class will complain if you provide an incompatible
    form object as input. The `Vector` class expects a form of rank 1.

    :arg form: a :class:`ufl.Form` object representing a 1-form.
    """

    def __init__(self, form):
        """Constructor for the Vector class."""
        r = len(form.arguments())
        if r != 1:
            raise Exception("Cannot create a `slate.Vector` from a form with rank %d" % r)
        self.form = form
        self.check_integrals(form.integrals())
        super(Vector, self).__init__()
        # Generate integral data after tensor has been initialized
        self.generate_integral_info(form.integrals())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.form.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.form.coefficients()

    def __str__(self, prec=None):
        """String representation of a SLATE Vector object."""
        return "V_%d" % self.id

    __repr__ = __str__


class Matrix(Tensor):
    """A matrix representation of a 2-form. This class wraps the
    relevant information provided from a :class:`ufl.Form` object
    for use in symbolic linear algebra computations.

    Note that this class will complain if you provide an incompatible
    form object as input. The `Matrix` class expects a form of rank 2.

    :arg form: a :class:`ufl.Form` object representing a 2-form.
    """

    def __init__(self, form):
        """Constructor for the Matrix class."""
        r = len(form.arguments())
        if r != 2:
            raise Exception("Cannot create a `slate.Matrix` from a form with rank %d" % r)
        self.form = form
        self.check_integrals(form.integrals())
        super(Matrix, self).__init__()
        # Generate integral data after tensor has been initialized
        self.generate_integral_info(form.integrals())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.form.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.form.coefficients()

    def __str__(self, prec=None):
        """String representation of a rank-2 tensor object in SLATE."""
        return "M_%d" % self.id

    __repr__ = __str__


class Inverse(Tensor):
    """An abstract SLATE class representing the tensor inverse."""

    def __init__(self, A):
        """Constructor for the Inverse class.

        :arg A: a SLATE tensor."""
        if isinstance(A, (Scalar, Vector)):
            raise Exception("Expecting a `slate.Matrix` object, not %r" % A)
        assert A.shape[0] == A.shape[1], "The inverse can only be computed on square tensors."
        self.children = A
        super(Inverse, self).__init__()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.children.arguments()[::-1]

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.children.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor."""
        return self.children.ufl_domains()

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found."""
        return self.children.ufl_domain()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`."""
        return self.children.subdomain_data()

    @property
    def operands(self):
        """Return the operand of the inverse operator."""
        return (self.children, )

    def __str__(self, prec=None):
        """String representation of the inverse of a SLATE tensor."""
        return "(%s).inv" % self.children

    def __repr__(self):
        """SLATE representation of the inverse of a tensor."""
        return "Inverse(%s)" % self.children


class Transpose(Tensor):
    """An abstract SLATE class representing the tranpose of a tensor."""

    def __init__(self, A):
        """Constructor for the Transpose class.

        :arg A: a SLATE tensor."""
        self.children = A
        super(Transpose, self).__init__()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.children.arguments()[::-1]

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.children.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor."""
        return self.children.ufl_domains()

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found."""
        return self.children.ufl_domain()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`."""
        return self.children.subdomain_data()

    @property
    def operands(self):
        """Returns the operand of the transpose operator."""
        return (self.children, )

    def __str__(self, prec=None):
        """String representation of a transposed tensor."""
        return "(%s).T" % self.children

    def __repr__(self):
        """SLATE representation of a transposed tensor."""
        return "Transpose(%s)" % self.children


class UnaryOp(Tensor):
    """An abstract SLATE class for representing unary operations on a
    Tensor object."""

    def __init__(self, A):
        """Constructor for the UnaryOp class.

        :arg tensor: a SLATE tensor."""

        if not isinstance(A, Tensor):
            raise Exception("Expecting a `slate.Tensor` object, not %r" % A)
        self.children = A
        super(UnaryOp, self).__init__()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.children.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.children.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor."""
        return self.children.ufl_domains()

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found."""
        return self.children.ufl_domain()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`."""
        return self.children.subdomain_data()

    @property
    def operands(self):
        """Returns the operand of the unary operation."""
        return (self.children, )

    def __str__(self, prec=None):
        """String representation of a resulting tensor after a unary
        operation is performed."""
        ops = {operator.neg: '-',
               operator.pos: '+'}
        if prec is None or self.prec >= prec:
            pars = lambda x: x
        else:
            pars = lambda x: "(%s)" % x

        return pars("%s%s" % (ops[self.op], self.children.__str__(prec=self.prec)))

    def __repr__(self):
        """SLATE representation of the resulting tensor."""
        return "%s(%r)" % (type(self).__name__, self.children)


class Negative(UnaryOp):
    """Abstract SLATE class representing the negation of a tensor object."""

    prec = 1
    op = operator.neg


class Positive(UnaryOp):
    """Abstract SLATE class representing the positive operation on a tensor."""

    prec = 1
    op = operator.pos


class BinaryOp(Tensor):
    """An abstract SLATE class representing binary operations on tensors.
    Such operations take two operands and returns a tensor-valued expression."""

    def __init__(self, A, B):
        """Constructor for the BinaryOp class."""
        assert isinstance(A, Tensor) and isinstance(B, Tensor), ("Both operands must be SLATE tensors. The operands given are of type (%s, %s)" % (type(A), type(B)))

        self.check_dimensions(A, B)
        self.children = A, B
        super(BinaryOp, self).__init__()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        A, B = self.children
        return self.get_bop_arguments(A, B)

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        A, B = self.children
        return self.get_bop_coefficients(A, B)

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor."""
        A, B = self.children
        # TODO: We need to have a discussion on how we should handle ufl_domains
        assert len(A.ufl_domains()) == 1
        assert len(B.ufl_domains()) == 1
        assert A.ufl_domains() == B.ufl_domains()
        return A.ufl_domains()

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found."""
        domains = self.ufl_domains()
        assert all(domain == domains[0] for domain in domains), "All integrals must share the same domain of integration."

        return domains[0]

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`."""
        # TODO: We need to have a discussion on how we need to handle subdomain_data
        return self.children[0].subdomain_data()

    @classmethod
    def check_dimensions(cls, A, B):
        """Performs a check on the shapes of the tensors before
        performing the binary operation.

        Implemented in subclass."""
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
        the coefficients are handled the same way for all binary operations."""

        clist = []
        A_coeffs = A.coefficients()
        uniqueAcoeffs = set(A_coeffs)
        for c in B.coefficients():
            if c not in uniqueAcoeffs:
                clist.append(c)

        return tuple(list(A_coeffs) + clist)

    @property
    def operands(self):
        """Returns the operands of the binary operation."""
        return (self.children)

    def __str__(self, prec=None):
        """String representation of the binary operation."""

        ops = {operator.add: '+',
               operator.sub: '-',
               operator.mul: '*'}
        if prec is None or self.prec >= prec:
            pars = lambda x: x
        else:
            pars = lambda x: "(%s)" % x
        operand1 = self.children[0].__str__(prec=self.prec)
        operand2 = self.children[1].__str__(prec=self.prec)
        result = "%s %s %s" % (operand1, ops[self.op], operand2)

        return pars(result)

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__,
                               self.children[0],
                               self.children[1])


class TensorAdd(BinaryOp):
    """Abstract SLATE class representing tensor addition."""

    prec = 1
    op = operator.add

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor addition."""

        if A.shape != B.shape:
            raise Exception("Cannot perform the operation of addition on %s-tensor with a %s-tensor." % (A.shape, B.shape))

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the appropriate arguments of the resulting
        tensor via tensor addition."""

        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        return A.arguments()


class TensorSub(BinaryOp):
    """Abstract SLATE class representing tensor subtraction."""

    prec = 1
    op = operator.sub

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor subtraction."""

        if A.shape != B.shape:
            raise Exception("Cannot perform the operation of substraction on %s-tensor with a %s-tensor." % (A.shape, B.shape))

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the appropriate arguments of the resulting
        tensor via tensor subtraction."""

        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        return A.arguments()


class TensorMul(BinaryOp):
    """Abstract SLATE class representing standard tensor multiplication.

    This includes Matrix-Matrix and Matrix-Vector multiplication."""

    prec = 2
    op = operator.mul

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor multiplication."""

        if A.shape[1] != B.shape[0]:
            raise Exception("Cannot perform the operation of multiplication on %s-tensor with a %s-tensor." % (A.shape, B.shape))

    @classmethod
    def get_bop_arguments(cls, A, B):
        """Returns the arguments of a tensor resulting
        from multiplying two tensors A and B."""

        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        argsA = A.arguments()
        argsB = B.arguments()
        assert argsA[-1].function_space() == argsB[0].function_space(), ("Cannot perform the contraction over middle arguments. They need to be in the space function space.")

        return argsA[:-1] + argsB[1:]
