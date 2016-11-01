"""SLATE is a symbolic language defining a framework for
performing linear algebra operations on finite element
tensors. It is similar in principle to most linear algebra
libraries in notation.

The design of SLATE was heavily influenced by UFL, and
utilizes much of UFLs functionality for FEM-specific form
manipulation.

Unlike UFL, however, once forms are assembled into SLATE
`Tensor` objects, one can utilize the operations defined
in SLATE to express complicated linear algebra operations.
(Such as the Schur-complement reduction of a block-matrix
system)

All SLATE expressions are handled by a specialized linear
algebra compiler, the SLATE Linear Algebra Compiler (SLAC),
which interprets SLATE expressions and produces C++ kernel
functions to be executed within the Firedrake architecture.

Written by: Thomas Gibson (t.gibson15@imperial.ac.uk)
"""

import hashlib
import operator

from ufl import Coefficient
from slate_assertions import *
from slate_equation import SlateEquation

from ufl.algorithms.domain_analysis import canonicalize_metadata
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction
from ufl.algorithms.signature import compute_expression_hashdata, compute_terminal_hashdata
from ufl.domain import join_domains, sort_domains
from ufl.form import _sorted_integrals


__all__ = ['CheckRestrictions', 'RemoveRestrictions',
           'SlateIntegral', 'Tensor',
           'Scalar', 'Vector', 'Matrix',
           'Inverse', 'Transpose', 'UnaryOp', 'Negative', 'Positive',
           'BinaryOp', 'TensorAdd', 'TensorSub', 'TensorMul']


class CheckRestrictions(MultiFunction):
    """UFL MultiFunction for enforcing cell-wise integrals to contain
    only positive restrictions."""

    expr = MultiFunction.reuse_if_untouched

    def negative_restrictions(self, o):
        raise ValueError("Cell-wise integrals must contain only positive restrictions.")


class RemoveRestrictions(MultiFunction):
    """UFL MultiFunction for removing any restrictions on the integrals of forms."""

    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class SlateIntegral(object):
    """An integral wrapping class that behaves like a UFL integral,
    but doesn't have integrands in the same way. Its purpose is to
    provide the appropriate integration domains and subdomain data."""

    def __init__(self, ufl_integral):
        """Constructor for the SlateIntegral class.

        :arg ufl_integral: a ufl integral object."""

        self._itype = ufl_integral.integral_type()
        self._subdomain_data = ufl_integral.subdomain_data()
        super(SlateIntegral, self).__init__()

    def integral_type(self):
        """Returns the integral type of the "integral"."""
        return self._itype

    def subdomain_data(self):
        """Returns the subdomain data of the "integral"."""
        return self._subdomain_data


class Tensor(object):
    """An abstract representation of a finite element
    tensor in SLATE."""

    # Initializing tensor id class variable for output purposes
    tensor_id = 0

    def __init__(self, arguments, coefficients, integrals):
        """Constructor for the Tensor class.

        :arg arugments: list of arguments of the associated UFL form
        :arg coefficients: list of coefficients of the associated UFL form
        :arg integrals: list of integrals of the associated UFL form
        """

        self.tensor_id = Tensor.tensor_id
        self._arguments = arguments
        self._coefficients = coefficients
        self._integrals = _sorted_integrals(integrals)

        slate_integrals = []
        for integral in self._integrals:
            slate_integrals.append(SlateIntegral(integral))
        self._slate_integrals = tuple(slate_integrals)

        # Compute Tensor shape information
        shape = []
        shapes = {}

        for i, arg in enumerate(self._arguments):
            V = arg.function_space()
            shapelist = []

            for fs in V:
                shapelist.append(fs.fiat_element.space_dimension() * fs.dim)

            shapes[i] = tuple(shapelist)
            shape.append(sum(shapelist))

        self.shapes = shapes
        self.shape = tuple(shape)
        self.rank = len(self._arguments)

        # Compute integration domains
        integral_domains = join_domains([it.ufl_domain() for it in self._integrals])
        self._integral_domains = sort_domains(integral_domains)

        # Compute relevant attributes for signature computation
        self._coefficient_numbering = dict((c, i) for i, c in enumerate(self._coefficients))
        self._domain_numering = dict((d, i) for i, d in enumerate(self._integral_domains))

        self._hash = None
        self._signature = None

    # Overloaded operations for the tensor algebra
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

    def equals(self, other):
        """Evaluates the boolean expression `A == B`."""

        if type(other) != Tensor:
            return False
        if len(self._arguments) != len(other._arguments):
            return False
        if len(self._coefficients) != len(other._coefficients):
            return False
        if len(self._integrals) != len(other._integrals):
            return False
        if hash(self) != hash(other):
            return False
        # TODO: Is this sufficient/correct?
        return all(a == b for a, b in zip(self._arguments, other._arguments) and
                   c == d for c, d in zip(self._coefficients, other._coefficients) and
                   e == f for e, f in zip(self._integrals, other._integrals))

    def __ne__(self, other):
        return not self.equals(other)

    def __eq__(self, other):
        """Evaluation of the "==" operator using the SLATE class: SlateEquation."""
        return SlateEquation(self, other)

    # Essential properties
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

    # Analyze integrals of tensor
    @classmethod
    def check_integrals(cls, integrals):
        """Checks for positive restrictions on integrals."""

        mapper = CheckRestrictions()
        for it in integrals:
            map_integrand_dags(mapper, it)

    # Accessor methods
    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self._arguments

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self._coefficients

    def get_ufl_integrals(self):
        """Returns the associated ufl integrals with the tensor
        (This is necessary for form compilation).

        This function should really only be called on base-Tensor objects
        like an individual Vector or Matrix rather than a full SLATE expr."""
        return self._integrals

    def integrals(self):
        """Returns a tuple of SlateIntegrals associated with the tensor."""
        return self._slate_integrals

    def integrals_by_type(self, integral_type):
        """Returns a tuple of integrals corresponding with a particular domain type."""

        slate_assert(integral_type in ['cell', 'interior_facet', 'exterior_facet'],
                     "Integral type %s is not supported." % integral_type)
        return tuple(it for it in self.integrals() if it.integral_type() == integral_type)

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor."""
        return self._integral_domains

    def ufl_domain(self):
        """This function is written under the assumption that it will be called
        on a fully constructed SLATE expression (or an individual tensor object
        such as a matrix with one form associated with it), where the integrals
        have been contracted to integrals on a single domain. This function returns
        a single domain of integration occuring in the tensor.

        The function will fail if multiple domains are found."""

        domains = self.ufl_domains()
        slate_assert(all(domain == domains[0] for domain in domains),
                     "All integrals must share the same domain of integration.")
        return domains[0]

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        `{domain:{integral_type: subdomain_data}}`."""

        # Scalar case
        if self.rank == 0:
            return None
        integration_domains = self.ufl_domains()
        integrals = self._integrals

        # Initializing subdomain_data
        subdomain_data = {}
        for domain in integration_domains:
            subdomain_data[domain] = {}

        for integral in integrals:
            domain = integral.ufl_domain()
            it_type = integral.integral_type()
            subdata = integral.subdomain_data()

            data = subdomain_data[domain].get(it_type)
            if data is None:
                subdomain_data[domain][it_type] = subdata
            elif subdata is not None:
                slate_assert(data.ufl_id() == subdata.ufl_id(),
                             "Integrals in the tensor must have the same subdomain_data objects.")

        return subdomain_data

    # Signature and hash methods
    def _compute_renumbering(self):
        """Renumbers coefficients by including integration domains."""

        dn = self._domain_numering
        cn = self._coefficient_numbering
        renumbering = {}
        renumbering.update(dn)
        renumbering.update(cn)

        n = len(dn)
        for c in cn:
            d = c.ufl_domain()
            if d is not None and d not in renumbering:
                renumbering[d] = n
                n += 1

        return renumbering

    def _compute_signature(self, renumbering):
        """Computes the signature of the tensor from the integrals."""
        # TODO: Is there a better/cheaper way of doing this?!

        integrals = self._integrals
        integrands = [integral.integrand() for integral in integrals]

        terminal_hashdata = compute_terminal_hashdata(integrands, renumbering)

        hashdata = []
        for integral in integrals:
            integrand_hashdata = compute_expression_hashdata(integral.integrand(),
                                                             terminal_hashdata)
            domain_hashdata = integral.ufl_domain()._ufl_signature_data_(renumbering)

            integral_hashdata = (integrand_hashdata, domain_hashdata,
                                 integral.integral_type(),
                                 integral.subdomain_id(),
                                 canonicalize_metadata(integral.metadata()), )
            hashdata.append(integral_hashdata)

        return hashlib.sha512(str(hashdata).encode('utf-8')).hexdigest()

    def signature(self):
        """Signature for use in caching. Reproduced from ufl."""

        if self._signature is None:
            self._signature = self._compute_signature(self._compute_renumbering())

        return self._signature

    def __hash__(self):
        """Returns a hash code for use in dictionary objects."""
        if self._hash is None:
            self._hash = hash((type(self), )
                              + tuple(hash(it) for it in self._integrals))

        return self._hash


class Scalar(Tensor):
    """An abstract representation of a rank-0 tensor object in SLATE."""

    def __init__(self, coefficient):
        """Constructor for the Scalar class."""
        slate_assert(isinstance(coefficient, Coefficient), "Scalars need a coefficient as an argument.")

        self.coefficient = coefficient
        Tensor.tensor_id += 1
        super(Scalar, self).__init__(arguments=(),
                                     coefficients=(coefficient, ),
                                     integrals=())

    def __str__(self, prec=None):
        """String representation of a SLATE Scalar object."""
        return "S_%d" % self.tensor_id

    __repr__ = __str__


class Vector(Tensor):
    """An abstract representation of a rank-1 tensor object in SLATE."""

    def __init__(self, form):
        """Constructor for the Vector class.

        :arg form: a ufl form."""

        r = len(form.arguments())
        if r != 1:
            rank_error(1, r)
        self.form = form
        self.check_integrals(form.integrals())
        Tensor.tensor_id += 1
        super(Vector, self).__init__(arguments=form.arguments(),
                                     coefficients=form.coefficients(),
                                     integrals=form.integrals())

    def __str__(self, prec=None):
        """String representation of a SLATE Vector object."""
        return "V_%d" % self.tensor_id

    __repr__ = __str__


class Matrix(Tensor):
    """An abstract representation of a rank-2 tensor object in SLATE."""

    def __init__(self, form):
        """Constructor for the Matrix class.

        :arg form: a ufl form."""

        r = len(form.arguments())
        if r != 2:
            rank_error(2, r)
        self.form = form
        self.check_integrals(form.integrals())
        Tensor.tensor_id += 1
        super(Matrix, self).__init__(arguments=form.arguments(),
                                     coefficients=form.coefficients(),
                                     integrals=form.integrals())

    def __str__(self, prec=None):
        """String representation of a rank-2 tensor object in SLATE."""
        return "M_%d" % self.tensor_id

    __repr__ = __str__


class Inverse(Tensor):
    """An abstract SLATE class representing the tensor inverse."""

    def __init__(self, A):
        """Constructor for the Inverse class.

        :arg A: a SLATE tensor."""

        if isinstance(A, (Scalar, Vector)):
            expecting_slate_object(Matrix, A)
        slate_assert(A.shape[0] == A.shape[1],
                     "The inverse can only be computed on square tensors.")

        self.children = A
        Tensor.tensor_id += 1
        super(Inverse, self).__init__(arguments=A.arguments()[::-1],
                                      coefficients=A.coefficients(),
                                      integrals=A.get_ufl_integrals())

    @property
    def operands(self):
        """Return the operand of the inverse operator."""
        return (self.children, )

    def __str__(self, prec=None):
        """String representation of the inverse of a SLATE tensor."""
        return "%s.inv" % self.children

    def __repr__(self):
        """SLATE representation of the inverse of a tensor."""
        return "Inverse(%s)" % self.children


class Transpose(Tensor):
    """An abstract SLATE class representing the tranpose of a tensor."""

    def __init__(self, A):
        """Constructor for the Transpose class.

        :arg A: a SLATE tensor."""

        slate_assert(not isinstance(A, Scalar),
                     "Cannot take the transpose of a scalar.")
        self.children = A
        Tensor.tensor_id += 1
        super(Transpose, self).__init__(arguments=A.arguments()[::-1],
                                        coefficients=A.coefficients(),
                                        integrals=A.get_ufl_integrals())

    @property
    def operands(self):
        """Returns the operand of the transpose operator."""
        return (self.children, )

    def __str__(self, prec=None):
        """String representation of a transposed tensor."""
        return "%s.T" % self.children

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
            expecting_slate_object(Tensor, type(A))
        self.children = A
        Tensor.tensor_id += 1
        super(UnaryOp, self).__init__(arguments=A.arguments(),
                                      coefficients=A.coefficients(),
                                      integrals=self.get_uop_integrals(A))

    @classmethod
    def get_uop_integrals(cls, A):
        """Returns the integrals of the resulting tensor after the
        unary operation (Positive or Negative)
        is performed.

        Method is implemented in subclass."""
        pass

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

    @classmethod
    def get_uop_integrals(cls, A):
        """Returns the integrals of Negative(A)."""

        integrals = []
        for it in A.get_ufl_integrals():
            integrals.append(-it)
        return tuple(integrals)


class Positive(UnaryOp):
    """Abstract SLATE class representing the positive operation on a tensor."""

    prec = 1
    op = operator.pos

    @classmethod
    def get_uop_integrals(cls, A):
        """Returns the integrals of Positive(A)."""
        return A.get_ufl_integrals()


class BinaryOp(Tensor):
    """An abstract SLATE class representing binary operations on tensors.
    Such operations take two operands and returns a tensor-valued expression."""

    def __init__(self, A, B):
        """Constructor for the BinaryOp class.

        :arg A: a SLATE tensor.
        :arg B: a SLATE tenosr."""

        slate_assert((isinstance(A, Tensor) and isinstance(B, Tensor)),
                     "Both operands must be SLATE tensors. The operands given are of type (%s, %s)" % (type(A), type(B)))

        self.check_dimensions(A, B)
        self.children = A, B
        Tensor.tensor_id += 1
        super(BinaryOp, self).__init__(arguments=self.get_bop_arguments(A, B),
                                       coefficients=self.get_bop_coefficients(A, B),
                                       integrals=self.get_bop_integrals(A, B))

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

    @classmethod
    def get_bop_integrals(cls, A, B):
        """Returns the expected integrals of the resulting tensor
        of performing a specific binary operation on two tensors.

        Implemented in subclass."""
        pass

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
            dimension_error(A.shape, B.shape)

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

    @classmethod
    def get_bop_integrals(cls, A, B):
        """Returns the appropriate integrals of the resulting
        tensor via tensor addition."""
        return A.get_ufl_integrals() + B.get_ufl_integrals()


class TensorSub(BinaryOp):
    """Abstract SLATE class representing tensor subtraction."""

    prec = 1
    op = operator.sub

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor subtraction."""

        if A.shape != B.shape:
            dimension_error(A.shape, B.shape)

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

    @classmethod
    def get_bop_integrals(cls, A, B):
        """Returns the appropriate integrals of the resulting
        tensor via tensor subtraction."""
        # TODO: Is it wasteful to declare an Negative tensor?
        return A.get_ufl_integrals() + Negative(B).get_ufl_integrals()


class TensorMul(BinaryOp):
    """Abstract SLATE class representing standard tensor multiplication."""

    prec = 2
    op = operator.mul

    @classmethod
    def check_dimensions(cls, A, B):
        """Checks the shapes of the tensors A and B before
        attempting to perform tensor multiplication."""

        if A.shape[1] != B.shape[0]:
            dimension_error(A.shape, B.shape)

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
        slate_assert(argsA[-1].function_space() == argsB[0].function_space(),
                     "Cannot perform the contraction over middle arguments. They need to be in the space function space.")
        return argsA[:-1] + argsB[1:]

    @classmethod
    def get_bop_integrals(cls, A, B):
        """Returns the integrals of a tensor resulting
        from multiplying two tensors A and B."""

        # If no middled integrals to contract, just concatenate
        if len(A.get_ufl_integrals()) == 1 and len(B.get_ufl_integrals()) == 1:
            return A.get_ufl_integrals() + B.get_ufl_integrals()
        integrandA = A.get_ufl_integrals()[-1].integrand()
        integrandB = B.get_ufl_integrals()[0].integrand()
        slate_assert(integrandA.ufl_domain().coordinates.function_space() ==
                     integrandB.ufl_domain().coordinates.function_space(),
                     "Cannot perform contraction over middle integrals. The integrands must be in the space function space.")
        return A.get_ufl_integrals()[:-1] + B.get_ufl_integrals()[1:]
