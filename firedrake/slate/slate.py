"""SLATE is a symbolic language defining a framework for performing linear algebra
operations on finite element tensors. It is similar in principle to most linear
algebra libraries in notation.

The design of SLATE was heavily influenced by UFL, and utilizes much of UFL's
functionality for FEM-specific form manipulation.

Unlike UFL, however, once forms are assembled into SLATE `Tensor` objects, one can
utilize the operations defined in SLATE to express complicated linear algebra operations
(such as the Schur-complement reduction of a block-matrix system).

All SLATE expressions are handled by a specialized linear algebra compiler, which interprets
SLATE expressions and produces C++ kernel functions to be executed within the Firedrake
architecture.
"""
from __future__ import absolute_import, print_function, division

from collections import OrderedDict

from firedrake.utils import cached_property

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction
from ufl.coefficient import Coefficient
from ufl.form import Form
from ufl.domain import join_domains


__all__ = ['Tensor', 'Inverse', 'Transpose', 'Negative',
           'Add', 'Sub', 'Mul', 'Action']


class CheckRestrictions(MultiFunction):
    """UFL MultiFunction for enforcing cell-wise integrals to contain
    only positive (outward) restrictions.
    """
    expr = MultiFunction.reuse_if_untouched

    def negative_restrictions(self, o):
        raise ValueError("Cell-wise integrals must contain only positive restrictions.")


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

    @cached_property
    def shapes(self):
        """Computes the internal shape information of its components.
        This is particularly useful to know if the tensor comes from a
        mixed form.
        """
        shapes = {}
        for i, arg in enumerate(self.arguments()):
            shapes[i] = tuple(fs.fiat_element.space_dimension() * fs.dim
                              for fs in arg.function_space())
        return shapes

    @cached_property
    def shape(self):
        """Computes the shape information of the local tensor."""
        return tuple(sum(shapelist) for shapelist in self.shapes.values())

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
        if isinstance(other, TensorBase):
            return Add(self, other)
        else:
            raise NotImplementedError("Operand type(s) for + not implemented: '%s' '%s'"
                                      % (type(self), type(other)))

    def __radd__(self, other):
        # If other is not a TensorBase, raise NotImplementedError. Otherwise,
        # delegate action to other.
        if not isinstance(other, TensorBase):
            raise NotImplementedError("Operand type(s) for + not implemented: '%s' '%s'"
                                      % (type(other), type(self)))
        else:
            other.__add__(self)

    def __sub__(self, other):
        if isinstance(other, TensorBase):
            return Sub(self, other)
        else:
            raise NotImplementedError("Operand type(s) for - not implemented: '%s' '%s'"
                                      % (type(self), type(other)))

    def __rsub__(self, other):
        # If other is not a TensorBase, raise NotImplementedError. Otherwise,
        # delegate action to other.
        if not isinstance(other, TensorBase):
            raise NotImplementedError("Operand type(s) for - not implemented: '%s' '%s'"
                                      % (type(other), type(self)))
        else:
            other.__sub__(self)

    def __mul__(self, other):
        # if other is a ufl.Coefficient, return action
        if isinstance(other, Coefficient):
            return Action(self, other)
        return Mul(self, other)

    def __rmul__(self, other):
        # If other is not a TensorBase, raise NotImplementedError. Otherwise,
        # delegate action to other.
        if not isinstance(other, TensorBase):
            raise NotImplementedError("Operand type(s) for * not implemented: '%s' '%s'"
                                      % (type(other), type(self)))
        else:
            other.__mul__(self)

    def __neg__(self):
        return Negative(self)

    def __eq__(self, other):
        """Determines whether two TensorBase objects are equal using their
        associated keys.
        """
        return self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found.
        """
        domains = self.ufl_domains()
        assert all(domain == domains[0] for domain in domains), (
            "All integrals must share the same domain of integration."
        )
        return domains[0]

    def __str__(self):
        """Returns a string representation."""
        return self._output_string(self.prec)

    @cached_property
    def _hash_id(self):
        """Returns a hash id for use in dictionary objects."""
        return hash(self._key())

    def __hash__(self):
        return self._hash_id


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

    prec = None

    def __init__(self, form):
        """Constructor for the Tensor class."""
        if not isinstance(form, Form):
            raise ValueError("Only UFL forms are acceptable inputs for creating terminal tensors.")

        r = len(form.arguments())
        if r not in (0, 1, 2):
            raise NotImplementedError("Currently don't support tensors of rank %d." % r)

        # Checks for positive restrictions on integrals
        integrals = form.integrals()
        mapper = CheckRestrictions()
        for it in integrals:
            map_integrand_dags(mapper, it)

        super(Tensor, self).__init__()

        self.form = form

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.form.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.form.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        return self.form.ufl_domains()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        return self.form.subdomain_data()

    def _output_string(self, prec=None):
        """Creates a string representation of the tensor."""
        return ["S", "V", "M"][self.rank] + "_%d" % self.id

    def __repr__(self):
        """SLATE representation of the tensor object."""
        return ["Scalar", "Vector", "Matrix"][self.rank] + "(%r)" % self.form

    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.form)


class UnaryOp(TensorBase):
    """An abstract SLATE class for representing unary operations on a
    Tensor object.

    :arg A: a :class:`TensorBase` object. This can be a terminal tensor object
            (:class:`Tensor`) or any derived expression resulting from any number
            of linear algebra operations on `Tensor` objects. For example,
            another instance of a `UnaryOp` object is an acceptable input, or
            a `BinaryOp` object.
    """

    def __init__(self, A):
        """Constructor for the UnaryOp class."""
        super(UnaryOp, self).__init__()
        self.tensor = A

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
        ``{domain:{integral_type: subdomain_data}}``.
        """
        return self.tensor.subdomain_data()

    @property
    def operands(self):
        """Returns an iterable of the operands of this operation."""
        return (self.tensor,)

    def __repr__(self):
        """SLATE representation of the resulting tensor."""
        return "%s(%r)" % (type(self).__name__, self.tensor)

    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.tensor)


class Inverse(UnaryOp):
    """An abstract SLATE class representing the inverse of a tensor.

    .. warning::

       This class will raise an error if the tensor is not square.
    """

    prec = None

    def __init__(self, A):
        """Constructor for the Inverse class."""
        assert A.rank == 2, "The tensor must be rank 2."
        assert A.shape[0] == A.shape[1], (
            "The inverse can only be computed on square tensors."
        )
        super(Inverse, self).__init__(A)

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        return self.tensor.arguments()[::-1]

    def _output_string(self, prec=None):
        """Creates a string representation of the inverse of a square tensor."""
        return "(%s).inv" % self.tensor


class Transpose(UnaryOp):
    """An abstract SLATE class representing the transpose of a tensor."""

    prec = None

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        return self.tensor.arguments()[::-1]

    def _output_string(self, prec=None):
        """Creates a string representation of the transpose of a tensor."""
        return "(%s).T" % self.tensor


class Negative(UnaryOp):
    """Abstract SLATE class representing the negation of a tensor object."""

    prec = 1

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        return self.tensor.arguments()

    def _output_string(self, prec=None):
        """String representation of a resulting tensor after a unary
        operation is performed.
        """
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x

        return par("-%s" % self.tensor._output_string(prec=self.prec))


class BinaryOp(TensorBase):
    """An abstract SLATE class representing binary operations on tensors.
    Such operations take two operands and returns a tensor-valued expression.

    :arg A: a :class:`TensorBase` object. This can be a terminal tensor object
            (:class:`Tensor`) or any derived expression resulting from any number
            of linear algebra operations on `Tensor` objects. For example,
            another instance of a `BinaryOp` object is an acceptable input, or
            a `UnaryOp` object.
    :arg B: a :class:`TensorBase` object.
    """

    def __init__(self, A, B):
        """Constructor for the BinaryOp class."""
        super(BinaryOp, self).__init__()
        self.tensors = A, B

    def coefficients(self):
        """Returns the expected coefficients of the resulting tensor
        of performing a binary operation on two tensors. Note that
        the coefficients are handled the same way for all binary operations.
        """
        A, B = self.tensors
        # Returns an ordered tuple of coefficients (no duplicates)
        return tuple(OrderedDict.fromkeys(A.coefficients() + B.coefficients()))

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        A, B = self.tensors
        return join_domains(A.ufl_domains() + B.ufl_domains())

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        A, B = self.tensors
        # Join subdomain_data
        sd_dict = A.subdomain_data()[A.ufl_domain()]
        for int_type_B in B.subdomain_data()[B.ufl_domain()].keys():
            if int_type_B not in sd_dict.keys():
                sd_dict.update({int_type_B: B.subdomain_data()[B.ufl_domain()][int_type_B]})

        return {self.ufl_domain(): sd_dict}

    @property
    def operands(self):
        """Returns an iterable of the operands of the binary operation."""
        return self.tensors

    def _output_string(self, prec=None):
        """Creates a string representation of the binary operation."""
        ops = {Add: '+',
               Sub: '-',
               Mul: '*'}
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x
        operand1 = self.operands[0]._output_string(prec=self.prec)
        operand2 = self.operands[1]._output_string(prec=self.prec)

        result = "%s %s %s" % (operand1, ops[type(self)], operand2)

        return par(result)

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.operands[0], self.operands[1])

    def _key(self):
        """Returns a key for hash and equality."""
        A, B = self.tensors
        return (type(self), A, B)


class Add(BinaryOp):
    """Abstract SLATE class representing matrix-matrix, vector-vector
     or scalar-scalar addition.

    :arg A: a :class:`TensorBase` object.
    :arg B: another :class:`TensorBase` object.
    """

    prec = 1

    def __init__(self, A, B):
        """Constructor for the Add class."""
        if A.shape != B.shape:
            raise ValueError("Cannot perform the operation on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))
        super(Add, self).__init__(A, B)

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        A, B = self.tensors
        assert [argA.function_space() == argB.function_space()
                for argA in A.arguments()
                for argB in B.arguments()], "Arguments must share the same function space."
        return A.arguments()


class Sub(BinaryOp):
    """Abstract SLATE class representing matrix-matrix, vector-vector
     or scalar-scalar subtraction.

    :arg A: a :class:`TensorBase` object.
    :arg B: another :class:`TensorBase` object.
    """

    prec = 1

    def __init__(self, A, B):
        """Constructor for the Sub class."""
        if A.shape != B.shape:
            raise ValueError("Cannot perform the operation on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))
        super(Sub, self).__init__(A, B)

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        A, B = self.tensors
        assert [argA.function_space() == argB.function_space()
                for argA in A.arguments()
                for argB in B.arguments()], "Arguments must share the same function space."
        return A.arguments()


class Mul(BinaryOp):
    """Abstract SLATE class representing the interior product or two tensors.
    By interior product, we mean an operation that results in a tensor of equal or
    lower rank via performing a contraction on arguments. This includes Matrix-Matrix
    and Matrix-Vector multiplication.

    :arg A: a :class:`TensorBase` object.
    :arg B: another :class:`TensorBase` object.
    """

    prec = 2

    def __init__(self, A, B):
        """Constructor for the Mul class."""
        if A.shape[1] != B.shape[0]:
            raise ValueError("Cannot perform the operation on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))
        super(Mul, self).__init__(A, B)

    def arguments(self):
        """Returns the arguments of a tensor resulting
        from multiplying two tensors A and B.
        """
        A, B = self.tensors
        argsA = A.arguments()
        argsB = B.arguments()
        assert argsA[-1].function_space() == argsB[0].function_space(), (
            "Cannot perform the contraction over middle arguments. "
            "They need to be in the same function space."
        )
        return argsA[:-1] + argsB[1:]


class Action(TensorBase):
    """Abstract SLATE class representing the action of a SLATE tensor on a
    UFL coefficient. This class can be interpreted as representing standard
    matrix-vector multiplication, except the vector is an assembled coefficient
    rather than a SLATE object.

    :arg tensor: a :class:`TensorBase` object.
    :arg coefficient: a :class:`ufl.Coefficient` object.
    """

    prec = 2

    def __init__(self, tensor, coefficient):
        """Constructor for the Action class."""
        assert isinstance(coefficient, Coefficient), (
            "Action can only be performed on a ufl.Coefficient object."
        )
        assert isinstance(tensor, TensorBase), (
            "The tensor must be a SLATE `TensorBase` object."
        )
        V = coefficient.function_space()
        assert tensor.arguments()[-1].function_space() == V, (
            "Argument function space must be the same as the "
            "coefficient function space."
        )
        super(Action, self).__init__()
        self.tensor = tensor
        self._acting_coefficient = coefficient

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self.tensor.arguments()[:-1]

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return tuple(OrderedDict.fromkeys(self.tensor.coefficients()
                                          + (self._acting_coefficient,)))

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        return self.tensor.ufl_domains()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        return self.tensor.subdomain_data()

    def _output_string(self, prec=None):
        """Creates a string representation."""
        return "(%s) * %s" % (self.tensor, self._acting_coefficient)

    def __repr__(self):
        """SLATE representation of the action of a tensor on a coefficient."""
        return "Action(%r, %r)" % (self.tensor, self._acting_coefficient)

    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.tensor, self._acting_coefficient)
