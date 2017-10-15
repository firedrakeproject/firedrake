"""Slate is a symbolic language defining a framework for performing
linear algebra operations on finite element tensors. It is similar
in principle to most linear algebra libraries in notation.

The design of Slate was heavily influenced by UFL, and utilizes
much of UFL's functionality for FEM-specific form manipulation.

Unlike UFL, however, once forms are assembled into Slate `Tensor`
objects, one can utilize the operations defined in Slate to express
complicated linear algebra operations (such as the Schur-complement
reduction of a block-matrix system).

All Slate expressions are handled by a specialized linear algebra
compiler, which interprets expressions and produces C++ kernel
functions to be executed within the Firedrake architecture.
"""
from abc import ABCMeta, abstractproperty, abstractmethod

from collections import OrderedDict

from firedrake.function import Function
from firedrake.utils import cached_property

from itertools import chain

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction
from ufl.domain import join_domains
from ufl.form import Form


__all__ = ['AssembledVector', 'Tensor',
           'Inverse', 'Transpose', 'Negative',
           'Add', 'Mul', 'Action']


class CheckRestrictions(MultiFunction):
    """UFL MultiFunction for enforcing cell-wise integrals to contain
    only positive (outward) restrictions.
    """
    expr = MultiFunction.reuse_if_untouched

    def negative_restrictions(self, o):
        raise ValueError("Must contain only positive restrictions!")


class TensorBase(object, metaclass=ABCMeta):
    """An abstract Slate node class.

    .. warning::

       Do not instantiate this class on its own. This is an abstract
       node class; is not meant to be worked with directly. Only use
       the appropriate subclasses.
    """

    id = 0

    def __init__(self):
        """Constructor for the TensorBase abstract class."""
        # NOTE: This attribute is for caching kernels after
        # an expression has been compiled.
        self._metakernel_cache = None

        self.id = TensorBase.id
        TensorBase.id += 1

    @abstractmethod
    def arg_function_spaces(self):
        """
        """

    @abstractmethod
    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""

    @cached_property
    def shapes(self):
        """Computes the internal shape information of its components.
        This is particularly useful to know if the tensor comes from a
        mixed form.
        """
        shapes = OrderedDict()
        for i, fs in enumerate(self.arg_function_spaces()):
            shapes[i] = tuple(V.finat_element.space_dimension() * V.value_size
                              for V in fs)
        return shapes

    @cached_property
    def shape(self):
        """Computes the shape information of the local tensor."""
        return tuple(sum(shapelist) for shapelist in self.shapes.values())

    @cached_property
    def rank(self):
        """Returns the rank information of the tensor object."""
        return len(self.arguments())

    @abstractmethod
    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""

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

    @abstractmethod
    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """

    @abstractmethod
    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """

    @cached_property
    def is_mixed(self):
        """Returns `True` if the tensor has mixed arguments and `False` otherwise.
        """
        return any(len(fs) > 1 for fs in self.arg_function_spaces())

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
            raise NotImplementedError("Type(s) for + not supported: '%s' '%s'"
                                      % (type(self), type(other)))

    def __radd__(self, other):
        # If other is not a TensorBase, raise NotImplementedError. Otherwise,
        # delegate action to other.
        if not isinstance(other, TensorBase):
            raise NotImplementedError("Type(s) for + not supported: '%s' '%s'"
                                      % (type(other), type(self)))
        else:
            other.__add__(self)

    def __sub__(self, other):
        if isinstance(other, TensorBase):
            return Add(self, Negative(other))
        else:
            raise NotImplementedError("Type(s) for - not supported: '%s' '%s'"
                                      % (type(self), type(other)))

    def __rsub__(self, other):
        # If other is not a TensorBase, raise NotImplementedError. Otherwise,
        # delegate action to other.
        if not isinstance(other, TensorBase):
            raise NotImplementedError("Type(s) for - not supported: '%s' '%s'"
                                      % (type(other), type(self)))
        else:
            other.__sub__(self)

    def __mul__(self, other):
        # if other is a firedrake.Function, return action
        if isinstance(other, Function):
            return Action(self, other)
        return Mul(self, other)

    def __rmul__(self, other):
        # If other is not a TensorBase, raise NotImplementedError. Otherwise,
        # delegate action to other.
        if not isinstance(other, TensorBase):
            raise NotImplementedError("Type(s) for * not supported: '%s' '%s'"
                                      % (type(other), type(self)))
        else:
            other.__mul__(self)

    def __neg__(self):
        return Negative(self)

    def __eq__(self, other):
        """Determines whether two TensorBase objects are equal using their
        associated keys.
        """
        return self._key == other._key

    def __ne__(self, other):
        return not self.__eq__(other)

    @cached_property
    def _hash_id(self):
        """Returns a hash id for use in dictionary objects."""
        return hash(self._key)

    @abstractproperty
    def _key(self):
        """Returns a key for hash and equality.

        This is used to generate a unique id associated with the
        TensorBase object.
        """

    @abstractmethod
    def _output_string(self):
        """Creates a string representation of the tensor.

        This is used when calling the `__str__` method on
        TensorBase objects.
        """

    def __str__(self):
        """Returns a string representation."""
        return self._output_string(self.prec)

    def __hash__(self):
        """Generates a hash for the TensorBase object."""
        return self._hash_id


class AssembledVector(TensorBase):
    """
    """

    operands = ()

    def __init__(self, function):
        """
        """
        if not isinstance(function, Function):
            raise TypeError("Object must be a firedrake function.")

        super(AssembledVector, self).__init__()

        self._function = function

    def arg_function_spaces(self):
        """
        """
        return (self._function.function_space(),)

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return ()

    @cached_property
    def rank(self):
        """Returns the rank information of the tensor object."""
        return 1

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return (self._function,)

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        return self._function.function_space().mesh()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        # FIXME: ???
        return None

    def _output_string(self, prec=None):
        """Creates a string representation of the tensor."""
        return "AV_%d" % self.id

    def __repr__(self):
        """Slate representation of the tensor object."""
        return "AssembledVector(%r)" % self._function

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self._function)


class Tensor(TensorBase):
    """This class is a symbolic representation of a finite element tensor
    derived from a bilinear or linear form. This class implements all
    supported ranks of general tensor (rank-0, rank-1 and rank-2 tensor
    objects). This class is the primary user-facing class that the Slate
    symbolic algebra supports.

    :arg form: a :class:`ufl.Form` object.

    A :class:`ufl.Form` is currently the only supported input of creating
    a `slate.Tensor` object:

        (1) If the form is a bilinear form, namely a form with two
            :class:`ufl.Argument` objects, then the Slate Tensor will be
            a rank-2 Matrix.
        (2) If the form has one `ufl.Argument` as in the case of a typical
            linear form, then this will create a rank-1 Vector.
        (3) A zero-form will create a rank-0 Scalar.

    These are all under the same type `slate.Tensor`. The attribute `self.rank`
    is used to determine what kind of tensor object is being handled.
    """

    operands = ()

    def __init__(self, form):
        """Constructor for the Tensor class."""
        if not isinstance(form, Form):
            if isinstance(form, Function):
                raise TypeError("Use AssembledVector instead of Tensor.")
            raise TypeError("Only UFL forms are acceptable inputs.")

        r = len(form.arguments())
        if r not in (0, 1, 2):
            raise NotImplementedError("No support for tensors of rank %d." % r)

        # Checks for positive restrictions on integrals
        integrals = form.integrals()
        mapper = CheckRestrictions()
        for it in integrals:
            map_integrand_dags(mapper, it)

        super(Tensor, self).__init__()

        self.form = form

    def arg_function_spaces(self):
        """
        """
        return tuple(arg.function_space() for arg in self.arguments())

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
        """Slate representation of the tensor object."""
        return ["Scalar", "Vector", "Matrix"][self.rank] + "(%r)" % self.form

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.form)


class TensorOp(TensorBase):
    """An abstract Slate class representing general operations on
    existing Slate tensors.

    :arg operands: an iterable of operands that are :class:`TensorBase`
                   objects.
    """

    def __init__(self, *operands):
        """Constructor for the TensorOp class."""
        super(TensorOp, self).__init__()
        self.operands = tuple(operands)

    def coefficients(self):
        """Returns the expected coefficients of the resulting tensor."""
        coeffs = [op.coefficients() for op in self.operands]
        return tuple(OrderedDict.fromkeys(chain(*coeffs)))

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        collected_domains = [op.ufl_domains() for op in self.operands]
        return join_domains(chain(*collected_domains))

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        sd = {}
        for op in self.operands:
            op_sd = op.subdomain_data()[op.ufl_domain()]

            for it_type, domain in op_sd.items():
                if it_type not in sd:
                    sd[it_type] = domain

                else:
                    assert sd[it_type] == domain, (
                        "Domains must agree!"
                    )

        return {self.ufl_domain(): sd}

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.operands)


class UnaryOp(TensorOp):
    """An abstract Slate class for representing unary operations on a
    Tensor object.

    :arg A: a :class:`TensorBase` object. This can be a terminal tensor object
            (:class:`Tensor`) or any derived expression resulting from any
            number of linear algebra operations on `Tensor` objects. For
            example, another instance of a `UnaryOp` object is an acceptable
            input, or a `BinaryOp` object.
    """

    def __repr__(self):
        """Slate representation of the resulting tensor."""
        tensor, = self.operands
        return "%s(%r)" % (type(self).__name__, tensor)


class Inverse(UnaryOp):
    """An abstract Slate class representing the inverse of a tensor.

    .. warning::

       This class will raise an error if the tensor is not square.
    """

    def __init__(self, A):
        """Constructor for the Inverse class."""
        assert A.rank == 2, "The tensor must be rank 2."
        assert A.shape[0] == A.shape[1], (
            "The inverse can only be computed on square tensors."
        )
        super(Inverse, self).__init__(A)

    def arg_function_spaces(self):
        """
        """
        tensor, = self.operands
        return tensor.arg_function_spaces()[::-1]

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        tensor, = self.operands
        return tensor.arguments()[::-1]

    def _output_string(self, prec=None):
        """Creates a string representation of the inverse of a tensor."""
        tensor, = self.operands
        return "(%s).inv" % tensor


class Transpose(UnaryOp):
    """An abstract Slate class representing the transpose of a tensor."""

    def arg_function_spaces(self):
        """
        """
        tensor, = self.operands
        return tensor.arg_function_spaces()[::-1]

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        tensor, = self.operands
        return tensor.arguments()[::-1]

    def _output_string(self, prec=None):
        """Creates a string representation of the transpose of a tensor."""
        tensor, = self.operands
        return "(%s).T" % tensor


class Negative(UnaryOp):
    """Abstract Slate class representing the negation of a tensor object."""

    def arg_function_spaces(self):
        """
        """
        tensor, = self.operands
        return tensor.arg_function_spaces()

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        tensor, = self.operands
        return tensor.arguments()

    def _output_string(self, prec=None):
        """String representation of a resulting tensor after a unary
        operation is performed.
        """
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x

        tensor, = self.operands
        return par("-%s" % tensor._output_string(prec=self.prec))


class BinaryOp(TensorOp):
    """An abstract Slate class representing binary operations on tensors.
    Such operations take two operands and returns a tensor-valued expression.

    :arg A: a :class:`TensorBase` object. This can be a terminal tensor object
            (:class:`Tensor`) or any derived expression resulting from any
            number of linear algebra operations on `Tensor` objects. For
            example, another instance of a `BinaryOp` object is an acceptable
            input, or a `UnaryOp` object.
    :arg B: a :class:`TensorBase` object.
    """

    def _output_string(self, prec=None):
        """Creates a string representation of the binary operation."""
        ops = {Add: '+',
               Mul: '*'}
        if prec is None or self.prec >= prec:
            par = lambda x: x
        else:
            par = lambda x: "(%s)" % x
        A, B = self.operands
        operand1 = A._output_string(prec=self.prec)
        operand2 = B._output_string(prec=self.prec)

        result = "%s %s %s" % (operand1, ops[type(self)], operand2)

        return par(result)

    def __repr__(self):
        A, B = self.operands
        return "%s(%r, %r)" % (type(self).__name__, A, B)


class Add(BinaryOp):
    """Abstract Slate class representing matrix-matrix, vector-vector
     or scalar-scalar addition.

    :arg A: a :class:`TensorBase` object.
    :arg B: another :class:`TensorBase` object.
    """

    def __init__(self, A, B):
        """Constructor for the Add class."""
        if A.shape != B.shape:
            raise ValueError("Illegal op on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))

        assert A.arg_function_spaces() == B.arg_function_spaces(), (
            "Function spaces associated with operands must match."
        )

        super(Add, self).__init__(A, B)

        # Function space check above ensures that the arguments of the
        # operands are identical (in the sense that they are arguments
        # defined on the same function space).
        self._args = A.arguments()

    def arg_function_spaces(self):
        """
        """
        A, _ = self.operands
        return A.arg_function_spaces()

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self._args


class Mul(BinaryOp):
    """Abstract Slate class representing the interior product or two tensors.
    By interior product, we mean an operation that results in a tensor of
    equal or lower rank via performing a contraction on arguments. This
    includes Matrix-Matrix and Matrix-Vector multiplication.

    :arg A: a :class:`TensorBase` object.
    :arg B: another :class:`TensorBase` object.
    """

    def __init__(self, A, B):
        """Constructor for the Mul class."""
        if A.shape[1] != B.shape[0]:
            raise ValueError("Illegal op on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))

        assert A.arg_function_spaces()[-1] == B.arg_function_spaces()[0], (
            "Cannot perform argument contraction over middle indices. "
            "They must be in the same function space."
        )

        super(Mul, self).__init__(A, B)

        # Function space check above ensures that middle arguments can
        # be 'eliminated'.
        self._args = A.arguments()[:-1] + B.arguments()[1:]

    def arg_function_spaces(self):
        """
        """
        A, B = self.operands
        return A.arg_function_spaces()[:-1] + B.arg_function_spaces()[1:]

    def arguments(self):
        """Returns the arguments of a tensor resulting
        from multiplying two tensors A and B.
        """
        return self._args


class Action(TensorOp):
    """Abstract Slate class representing the action of a Slate tensor on a
    UFL coefficient. This class can be interpreted as representing standard
    matrix-vector multiplication, except the vector is an assembled coefficient
    rather than a Slate object.

    :arg tensor: a :class:`TensorBase` object.
    :arg function: a :class:`firedrake.Function` object.
    """

    def __init__(self, tensor, function):
        """Constructor for the Action class."""
        assert isinstance(function, Function), (
            "Action can only be performed on a firedrake.Function object."
        )
        assert isinstance(tensor, TensorBase), (
            "The tensor must be a Slate `TensorBase` object."
        )
        V = function.function_space()
        assert tensor.arguments()[-1].function_space() == V, (
            "Argument function space must be the same as the "
            "coefficient function space."
        )
        super(Action, self).__init__(tensor)
        self.actee = function,

    def arg_function_spaces(self):
        """
        """
        # TODO: This class will be deleted after Mul is fully closed.
        return (self.arguments()[-1].function_space(),)

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        tensor, = self.operands
        return tensor.arguments()[:-1]

    def coefficients(self):
        """Returns the expected coefficients of the resulting tensor."""
        coeffs = [op.coefficients() for op in self.operands] + [self.actee]
        return tuple(OrderedDict.fromkeys(chain(*coeffs)))

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        collected_domains = [obj.ufl_domains() for obj in self.operands
                             + self.actee]
        return join_domains(chain(*collected_domains))

    def _output_string(self, prec=None):
        """Creates a string representation."""
        tensor, = self.operands
        function, = self.actee
        return "(%s) * %s" % (tensor, function)

    def __repr__(self):
        """Slate representation of the action of a tensor on a coefficient."""
        tensor, = self.operands
        function, = self.actee
        return "Action(%r, %r)" % (tensor, function)

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.operands, self.actee)


# Establishes levels of precedence for Slate tensors
precedences = [
    [Tensor, AssembledVector],
    [UnaryOp],
    [Add],
    [Mul, Action]
]

# Here we establish the precedence class attribute for a given
# Slate TensorOp class.
for level, group in enumerate(precedences):
    for tensor in group:
        tensor.prec = level
