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

from collections import OrderedDict, namedtuple, defaultdict

from ufl import Constant
from ufl.coefficient import BaseCoefficient

from firedrake.formmanipulation import ExtractSubBlock, subspace
from firedrake.function import Function, Cofunction
from firedrake.ufl_expr import TestFunction
from firedrake.utils import cached_property, unique

from itertools import chain, count

from pyop2.utils import as_tuple

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import Zero
from ufl.domain import join_domains, sort_domains
from ufl.form import BaseForm, Form, ZeroBaseForm
import hashlib

from tsfc.ufl_utils import extract_firedrake_constants


__all__ = ['AssembledVector', 'Block', 'Factorization', 'Tensor',
           'Inverse', 'Transpose', 'Negative',
           'Add', 'Mul', 'Solve', 'BlockAssembledVector', 'DiagonalTensor',
           'Reciprocal']

# BlockFunction description type
BlockFunction = namedtuple('BlockFunction', ['split_function', 'indices', 'orig_function'])
BlockFunction.__doc__ = """\
Context that carries information for a block on an assembled vector.

:param split_function: The splits of the orig_function corresponding to the block.
:param indices: The indices of the block.
:param orig_function: The  (unsplit) function corresponding the assembled vector.
"""


class RemoveNegativeRestrictions(MultiFunction):
    """UFL MultiFunction which removes any negative restrictions
    in a form.
    """
    expr = MultiFunction.reuse_if_untouched

    def negative_restricted(self, o):
        return Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)


class BlockIndexer(object):
    """Container class which only exists to enable smart indexing of :class:`Tensor`

    .. warning::

       This class is not intended for user instatiation.
    """

    __slots__ = ['tensor', 'block_cache']

    def __init__(self, tensor):
        self.tensor = tensor
        self.block_cache = {}

    def __getitem__(self, key):
        key = as_tuple(key)
        # Make indexing with too few indices legal.
        key = key + tuple(slice(None) for i in range(self.tensor.rank - len(key)))
        if len(key) > self.tensor.rank:
            raise ValueError("Attempting to index a rank-%s tensor with %s indices."
                             % (self.tensor.rank, len(key)))

        block_shape = tuple(len(V) for V in self.tensor.arg_function_spaces)
        # Convert slice indices to tuple of indices.
        blocks = tuple(tuple(range(k.start or 0, k.stop or n, k.step or 1))
                       if isinstance(k, slice)
                       else (k,)
                       for k, n in zip(key, block_shape))

        if blocks == tuple(tuple(range(n)) for n in block_shape):
            return self.tensor
        # Avoid repeated instantiation of an equivalent block
        try:
            block = self.block_cache[blocks]
        except KeyError:
            block = Block(tensor=self.tensor, indices=blocks)
            self.block_cache[blocks] = block
        return block


class MockCellIntegral(object):
    def integral_type(self):
        return "cell"

    def __iter__(self):
        yield self

    def __call__(self):
        return self


class TensorBase(object, metaclass=ABCMeta):
    """An abstract Slate node class.

    .. warning::

       Do not instantiate this class on its own. This is an abstract
       node class; is not meant to be worked with directly. Only use
       the appropriate subclasses.
    """

    integrals = MockCellIntegral()
    """A mock object that provides enough compatibility with ufl.Form
    that one can assemble a tensor."""

    terminal = False
    assembled = False
    diagonal = False

    _id = count()

    def __init__(self, *_):
        """Initialise a cache for stashing results.

        Mirrors :class:`~ufl.form.Form`.
        """
        self._cache = {}

    @cached_property
    def id(self):
        return next(TensorBase._id)

    @cached_property
    def _metakernel_cache(self):
        return {}

    @property
    def children(self):
        return self.operands

    @cached_property
    def expression_hash(self):
        from firedrake.slate.slac.utils import traverse_dags
        hashdata = []
        for op in traverse_dags([self]):
            if isinstance(op, AssembledVector):
                data = (type(op).__name__, op.arg_function_spaces[0].ufl_element()._ufl_signature_data_(), )
            elif isinstance(op, Block):
                data = (type(op).__name__, op._indices, )
            elif isinstance(op, BlockAssembledVector):
                data = (type(op).__name__, op._indices, op._original_function, op._function)
            elif isinstance(op, Factorization):
                data = (type(op).__name__, op.decomposition, )
            elif isinstance(op, Tensor):
                data = (op.form.signature(), op.diagonal, )
            elif isinstance(op, (UnaryOp, BinaryOp)):
                data = (type(op).__name__, )
            else:
                raise ValueError("Unhandled type %r" % type(op))
            hashdata.append(data + (op.prec, ))
        hashdata = "".join("%s" % (s, ) for s in hashdata)
        return hashlib.sha512(hashdata.encode("utf-8")).hexdigest()

    @abstractproperty
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on. For example, if A is a rank-2 tensor
        defined on V x W, then this method returns (V, W).
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
        for i, fs in enumerate(self.arg_function_spaces):
            shapes[i] = tuple(int(V.finat_element.space_dimension() * V.block_size)
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

    @abstractmethod
    def constants(self):
        """Returns a tuple of constants associated with the tensor."""

    @abstractmethod
    def slate_coefficients(self):
        """Returns a tuple of Slate coefficients associated with the tensor."""

    @property
    def coeff_map(self):
        """A map from local coefficient numbers
        to the split global coefficient numbers.
        The split coefficients are defined on the pieces of the originally mixed function spaces.
        """
        coeff_map = defaultdict(set)
        for c in self.slate_coefficients():
            if isinstance(c, BlockFunction):  # for block assembled vectors
                m = self.coefficients().index(c.orig_function)
                coeff_map[m].update(c.indices[0])
            else:
                m = self.coefficients().index(c)
                split_map = tuple(range(len(c.subfunctions))) if isinstance(c, (Function, Constant, Cofunction)) else (0,)
                coeff_map[m].update(split_map)
        return tuple((k, tuple(sorted(v)))for k, v in coeff_map.items())

    def ufl_domain(self):
        """This function returns a single domain of integration occuring
        in the tensor.

        The function will fail if multiple domains are found.
        """
        try:
            domain, = self.ufl_domains()
        except ValueError:
            raise ValueError("All integrals must share the same domain of integration.")
        return domain

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
        return any(len(fs) > 1 for fs in self.arg_function_spaces)

    @property
    def inv(self):
        return Inverse(self)

    @property
    def T(self):
        return Transpose(self)

    def solve(self, B, decomposition=None):
        """Solve a system of equations with
        a specified right-hand side.

        :arg B: a Slate expression. This can be either a
            vector or a matrix.
        :arg decomposition: A string describing the type of
            factorization to use when inverting the local
            systems. A complete list of available matrix
            decompositions are outlined in
            :class:`Factorization`.
        """
        return Solve(self, B, decomposition=decomposition)

    def empty(self):
        """Returns whether the form associated with the tensor is empty."""
        return False

    @cached_property
    def blocks(self):
        """Returns an object containing the blocks of the tensor defined
        on a mixed space. Indices can then be provided to extract a
        particular sub-block.

        For example, consider the rank-2 tensor described by:

        .. code-block:: python3

           V = FunctionSpace(m, "CG", 1)
           W = V * V * V
           u, p, r = TrialFunctions(W)
           w, q, s = TestFunctions(W)
           A = Tensor(u*w*dx + p*q*dx + r*s*dx)

        The tensor `A` has 3x3 block structure. The block defined
        by the form `u*w*dx` could be extracted with:

        .. code-block:: python3

           A.blocks[0, 0]

        While the block coupling `p`, `r`, `q`, and `s` could be
        extracted with:

        .. code-block:: python3

           A.block[1:, 1:]

        The usual Python slicing operations apply.
        """
        return BlockIndexer(self)

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
        if isinstance(other, TensorBase):
            return Mul(self, other)
        else:
            raise NotImplementedError("Type(s) for * not supported: '%s' '%s'"
                                      % (type(self), type(other)))

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
        if isinstance(other, (int, float)) and other == 0:
            if isinstance(self, Tensor):
                return isinstance(self.form, ZeroBaseForm) or self.form.empty()
            return False
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
    """This class is a symbolic representation of an assembled
    vector of data contained in a :class:`~.Function`.

    :arg function: A firedrake function.
    """

    @property
    def integrals(self):
        raise ValueError("AssembledVector has no integrals")

    operands = ()
    terminal = True
    assembled = True

    def __new__(cls, function):
        if isinstance(function, AssembledVector):
            return function
        elif isinstance(function, BaseCoefficient):
            self = super().__new__(cls)
            self._function = function
            return self
        else:
            raise TypeError("Expecting a BaseCoefficient or AssembledVector (not a %r)" %
                            type(function))

    @cached_property
    def form(self):
        return self._function

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor = self._function
        if isinstance(tensor, BaseForm):
            return tuple(a.function_space() for a in tensor.arguments())
        else:
            return (tensor.ufl_function_space(),)

    @cached_property
    def _argument(self):
        """Generates a 'test function' associated with this class."""
        V, = self.arg_function_spaces
        return TestFunction(V)

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return (self._argument,)

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return (self._function,)

    def constants(self):
        return ()

    def slate_coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        return self._function.ufl_domains()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        return {self.ufl_domain(): {"cell": [None]}}

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


class BlockAssembledVector(AssembledVector):
    """This class is a symbolic representation of an assembled
    vector of data contained in a set of :class:`~.Function` s
    defined on pieces of a split mixed function space.

    :arg functions: A tuple of firedrake functions.
    """

    def __new__(cls, function, expr, indices):
        block = Block(expr, indices)
        split_functions = block.form
        if isinstance(split_functions, tuple) \
           and all(isinstance(f, BaseCoefficient) for f in split_functions):
            self = TensorBase.__new__(cls)
            self._function = split_functions
            self._indices = indices
            self._original_function = function
            self._block = block
            return self
        else:
            raise TypeError("Expecting a tuple of BaseCoefficients (not a %r)" %
                            type(split_functions))

    @cached_property
    def form(self):
        return self._original_function

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces associated to the corresponding block.
        """
        return self._block.arg_function_spaces

    @cached_property
    def _argument(self):
        """Generates a tuple of 'test function' associated with this class."""
        return tuple(TestFunction(fs) for fs in self.arg_function_spaces)

    def arguments(self):
        """Returns a tuple of arguments associated with the corresponding block."""
        return self._block.arguments()

    def coefficients(self):
        return (self._original_function, )

    def slate_coefficients(self):
        """Returns a BlockFunction in a tuple which carries all information to generate the right coefficients and maps."""
        return (BlockFunction(self._function, self._indices, self._original_function),)

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with the tensor.
        """
        return tuple(domain for fs in self.arg_function_spaces for domain in fs.ufl_domains())

    def subdomain_data(self):
        """Returns mappings on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        return tuple({domain: {"cell": [None]}} for domain in self.ufl_domain())

    def _output_string(self, prec=None):
        """Creates a string representation of the tensor."""
        return "BAV_%d" % self.id

    def __repr__(self):
        """Slate representation of the tensor object."""
        return "BlockAssembledVector(%r)" % self._function

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self._function, self._original_function, self._indices)


class Block(TensorBase):
    r"""This class represents a tensor corresponding
    to particular block of a mixed tensor. Depending on
    the indices provided, the subblocks can span multiple
    test/trial spaces.

    :arg tensor: A (mixed) tensor.
    :arg indices: Indices of the test and trial function
        spaces to extract. This should be a 0-, 1-, or
        2-tuple (whose length is equal to the rank of the
        tensor.) The entries should be an iterable of integer
        indices.

    For example, consider the mixed tensor defined by:

    .. code-block:: python3

       n = FacetNormal(m)
       U = FunctionSpace(m, "DRT", 1)
       V = FunctionSpace(m, "DG", 0)
       M = FunctionSpace(m, "DGT", 0)
       W = U * V * M
       u, p, r = TrialFunctions(W)
       w, q, s = TestFunctions(W)
       A = Tensor(dot(u, w)*dx + p*div(w)*dx + r*dot(w, n)*dS
                  + div(u)*q*dx + p*q*dx + r*s*ds)

    This describes a block 3x3 mixed tensor of the form:

    .. math::

      \begin{bmatrix}
            A & B & C \\
            D & E & F \\
            G & H & J
      \end{bmatrix}

    Providing the 2-tuple ((0, 1), (0, 1)) returns a tensor
    corresponding to the upper 2x2 block:

    .. math::

       \begin{bmatrix}
            A & B \\
            D & E
       \end{bmatrix}

    More generally, argument indices of the form `(idr, idc)`
    produces a tensor of block-size `len(idr)` x `len(idc)`
    spanning the specified test/trial spaces.
    """

    def __new__(cls, tensor, indices):
        if not isinstance(tensor, TensorBase):
            raise TypeError("Can only extract blocks of Slate tensors.")

        if len(indices) != tensor.rank:
            raise ValueError("Length of indices must be equal to the tensor rank.")

        if not all(0 <= i < len(arg.function_space())
                   for arg, idx in zip(tensor.arguments(), indices) for i in as_tuple(idx)):
            raise ValueError("Indices out of range.")

        if not tensor.is_mixed:
            return tensor

        return super().__new__(cls)

    def __init__(self, tensor, indices):
        """Constructor for the Block class."""
        super(Block, self).__init__()
        self.operands = (tensor,)
        self._blocks = dict(enumerate(map(as_tuple, indices)))
        self._indices = indices

    @cached_property
    def terminal(self):
        """Blocks are only terminal when they sit on Tensors or AssembledVectors"""
        tensor, = self.operands
        return tensor.terminal

    @cached_property
    def _split_arguments(self):
        """Splits the function space and stores the component
        spaces determined by the indices.
        """
        tensor, = self.operands
        return tuple(type(a)(subspace(a.function_space(), self._blocks[i]),
                             a.number(), part=a.part())
                     for i, a in enumerate(tensor.arguments()))

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        return tuple(arg.function_space() for arg in self.arguments())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self._split_arguments

    @cached_property
    def form(self):
        tensor, = self.operands
        assert tensor.terminal
        if not tensor.assembled:
            # turns a Block on a Tensor into an indexed ufl form
            return ExtractSubBlock().split(tensor.form, self._indices)
        else:
            # turns the Block on an AssembledVector into a set off coefficients
            # corresponding to the indices of the Block
            return tuple(tensor._function.subfunctions[i] for i in chain(*self._indices))

    @cached_property
    def assembled(self):
        tensor, = self.operands
        return tensor.assembled

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        tensor, = self.operands
        return tensor.coefficients()

    def constants(self):
        """Returns a tuple of constants associated with the tensor."""
        tensor, = self.operands
        return tensor.constants()

    def slate_coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        tensor, = self.operands
        return tensor.ufl_domains()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        tensor, = self.operands
        return tensor.subdomain_data()

    def _output_string(self, prec=None):
        """Creates a string representation of the tensor."""
        tensor, = self.operands
        return "%s[%s]_%d" % (tensor, self._indices, self.id)

    def __repr__(self):
        """Slate representation of the tensor object."""
        tensor, = self.operands
        return "%s(%r, idx=%s)" % (type(self).__name__, tensor, self._indices)

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        tensor, = self.operands
        return (type(self), tensor, self._indices)


class Factorization(TensorBase):
    """An abstract Slate class for the factorization of matrices. The
    factorizations available are the following:

        (1) LU with full or partial pivoting ('FullPivLU' and 'PartialPivLU');
        (2) QR using Householder reflectors ('HouseholderQR') with the option
            to use column pivoting ('ColPivHouseholderQR') or full pivoting
            ('FullPivHouseholderQR');
        (3) standard Cholesky ('LLT') and stabilized Cholesky factorizations
            with pivoting ('LDLT');
        (4) a rank-revealing complete orthogonal decomposition using
            Householder transformations ('CompleteOrthogonalDecomposition');
            and
        (5) singular-valued decompositions ('JacobiSVD' and 'BDCSVD'). For
            larger matrices, 'BDCSVD' is recommended.
    """

    def __init__(self, tensor, decomposition=None):
        """Constructor for the Factorization class."""

        decomposition = decomposition or "PartialPivLU"

        if decomposition not in ["PartialPivLU", "FullPivLU",
                                 "HouseholderQR", "ColPivHouseholderQR",
                                 "FullPivHouseholderQR", "LLT", "LDLT",
                                 "CompleteOrthogonalDecomposition",
                                 "BDCSVD", "JacobiSVD"]:
            raise ValueError("Decomposition '%s' not supported" % decomposition)

        if tensor.rank != 2:
            raise ValueError("Can only decompose matrices.")

        super(Factorization, self).__init__()

        self.operands = (tensor,)
        self.decomposition = decomposition

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor, = self.operands
        return tensor.arg_function_spaces

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        tensor, = self.operands
        return tensor.arguments()

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        tensor, = self.operands
        return tensor.coefficients()

    def constants(self):
        """Returns a tuple of constants associated with the tensor."""
        tensor, = self.operands
        return tensor.constants()

    def slate_coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.coefficients()

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        tensor, = self.operands
        return tensor.ufl_domains()

    def subdomain_data(self):
        """Returns a mapping on the tensor:
        ``{domain:{integral_type: subdomain_data}}``.
        """
        tensor, = self.operands
        return tensor.subdomain_data()

    def _output_string(self, prec=None):
        """Creates a string representation of the tensor."""
        tensor, = self.operands
        return "%s(%s)_%d" % (self.decomposition, tensor, self.id)

    def __repr__(self):
        """Slate representation of the tensor object."""
        tensor, = self.operands
        return "%s(%r, %s)" % (type(self).__name__, tensor, self.decomposition)

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        tensor, = self.operands
        return (type(self), tensor, self.decomposition)


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
    terminal = True

    def __init__(self, form, diagonal=False):
        """Constructor for the Tensor class."""
        if not isinstance(form, (Form, ZeroBaseForm)):
            if isinstance(form, Function):
                raise TypeError("Use AssembledVector instead of Tensor.")
            raise TypeError("Only UFL forms are acceptable inputs.")

        if self.diagonal:
            assert len(form.arguments()) > 1, "Diagonal option only makes sense on rank-2 tensors."

        r = len(form.arguments()) - diagonal
        if r not in (0, 1, 2):
            raise NotImplementedError("No support for tensors of rank %d." % r)

        # Remove any negative restrictions and replace with zero
        form = map_integrand_dags(RemoveNegativeRestrictions(), form)

        super(Tensor, self).__init__()

        self.form = form
        self.diagonal = diagonal

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        return tuple(arg.function_space() for arg in self.arguments())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        r = len(self.form.arguments()) - self.diagonal
        return self.form.arguments()[0:r]

    def coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.form.coefficients()

    def constants(self):
        """Returns a tuple of constants associated with the tensor."""
        return unique(extract_firedrake_constants(self.form))

    def slate_coefficients(self):
        """Returns a tuple of coefficients associated with the tensor."""
        return self.coefficients()

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

    def empty(self):
        """Returns whether the form associated with the tensor is empty."""
        return self.form.empty()

    def _output_string(self, prec=None):
        """Creates a string representation of the tensor."""
        return ["S", "V", "M"][self.rank] + "_%d" % self.id

    def __repr__(self):
        """Slate representation of the tensor object."""
        return ["Scalar", "Vector", "Matrix"][self.rank] + "(%r)" % self.form

    @cached_property
    def _key(self):
        """Returns a key for hash and equality."""
        return (type(self), self.form, self.diagonal)


class TensorOp(TensorBase):
    """An abstract Slate class representing general operations on
    existing Slate tensors.

    :arg operands: an iterable of operands that are :class:`~.firedrake.slate.TensorBase`
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

    def constants(self):
        """Returns a tuple of constants associated with the tensor."""
        const = [op.constants() for op in self.operands]
        return unique(chain(*const))

    def slate_coefficients(self):
        """Returns the expected coefficients of the resulting tensor."""
        coeffs = [op.slate_coefficients() for op in self.operands]
        return tuple(OrderedDict.fromkeys(chain(*coeffs)))

    def ufl_domains(self):
        """Returns the integration domains of the integrals associated with
        the tensor.
        """
        collected_domains = [op.ufl_domains() for op in self.operands]
        return sort_domains(join_domains(chain(*collected_domains)))

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
                    if not all(d is None for d in sd[it_type]) or not all(d is None for d in domain):
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

    :arg A: a :class:`~.firedrake.slate.TensorBase` object. This can be a terminal tensor object
        (:class:`Tensor`) or any derived expression resulting from any
        number of linear algebra operations on `Tensor` objects. For
        example, another instance of a `UnaryOp` object is an acceptable
        input, or a `BinaryOp` object.
    """

    def __repr__(self):
        """Slate representation of the resulting tensor."""
        tensor, = self.operands
        return "%s(%r)" % (type(self).__name__, tensor)


class Reciprocal(UnaryOp):
    """An abstract Slate class representing the reciprocal of a vector.
    """

    def __init__(self, A):
        """Constructor for the Inverse class."""
        assert A.rank == 1, "The tensor must be rank 1."

        super(Reciprocal, self).__init__(A)

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor, = self.operands
        return tensor.arg_function_spaces

    def arguments(self):
        """Returns the expected arguments of the resulting tensor of
        performing a specific unary operation on a tensor.
        """
        tensor, = self.operands
        return tensor.arguments()

    def _output_string(self, prec=None):
        """Creates a string representation of the inverse of a tensor."""
        tensor, = self.operands
        return "(%s).reciprocal" % tensor


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
        self.diagonal = A.diagonal

        if A.shape > (4, 4) and not isinstance(A, Factorization) and not self.diagonal:
            A = Factorization(A, decomposition="PartialPivLU")

        super(Inverse, self).__init__(A)

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor, = self.operands
        return tensor.arg_function_spaces[::-1]

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
    def __new__(cls, A):
        if A == 0:
            return Tensor(ZeroBaseForm(A.arguments()[::-1]))
        if isinstance(A, Transpose):
            tensor, = A.operands
            return tensor
        return BinaryOp.__new__(cls)

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor, = self.operands
        return tensor.arg_function_spaces[::-1]

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
    def __new__(cls, A):
        if A == 0:
            return A
        return BinaryOp.__new__(cls)

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor, = self.operands
        return tensor.arg_function_spaces

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

    :arg A: a :class:`~.firedrake.slate.TensorBase` object. This can be a terminal tensor object
        (:class:`Tensor`) or any derived expression resulting from any
        number of linear algebra operations on `Tensor` objects. For
        example, another instance of a `BinaryOp` object is an acceptable
        input, or a `UnaryOp` object.
    :arg B: a :class:`~.firedrake.slate.TensorBase` object.
    """

    def _output_string(self, prec=None):
        """Creates a string representation of the binary operation."""
        ops = {Add: '+',
               Mul: '*',
               Solve: '\\'}
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

    :arg A: a :class:`~.firedrake.slate.TensorBase` object.
    :arg B: another :class:`~.firedrake.slate.TensorBase` object.
    """
    def __new__(cls, A, B):
        if A == 0:
            return B
        elif B == 0:
            return A
        return BinaryOp.__new__(cls)

    def __init__(self, A, B):
        """Constructor for the Add class."""
        if A.shape != B.shape:
            raise ValueError("Illegal op on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))

        assert all(space_equivalence(fsA, fsB) for fsA, fsB in
                   zip(A.arg_function_spaces, B.arg_function_spaces)), (
            "Function spaces associated with operands must match."
        )

        super(Add, self).__init__(A, B)

        # Function space check above ensures that the arguments of the
        # operands are identical (in the sense that they are arguments
        # defined on the same function space).
        self._args = A.arguments()

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        A, _ = self.operands
        return A.arg_function_spaces

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        return self._args


class Mul(BinaryOp):
    """Abstract Slate class representing the interior product or two tensors.
    By interior product, we mean an operation that results in a tensor of
    equal or lower rank via performing a contraction on arguments. This
    includes Matrix-Matrix and Matrix-Vector multiplication.

    :arg A: a :class:`~.firedrake.slate.TensorBase` object.
    :arg B: another :class:`~.firedrake.slate.TensorBase` object.
    """
    def __new__(cls, A, B):
        if A == 0 or B == 0:
            return Tensor(ZeroBaseForm(A.arguments()[:-1] + B.arguments()[1:]))
        return BinaryOp.__new__(cls)

    def __init__(self, A, B):
        """Constructor for the Mul class."""
        if A.shape[-1] != B.shape[0]:
            raise ValueError("Illegal op on a %s-tensor with a %s-tensor."
                             % (A.shape, B.shape))

        fsA = A.arg_function_spaces[-1]
        fsB = B.arg_function_spaces[0]

        assert space_equivalence(fsA, fsB), (
            "Cannot perform argument contraction over middle indices. "
            "They must be in the same function space."
        )

        super(Mul, self).__init__(A, B)

        # Function space check above ensures that middle arguments can
        # be 'eliminated'.
        self._args = A.arguments()[:-1] + B.arguments()[1:]

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        A, B = self.operands
        return A.arg_function_spaces[:-1] + B.arg_function_spaces[1:]

    def arguments(self):
        """Returns the arguments of a tensor resulting
        from multiplying two tensors A and B.
        """
        return self._args


class Solve(BinaryOp):
    """Abstract Slate class describing a local linear system of equations.
    This object is a direct solver, utilizing the application of the inverse
    of matrix in a decomposed form.

    :arg A: The left-hand side operator.
    :arg B: The right-hand side.
    :arg decomposition: A string denoting the type of matrix decomposition
        to used. The factorizations available are detailed in the
        :class:`Factorization` documentation.
    """

    def __new__(cls, A, B, decomposition=None):
        assert A.rank == 2, "Operator must be a matrix."
        assert B.rank >= 1, "RHS must be a vector or matrix."

        # Same rules for performing multiplication on Slate tensors
        # applies here.
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Illegal op on a {A.shape}-tensor with a {B.shape}-tensor.")

        fsA = A.arg_function_spaces[0]
        fsB = B.arg_function_spaces[0]

        assert space_equivalence(fsA, fsB), (
            "Cannot perform argument contraction over middle indices. "
            "They must be in the same function space."
        )

        # For matrices smaller than 5x5, exact formulae can be used
        # to evaluate the inverse. Otherwise, this class will trigger
        # a factorization method in the code-generation.
        if A.shape < (5, 5):
            return A.inv * B

        return super().__new__(cls)

    def __init__(self, A, B, decomposition=None):
        """Constructor for the Solve class."""

        # LU with partial pivoting is a stable default.
        decomposition = decomposition or "PartialPivLU"

        # Create a matrix factorization
        A_factored = Factorization(A, decomposition=decomposition) if not A.diagonal else A

        super(Solve, self).__init__(A_factored, B)

        self._args = A_factored.arguments()[::-1][:-1] + B.arguments()[1:]
        self._arg_fs = [arg.function_space() for arg in self._args]

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        return tuple(self._arg_fs)

    def arguments(self):
        """Returns the arguments of a tensor resulting
        from applying the inverse of A onto B.
        """
        return self._args


class DiagonalTensor(UnaryOp):
    """An abstract Slate class representing the diagonal of a tensor.

    .. warning::

       This class will raise an error if the tensor is not square.
    """
    diagonal = True

    def __new__(cls, A):
        if A == 0:
            return Tensor(ZeroBaseForm(A.arguments()[:1]))
        return BinaryOp.__new__(cls)

    def __init__(self, A):
        """Constructor for the Diagonal class."""
        assert A.rank == 2, "The tensor must be rank 2."
        assert A.shape[0] == A.shape[1], (
            "The diagonal can only be computed on square tensors."
        )

        super(DiagonalTensor, self).__init__(A)

    @cached_property
    def arg_function_spaces(self):
        """Returns a tuple of function spaces that the tensor
        is defined on.
        """
        tensor, = self.operands
        return tuple(arg.function_space() for arg in tensor.arguments())

    def arguments(self):
        """Returns a tuple of arguments associated with the tensor."""
        tensor, = self.operands
        return tensor.arguments()

    def _output_string(self, prec=None):
        """Creates a string representation of the diagonal of a tensor."""
        tensor, = self.operands
        return "(%s).diag" % tensor


def space_equivalence(A, B):
    """Checks that two function spaces are equivalent.

    :arg A: A function space.
    :arg B: Another function space.

    Returns `True` if they have matching meshes, elements, and rank. Otherwise,
    `False` is returned.
    """

    return A.mesh() == B.mesh() and A.ufl_element() == B.ufl_element()


# Establishes levels of precedence for Slate tensors
precedences = [
    [AssembledVector, Block, Factorization, Tensor, DiagonalTensor, Reciprocal],
    [Add],
    [Mul],
    [Solve],
    [UnaryOp],
]

# Here we establish the precedence class attribute for a given
# Slate TensorOp class.
for level, group in enumerate(precedences):
    for tensor in group:
        tensor.prec = level
