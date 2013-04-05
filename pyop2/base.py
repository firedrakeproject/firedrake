# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

""" Base classes for OP2 objects. The versions here deal only with metadata and perform no processing of the data itself. This enables these objects to be used in static analysis mode where no runtime information is available. """

import numpy as np

from exceptions import *
from utils import *
from backends import _make_object

def set_mpi_communicator(comm):
    pass

# Data API

class Access(object):
    """OP2 access type. In an :py:class:`Arg`, this describes how the :py:class:`DataCarrier` will be accessed.

    .. warning :: Access should not be instantiated by user code. Instead, use the predefined values: :const:`READ`, :const:`WRITE`, :const:`RW`, :const:`INC`, :const:`MIN`, :const:`MAX`
"""

    _modes = ["READ", "WRITE", "RW", "INC", "MIN", "MAX"]

    @validate_in(('mode', _modes, ModeValueError))
    def __init__(self, mode):
        self._mode = mode

    def __str__(self):
        return "OP2 Access: %s" % self._mode

    def __repr__(self):
        return "Access('%s')" % self._mode

READ  = Access("READ")
"""The :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed read-only."""

WRITE = Access("WRITE")
"""The  :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed write-only, and OP2 is not required to handle write conflicts."""

RW    = Access("RW")
"""The  :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed for reading and writing, and OP2 is not required to handle write conflicts."""

INC   = Access("INC")
"""The kernel computes increments to be summed onto a :class:`Global`, :class:`Dat`, or :class:`Mat`. OP2 is responsible for managing the write conflicts caused."""

MIN   = Access("MIN")
"""The kernel contributes to a reduction into a :class:`Global` using a ``min`` operation. OP2 is responsible for reducing over the different kernel invocations."""

MAX   = Access("MAX")
"""The kernel contributes to a reduction into a :class:`Global` using a ``max`` operation. OP2 is responsible for reducing over the different kernel invocations."""

# Data API

class Arg(object):
    """An argument to a :func:`par_loop`.

    .. warning:: User code should not directly instantiate :class:`Arg`. Instead, use the call syntax on the :class:`DataCarrier`.
    """
    def __init__(self, data=None, map=None, idx=None, access=None):
        self._dat = data
        self._map = map
        self._idx = idx
        self._access = access
        self._lib_handle = None
        self._in_flight = False # some kind of comms in flight for this arg

    def __str__(self):
        return "OP2 Arg: dat %s, map %s, index %s, access %s" % \
                   (self._dat, self._map, self._idx, self._access)

    def __repr__(self):
        return "Arg(%r, %r, %r, %r)" % \
                   (self._dat, self._map, self._idx, self._access)

    @property
    def data(self):
        """Data carrier: :class:`Dat`, :class:`Mat`, :class:`Const` or :class:`Global`."""
        return self._dat

    @property
    def ctype(self):
        """String representing the C type of the data in this ``Arg``."""
        return self.data.ctype

    @property
    def dtype(self):
        """Numpy datatype of this Arg"""
        return self.data.dtype

    @property
    def map(self):
        """The :class:`Map` via which the data is to be accessed."""
        return self._map

    @property
    def idx(self):
        """Index into the mapping."""
        return self._idx

    @property
    def access(self):
        """Access descriptor. One of the constants of type :class:`Access`"""
        return self._access

    @property
    def _is_soa(self):
        return self._is_dat and self._dat.soa

    @property
    def _is_vec_map(self):
        return self._is_indirect and self._idx is None

    @property
    def _is_mat(self):
        return isinstance(self._dat, Mat)

    @property
    def _is_global(self):
        return isinstance(self._dat, Global)

    @property
    def _is_global_reduction(self):
        return self._is_global and self._access in [INC, MIN, MAX]

    @property
    def _is_dat(self):
        return isinstance(self._dat, Dat)

    @property
    def _is_INC(self):
        return self._access == INC

    @property
    def _is_MIN(self):
        return self._access == MIN

    @property
    def _is_MAX(self):
        return self._access == MAX

    @property
    def _is_direct(self):
        return isinstance(self._dat, Dat) and self._map is IdentityMap

    @property
    def _is_indirect(self):
        return isinstance(self._dat, Dat) and self._map not in [None, IdentityMap]

    @property
    def _is_indirect_and_not_read(self):
        return self._is_indirect and self._access is not READ

    @property
    def _is_indirect_reduction(self):
        return self._is_indirect and self._access is INC

    @property
    def _uses_itspace(self):
        return self._is_mat or isinstance(self.idx, IterationIndex)

    def halo_exchange_begin(self):
        pass

    def halo_exchange_end(self):
        pass

    def reduction_begin(self):
        pass

    def reduction_end(self):
        pass

class Set(object):
    """OP2 set.

    When the set is employed as an iteration space in a
    :func:`par_loop`, the extent of any local iteration space within
    each set entry is indicated in brackets. See the example in
    :func:`pyop2.op2.par_loop` for more details.

    The size of the set can either be an integer, or a list of four
    integers.  The latter case is used for running in parallel where
    we distinguish between:

      - CORE (owned and not touching halo)
      - OWNED (owned, touching halo)
      - EXECUTE HALO (not owned, but executed over redundantly)
      - NON EXECUTE HALO (not owned, read when executing in the
                          execute halo)

    If a single integer is passed, we assume that we're running in
    serial and there is no distinction.

    The division of set elements is:

    [0, CORE)
    [CORE, OWNED)
    [OWNED, EXECUTE HALO)
    [EXECUTE HALO, NON EXECUTE HALO).

    Halo send/receive data is stored on sets in a :class:`Halo`.
    """

    _globalcount = 0

    CORE_SIZE = 0
    OWNED_SIZE = 1
    IMPORT_EXEC_SIZE = 2
    IMPORT_NON_EXEC_SIZE = 3
    @validate_type(('name', str, NameTypeError))
    def __init__(self, size=None, name=None, halo=None):
        if type(size) is int:
            size = [size]*4
        size = as_tuple(size, int, 4)
        assert size[Set.CORE_SIZE] <= size[Set.OWNED_SIZE] <= \
                size[Set.IMPORT_EXEC_SIZE] <= size[Set.IMPORT_NON_EXEC_SIZE], \
                "Set received invalid sizes: %s" % size
        self._core_size = size[Set.CORE_SIZE]
        self._size = size[Set.OWNED_SIZE]
        self._ieh_size = size[Set.IMPORT_EXEC_SIZE]
        self._inh_size = size[Set.IMPORT_NON_EXEC_SIZE]
        self._name = name or "set_%d" % Set._globalcount
        self._lib_handle = None
        self._halo = halo
        if self.halo:
            self.halo.verify(self)
        Set._globalcount += 1

    def __call__(self, *dims):
        return IterationSpace(self, dims)

    @property
    def core_size(self):
        """Core set size.  Owned elements not touching halo elements."""
        return self._core_size

    @property
    def size(self):
        """Set size, owned elements."""
        return self._size

    @property
    def exec_size(self):
        """Set size including execute halo elements.

        If a :class:`ParLoop` is indirect, we do redundant computation
        by executing over these set elements as well as owned ones.
        """
        return self._ieh_size

    @property
    def total_size(self):
        """Total set size, including halo elements."""
        return self._inh_size

    @property
    def name(self):
        """User-defined label"""
        return self._name

    @property
    def halo(self):
        """:class:`Halo` associated with this Set"""
        return self._halo

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%s, '%s')" % (self._size, self._name)

class Halo(object):
    """A description of a halo associated with a :class:`Set`.

    The halo object describes which :class:`Set` elements are sent
    where, and which :class:`Set` elements are received from where.

    For each process to send to, `sends[process]` should be a numpy
    arraylike (tuple, list, iterable, numpy array) of the set elements
    to send to `process`.  Similarly `receives[process]` should be the
    set elements that will be received from `process`.

    To send/receive no set elements to/from a process, pass an empty
    list in that position.

    The gnn2unn array is a map from process-local set element
    numbering to cross-process set element numbering.  It must
    correctly number all the set elements in the halo region as well
    as owned elements.  Providing this array is only necessary if you
    will access :class:`Mat` objects on the :class:`Set` this `Halo`
    lives on.  Insertion into :class:`Dat`s always uses process-local
    numbering, however insertion into :class:`Mat`s uses cross-process
    numbering under the hood.
    """
    def __init__(self, sends, receives, gnn2unn=None):
        self._sends = tuple(np.asarray(x, dtype=np.int32) for x in sends)
        self._receives = tuple(np.asarray(x, dtype=np.int32) for x in receives)
        self._global_to_petsc_numbering = gnn2unn

    @property
    def sends(self):
        """Return the sends associated with this :class:`Halo`.

        A tuple of numpy arrays, one entry for each rank, with each
        array indicating the :class:`Set` elements to send.

        For example, to send no elements to rank 0, elements 1 and 2
        to rank 1 and no elements to rank 2 (with comm.size == 3) we
        would have:

        (np.empty(0, dtype=np.int32), np.array([1,2], dtype=np.int32),
         np.empty(0, dtype=np.int32)."""
        return self._sends

    @property
    def receives(self):
        """Return the receives associated with this :class:`Halo`.

        A tuple of numpy arrays, one entry for each rank, with each
        array indicating the :class:`Set` elements to receive.

        See `Halo.sends` for an example"""
        return self._receives

    @property
    def global_to_petsc_numbering(self):
        """The mapping from global (per-process) dof numbering to
    petsc (cross-process) dof numbering."""
        return self._global_to_petsc_numbering

class IterationSpace(object):
    """OP2 iteration space type.

    .. Warning:: User code should not directly instantiate IterationSpace. Instead use the call syntax on the iteration set in the :func:`par_loop` call.
"""

    @validate_type(('iterset', Set, SetTypeError))
    def __init__(self, iterset, extents=()):
        self._iterset = iterset
        self._extents = as_tuple(extents, int)

    @property
    def iterset(self):
        """The :class:`Set` over which this IterationSpace is defined."""
        return self._iterset

    @property
    def extents(self):
        """Extents of the IterationSpace within each item of ``iterset``"""
        return self._extents

    @property
    def name(self):
        """The name of the :class:`Set` over which this IterationSpace is defined."""
        return self._iterset.name

    @property
    def core_size(self):
        """The number of :class:`Set` elements which don't touch halo elements in the set over which this IterationSpace is defined"""
        return self._iterset.core_size

    @property
    def size(self):
        """The size of the :class:`Set` over which this IterationSpace is defined."""
        return self._iterset.size

    @property
    def exec_size(self):
        """The size of the :class:`Set` over which this IterationSpace
    is defined, including halo elements to be executed over"""
        return self._iterset.exec_size

    @property
    def total_size(self):
        """The total size of :class:`Set` over which this IterationSpace is defined.

        This includes all halo set elements."""
        return self._iterset.total_size

    @property
    def _extent_ranges(self):
        return [e for e in self.extents]

    def __str__(self):
        return "OP2 Iteration Space: %s with extents %s" % (self._iterset, self._extents)

    def __repr__(self):
        return "IterationSpace(%r, %r)" % (self._iterset, self._extents)

class DataCarrier(object):
    """Abstract base class for OP2 data. Actual objects will be
    ``DataCarrier`` objects of rank 0 (:class:`Const` and
    :class:`Global`), rank 1 (:class:`Dat`), or rank 2
    (:class:`Mat`)"""

    @property
    def dtype(self):
        """The Python type of the data."""
        return self._data.dtype

    @property
    def ctype(self):
        """The c type of the data."""
        # FIXME: Complex and float16 not supported
        typemap = { "bool":    "unsigned char",
                    "int":     "int",
                    "int8":    "char",
                    "int16":   "short",
                    "int32":   "int",
                    "int64":   "long long",
                    "uint8":   "unsigned char",
                    "uint16":  "unsigned short",
                    "uint32":  "unsigned int",
                    "uint64":  "unsigned long",
                    "float":   "double",
                    "float32": "float",
                    "float64": "double" }
        return typemap[self.dtype.name]

    @property
    def name(self):
        """User-defined label."""
        return self._name

    @property
    def dim(self):
        """The shape of the values for each element of the object."""
        return self._dim

    @property
    def cdim(self):
        """The number of values for each member of the object. This is the product of the dims."""
        return np.asscalar(np.prod(self.dim))

class Dat(DataCarrier):
    """OP2 vector data. A ``Dat`` holds ``dim`` values for every member of a :class:`Set`.

    When a ``Dat`` is passed to :func:`par_loop`, the map via which
    indirection occurs and the access descriptor are passed by
    `calling` the ``Dat``. For instance, if a ``Dat`` named ``D`` is
    to be accessed for reading via a :class:`Map` named ``M``, this is
    accomplished by::

      D(M, pyop2.READ)

    The :class:`Map` through which indirection occurs can be indexed
    using the index notation described in the documentation for the
    :class:`Map` class. Direct access to a Dat can be accomplished by
    using the :data:`IdentityMap` as the indirection.

    ``Dat`` objects support the pointwise linear algebra operations +=, *=,
    -=, /=, where *= and /= also support multiplication/division by a scalar.
    """

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]

    @validate_type(('dataset', Set, SetTypeError), ('name', str, NameTypeError))
    def __init__(self, dataset, dim, data=None, dtype=None, name=None,
                 soa=None, uid=None):
        self._dataset = dataset
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, (dataset.total_size,)+self._dim, allow_none=True)
        # Are these data to be treated as SoA on the device?
        self._soa = bool(soa)
        self._lib_handle = None
        self._needs_halo_update = False
        # If the uid is not passed in from outside, assume that Dats
        # have been declared in the same order everywhere.
        if uid is None:
            self._id = Dat._globalcount
            Dat._globalcount += 1
        else:
            self._id = uid
        self._name = name or "dat_%d" % self._id

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, path, access):
        if isinstance(path, Map):
            if path._dataset != self._dataset and path != IdentityMap:
                raise MapValueError("Dataset of Map does not match Dataset of Dat.")
            return _make_object('Arg', data=self, map=path, access=access)
        else:
            path._dat = self
            path._access = access
            return path

    @property
    def dataset(self):
        """:class:`Set` on which the Dat is defined."""
        return self._dataset

    @property
    def soa(self):
        """Are the data in SoA format?"""
        return self._soa

    @property
    def data(self):
        """Numpy array containing the data values."""
        if self.dataset.total_size > 0 and self._data.size == 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        self.needs_halo_update = True
        return self._data

    @property
    def data_ro(self):
        """Numpy array containing the data values.  Read-only"""
        if self.dataset.total_size > 0 and self._data.size == 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        maybe_setflags(self._data, write=False)
        return self._data

    @property
    def dim(self):
        '''The number of values at each member of the dataset.'''
        return self._dim

    @property
    def needs_halo_update(self):
        '''Has this Dat been written to since the last halo exchange?'''
        return self._needs_halo_update

    @needs_halo_update.setter
    def needs_halo_update(self, val):
        self._needs_halo_update = val

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        raise NotImplementedError("Norm is not implemented.")

    def halo_exchange_begin(self):
        pass

    def halo_exchange_end(self):
        pass

    def zero(self):
        """Zero the data associated with this :class:`Dat`"""
        if not hasattr(self, '_zero_kernel'):
            k = """void zero(%(t)s *dat) {
                for (int n = 0; n < %(dim)s; ++n) {
                    dat[n] = (%(t)s)0;
                }
            }""" % { 't': self.ctype, 'dim' : self.cdim }
            self._zero_kernel = _make_object('Kernel', k, 'zero')
        _make_object('ParLoop', self._zero_kernel, self.dataset,
                     self(IdentityMap, WRITE)).compute()

    def __str__(self):
        return "OP2 Dat: %s on (%s) with dim %s and datatype %s" \
               % (self._name, self._dataset, self._dim, self._data.dtype.name)

    def __repr__(self):
        return "Dat(%r, %s, '%s', None, '%s')" \
               % (self._dataset, self._dim, self._data.dtype, self._name)

class Const(DataCarrier):
    """Data that is constant for any element of any set."""

    class NonUniqueNameError(ValueError):
        """The Names of const variables are required to be globally unique. This exception is raised if the name is already in use."""

    _defs = set()
    _globalcount = 0

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, name=None, dtype=None):
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, self._dim, allow_none=True)
        self._name = name or "const_%d" % Const._globalcount
        if any(self._name is const._name for const in Const._defs):
            raise Const.NonUniqueNameError(
                "OP2 Constants are globally scoped, %s is already in use" % self._name)
        Const._defs.add(self)
        Const._globalcount += 1


    @property
    def data(self):
        """Data array."""
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Const!")
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

    def __str__(self):
        return "OP2 Const: %s of dim %s and type %s with value %s" \
               % (self._name, self._dim, self._data.dtype.name, self._data)

    def __repr__(self):
        return "Const(%s, %s, '%s')" \
               % (self._dim, self._data, self._name)

    @classmethod
    def _definitions(cls):
        return sorted(Const._defs, key=lambda c: c.name)

    def remove_from_namespace(self):
        """Remove this Const object from the namespace

        This allows the same name to be redeclared with a different shape."""
        Const._defs.discard(self)

    def _format_declaration(self):
        d = {'type' : self.ctype,
             'name' : self.name,
             'dim' : self.cdim}

        if self.cdim == 1:
            return "static %(type)s %(name)s;" % d

        return "static %(type)s %(name)s[%(dim)s];" % d

class Global(DataCarrier):
    """OP2 global value.

    When a ``Global`` is passed to a :func:`par_loop`, the access
    descriptor is passed by `calling` the ``Global``.  For example, if
    a ``Global`` named ``G`` is to be accessed for reading, this is
    accomplished by::

      G(pyop2.READ)
    """

    _globalcount = 0
    _modes = [READ, INC, MIN, MAX]

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None):
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, self._dim, allow_none=True)
        self._buf = np.empty_like(self._data)
        self._name = name or "global_%d" % Global._globalcount
        Global._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access):
        return _make_object('Arg', data=self, access=access)

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
                % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global('%s', %r, %r)" % (self._name, self._dim, self._data)

    @property
    def data(self):
        """Data array."""
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Global!")
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

    @property
    def soa(self):
        """Are the data in SoA format? This is always false for :class:`Global`
        objects."""
        return False

#FIXME: Part of kernel API, but must be declared before Map for the validation.

class IterationIndex(object):
    """OP2 iteration space index

    Users should not directly instantiate :class:`IterationIndex` objects. Use
    ``op2.i`` instead."""

    def __init__(self, index=None):
        assert index is None or isinstance(index, int), "i must be an int"
        self._index = index

    def __str__(self):
        return "OP2 IterationIndex: %d" % self._index

    def __repr__(self):
        return "IterationIndex(%d)" % self._index

    @property
    def index(self):
        """Return the integer value of this index."""
        return self._index

    def __getitem__(self, idx):
        return IterationIndex(idx)

    # This is necessary so that we can convert an IterationIndex to a
    # tuple.  Because, __getitem__ returns a new IterationIndex
    # we have to explicitly provide an iterable interface
    def __iter__(self):
        yield self

i = IterationIndex()
"""Shorthand for constructing :class:`IterationIndex` objects.

``i[idx]`` builds an :class:`IterationIndex` object for which the `index`
property is `idx`.
"""

class Map(object):
    """OP2 map, a relation between two :class:`Set` objects.

    Each entry in the ``iterset`` maps to ``dim`` entries in the
    ``dataset``. When a map is used in a :func:`par_loop`,
    it is possible to use Python index notation to select an
    individual entry on the right hand side of this map. There are three possibilities:

    * No index. All ``dim`` :class:`Dat` entries will be passed to the
      kernel.
    * An integer: ``some_map[n]``. The ``n`` th entry of the
      map result will be passed to the kernel.
    * An :class:`IterationIndex`, ``some_map[pyop2.i[n]]``. ``n``
      will take each value from ``0`` to ``e-1`` where ``e`` is the
      ``n`` th extent passed to the iteration space for this :func:`par_loop`.
      See also :data:`i`.
    """

    _globalcount = 0

    @validate_type(('iterset', Set, SetTypeError), ('dataset', Set, SetTypeError), \
            ('dim', int, DimTypeError), ('name', str, NameTypeError))
    def __init__(self, iterset, dataset, dim, values=None, name=None):
        self._iterset = iterset
        self._dataset = dataset
        self._dim = dim
        self._values = verify_reshape(values, np.int32, (iterset.total_size, dim), \
                                      allow_none=True)
        self._name = name or "map_%d" % Map._globalcount
        self._lib_handle = None
        Map._globalcount += 1

    @validate_type(('index', (int, IterationIndex), IndexTypeError))
    def __getitem__(self, index):
        if isinstance(index, int) and not (0 <= index < self._dim):
            raise IndexValueError("Index must be in interval [0,%d]" % (self._dim-1))
        if isinstance(index, IterationIndex) and index.index not in [0, 1]:
            raise IndexValueError("IterationIndex must be in interval [0,1]")
        return _make_object('Arg', map=self, idx=index)

    # This is necessary so that we can convert a Map to a tuple
    # (needed in as_tuple).  Because, __getitem__ no longer returns a
    # Map we have to explicitly provide an iterable interface
    def __iter__(self):
        yield self

    def __getslice__(self, i, j):
        raise NotImplementedError("Slicing maps is not currently implemented")

    @property
    def iterset(self):
        """:class:`Set` mapped from."""
        return self._iterset

    @property
    def dataset(self):
        """:class:`Set` mapped to."""
        return self._dataset

    @property
    def dim(self):
        """Dimension of the mapping: number of dataset elements mapped to per
        iterset element."""
        return self._dim

    @property
    def values(self):
        """Mapping array."""
        return self._values

    @property
    def name(self):
        """User-defined label"""
        return self._name

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with dim %s" \
               % (self._name, self._iterset, self._dataset, self._dim)

    def __repr__(self):
        return "Map(%r, %r, %s, None, '%s')" \
               % (self._iterset, self._dataset, self._dim, self._name)

    def __eq__(self, o):
        try:
            return (self._iterset == o._iterset and self._dataset == o._dataset and \
               self._dim == o.dim and self._name == o.name)
        except AttributeError:
            return False

    def __ne__(self, o):
        return not self.__eq__(o)

IdentityMap = Map(Set(0), Set(0), 1, [], 'identity')
"""The identity map.  Used to indicate direct access to a :class:`Dat`."""

class Sparsity(object):
    """OP2 Sparsity, a matrix structure derived from the union of the outer
    product of pairs of :class:`Map` objects.

    :param maps: :class:`Maps` to build the :class:`Sparsity` from
    :type maps: a pair of :class:`Maps` specifying a row map and a column map,
        or a tuple of pairs of :class:`Maps` specifying multiple row and
        column maps - if a single :class:`Map` is passed, it is used as both a
        row map and a column map
    :param dims: row and column dimensions of a single :class:`Sparsity` entry
    :type dims: pair of integers or integer used for rows and columns
    :param string name: user-defined label (optional)

    Examples of constructing a Sparsity: ::

        Sparsity(single_map, 1, 'mass')
        Sparsity((single_rowmap, single_colmap), (2,1))
        Sparsity(((first_rowmap, first_colmap), (second_rowmap, second_colmap)), 2)
    """

    _globalcount = 0

    @validate_type(('maps', (Map, tuple), MapTypeError), \
                   ('dims', (int, tuple), TypeError))
    def __init__(self, maps, dims, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"

        maps = (maps,maps) if isinstance(maps, Map) else maps
        lmaps = (maps,) if isinstance(maps[0], Map) else maps
        self._rmaps, self._cmaps = map (lambda x : as_tuple(x, Map), zip(*lmaps))

        assert len(self._rmaps) == len(self._cmaps), \
            "Must pass equal number of row and column maps"

        for pair in lmaps:
            if pair[0].iterset is not pair[1].iterset:
                raise RuntimeError("Iterset of both maps in a pair must be the same")

        if not all(m.dataset is self._rmaps[0].dataset for m in self._rmaps):
            raise RuntimeError("Dataset of all row maps must be the same")

        if not all(m.dataset is self._cmaps[0].dataset for m in self._cmaps):
            raise RuntimeError("Dataset of all column maps must be the same")

        # All rmaps and cmaps have the same dataset - just use the first.
        self._nrows = self._rmaps[0].dataset.size
        self._ncols = self._cmaps[0].dataset.size

        self._dims = as_tuple(dims, int, 2)
        self._name = name or "sparsity_%d" % Sparsity._globalcount
        self._lib_handle = None
        Sparsity._globalcount += 1

    @property
    def _nmaps(self):
        return len(self._rmaps)

    @property
    def maps(self):
        """A list of pairs (rmap, cmap) where each pair of
        :class:`Map` objects will later be used to assemble into this
        matrix. The iterset of each of the maps in a pair must be the
        same, while the dataset of all the maps which appear first
        must be common, this will form the row :class:`Set` of the
        sparsity. Similarly, the dataset of all the maps which appear
        second must be common and will form the column :class:`Set` of
        the ``Sparsity``."""
        return zip(self._rmaps, self._cmaps)

    @property
    def dims(self):
        """A pair giving the number of rows per entry of the row
        :class:`Set` and the number of columns per entry of the column
        :class:`Set` of the ``Sparsity``."""
        return self._dims

    @property
    def nrows(self):
        """The number of rows in the ``Sparsity``."""
        return self._nrows

    @property
    def ncols(self):
        """The number of columns in the ``Sparsity``."""
        return self._ncols

    @property
    def name(self):
        """A user-defined label."""
        return self._name

    def __str__(self):
        return "OP2 Sparsity: rmaps %s, cmaps %s, dims %s, name %s" % \
               (self._rmaps, self._cmaps, self._dims, self._name)

    def __repr__(self):
        return "Sparsity(%s,%s,%s,%s)" % \
               (self._rmaps, self._cmaps, self._dims, self._name)

class Mat(DataCarrier):
    """OP2 matrix data. A ``Mat`` is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`.

    When a ``Mat`` is passed to :func:`par_loop`, the maps via which
    indirection occurs for the row and column space, and the access
    descriptor are passed by `calling` the ``Mat``. For instance, if a
    ``Mat`` named ``A`` is to be accessed for reading via a row :class:`Map`
    named ``R`` and a column :class:`Map` named ``C``, this is accomplished by::

     A( (R[pyop2.i[0]], C[pyop2.i[1]]), pyop2.READ)

    Notice that it is `always` necessary to index the indirection maps
    for a ``Mat``. See the :class:`Mat` documentation for more
    details."""

    _globalcount = 0
    _modes = [WRITE, INC]

    @validate_type(('sparsity', Sparsity, SparsityTypeError), \
                   ('name', str, NameTypeError))
    def __init__(self, sparsity, dtype=None, name=None):
        self._sparsity = sparsity
        self._datatype = np.dtype(dtype)
        self._name = name or "mat_%d" % Mat._globalcount
        self._lib_handle = None
        Mat._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, path, access):
        path = as_tuple(path, Arg, 2)
        path_maps = [arg.map for arg in path]
        path_idxs = [arg.idx for arg in path]
        if tuple(path_maps) not in self.sparsity.maps:
            raise MapValueError("Path maps not in sparsity maps")
        return _make_object('Arg', data=self, map=path_maps, access=access, idx=path_idxs)

    @property
    def dims(self):
        """A pair of integers giving the number of matrix rows and columns for each member of the row :class:`Set`  and column :class:`Set` respectively. This corresponds to the ``dim`` member of a :class:`Dat`. Note that ``dims`` is actually specified at the :class:`Sparsity` level and inherited by the ``Mat``."""
        return self._sparsity._dims

    @property
    def sparsity(self):
        """:class:`Sparsity` on which the ``Mat`` is defined."""
        return self._sparsity

    @property
    def _is_scalar_field(self):
        return np.prod(self.dims) == 1

    @property
    def _is_vector_field(self):
        return not self._is_scalar_field

    @property
    def values(self):
        """A numpy array of matrix values.

        .. warning ::
            This is a dense array, so will need a lot of memory.  It's
            probably not a good idea to access this property if your
            matrix has more than around 10000 degrees of freedom.

        """
        raise NotImplementedError("Abstract base Mat does not implement values()")

    @property
    def dtype(self):
        """The Python type of the data."""
        return self._datatype

    def __str__(self):
        return "OP2 Mat: %s, sparsity (%s), datatype %s" \
               % (self._name, self._sparsity, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, '%s', '%s')" \
               % (self._sparsity, self._datatype, self._name)

# Kernel API

class Kernel(object):
    """OP2 kernel type."""

    _globalcount = 0

    @validate_type(('name', str, NameTypeError))
    def __init__(self, code, name):
        self._name = name or "kernel_%d" % Kernel._globalcount
        self._code = code
        Kernel._globalcount += 1

    @property
    def name(self):
        """Kernel name, must match the kernel function name in the code."""
        return self._name

    @property
    def code(self):
        """String containing the c code for this kernel routine. This
        code must conform to the OP2 user kernel API."""
        return self._code

    @property
    def md5(self):
        """MD5 digest of kernel code and name."""
        if not hasattr(self, '_md5'):
            import md5
            self._md5 = md5.new(self._code + self._name).hexdigest()
        return self._md5

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", "%s")' % (self._code, self._name)

_parloop_cache = dict()

def _empty_parloop_cache():
    _parloop_cache.clear()

def _parloop_cache_size():
    return len(_parloop_cache)

class ParLoop(object):
    """Represents the kernel, iteration space and arguments of a parallel loop
    invocation.

    Users should not directly construct :class:`ParLoop` objects, but use
    ``op2.par_loop()`` instead."""
    def __init__(self, kernel, itspace, *args):
        self._kernel = kernel
        if isinstance(itspace, IterationSpace):
            self._it_space = itspace
        else:
            self._it_space = IterationSpace(itspace)
        self._actual_args = list(args)

        self.check_args()

    def halo_exchange_begin(self):
        """Start halo exchanges."""
        if self.is_direct:
            # No need for halo exchanges for a direct loop
            return
        for arg in self.args:
            if arg._is_dat:
                arg.halo_exchange_begin()

    def halo_exchange_end(self):
        """Finish halo exchanges (wait on irecvs)"""
        if self.is_direct:
            return
        for arg in self.args:
            if arg._is_dat:
                arg.halo_exchange_end()

    def reduction_begin(self):
        """Start reductions"""
        for arg in self.args:
            if arg._is_global_reduction:
                arg.reduction_begin()

    def reduction_end(self):
        """End reductions"""
        for arg in self.args:
            if arg._is_global_reduction:
                arg.reduction_end()

    def maybe_set_halo_update_needed(self):
        """Set halo update needed for :class:`Dat` arguments that are written to
        in this parallel loop."""
        for arg in self.args:
            if arg._is_dat and arg.access in [INC, WRITE, RW]:
                arg.data.needs_halo_update = True

    def check_args(self):
        """Checks the following:

        1. That the iteration set of the :class:`ParLoop` matches the iteration
           set of all its arguments.
        2. For each argument, check that the dataset of the map used to access
           it matches the dataset it is defined on.

        A :class:`MapValueError` is raised if these conditions are not met."""
        iterset = self._it_space._iterset
        for i, arg in enumerate(self._actual_args):
            if arg._is_global or arg._map == IdentityMap:
                continue
            for j, m in enumerate(arg._map):
                if m._iterset != iterset:
                    raise MapValueError( \
                        "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))
                else:
                    if arg._is_mat:
                        continue
                    if m._dataset != arg.data._dataset:
                        raise MapValueError( \
                            "Dataset of arg %s map %sdoesn't match the set of its Dat." % (i, j))

    def generate_code(self):
        raise RuntimeError('Must select a backend')

    @property
    def it_space(self):
        """Iteration space of the parallel loop."""
        return self._it_space

    @property
    def is_direct(self):
        """Is this parallel loop direct? I.e. are all the arguments either
        :class:Dats accessed through the identity map, or :class:Global?"""
        return all(a.map in [None, IdentityMap] for a in self.args)

    @property
    def is_indirect(self):
        """Is the parallel loop indirect?"""
        return not self.is_direct

    @property
    def needs_exec_halo(self):
        """Does the parallel loop need an exec halo?"""
        return any(arg._is_indirect_and_not_read or arg._is_mat
                   for arg in self.args)

    @property
    def kernel(self):
        """Kernel executed by this parallel loop."""
        return self._kernel

    @property
    def args(self):
        """Arguments to this parallel loop."""
        return self._actual_args

    @property
    def _has_soa(self):
        return any(a._is_soa for a in self._actual_args)

    @property
    def _cache_key(self):
        key = (self._kernel.md5, )

        key += (self._it_space.extents, )
        for arg in self.args:
            if arg._is_global:
                key += (arg.data.dim, arg.data.dtype, arg.access)
            elif arg._is_dat:
                if isinstance(arg.idx, IterationIndex):
                    idx = (arg.idx.__class__, arg.idx.index)
                else:
                    idx = arg.idx
                if arg.map is IdentityMap:
                    map_dim = None
                else:
                    map_dim = arg.map.dim
                key += (arg.data.dim, arg.data.dtype, map_dim, idx, arg.access)
            elif arg._is_mat:
                idxs = (arg.idx[0].__class__, arg.idx[0].index,
                        arg.idx[1].index)
                map_dims = (arg.map[0].dim, arg.map[1].dim)
                key += (arg.data.dims, arg.data.dtype, idxs,
                      map_dims, arg.access)

        for c in Const._definitions():
            key += (c.name, c.dtype, c.cdim)

        return key

DEFAULT_SOLVER_PARAMETERS = {'linear_solver':      'cg',
                             'preconditioner':     'jacobi',
                             'relative_tolerance': 1.0e-7,
                             'absolute_tolerance': 1.0e-50,
                             'divergence_tolerance': 1.0e+4,
                             'maximum_iterations': 1000,
                             'monitor_convergence': False,
                             'plot_convergence': False,
                             'plot_prefix': '',
                             'error_on_nonconvergence': True,
                             'gmres_restart': 30}
"""The default parameters for the solver are the same as those used in PETSc
3.3. Note that the parameters accepted by :class:`op2.Solver` are only a subset
of all PETSc parameters."""

class Solver(object):
    """OP2 Solver object. The :class:`Solver` holds a set of parameters that are
    passed to the underlying linear algebra library when the ``solve`` method
    is called. These can either be passed as a dictionary ``parameters`` *or*
    as individual keyword arguments (combining both will cause an exception).

    Recognized parameters either as dictionary keys or keyword arguments are:

    :arg linear_solver: the solver type ('cg')
    :arg preconditioner: the preconditioner type ('jacobi')
    :arg relative_tolerance: relative solver tolerance (1e-7)
    :arg absolute_tolerance: absolute solver tolerance (1e-50)
    :arg divergence_tolerance: factor by which the residual norm may exceed
        the right-hand-side norm before the solve is considered to have
        diverged: ``norm(r) >= dtol*norm(b)`` (1e4)
    :arg maximum_iterations: maximum number of solver iterations (1000)
    :arg error_on_nonconvergence: abort if the solve does not converge in the
      maximum number of iterations (True, if False only a warning is printed)
    :arg monitor_convergence: print the residual norm after each iteration
        (False)
    :arg plot_convergence: plot a graph of the convergence history after the
        solve has finished and save it to file (False, implies monitor_convergence)
    :arg plot_prefix: filename prefix for plot files ('')
    :arg gmres_restart: restart period when using GMRES

    """

    def __init__(self, parameters=None, **kwargs):
        self.parameters = DEFAULT_SOLVER_PARAMETERS.copy()
        if parameters and kwargs:
            raise RuntimeError("Solver options are set either by parameters or kwargs")
        if parameters:
            self.parameters.update(parameters)
        else:
            self.parameters.update(kwargs)

    def update_parameters(self, parameters):
        """Update solver parameters

        :arg parameters: Dictionary containing the parameters to update.
        """
        self.parameters.update(parameters)

    def solve(self, A, x, b):
        """Solve a matrix equation.

        :arg A: The :class:`Mat` containing the matrix.
        :arg x: The :class:`Dat` to receive the solution.
        :arg b: The :class:`Dat` containing the RHS.
        """
        raise NotImplementedError("solve must be implemented by backend")
