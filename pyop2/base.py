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
WRITE = Access("WRITE")
RW    = Access("RW")
INC   = Access("INC")
MIN   = Access("MIN")
MAX   = Access("MAX")

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

    @property
    def data(self):
        """Data carrier: :class:`Dat`, :class:`Mat`, :class:`Const` or :class:`Global`."""
        return self._dat

    @property
    def ctype(self):
        """String representing the C type of the data in this ``Arg``."""
        return self.data.ctype

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
        return isinstance(self._dat, Dat) and self._dat.soa

    @property
    def _is_vec_map(self):
        return self._is_indirect and self._idx is None

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
    def _is_global(self):
        return isinstance(self._dat, Global)

    @property
    def _is_mat(self):
        return isinstance(self._dat, Mat)

class Set(object):
    """OP2 set.

    When the set is employed as an iteration space in a :func:`par_loop`, the extent of any local iteration space within each set entry is indicated in brackets. See the example in :func:`pyop2.op2.par_loop` for more details.
    """

    _globalcount = 0

    @validate_type(('name', str, NameTypeError))
    def __init__(self, size=None, name=None):
        self._size = size
        self._name = name or "set_%d" % Set._globalcount
        self._lib_handle = None
        Set._globalcount += 1

    def __call__(self, *dims):
        return IterationSpace(self, dims)

    @property
    def size(self):
        """Set size"""
        return self._size

    @property
    def name(self):
        """User-defined label"""
        return self._name

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%s, '%s')" % (self._size, self._name)

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
    def size(self):
        """The size of the :class:`Set` over which this IterationSpace is defined."""
        return self._iterset.size

    @property
    def _extent_ranges(self):
        return [e for e in self.extents]

    def __str__(self):
        return "OP2 Iteration Space: %s with extents %s" % self._extents

    def __repr__(self):
        return "IterationSpace(%r, %r)" % (self._iterset, self._extents)

class DataCarrier(object):
    """Abstract base class for OP2 data."""

    @property
    def dtype(self):
        """Data type."""
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
                    "uint64":  "unsigned long long",
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
        """Dimension/shape of a single data item."""
        return self._dim

    @property
    def cdim(self):
        """The size of a single data item, this is the product of the dims."""
        return np.prod(self.dim)

class Dat(DataCarrier):
    """OP2 vector data. A ``Dat`` holds a value for every member of a :class:`Set`."""

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]
    _arg_type = Arg

    @validate_type(('dataset', Set, SetTypeError), ('name', str, NameTypeError))
    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        self._dataset = dataset
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, (dataset.size,)+self._dim, allow_none=True)
        # Are these data in SoA format, rather than standard AoS?
        self._soa = bool(soa)
        # Make data "look" right
        if self._soa:
            self._data = self._data.T
        self._name = name or "dat_%d" % Dat._globalcount
        self._lib_handle = None
        Dat._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, path, access):
        if isinstance(path, Map):
            return self._arg_type(data=self, map=path, access=access)
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
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        return self._data

    @property
    def dim(self):
        '''The number of values at each member of the dataset.'''
        return self._dim

    def __str__(self):
        return "OP2 Dat: %s on (%s) with dim %s and datatype %s" \
               % (self._name, self._dataset, self._dim, self._data.dtype.name)

    def __repr__(self):
        return "Dat(%r, %s, '%s', None, '%s')" \
               % (self._dataset, self._dim, self._data.dtype, self._name)

class Const(DataCarrier):
    """Data that is constant for any element of any set."""

    class NonUniqueNameError(ValueError):
        """The Names of const variables are requried to be globally unique. This exception is raised if the name is already in use."""

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

    def remove_from_namespace(self):
        if self in Const._defs:
            Const._defs.remove(self)

    def _format_for_c(self):
        d = {'type' : self.ctype,
             'name' : self.name,
             'dim' : self.cdim}

        if self.cdim == 1:
            return "static %(type)s %(name)s;" % d

        return "static %(type)s %(name)s[%(dim)s];" % d

class Global(DataCarrier):
    """OP2 global value."""

    _globalcount = 0
    _modes = [READ, INC, MIN, MAX]
    _arg_type = Arg

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None):
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, self._dim, allow_none=True)
        self._name = name or "global_%d" % Global._globalcount
        Global._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access):
        return self._arg_type(data=self, access=access)

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

#FIXME: Part of kernel API, but must be declared before Map for the validation.

class IterationIndex(object):
    """OP2 iteration space index"""

    def __init__(self, index):
        assert isinstance(index, int), "i must be an int"
        self._index = index

    def __str__(self):
        return "OP2 IterationIndex: %d" % self._index

    def __repr__(self):
        return "IterationIndex(%d)" % self._index

    @property
    def index(self):
        return self._index

def i(index):
    """Shorthand for constructing :class:`IterationIndex` objects"""
    return IterationIndex(index)

class Map(object):
    """OP2 map, a relation between two :class:`Set` objects."""

    _globalcount = 0
    _arg_type = Arg

    @validate_type(('iterset', Set, SetTypeError), ('dataset', Set, SetTypeError), \
            ('dim', int, DimTypeError), ('name', str, NameTypeError))
    def __init__(self, iterset, dataset, dim, values=None, name=None):
        self._iterset = iterset
        self._dataset = dataset
        self._dim = dim
        self._values = verify_reshape(values, np.int32, (iterset.size, dim), \
                                      allow_none=True)
        self._name = name or "map_%d" % Map._globalcount
        self._lib_handle = None
        Map._globalcount += 1

    @validate_type(('index', (int, IterationIndex), IndexTypeError))
    def __call__(self, index):
        if isinstance(index, int) and not (0 <= index < self._dim):
            raise IndexValueError("Index must be in interval [0,%d]" % (self._dim-1))
        if isinstance(index, IterationIndex) and index.index not in [0, 1]:
            raise IndexValueError("IterationIndex must be in interval [0,1]")
        return self._arg_type(map=self, idx=index)

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

IdentityMap = Map(Set(0), Set(0), 1, [], 'identity')

class Sparsity(object):
    """OP2 Sparsity, a matrix structure derived from the union of the outer product of pairs of :class:`Map` objects."""

    _globalcount = 0

    @validate_type(('rmaps', (Map, tuple), MapTypeError), \
                   ('cmaps', (Map, tuple), MapTypeError), \
                   ('dims', (int, tuple), TypeError))
    def __init__(self, rmaps, cmaps, dims, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"

        self._rmaps = as_tuple(rmaps, Map)
        self._cmaps = as_tuple(cmaps, Map)
        assert len(self._rmaps) == len(self._cmaps), \
            "Must pass equal number of row and column maps"
        self._dims = as_tuple(dims, int, 2)
        self._name = name or "global_%d" % Sparsity._globalcount
        self._lib_handle = None
        Sparsity._globalcount += 1

    @property
    def _nmaps(self):
        return len(self._rmaps)

    @property
    def rmaps(self):
        return self._rmaps

    @property
    def cmaps(self):
        return self._cmaps

    @property
    def dims(self):
        return self._dims

    @property
    def name(self):
        return self._name

    def __str__(self):
        return "OP2 Sparsity: rmaps %s, cmaps %s, dims %s, name %s" % \
               (self._rmaps, self._cmaps, self._dims, self._name)

    def __repr__(self):
        return "Sparsity(%s,%s,%s,%s)" % \
               (self._rmaps, self._cmaps, self._dims, self._name)

class Mat(DataCarrier):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    _globalcount = 0
    _modes = [WRITE, INC]
    _arg_type = Arg

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
        # FIXME: do argument checking
        return self._arg_type(data=self, map=path_maps, access=access, idx=path_idxs)

    @property
    def dims(self):
        """A pair of integers giving the number of matrix rows and columns for each member of the row :class:`Set`  and column :class:`Set` respectively. This corresponds to the ``dim`` member of a :class:`Dat`. Note that ``dims`` is actually specified at the :class:`Sparsity` level and inherited by the ``Mat``."""
        return self._sparsity._dims

    @property
    def sparsity(self):
        """:class:`Sparsity` on which the ``Mat`` is defined."""
        return self._sparsity

    @property
    def values(self):
        """A numpy array of matrix values."""
        return self.c_handle.values

    @property
    def dtype(self):
        """Data type."""
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

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", "%s")' % (self._code, self._name)
