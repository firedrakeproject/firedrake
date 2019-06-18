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

"""Base classes for OP2 objects, containing metadata and runtime data
information which is backend independent. Individual runtime backends should
subclass these as required to implement backend-specific features.
"""
import abc

from enum import IntEnum
from collections import defaultdict
import itertools
import numpy as np
import ctypes
import numbers
import operator
import types
from hashlib import md5

from pyop2.datatypes import IntType, as_cstr, dtype_limits, ScalarType
from pyop2.configuration import configuration
from pyop2.caching import Cached, ObjectCached
from pyop2.exceptions import *
from pyop2.utils import *
from pyop2.mpi import MPI, collective, dup_comm
from pyop2.profiling import timed_region
from pyop2.sparsity import build_sparsity
from pyop2.version import __version__ as version

from coffee.base import Node
from coffee.visitors import EstimateFlops
from functools import reduce

import loopy


def _make_object(name, *args, **kwargs):
    from pyop2 import sequential
    return getattr(sequential, name)(*args, **kwargs)


# Data API

class Access(IntEnum):
    READ = 1
    WRITE = 2
    RW = 3
    INC = 4
    MIN = 5
    MAX = 6


READ = Access.READ
"""The :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed read-only."""

WRITE = Access.WRITE
"""The  :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed write-only,
and OP2 is not required to handle write conflicts."""

RW = Access.RW
"""The  :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed for reading
and writing, and OP2 is not required to handle write conflicts."""

INC = Access.INC
"""The kernel computes increments to be summed onto a :class:`Global`,
:class:`Dat`, or :class:`Mat`. OP2 is responsible for managing the write
conflicts caused."""

MIN = Access.MIN
"""The kernel contributes to a reduction into a :class:`Global` using a ``min``
operation. OP2 is responsible for reducing over the different kernel
invocations."""

MAX = Access.MAX
"""The kernel contributes to a reduction into a :class:`Global` using a ``max``
operation. OP2 is responsible for reducing over the different kernel
invocations."""

# Data API


class Arg(object):

    """An argument to a :func:`pyop2.op2.par_loop`.

    .. warning ::
        User code should not directly instantiate :class:`Arg`.
        Instead, use the call syntax on the :class:`DataCarrier`.
    """

    def __init__(self, data=None, map=None, access=None, lgmaps=None, unroll_map=False):
        """
        :param data: A data-carrying object, either :class:`Dat` or class:`Mat`
        :param map:  A :class:`Map` to access this :class:`Arg` or the default
                     if the identity map is to be used.
        :param access: An access descriptor of type :class:`Access`
        :param lgmaps: For :class:`Mat` objects, a 2-tuple of local to
            global maps used during assembly.

        Checks that:

        1. the maps used are initialized i.e. have mapping data associated, and
        2. the to Set of the map used to access it matches the Set it is
           defined on.

        A :class:`MapValueError` is raised if these conditions are not met."""
        self.data = data
        self._map = map
        if map is None:
            self.map_tuple = ()
        elif isinstance(map, Map):
            self.map_tuple = (map, )
        else:
            self.map_tuple = tuple(map)
        self._access = access

        self.unroll_map = unroll_map
        self.lgmaps = None
        if self._is_mat and lgmaps is not None:
            self.lgmaps = as_tuple(lgmaps)
        else:
            if lgmaps is not None:
                raise ValueError("Local to global maps only for matrices")

        # Check arguments for consistency
        if configuration["type_check"] and not (self._is_global or map is None):
            for j, m in enumerate(map):
                if m.iterset.total_size > 0 and len(m.values_with_halo) == 0:
                    raise MapValueError("%s is not initialized." % map)
                if self._is_mat and m.toset != data.sparsity.dsets[j].set:
                    raise MapValueError(
                        "To set of %s doesn't match the set of %s." % (map, data))
            if self._is_dat and map.toset != data.dataset.set:
                raise MapValueError(
                    "To set of %s doesn't match the set of %s." % (map, data))

    @cached_property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @cached_property
    def _argtypes_(self):
        return self.data._argtypes_

    @cached_property
    def _wrapper_cache_key_(self):
        if self.map is not None:
            map_ = tuple(None if m is None else m._wrapper_cache_key_ for m in self.map)
        else:
            map_ = self.map
        return (type(self), self.access, self.data._wrapper_cache_key_, map_, self.unroll_map)

    @property
    def _key(self):
        return (self.data, self._map, self._access)

    def __eq__(self, other):
        r""":class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return self._key == other._key

    def __ne__(self, other):
        r""":class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return not self.__eq__(other)

    def __str__(self):
        return "OP2 Arg: dat %s, map %s, access %s" % \
            (self.data, self._map, self._access)

    def __repr__(self):
        return "Arg(%r, %r, %r)" % \
            (self.data, self._map, self._access)

    def __iter__(self):
        for arg in self.split:
            yield arg

    @cached_property
    def split(self):
        """Split a mixed argument into a tuple of constituent arguments."""
        if self._is_mixed_dat:
            return tuple(_make_object('Arg', d, m, self._access)
                         for d, m in zip(self.data, self._map))
        elif self._is_mixed_mat:
            rows, cols = self.data.sparsity.shape
            mr, mc = self.map
            return tuple(_make_object('Arg', self.data[i, j], (mr.split[i], mc.split[j]),
                                      self._access)
                         for i in range(rows) for j in range(cols))
        else:
            return (self,)

    @cached_property
    def name(self):
        """The generated argument name."""
        return "arg%d" % self.position

    @cached_property
    def ctype(self):
        """String representing the C type of the data in this ``Arg``."""
        return self.data.ctype

    @cached_property
    def dtype(self):
        """Numpy datatype of this Arg"""
        return self.data.dtype

    @cached_property
    def map(self):
        """The :class:`Map` via which the data is to be accessed."""
        return self._map

    @cached_property
    def access(self):
        """Access descriptor. One of the constants of type :class:`Access`"""
        return self._access

    @cached_property
    def _is_dat_view(self):
        return isinstance(self.data, DatView)

    @cached_property
    def _is_mat(self):
        return isinstance(self.data, Mat)

    @cached_property
    def _is_mixed_mat(self):
        return self._is_mat and self.data.sparsity.shape > (1, 1)

    @cached_property
    def _is_global(self):
        return isinstance(self.data, Global)

    @cached_property
    def _is_global_reduction(self):
        return self._is_global and self._access in [INC, MIN, MAX]

    @cached_property
    def _is_dat(self):
        return isinstance(self.data, Dat)

    @cached_property
    def _is_mixed_dat(self):
        return isinstance(self.data, MixedDat)

    @cached_property
    def _is_mixed(self):
        return self._is_mixed_dat or self._is_mixed_mat

    @cached_property
    def _is_direct(self):
        return isinstance(self.data, Dat) and self.map is None

    @cached_property
    def _is_indirect(self):
        return isinstance(self.data, Dat) and self.map is not None

    @collective
    def global_to_local_begin(self):
        """Begin halo exchange for the argument if a halo update is required.
        Doing halo exchanges only makes sense for :class:`Dat` objects.
        """
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access is not WRITE:
            self.data.global_to_local_begin(self.access)

    @collective
    def global_to_local_end(self):
        """Finish halo exchange for the argument if a halo update is required.
        Doing halo exchanges only makes sense for :class:`Dat` objects.
        """
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access is not WRITE:
            self.data.global_to_local_end(self.access)

    @collective
    def local_to_global_begin(self):
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access in {INC, MIN, MAX}:
            self.data.local_to_global_begin(self.access)

    @collective
    def local_to_global_end(self):
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access in {INC, MIN, MAX}:
            self.data.local_to_global_end(self.access)

    @collective
    def reduction_begin(self, comm):
        """Begin reduction for the argument if its access is INC, MIN, or MAX.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not READ:
            if self.access is INC:
                op = MPI.SUM
            elif self.access is MIN:
                op = MPI.MIN
            elif self.access is MAX:
                op = MPI.MAX
            if MPI.VERSION >= 3:
                self._reduction_req = comm.Iallreduce(self.data._data, self.data._buf, op=op)
            else:
                comm.Allreduce(self.data._data, self.data._buf, op=op)

    @collective
    def reduction_end(self, comm):
        """End reduction for the argument if it is in flight.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not READ:
            if MPI.VERSION >= 3:
                self._reduction_req.Wait()
                self._reduction_req = None
            self.data._data[:] = self.data._buf[:]


class Set(object):

    """OP2 set.

    :param size: The size of the set.
    :type size: integer or list of four integers.
    :param string name: The name of the set (optional).
    :param halo: An exisiting halo to use (optional).

    When the set is employed as an iteration space in a
    :func:`pyop2.op2.par_loop`, the extent of any local iteration space within
    each set entry is indicated in brackets. See the example in
    :func:`pyop2.op2.par_loop` for more details.

    The size of the set can either be an integer, or a list of four
    integers.  The latter case is used for running in parallel where
    we distinguish between:

      - `CORE` (owned and not touching halo)
      - `OWNED` (owned, touching halo)
      - `EXECUTE HALO` (not owned, but executed over redundantly)
      - `NON EXECUTE HALO` (not owned, read when executing in the execute halo)

    If a single integer is passed, we assume that we're running in
    serial and there is no distinction.

    The division of set elements is: ::

        [0, CORE)
        [CORE, OWNED)
        [OWNED, GHOST)

    Halo send/receive data is stored on sets in a :class:`Halo`.
    """

    _CORE_SIZE = 0
    _OWNED_SIZE = 1
    _GHOST_SIZE = 2

    _extruded = False

    _kernel_args_ = ()
    _argtypes_ = ()

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), )

    @validate_type(('size', (numbers.Integral, tuple, list, np.ndarray), SizeTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, size, name=None, halo=None, comm=None):
        self.comm = dup_comm(comm)
        if isinstance(size, numbers.Integral):
            size = [size] * 3
        size = as_tuple(size, numbers.Integral, 3)
        assert size[Set._CORE_SIZE] <= size[Set._OWNED_SIZE] <= \
            size[Set._GHOST_SIZE], "Set received invalid sizes: %s" % size
        self._sizes = size
        self._name = name or "set_#x%x" % id(self)
        self._halo = halo
        self._partition_size = 1024
        # A cache of objects built on top of this set
        self._cache = {}

    @cached_property
    def core_size(self):
        """Core set size.  Owned elements not touching halo elements."""
        return self._sizes[Set._CORE_SIZE]

    @cached_property
    def size(self):
        """Set size, owned elements."""
        return self._sizes[Set._OWNED_SIZE]

    @cached_property
    def total_size(self):
        """Set size including ghost elements.
        """
        return self._sizes[Set._GHOST_SIZE]

    @cached_property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return self._sizes

    @cached_property
    def core_part(self):
        return SetPartition(self, 0, self.core_size)

    @cached_property
    def owned_part(self):
        return SetPartition(self, self.core_size, self.size - self.core_size)

    @cached_property
    def name(self):
        """User-defined label"""
        return self._name

    @cached_property
    def halo(self):
        """:class:`Halo` associated with this Set"""
        return self._halo

    @property
    def partition_size(self):
        """Default partition size"""
        return self._partition_size

    @partition_size.setter
    def partition_size(self, partition_value):
        """Set the partition size"""
        self._partition_size = partition_value

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __getitem__(self, idx):
        """Allow indexing to return self"""
        assert idx == 0
        return self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self.size)

    def __repr__(self):
        return "Set(%r, %r)" % (self._sizes, self._name)

    def __call__(self, *indices):
        """Build a :class:`Subset` from this :class:`Set`

        :arg indices: The elements of this :class:`Set` from which the
                      :class:`Subset` should be formed.

        """
        if len(indices) == 1:
            indices = indices[0]
            if np.isscalar(indices):
                indices = [indices]
        return _make_object('Subset', self, indices)

    def __contains__(self, dset):
        """Indicate whether a given DataSet is compatible with this Set."""
        if isinstance(dset, DataSet):
            return dset.set is self
        else:
            return False

    def __pow__(self, e):
        """Derive a :class:`DataSet` with dimension ``e``"""
        return _make_object('DataSet', self, dim=e)

    @cached_property
    def layers(self):
        """Return None (not an :class:`ExtrudedSet`)."""
        return None


class GlobalSet(Set):

    _extruded = False

    """A proxy set allowing a :class:`Global` to be used in place of a
    :class:`Dat` where appropriate."""

    _kernel_args_ = ()
    _argtypes_ = ()

    def __init__(self, comm=None):
        self.comm = dup_comm(comm)
        self._cache = {}

    @cached_property
    def core_size(self):
        return 0

    @cached_property
    def size(self):
        return 1 if self.comm.rank == 0 else 0

    @cached_property
    def total_size(self):
        """Total set size, including halo elements."""
        return 1 if self.comm.rank == 0 else 0

    @cached_property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return (self.core_size, self.size, self.total_size)

    @cached_property
    def name(self):
        """User-defined label"""
        return "GlobalSet"

    @cached_property
    def halo(self):
        """:class:`Halo` associated with this Set"""
        return None

    @property
    def partition_size(self):
        """Default partition size"""
        return None

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __getitem__(self, idx):
        """Allow indexing to return self"""
        assert idx == 0
        return self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 GlobalSet"

    def __repr__(self):
        return "GlobalSet()"

    def __eq__(self, other):
        # Currently all GlobalSets compare equal.
        return isinstance(other, GlobalSet)

    def __hash__(self):
        # Currently all GlobalSets compare equal.
        return hash(type(self))


class ExtrudedSet(Set):

    """OP2 ExtrudedSet.

    :param parent: The parent :class:`Set` to build this :class:`ExtrudedSet` on top of
    :type parent: a :class:`Set`.
    :param layers: The number of layers in this :class:`ExtrudedSet`.
    :type layers: an integer, indicating the number of layers for every entity,
        or an array of shape (parent.total_size, 2) giving the start
        and one past the stop layer for every entity.  An entry
        ``a, b = layers[e, ...]`` means that the layers for entity
        ``e`` run over :math:`[a, b)`.

    The number of layers indicates the number of time the base set is
    extruded in the direction of the :class:`ExtrudedSet`.  As a
    result, there are ``layers-1`` extruded "cells" in an extruded set.
    """

    @validate_type(('parent', Set, TypeError))
    def __init__(self, parent, layers):
        self._parent = parent
        try:
            layers = verify_reshape(layers, IntType, (parent.total_size, 2))
            self.constant_layers = False
            if layers.min() < 0:
                raise SizeTypeError("Bottom of layers must be >= 0")
            if any(layers[:, 1] - layers[:, 0] < 1):
                raise SizeTypeError("Number of layers must be >= 0")
        except DataValueError:
            # Legacy, integer
            layers = np.asarray(layers, dtype=IntType)
            if layers.shape:
                raise SizeTypeError("Specifying layers per entity, but provided %s, needed (%d, 2)",
                                    layers.shape, parent.total_size)
            if layers < 2:
                raise SizeTypeError("Need at least two layers, not %d", layers)
            layers = np.asarray([[0, layers]], dtype=IntType)
            self.constant_layers = True

        self._layers = layers
        self._extruded = True

    @cached_property
    def _kernel_args_(self):
        return (self.layers_array.ctypes.data, )

    @cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @cached_property
    def _wrapper_cache_key_(self):
        return self.parent._wrapper_cache_key_ + (self.constant_layers, )

    def __getattr__(self, name):
        """Returns a :class:`Set` specific attribute."""
        value = getattr(self._parent, name)
        setattr(self, name, value)
        return value

    def __contains__(self, set):
        return set is self.parent

    def __str__(self):
        return "OP2 ExtrudedSet: %s with size %s (%s layers)" % \
            (self._name, self.size, self._layers)

    def __repr__(self):
        return "ExtrudedSet(%r, %r)" % (self._parent, self._layers)

    @cached_property
    def parent(self):
        return self._parent

    @cached_property
    def layers(self):
        """The layers of this extruded set."""
        if self.constant_layers:
            # Backwards compat
            return self.layers_array[0, 1]
        else:
            raise ValueError("No single layer, use layers_array attribute")

    @cached_property
    def layers_array(self):
        return self._layers


class Subset(ExtrudedSet):

    """OP2 subset.

    :param superset: The superset of the subset.
    :type superset: a :class:`Set` or a :class:`Subset`.
    :param indices: Elements of the superset that form the
        subset. Duplicate values are removed when constructing the subset.
    :type indices: a list of integers, or a numpy array.
    """
    @validate_type(('superset', Set, TypeError),
                   ('indices', (list, tuple, np.ndarray), TypeError))
    def __init__(self, superset, indices):
        # sort and remove duplicates
        indices = np.unique(indices)
        if isinstance(superset, Subset):
            # Unroll indices to point to those in the parent
            indices = superset.indices[indices]
            superset = superset.superset
        assert type(superset) is Set or type(superset) is ExtrudedSet, \
            'Subset construction failed, should not happen'

        self._superset = superset
        self._indices = verify_reshape(indices, IntType, (len(indices),))

        if len(self._indices) > 0 and (self._indices[0] < 0 or self._indices[-1] >= self._superset.total_size):
            raise SubsetIndexOutOfBounds(
                'Out of bounds indices in Subset construction: [%d, %d) not [0, %d)' %
                (self._indices[0], self._indices[-1], self._superset.total_size))

        self._sizes = ((self._indices < superset.core_size).sum(),
                       (self._indices < superset.size).sum(),
                       len(self._indices))
        self._extruded = superset._extruded

    @cached_property
    def _kernel_args_(self):
        return self._superset._kernel_args_ + (self._indices.ctypes.data, )

    @cached_property
    def _argtypes_(self):
        return self._superset._argtypes_ + (ctypes.c_voidp, )

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a :class:`Set` specific attribute."""
        value = getattr(self._superset, name)
        setattr(self, name, value)
        return value

    def __pow__(self, e):
        """Derive a :class:`DataSet` with dimension ``e``"""
        raise NotImplementedError("Deriving a DataSet from a Subset is unsupported")

    def __str__(self):
        return "OP2 Subset: %s with sizes %s" % \
            (self._name, self._sizes)

    def __repr__(self):
        return "Subset(%r, %r)" % (self._superset, self._indices)

    def __call__(self, *indices):
        """Build a :class:`Subset` from this :class:`Subset`

        :arg indices: The elements of this :class:`Subset` from which the
                      :class:`Subset` should be formed.

        """
        if len(indices) == 1:
            indices = indices[0]
            if np.isscalar(indices):
                indices = [indices]
        return _make_object('Subset', self, indices)

    @cached_property
    def superset(self):
        """Returns the superset Set"""
        return self._superset

    @cached_property
    def indices(self):
        """Returns the indices pointing in the superset."""
        return self._indices

    @cached_property
    def layers_array(self):
        if self._superset.constant_layers:
            return self._superset.layers_array
        else:
            return self._superset.layers_array[self.indices, ...]


class SetPartition(object):
    def __init__(self, set, offset, size):
        self.set = set
        self.offset = offset
        self.size = size


class MixedSet(Set, ObjectCached):
    r"""A container for a bag of :class:`Set`\s."""

    def __init__(self, sets):
        r""":param iterable sets: Iterable of :class:`Set`\s or :class:`ExtrudedSet`\s"""
        if self._initialized:
            return
        self._sets = sets
        assert all(s is None or isinstance(s, GlobalSet) or ((s.layers == self._sets[0].layers).all() if s.layers is not None else True) for s in sets), \
            "All components of a MixedSet must have the same number of layers."
        # TODO: do all sets need the same communicator?
        self.comm = reduce(lambda a, b: a or b, map(lambda s: s if s is None else s.comm, sets))
        self._initialized = True

    @cached_property
    def _kernel_args_(self):
        raise NotImplementedError

    @cached_property
    def _argtypes_(self):
        raise NotImplementedError

    @cached_property
    def _wrapper_cache_key_(self):
        raise NotImplementedError

    @classmethod
    def _process_args(cls, sets, **kwargs):
        sets = [s for s in sets]
        try:
            sets = as_tuple(sets, ExtrudedSet)
        except TypeError:
            sets = as_tuple(sets, (Set, type(None)))
        cache = sets[0]
        return (cache, ) + (sets, ), kwargs

    @classmethod
    def _cache_key(cls, sets, **kwargs):
        return sets

    def __getitem__(self, idx):
        """Return :class:`Set` with index ``idx`` or a given slice of sets."""
        return self._sets[idx]

    @cached_property
    def split(self):
        r"""The underlying tuple of :class:`Set`\s."""
        return self._sets

    @cached_property
    def core_size(self):
        """Core set size. Owned elements not touching halo elements."""
        return sum(s.core_size for s in self._sets)

    @cached_property
    def size(self):
        """Set size, owned elements."""
        return sum(0 if s is None else s.size for s in self._sets)

    @cached_property
    def total_size(self):
        """Total set size, including halo elements."""
        return sum(s.total_size for s in self._sets)

    @cached_property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return (self.core_size, self.size, self.total_size)

    @cached_property
    def name(self):
        """User-defined labels."""
        return tuple(s.name for s in self._sets)

    @cached_property
    def halo(self):
        r""":class:`Halo`\s associated with these :class:`Set`\s."""
        halos = tuple(s.halo for s in self._sets)
        return halos if any(halos) else None

    @cached_property
    def _extruded(self):
        return isinstance(self._sets[0], ExtrudedSet)

    @cached_property
    def layers(self):
        """Numbers of layers in the extruded mesh (or None if this MixedSet is not extruded)."""
        return self._sets[0].layers

    def __iter__(self):
        r"""Yield all :class:`Set`\s when iterated over."""
        for s in self._sets:
            yield s

    def __len__(self):
        """Return number of contained :class:`Set`s."""
        return len(self._sets)

    def __pow__(self, e):
        """Derive a :class:`MixedDataSet` with dimensions ``e``"""
        return _make_object('MixedDataSet', self._sets, e)

    def __str__(self):
        return "OP2 MixedSet composed of Sets: %s" % (self._sets,)

    def __repr__(self):
        return "MixedSet(%r)" % (self._sets,)

    def __eq__(self, other):
        return type(self) == type(other) and self._sets == other._sets


class DataSet(ObjectCached):
    """PyOP2 Data Set

    Set used in the op2.Dat structures to specify the dimension of the data.
    """

    @validate_type(('iter_set', Set, SetTypeError),
                   ('dim', (numbers.Integral, tuple, list), DimTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, iter_set, dim=1, name=None):
        if isinstance(iter_set, ExtrudedSet):
            raise NotImplementedError("Not allowed!")
        if self._initialized:
            return
        if isinstance(iter_set, Subset):
            raise NotImplementedError("Deriving a DataSet from a Subset is unsupported")
        self._set = iter_set
        self._dim = as_tuple(dim, numbers.Integral)
        self._cdim = np.prod(self._dim).item()
        self._name = name or "dset_#x%x" % id(self)
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (args[0], ) + args, kwargs

    @classmethod
    def _cache_key(cls, iter_set, dim=1, name=None):
        return (iter_set, as_tuple(dim, numbers.Integral))

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dim, self._set._wrapper_cache_key_)

    def __getstate__(self):
        """Extract state to pickle."""
        return self.__dict__

    def __setstate__(self, d):
        """Restore from pickled state."""
        self.__dict__.update(d)

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a Set specific attribute."""
        value = getattr(self.set, name)
        setattr(self, name, value)
        return value

    def __getitem__(self, idx):
        """Allow index to return self"""
        assert idx == 0
        return self

    @cached_property
    def dim(self):
        """The shape tuple of the values for each element of the set."""
        return self._dim

    @cached_property
    def cdim(self):
        """The scalar number of values for each member of the set. This is
        the product of the dim tuple."""
        return self._cdim

    @cached_property
    def name(self):
        """Returns the name of the data set."""
        return self._name

    @cached_property
    def set(self):
        """Returns the parent set of the data set."""
        return self._set

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 DataSet: %s on set %s, with dim %s" % \
            (self._name, self._set, self._dim)

    def __repr__(self):
        return "DataSet(%r, %r, %r)" % (self._set, self._dim, self._name)

    def __contains__(self, dat):
        """Indicate whether a given Dat is compatible with this DataSet."""
        return dat.dataset == self


class GlobalDataSet(DataSet):
    """A proxy :class:`DataSet` for use in a :class:`Sparsity` where the
    matrix has :class:`Global` rows or columns."""

    def __init__(self, global_):
        """
        :param global_: The :class:`Global` on which this object is based."""

        self._global = global_
        self._globalset = GlobalSet(comm=self.comm)

    @classmethod
    def _cache_key(cls, *args):
        return None

    @cached_property
    def dim(self):
        """The shape tuple of the values for each element of the set."""
        return self._global._dim

    @cached_property
    def cdim(self):
        """The scalar number of values for each member of the set. This is
        the product of the dim tuple."""
        return self._global._cdim

    @cached_property
    def name(self):
        """Returns the name of the data set."""
        return self._global._name

    @cached_property
    def comm(self):
        """Return the communicator on which the set is defined."""
        return self._global.comm

    @cached_property
    def set(self):
        """Returns the parent set of the data set."""
        return self._globalset

    @cached_property
    def size(self):
        """The number of local entries in the Dataset (1 on rank 0)"""
        return 1 if MPI.comm.rank == 0 else 0

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 GlobalDataSet: %s on Global %s" % \
            (self._name, self._global)

    def __repr__(self):
        return "GlobalDataSet(%r)" % (self._global)


class MixedDataSet(DataSet, ObjectCached):
    r"""A container for a bag of :class:`DataSet`\s.

    Initialized either from a :class:`MixedSet` and an iterable or iterator of
    ``dims`` of corresponding length ::

        mdset = op2.MixedDataSet(mset, [dim1, ..., dimN])

    or from a tuple of :class:`Set`\s and an iterable of ``dims`` of
    corresponding length ::

        mdset = op2.MixedDataSet([set1, ..., setN], [dim1, ..., dimN])

    If all ``dims`` are to be the same, they can also be given as an
    :class:`int` for either of above invocations ::

        mdset = op2.MixedDataSet(mset, dim)
        mdset = op2.MixedDataSet([set1, ..., setN], dim)

    Initialized from a :class:`MixedSet` without explicitly specifying ``dims``
    they default to 1 ::

        mdset = op2.MixedDataSet(mset)

    Initialized from an iterable or iterator of :class:`DataSet`\s and/or
    :class:`Set`\s, where :class:`Set`\s are implicitly upcast to
    :class:`DataSet`\s of dim 1 ::

        mdset = op2.MixedDataSet([dset1, ..., dsetN])
    """

    def __init__(self, arg, dims=None):
        r"""
        :param arg:  a :class:`MixedSet` or an iterable or a generator
                     expression of :class:`Set`\s or :class:`DataSet`\s or a
                     mixture of both
        :param dims: `None` (the default) or an :class:`int` or an iterable or
                     generator expression of :class:`int`\s, which **must** be
                     of same length as `arg`

        .. Warning ::
            When using generator expressions for ``arg`` or ``dims``, these
            **must** terminate or else will cause an infinite loop.
        """
        if self._initialized:
            return
        self._dsets = arg
        self._initialized = True

    @classmethod
    def _process_args(cls, arg, dims=None):
        # If the second argument is not None it is expect to be a scalar dim
        # or an iterable of dims and the first is expected to be a MixedSet or
        # an iterable of Sets
        if dims is not None:
            # If arg is a MixedSet, get its Sets tuple
            sets = arg.split if isinstance(arg, MixedSet) else tuple(arg)
            # If dims is a scalar, turn it into a tuple of right length
            dims = (dims,) * len(sets) if isinstance(dims, int) else tuple(dims)
            if len(sets) != len(dims):
                raise ValueError("Got MixedSet of %d Sets but %s dims" %
                                 (len(sets), len(dims)))
            dsets = tuple(s ** d for s, d in zip(sets, dims))
        # Otherwise expect the first argument to be an iterable of Sets and/or
        # DataSets and upcast Sets to DataSets as necessary
        else:
            arg = [s if isinstance(s, DataSet) else s ** 1 for s in arg]
            dsets = as_tuple(arg, type=DataSet)

        return (dsets[0].set, ) + (dsets, ), {}

    @classmethod
    def _cache_key(cls, arg, dims=None):
        return arg

    @cached_property
    def _wrapper_cache_key_(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Return :class:`DataSet` with index ``idx`` or a given slice of datasets."""
        return self._dsets[idx]

    @cached_property
    def split(self):
        r"""The underlying tuple of :class:`DataSet`\s."""
        return self._dsets

    @cached_property
    def dim(self):
        """The shape tuple of the values for each element of the sets."""
        return tuple(s.dim for s in self._dsets)

    @cached_property
    def cdim(self):
        """The sum of the scalar number of values for each member of the sets.
        This is the sum of products of the dim tuples."""
        return sum(s.cdim for s in self._dsets)

    @cached_property
    def name(self):
        """Returns the name of the data sets."""
        return tuple(s.name for s in self._dsets)

    @cached_property
    def set(self):
        """Returns the :class:`MixedSet` this :class:`MixedDataSet` is
        defined on."""
        return MixedSet(s.set for s in self._dsets)

    def __iter__(self):
        r"""Yield all :class:`DataSet`\s when iterated over."""
        for ds in self._dsets:
            yield ds

    def __len__(self):
        """Return number of contained :class:`DataSet`s."""
        return len(self._dsets)

    def __str__(self):
        return "OP2 MixedDataSet composed of DataSets: %s" % (self._dsets,)

    def __repr__(self):
        return "MixedDataSet(%r)" % (self._dsets,)


class Halo(object, metaclass=abc.ABCMeta):

    """A description of a halo associated with a :class:`Set`.

    The halo object describes which :class:`Set` elements are sent
    where, and which :class:`Set` elements are received from where.
    """

    @abc.abstractproperty
    def comm(self):
        """The MPI communicator for this halo."""
        pass

    @abc.abstractproperty
    def local_to_global_numbering(self):
        """The mapping from process-local to process-global numbers for this halo."""
        pass

    @abc.abstractmethod
    def global_to_local_begin(self, dat, insert_mode):
        """Begin an exchange from global (assembled) to local (ghosted) representation.

        :arg dat: The :class:`Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass

    @abc.abstractmethod
    def global_to_local_end(self, dat, insert_mode):
        """Finish an exchange from global (assembled) to local (ghosted) representation.

        :arg dat: The :class:`Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass

    @abc.abstractmethod
    def local_to_global_begin(self, dat, insert_mode):
        """Begin an exchange from local (ghosted) to global (assembled) representation.

        :arg dat: The :class:`Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass

    @abc.abstractmethod
    def local_to_global_end(self, dat, insert_mode):
        """Finish an exchange from local (ghosted) to global (assembled) representation.

        :arg dat: The :class:`Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass


class DataCarrier(object):

    """Abstract base class for OP2 data.

    Actual objects will be :class:`DataCarrier` objects of rank 0
    (:class:`Global`), rank 1 (:class:`Dat`), or rank 2
    (:class:`Mat`)"""

    @cached_property
    def dtype(self):
        """The Python type of the data."""
        return self._data.dtype

    @cached_property
    def ctype(self):
        """The c type of the data."""
        return as_cstr(self.dtype)

    @cached_property
    def name(self):
        """User-defined label."""
        return self._name

    @cached_property
    def dim(self):
        """The shape tuple of the values for each element of the object."""
        return self._dim

    @cached_property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self._cdim


class _EmptyDataMixin(object):
    """A mixin for :class:`Dat` and :class:`Global` objects that takes
    care of allocating data on demand if the user has passed nothing
    in.

    Accessing the :attr:`_data` property allocates a zeroed data array
    if it does not already exist.
    """
    def __init__(self, data, dtype, shape):
        if data is None:
            self._dtype = np.dtype(dtype if dtype is not None else np.float64)
        else:
            self._numpy_data = verify_reshape(data, dtype, shape, allow_none=True)
            self._dtype = self._data.dtype

    @cached_property
    def _data(self):
        """Return the user-provided data buffer, or a zeroed buffer of
        the correct size if none was provided."""
        if not self._is_allocated:
            self._numpy_data = np.zeros(self.shape, dtype=self._dtype)
        return self._numpy_data

    @property
    def _is_allocated(self):
        """Return True if the data buffer has been allocated."""
        return hasattr(self, '_numpy_data')


class Dat(DataCarrier, _EmptyDataMixin):
    """OP2 vector data. A :class:`Dat` holds values on every element of a
    :class:`DataSet`.

    If a :class:`Set` is passed as the ``dataset`` argument, rather
    than a :class:`DataSet`, the :class:`Dat` is created with a default
    :class:`DataSet` dimension of 1.

    If a :class:`Dat` is passed as the ``dataset`` argument, a copy is
    returned.

    It is permissible to pass `None` as the `data` argument.  In this
    case, allocation of the data buffer is postponed until it is
    accessed.

    .. note::
        If the data buffer is not passed in, it is implicitly
        initialised to be zero.

    When a :class:`Dat` is passed to :func:`pyop2.op2.par_loop`, the map via
    which indirection occurs and the access descriptor are passed by
    calling the :class:`Dat`. For instance, if a :class:`Dat` named ``D`` is
    to be accessed for reading via a :class:`Map` named ``M``, this is
    accomplished by ::

      D(pyop2.READ, M)

    The :class:`Map` through which indirection occurs can be indexed
    using the index notation described in the documentation for the
    :class:`Map`. Direct access to a Dat is accomplished by
    omitting the path argument.

    :class:`Dat` objects support the pointwise linear algebra operations
    ``+=``, ``*=``, ``-=``, ``/=``, where ``*=`` and ``/=`` also support
    multiplication / division by a scalar.
    """

    @cached_property
    def pack(self):
        from pyop2.codegen.builder import DatPack
        return DatPack

    _modes = [READ, WRITE, RW, INC, MIN, MAX]

    @validate_type(('dataset', (DataCarrier, DataSet, Set), DataSetTypeError),
                   ('name', str, NameTypeError))
    @validate_dtype(('dtype', None, DataTypeError))
    def __init__(self, dataset, data=None, dtype=None, name=None):

        if isinstance(dataset, Dat):
            self.__init__(dataset.dataset, None, dtype=dataset.dtype,
                          name="copy_of_%s" % dataset.name)
            dataset.copy(self)
            return
        if type(dataset) is Set or type(dataset) is ExtrudedSet:
            # If a Set, rather than a dataset is passed in, default to
            # a dataset dimension of 1.
            dataset = dataset ** 1
        self._shape = (dataset.total_size,) + (() if dataset.cdim == 1 else dataset.dim)
        _EmptyDataMixin.__init__(self, data, dtype, self._shape)

        self._dataset = dataset
        self.comm = dataset.comm
        self.halo_valid = True
        self._name = name or "dat_#x%x" % id(self)

    @cached_property
    def _kernel_args_(self):
        return (self._data.ctypes.data, )

    @cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self._dataset._wrapper_cache_key_)

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path=None):
        if configuration["type_check"] and path and path.toset != self.dataset.set:
            raise MapValueError("To Set of Map does not match Set of Dat.")
        return _make_object('Arg', data=self, map=path, access=access)

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise IndexValueError("Can only extract component 0 from %r" % self)
        return self

    @cached_property
    def split(self):
        """Tuple containing only this :class:`Dat`."""
        return (self,)

    @cached_property
    def dataset(self):
        """:class:`DataSet` on which the Dat is defined."""
        return self._dataset

    @cached_property
    def dim(self):
        """The shape of the values for each element of the object."""
        return self.dataset.dim

    @cached_property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self.dataset.cdim

    @property
    @collective
    def data(self):
        """Numpy array containing the data values.

        With this accessor you are claiming that you will modify
        the values you get back.  If you only need to look at the
        values, use :meth:`data_ro` instead.

        This only shows local values, to see the halo values too use
        :meth:`data_with_halos`.

        """
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        self.halo_valid = False
        v = self._data[:self.dataset.size].view()
        v.setflags(write=True)
        return v

    @property
    @collective
    def data_with_halos(self):
        r"""A view of this :class:`Dat`\s data.

        This accessor marks the :class:`Dat` as dirty, see
        :meth:`data` for more details on the semantics.

        With this accessor, you get to see up to date halo values, but
        you should not try and modify them, because they will be
        overwritten by the next halo exchange."""
        self.global_to_local_begin(RW)
        self.global_to_local_end(RW)
        self.halo_valid = False
        v = self._data.view()
        v.setflags(write=True)
        return v

    @property
    @collective
    def data_ro(self):
        """Numpy array containing the data values.  Read-only.

        With this accessor you are not allowed to modify the values
        you get back.  If you need to do so, use :meth:`data` instead.

        This only shows local values, to see the halo values too use
        :meth:`data_ro_with_halos`.

        """
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        v = self._data[:self.dataset.size].view()
        v.setflags(write=False)
        return v

    @property
    @collective
    def data_ro_with_halos(self):
        r"""A view of this :class:`Dat`\s data.

        This accessor does not mark the :class:`Dat` as dirty, and is
        a read only view, see :meth:`data_ro` for more details on the
        semantics.

        With this accessor, you get to see up to date halo values, but
        you should not try and modify them, because they will be
        overwritten by the next halo exchange.

        """
        self.global_to_local_begin(READ)
        self.global_to_local_end(READ)
        v = self._data.view()
        v.setflags(write=False)
        return v

    def save(self, filename):
        """Write the data array to file ``filename`` in NumPy format."""
        np.save(filename, self.data_ro)

    def load(self, filename):
        """Read the data stored in file ``filename`` into a NumPy array
        and store the values in :meth:`_data`.
        """
        # The np.save method appends a .npy extension to the file name
        # if the user has not supplied it. However, np.load does not,
        # so we need to handle this ourselves here.
        if(filename[-4:] != ".npy"):
            filename = filename + ".npy"

        if isinstance(self.data, tuple):
            # MixedDat case
            for d, d_from_file in zip(self.data, np.load(filename)):
                d[:] = d_from_file[:]
        else:
            self.data[:] = np.load(filename)

    @cached_property
    def shape(self):
        return self._shape

    @cached_property
    def dtype(self):
        return self._dtype

    @cached_property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Dat` in bytes. This will be the correct size of the data
        payload, but does not take into account the (presumably small)
        overhead of the object and its metadata.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return self.dtype.itemsize * self.dataset.total_size * self.dataset.cdim

    @collective
    def zero(self, subset=None):
        """Zero the data associated with this :class:`Dat`

        :arg subset: A :class:`Subset` of entries to zero (optional)."""
        if hasattr(self, "_zero_parloops"):
            loops = self._zero_parloops
        else:
            loops = {}
            self._zero_parloops = loops

        iterset = subset or self.dataset.set

        loop = loops.get(iterset, None)

        if loop is None:
            import islpy as isl
            import pymbolic.primitives as p

            inames = isl.make_zero_and_vars(["i"])
            domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
            x = p.Variable("dat")
            i = p.Variable("i")
            insn = loopy.Assignment(x.index(i), 0, within_inames=frozenset(["i"]))
            data = loopy.GlobalArg("dat", dtype=self.dtype, shape=(self.cdim,))
            knl = loopy.make_function([domain], [insn], [data], name="zero")

            knl = _make_object('Kernel', knl, 'zero')
            loop = _make_object('ParLoop', knl,
                                iterset,
                                self(WRITE))
            loops[iterset] = loop
        loop.compute()

    @collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`Dat` into another.

        :arg other: The destination :class:`Dat`
        :arg subset: A :class:`Subset` of elements to copy (optional)"""
        if other is self:
            return
        self._copy_parloop(other, subset=subset).compute()

    @collective
    def _copy_parloop(self, other, subset=None):
        """Create the :class:`ParLoop` implementing copy."""
        if not hasattr(self, '_copy_kernel'):
            import islpy as isl
            import pymbolic.primitives as p
            inames = isl.make_zero_and_vars(["i"])
            domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
            _other = p.Variable("other")
            _self = p.Variable("self")
            i = p.Variable("i")
            insn = loopy.Assignment(_other.index(i), _self.index(i), within_inames=frozenset(["i"]))
            data = [loopy.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,)),
                    loopy.GlobalArg("other", dtype=other.dtype, shape=(other.cdim,))]
            knl = loopy.make_function([domain], [insn], data, name="copy")

            self._copy_kernel = _make_object('Kernel', knl, 'copy')
        return _make_object('ParLoop', self._copy_kernel,
                            subset or self.dataset.set,
                            self(READ), other(WRITE))

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 Dat: %s on (%s) with datatype %s" \
               % (self._name, self._dataset, self.dtype.name)

    def __repr__(self):
        return "Dat(%r, None, %r, %r)" \
               % (self._dataset, self.dtype, self._name)

    def _check_shape(self, other):
        if other.dataset.dim != self.dataset.dim:
            raise ValueError('Mismatched shapes in operands %s and %s',
                             self.dataset.dim, other.dataset.dim)

    def _op_kernel(self, op, globalp, dtype):
        key = (op, globalp, dtype)
        try:
            if not hasattr(self, "_op_kernel_cache"):
                self._op_kernel_cache = {}
            return self._op_kernel_cache[key]
        except KeyError:
            pass
        import islpy as isl
        import pymbolic.primitives as p
        name = "binop_%s" % op.__name__
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        _other = p.Variable("other")
        _self = p.Variable("self")
        _ret = p.Variable("ret")
        i = p.Variable("i")
        lhs = _ret.index(i)
        if globalp:
            rhs = _other.index(0)
            rshape = (1, )
        else:
            rhs = _other.index(i)
            rshape = (self.cdim, )
        insn = loopy.Assignment(lhs, op(_self.index(i), rhs), within_inames=frozenset(["i"]))
        data = [loopy.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,)),
                loopy.GlobalArg("other", dtype=dtype, shape=rshape),
                loopy.GlobalArg("ret", dtype=self.dtype, shape=(self.cdim,))]
        knl = loopy.make_function([domain], [insn], data, name=name)
        return self._op_kernel_cache.setdefault(key, _make_object('Kernel', knl, name))

    def _op(self, other, op):
        ret = _make_object('Dat', self.dataset, None, self.dtype)
        if np.isscalar(other):
            other = _make_object('Global', 1, data=other)
            globalp = True
        else:
            self._check_shape(other)
            globalp = False
        par_loop(self._op_kernel(op, globalp, other.dtype),
                 self.dataset.set, self(READ), other(READ), ret(WRITE))
        return ret

    def _iop_kernel(self, op, globalp, other_is_self, dtype):
        key = (op, globalp, other_is_self, dtype)
        try:
            if not hasattr(self, "_iop_kernel_cache"):
                self._iop_kernel_cache = {}
            return self._iop_kernel_cache[key]
        except KeyError:
            pass
        import islpy as isl
        import pymbolic.primitives as p
        name = "iop_%s" % op.__name__
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        _other = p.Variable("other")
        _self = p.Variable("self")
        i = p.Variable("i")
        lhs = _self.index(i)
        rshape = (self.cdim, )
        if globalp:
            rhs = _other.index(0)
            rshape = (1, )
        elif other_is_self:
            rhs = _self.index(i)
        else:
            rhs = _other.index(i)
        insn = loopy.Assignment(lhs, op(lhs, rhs), within_inames=frozenset(["i"]))
        data = [loopy.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,))]
        if not other_is_self:
            data.append(loopy.GlobalArg("other", dtype=dtype, shape=rshape))
        knl = loopy.make_function([domain], [insn], data, name=name)
        return self._iop_kernel_cache.setdefault(key, _make_object('Kernel', knl, name))

    def _iop(self, other, op):
        globalp = False
        if np.isscalar(other):
            other = _make_object('Global', 1, data=other)
            globalp = True
        elif other is not self:
            self._check_shape(other)
        args = [self(INC)]
        if other is not self:
            args.append(other(READ))
        par_loop(self._iop_kernel(op, globalp, other is self, other.dtype), self.dataset.set, *args)
        return self

    def _inner_kernel(self, dtype):
        try:
            if not hasattr(self, "_inner_kernel_cache"):
                self._inner_kernel_cache = {}
            return self._inner_kernel_cache[dtype]
        except KeyError:
            pass
        import islpy as isl
        import pymbolic.primitives as p
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        _self = p.Variable("self")
        _other = p.Variable("other")
        _ret = p.Variable("ret")
        _conj = p.Variable("conj") if dtype.kind == "c" else lambda x: x
        i = p.Variable("i")
        insn = loopy.Assignment(_ret[0], _ret[0] + _self[i]*_conj(_other[i]),
                                within_inames=frozenset(["i"]))
        data = [loopy.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,)),
                loopy.GlobalArg("other", dtype=dtype, shape=(self.cdim,)),
                loopy.GlobalArg("ret", dtype=self.dtype, shape=(1,))]
        knl = loopy.make_function([domain], [insn], data, name="inner")
        k = _make_object('Kernel', knl, "inner")
        return self._inner_kernel_cache.setdefault(dtype, k)

    def inner(self, other):
        """Compute the l2 inner product of the flattened :class:`Dat`

        :arg other: the other :class:`Dat` to compute the inner
             product against. The complex conjugate of this is taken.

        """
        self._check_shape(other)
        ret = _make_object('Global', 1, data=0, dtype=self.dtype)
        par_loop(self._inner_kernel(other.dtype), self.dataset.set,
                 self(READ), other(READ), ret(INC))
        return ret.data_ro[0]

    @property
    def norm(self):
        """Compute the l2 norm of this :class:`Dat`

        .. note::

           This acts on the flattened data (see also :meth:`inner`)."""
        from math import sqrt
        return sqrt(self.inner(self).real)

    def __pos__(self):
        pos = _make_object('Dat', self)
        return pos

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self + other

    @cached_property
    def _neg_kernel(self):
        # Copy and negate in one go.
        import islpy as isl
        import pymbolic.primitives as p
        name = "neg"
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        lvalue = p.Variable("neg")
        rvalue = p.Variable("self")
        i = p.Variable("i")
        insn = loopy.Assignment(lvalue.index(i), -rvalue.index(i), within_inames=frozenset(["i"]))
        data = [loopy.GlobalArg("neg", dtype=self.dtype, shape=(self.cdim,)),
                loopy.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,))]
        knl = loopy.make_function([domain], [insn], data, name=name)
        return _make_object('Kernel', knl, name)

    def __neg__(self):
        neg = _make_object('Dat', self.dataset, dtype=self.dtype)
        par_loop(self._neg_kernel, self.dataset.set, neg(WRITE), self(READ))
        return neg

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        ret = -self
        ret += other
        return ret

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.truediv)

    __div__ = __truediv__  # Python 2 compatibility

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __itruediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.itruediv)

    __idiv__ = __itruediv__  # Python 2 compatibility

    @collective
    def global_to_local_begin(self, access_mode):
        """Begin a halo exchange from global to ghosted representation.

        :kwarg access_mode: Mode with which the data will subsequently
           be accessed."""
        halo = self.dataset.halo
        if halo is None:
            return
        if not self.halo_valid and access_mode in {READ, RW}:
            halo.global_to_local_begin(self, WRITE)
        elif access_mode in {INC, MIN, MAX}:
            min_, max_ = dtype_limits(self.dtype)
            val = {MAX: min_, MIN: max_, INC: 0}[access_mode]
            self._data[self.dataset.size:] = val
        else:
            # WRITE
            pass

    @collective
    def global_to_local_end(self, access_mode):
        """End a halo exchange from global to ghosted representation.

        :kwarg access_mode: Mode with which the data will subsequently
           be accessed."""
        halo = self.dataset.halo
        if halo is None:
            return
        if not self.halo_valid and access_mode in {READ, RW}:
            halo.global_to_local_end(self, WRITE)
            self.halo_valid = True
        elif access_mode in {INC, MIN, MAX}:
            self.halo_valid = False
        else:
            # WRITE
            pass

    @collective
    def local_to_global_begin(self, insert_mode):
        """Begin a halo exchange from ghosted to global representation.

        :kwarg insert_mode: insertion mode (an access descriptor)"""
        halo = self.dataset.halo
        if halo is None:
            return
        halo.local_to_global_begin(self, insert_mode)

    @collective
    def local_to_global_end(self, insert_mode):
        """End a halo exchange from ghosted to global representation.

        :kwarg insert_mode: insertion mode (an access descriptor)"""
        halo = self.dataset.halo
        if halo is None:
            return
        halo.local_to_global_end(self, insert_mode)
        self.halo_valid = False


class DatView(Dat):
    """An indexed view into a :class:`Dat`.

    This object can be used like a :class:`Dat` but the kernel will
    only see the requested index, rather than the full data.

    :arg dat: The :class:`Dat` to create a view into.
    :arg index: The component to select a view of.
    """
    def __init__(self, dat, index):
        index = as_tuple(index)
        assert len(index) == len(dat.dim)
        for i, d in zip(index, dat.dim):
            if not (0 <= i < d):
                raise IndexValueError("Can't create DatView with index %s for Dat with shape %s" % (index, dat.dim))
        self.index = index
        # Point at underlying data
        super(DatView, self).__init__(dat.dataset,
                                      dat._data,
                                      dtype=dat.dtype,
                                      name="view[%s](%s)" % (index, dat.name))
        self._parent = dat

    @cached_property
    def _kernel_args_(self):
        return self._parent._kernel_args_

    @cached_property
    def _argtypes_(self):
        return self._parent._argtypes_

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.index, self._parent._wrapper_cache_key_)

    @cached_property
    def cdim(self):
        return 1

    @cached_property
    def dim(self):
        return (1, )

    @cached_property
    def shape(self):
        return (self.dataset.total_size, )

    @property
    def data(self):
        full = self._parent.data
        idx = (slice(None), *self.index)
        return full[idx]

    @property
    def data_ro(self):
        full = self._parent.data_ro
        idx = (slice(None), *self.index)
        return full[idx]

    @property
    def data_with_halos(self):
        full = self._parent.data_with_halos
        idx = (slice(None), *self.index)
        return full[idx]

    @property
    def data_ro_with_halos(self):
        full = self._parent.data_ro_with_halos
        idx = (slice(None), *self.index)
        return full[idx]


class MixedDat(Dat):
    r"""A container for a bag of :class:`Dat`\s.

    Initialized either from a :class:`MixedDataSet`, a :class:`MixedSet`, or
    an iterable of :class:`DataSet`\s and/or :class:`Set`\s, where all the
    :class:`Set`\s are implcitly upcast to :class:`DataSet`\s ::

        mdat = op2.MixedDat(mdset)
        mdat = op2.MixedDat([dset1, ..., dsetN])

    or from an iterable of :class:`Dat`\s ::

        mdat = op2.MixedDat([dat1, ..., datN])
    """

    def __init__(self, mdset_or_dats):
        def what(x):
            if isinstance(x, (Global, GlobalDataSet, GlobalSet)):
                return "Global"
            elif isinstance(x, (Dat, DataSet, Set)):
                return "Dat"
            else:
                raise DataSetTypeError("Huh?!")
        if isinstance(mdset_or_dats, MixedDat):
            self._dats = tuple(_make_object(what(d), d) for d in mdset_or_dats)
        else:
            self._dats = tuple(d if isinstance(d, (Dat, Global)) else _make_object(what(d), d) for d in mdset_or_dats)
        if not all(d.dtype == self._dats[0].dtype for d in self._dats):
            raise DataValueError('MixedDat with different dtypes is not supported')
        # TODO: Think about different communicators on dats (c.f. MixedSet)
        self.comm = self._dats[0].comm

    @cached_property
    def _kernel_args_(self):
        return tuple(itertools.chain(*(d._kernel_args_ for d in self)))

    @cached_property
    def _argtypes_(self):
        return tuple(itertools.chain(*(d._argtypes_ for d in self)))

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self),) + tuple(d._wrapper_cache_key_ for d in self)

    def __getitem__(self, idx):
        """Return :class:`Dat` with index ``idx`` or a given slice of Dats."""
        return self._dats[idx]

    @cached_property
    def dtype(self):
        """The NumPy dtype of the data."""
        return self._dats[0].dtype

    @cached_property
    def split(self):
        r"""The underlying tuple of :class:`Dat`\s."""
        return self._dats

    @cached_property
    def dataset(self):
        r""":class:`MixedDataSet`\s this :class:`MixedDat` is defined on."""
        return _make_object('MixedDataSet', tuple(s.dataset for s in self._dats))

    @cached_property
    def _data(self):
        """Return the user-provided data buffer, or a zeroed buffer of
        the correct size if none was provided."""
        return tuple(d._data for d in self)

    @property
    @collective
    def data(self):
        """Numpy arrays containing the data excluding halos."""
        return tuple(s.data for s in self._dats)

    @property
    @collective
    def data_with_halos(self):
        """Numpy arrays containing the data including halos."""
        return tuple(s.data_with_halos for s in self._dats)

    @property
    @collective
    def data_ro(self):
        """Numpy arrays with read-only data excluding halos."""
        return tuple(s.data_ro for s in self._dats)

    @property
    @collective
    def data_ro_with_halos(self):
        """Numpy arrays with read-only data including halos."""
        return tuple(s.data_ro_with_halos for s in self._dats)

    @property
    def halo_valid(self):
        """Does this Dat have up to date halos?"""
        return all(s.halo_valid for s in self)

    @halo_valid.setter
    def halo_valid(self, val):
        """Indictate whether this Dat requires a halo update"""
        for d in self:
            d.halo_valid = val

    @collective
    def global_to_local_begin(self, access_mode):
        for s in self:
            s.global_to_local_begin(access_mode)

    @collective
    def global_to_local_end(self, access_mode):
        for s in self:
            s.global_to_local_end(access_mode)

    @collective
    def local_to_global_begin(self, insert_mode):
        for s in self:
            s.local_to_global_begin(insert_mode)

    @collective
    def local_to_global_end(self, insert_mode):
        for s in self:
            s.local_to_global_end(insert_mode)

    @collective
    def zero(self, subset=None):
        """Zero the data associated with this :class:`MixedDat`.

        :arg subset: optional subset of entries to zero (not implemented)."""
        if subset is not None:
            raise NotImplementedError("Subsets of mixed sets not implemented")
        for d in self._dats:
            d.zero()

    @cached_property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`MixedDat` in bytes. This will be the correct size of the data
        payload, but does not take into account the (presumably small)
        overhead of the object and its metadata.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return np.sum([d.nbytes for d in self._dats])

    @collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`MixedDat` into another.

        :arg other: The destination :class:`MixedDat`
        :arg subset: Subsets are not supported, this must be :class:`None`"""

        if subset is not None:
            raise NotImplementedError("MixedDat.copy with a Subset is not supported")
        for s, o in zip(self, other):
            s.copy(o)

    def __iter__(self):
        r"""Yield all :class:`Dat`\s when iterated over."""
        for d in self._dats:
            yield d

    def __len__(self):
        r"""Return number of contained :class:`Dats`\s."""
        return len(self._dats)

    def __hash__(self):
        return hash(self._dats)

    def __eq__(self, other):
        r""":class:`MixedDat`\s are equal if all their contained :class:`Dat`\s
        are."""
        return type(self) == type(other) and self._dats == other._dats

    def __ne__(self, other):
        r""":class:`MixedDat`\s are equal if all their contained :class:`Dat`\s
        are."""
        return not self.__eq__(other)

    def __str__(self):
        return "OP2 MixedDat composed of Dats: %s" % (self._dats,)

    def __repr__(self):
        return "MixedDat(%r)" % (self._dats,)

    def inner(self, other):
        """Compute the l2 inner product.

        :arg other: the other :class:`MixedDat` to compute the inner product against"""
        ret = 0
        for s, o in zip(self, other):
            ret += s.inner(o)
        return ret

    def _op(self, other, op):
        ret = []
        if np.isscalar(other):
            for s in self:
                ret.append(op(s, other))
        else:
            self._check_shape(other)
            for s, o in zip(self, other):
                ret.append(op(s, o))
        return _make_object('MixedDat', ret)

    def _iop(self, other, op):
        if np.isscalar(other):
            for s in self:
                op(s, other)
        else:
            self._check_shape(other)
            for s, o in zip(self, other):
                op(s, o)
        return self

    def __pos__(self):
        ret = []
        for s in self:
            ret.append(s.__pos__())
        return _make_object('MixedDat', ret)

    def __neg__(self):
        ret = []
        for s in self:
            ret.append(s.__neg__())
        return _make_object('MixedDat', ret)

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self._op(other, operator.add)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self._op(other, operator.mul)

    def __div__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.div)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.idiv)


class Global(DataCarrier, _EmptyDataMixin):

    """OP2 global value.

    When a ``Global`` is passed to a :func:`pyop2.op2.par_loop`, the access
    descriptor is passed by `calling` the ``Global``.  For example, if
    a ``Global`` named ``G`` is to be accessed for reading, this is
    accomplished by::

      G(pyop2.READ)

    It is permissible to pass `None` as the `data` argument.  In this
    case, allocation of the data buffer is postponed until it is
    accessed.

    .. note::
        If the data buffer is not passed in, it is implicitly
        initialised to be zero.
    """

    _modes = [READ, INC, MIN, MAX]

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None, comm=None):
        if isinstance(dim, Global):
            # If g is a Global, Global(g) performs a deep copy. This is for compatibility with Dat.
            self.__init__(dim._dim, None, dtype=dim.dtype,
                          name="copy_of_%s" % dim.name, comm=dim.comm)
            dim.copy(self)
            return
        self._dim = as_tuple(dim, int)
        self._cdim = np.prod(self._dim).item()
        _EmptyDataMixin.__init__(self, data, dtype, self._dim)
        self._buf = np.empty(self.shape, dtype=self.dtype)
        self._name = name or "global_#x%x" % id(self)
        self.comm = comm

    @cached_property
    def _kernel_args_(self):
        return (self._data.ctypes.data, )

    @cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self.shape)

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path=None):
        return _make_object('Arg', data=self, access=access)

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise IndexValueError("Can only extract component 0 from %r" % self)
        return self

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
            % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global(%r, %r, %r, %r)" % (self._dim, self._data,
                                           self._data.dtype, self._name)

    @cached_property
    def dataset(self):
        return _make_object('GlobalDataSet', self)

    @property
    def shape(self):
        return self._dim

    @property
    def data(self):
        """Data array."""
        if len(self._data) == 0:
            raise RuntimeError("Illegal access: No data associated with this Global!")
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_ro(self):
        """Data array."""
        view = self.data.view()
        view.setflags(write=False)
        return view

    @data.setter
    def data(self, value):
        self._data[:] = verify_reshape(value, self.dtype, self.dim)

    @property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Global` in bytes. This will be the correct size of the
        data payload, but does not take into account the overhead of
        the object and its metadata. This renders this method of
        little statistical significance, however it is included to
        make the interface consistent.
        """

        return self.dtype.itemsize * self._cdim

    @collective
    def duplicate(self):
        """Return a deep copy of self."""
        return type(self)(self.dim, data=np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)

    @collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`Global` into another.

        :arg other: The destination :class:`Global`
        :arg subset: A :class:`Subset` of elements to copy (optional)"""

        other.data = np.copy(self.data_ro)

    @collective
    def zero(self):
        self._data[...] = 0

    @collective
    def global_to_local_begin(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @collective
    def global_to_local_end(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @collective
    def local_to_global_begin(self, insert_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @collective
    def local_to_global_end(self, insert_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    def _op(self, other, op):
        ret = type(self)(self.dim, dtype=self.dtype, name=self.name, comm=self.comm)
        if isinstance(other, Global):
            ret.data[:] = op(self.data_ro, other.data_ro)
        else:
            ret.data[:] = op(self.data_ro, other)
        return ret

    def _iop(self, other, op):
        if isinstance(other, Global):
            op(self.data[:], other.data_ro)
        else:
            op(self.data[:], other)
        return self

    def __pos__(self):
        return self.duplicate()

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self + other

    def __neg__(self):
        return type(self)(self.dim, data=-np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        ret = -self
        ret += other
        return ret

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.truediv)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __itruediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.itruediv)

    def inner(self, other):
        assert isinstance(other, Global)
        return np.dot(self.data_ro, np.conj(other.data_ro))


class Map(object):

    """OP2 map, a relation between two :class:`Set` objects.

    Each entry in the ``iterset`` maps to ``arity`` entries in the
    ``toset``. When a map is used in a :func:`pyop2.op2.par_loop`, it is
    possible to use Python index notation to select an individual entry on the
    right hand side of this map. There are three possibilities:

    * No index. All ``arity`` :class:`Dat` entries will be passed to the
      kernel.
    * An integer: ``some_map[n]``. The ``n`` th entry of the
      map result will be passed to the kernel.
    """

    dtype = IntType

    @validate_type(('iterset', Set, SetTypeError), ('toset', Set, SetTypeError),
                   ('arity', numbers.Integral, ArityTypeError), ('name', str, NameTypeError))
    def __init__(self, iterset, toset, arity, values=None, name=None, offset=None):
        self._iterset = iterset
        self._toset = toset
        self.comm = toset.comm
        self._arity = arity
        self._values = verify_reshape(values, IntType,
                                      (iterset.total_size, arity),
                                      allow_none=True)
        self.shape = (iterset.total_size, arity)
        self._name = name or "map_#x%x" % id(self)
        if offset is None or len(offset) == 0:
            self._offset = None
        else:
            self._offset = verify_reshape(offset, IntType, (arity, ))
        # A cache for objects built on top of this map
        self._cache = {}

    @cached_property
    def _kernel_args_(self):
        return (self._values.ctypes.data, )

    @cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.arity, tuplify(self.offset))

    # This is necessary so that we can convert a Map to a tuple
    # (needed in as_tuple).  Because, __getitem__ no longer returns a
    # Map we have to explicitly provide an iterable interface
    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    @cached_property
    def split(self):
        return (self,)

    @cached_property
    def iterset(self):
        """:class:`Set` mapped from."""
        return self._iterset

    @cached_property
    def toset(self):
        """:class:`Set` mapped to."""
        return self._toset

    @cached_property
    def arity(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element."""
        return self._arity

    @cached_property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return (self._arity,)

    @cached_property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0, self._arity)

    @cached_property
    def values(self):
        """Mapping array.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return self._values[:self.iterset.size]

    @cached_property
    def values_with_halo(self):
        """Mapping array.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return self._values

    @cached_property
    def name(self):
        """User-defined label"""
        return self._name

    @cached_property
    def offset(self):
        """The vertical offset."""
        return self._offset

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with arity %s" \
               % (self._name, self._iterset, self._toset, self._arity)

    def __repr__(self):
        return "Map(%r, %r, %r, None, %r)" \
               % (self._iterset, self._toset, self._arity, self._name)

    def __le__(self, o):
        """self<=o if o equals self or self._parent <= o."""
        return self == o


class MixedMap(Map, ObjectCached):
    r"""A container for a bag of :class:`Map`\s."""

    def __init__(self, maps):
        r""":param iterable maps: Iterable of :class:`Map`\s"""
        if self._initialized:
            return
        self._maps = maps
        if not all(m is None or m.iterset == self.iterset for m in self._maps):
            raise MapTypeError("All maps in a MixedMap need to share the same iterset")
        # TODO: Think about different communicators on maps (c.f. MixedSet)
        # TODO: What if all maps are None?
        comms = tuple(m.comm for m in self._maps if m is not None)
        if not all(c == comms[0] for c in comms):
            raise MapTypeError("All maps needs to share a communicator")
        if len(comms) == 0:
            raise MapTypeError("Don't know how to make communicator")
        self.comm = comms[0]
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        maps = as_tuple(args[0], type=Map, allow_none=True)
        cache = maps[0]
        return (cache, ) + (maps, ), kwargs

    @classmethod
    def _cache_key(cls, maps):
        return maps

    @cached_property
    def _kernel_args_(self):
        return tuple(itertools.chain(*(m._kernel_args_ for m in self if m is not None)))

    @cached_property
    def _argtypes_(self):
        return tuple(itertools.chain(*(m._argtypes_ for m in self if m is not None)))

    @cached_property
    def _wrapper_cache_key_(self):
        return tuple(m._wrapper_cache_key_ for m in self if m is not None)

    @cached_property
    def split(self):
        r"""The underlying tuple of :class:`Map`\s."""
        return self._maps

    @cached_property
    def iterset(self):
        """:class:`MixedSet` mapped from."""
        return reduce(lambda a, b: a or b, map(lambda s: s if s is None else s.iterset, self._maps))

    @cached_property
    def toset(self):
        """:class:`MixedSet` mapped to."""
        return MixedSet(tuple(GlobalSet(comm=self.comm) if m is None else
                              m.toset for m in self._maps))

    @cached_property
    def arity(self):
        """Arity of the mapping: total number of toset elements mapped to per
        iterset element."""
        return sum(m.arity for m in self._maps)

    @cached_property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return tuple(m.arity for m in self._maps)

    @cached_property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0,) + tuple(np.cumsum(self.arities))

    @cached_property
    def values(self):
        """Mapping arrays excluding data for halos.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return tuple(m.values for m in self._maps)

    @cached_property
    def values_with_halo(self):
        """Mapping arrays including data for halos.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return tuple(None if m is None else
                     m.values_with_halo for m in self._maps)

    @cached_property
    def name(self):
        """User-defined labels"""
        return tuple(m.name for m in self._maps)

    @cached_property
    def offset(self):
        """Vertical offsets."""
        return tuple(0 if m is None else m.offset for m in self._maps)

    def __iter__(self):
        r"""Yield all :class:`Map`\s when iterated over."""
        for m in self._maps:
            yield m

    def __len__(self):
        r"""Number of contained :class:`Map`\s."""
        return len(self._maps)

    def __le__(self, o):
        """self<=o if o equals self or its self._parent==o."""
        return self == o or all(m <= om for m, om in zip(self, o))

    def __str__(self):
        return "OP2 MixedMap composed of Maps: %s" % (self._maps,)

    def __repr__(self):
        return "MixedMap(%r)" % (self._maps,)


class Sparsity(ObjectCached):

    """OP2 Sparsity, the non-zero structure a matrix derived from the union of
    the outer product of pairs of :class:`Map` objects.

    Examples of constructing a Sparsity: ::

        Sparsity(single_dset, single_map, 'mass')
        Sparsity((row_dset, col_dset), (single_rowmap, single_colmap))
        Sparsity((row_dset, col_dset),
                 [(first_rowmap, first_colmap), (second_rowmap, second_colmap)])

    .. _MatMPIAIJSetPreallocation: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
    """

    def __init__(self, dsets, maps, *, iteration_regions=None, name=None, nest=None, block_sparse=None):
        r"""
        :param dsets: :class:`DataSet`\s for the left and right function
            spaces this :class:`Sparsity` maps between
        :param maps: :class:`Map`\s to build the :class:`Sparsity` from
        :type maps: a pair of :class:`Map`\s specifying a row map and a column
            map, or an iterable of pairs of :class:`Map`\s specifying multiple
            row and column maps - if a single :class:`Map` is passed, it is
            used as both a row map and a column map
        :param iteration_regions: regions that select subsets of extruded maps to iterate over.
        :param string name: user-defined label (optional)
        :param nest: Should the sparsity over mixed set be built as nested blocks?
        :param block_sparse: Should the sparsity for datasets with
            cdim > 1 be built as a block sparsity?
        """
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return

        self._block_sparse = block_sparse
        # Split into a list of row maps and a list of column maps
        maps, iteration_regions = zip(*maps)
        self._rmaps, self._cmaps = zip(*maps)
        self._dsets = dsets

        if isinstance(dsets[0], GlobalDataSet) or isinstance(dsets[1], GlobalDataSet):
            self._dims = (((1, 1),),)
            self._d_nnz = None
            self._o_nnz = None
            self._nrows = None if isinstance(dsets[0], GlobalDataSet) else self._rmaps[0].toset.size
            self._ncols = None if isinstance(dsets[1], GlobalDataSet) else self._cmaps[0].toset.size
            self.lcomm = dsets[0].comm if isinstance(dsets[0], GlobalDataSet) else self._rmaps[0].comm
            self.rcomm = dsets[1].comm if isinstance(dsets[1], GlobalDataSet) else self._cmaps[0].comm
        else:
            self.lcomm = self._rmaps[0].comm
            self.rcomm = self._cmaps[0].comm

            rset, cset = self.dsets
            # All rmaps and cmaps have the same data set - just use the first.
            self._nrows = rset.size
            self._ncols = cset.size

            self._has_diagonal = (rset == cset)

            tmp = itertools.product([x.cdim for x in self._dsets[0]],
                                    [x.cdim for x in self._dsets[1]])

            dims = [[None for _ in range(self.shape[1])] for _ in range(self.shape[0])]
            for r in range(self.shape[0]):
                for c in range(self.shape[1]):
                    dims[r][c] = next(tmp)

            self._dims = tuple(tuple(d) for d in dims)

        if self.lcomm != self.rcomm:
            raise ValueError("Haven't thought hard enough about different left and right communicators")
        self.comm = self.lcomm

        self._name = name or "sparsity_#x%x" % id(self)

        self.iteration_regions = iteration_regions
        # If the Sparsity is defined on MixedDataSets, we need to build each
        # block separately
        if (isinstance(dsets[0], MixedDataSet) or isinstance(dsets[1], MixedDataSet)) \
           and nest:
            self._nested = True
            self._blocks = []
            for i, rds in enumerate(dsets[0]):
                row = []
                for j, cds in enumerate(dsets[1]):
                    row.append(Sparsity((rds, cds), [(rm.split[i], cm.split[j]) for
                                                     rm, cm in maps],
                                        iteration_regions=iteration_regions,
                                        block_sparse=block_sparse))
                self._blocks.append(row)
            self._d_nnz = tuple(s._d_nnz for s in self)
            self._o_nnz = tuple(s._o_nnz for s in self)
        elif isinstance(dsets[0], GlobalDataSet) or isinstance(dsets[1], GlobalDataSet):
            # Where the sparsity maps either from or to a Global, we
            # don't really have any sparsity structure.
            self._blocks = [[self]]
            self._nested = False
        else:
            for dset in dsets:
                if isinstance(dset, MixedDataSet) and any([isinstance(d, GlobalDataSet) for d in dset]):
                    raise SparsityFormatError("Mixed monolithic matrices with Global rows or columns are not supported.")
            self._nested = False
            with timed_region("CreateSparsity"):
                nnz, onnz = build_sparsity(self)
                self._d_nnz = nnz
                self._o_nnz = onnz
            self._blocks = [[self]]
        self._initialized = True

    _cache = {}

    @classmethod
    @validate_type(('dsets', (Set, DataSet, tuple, list), DataSetTypeError),
                   ('maps', (Map, tuple, list), MapTypeError))
    def _process_args(cls, dsets, maps, *, iteration_regions=None, name=None, nest=None, block_sparse=None):
        "Turn maps argument into a canonical tuple of pairs."

        # A single data set becomes a pair of identical data sets
        dsets = [dsets, dsets] if isinstance(dsets, (Set, DataSet)) else list(dsets)
        # Upcast Sets to DataSets
        dsets = [s ** 1 if isinstance(s, Set) else s for s in dsets]

        # Check data sets are valid
        for dset in dsets:
            if not isinstance(dset, DataSet) and dset is not None:
                raise DataSetTypeError("All data sets must be of type DataSet, not type %r" % type(dset))

        # A single map becomes a pair of identical maps
        maps = (maps, maps) if isinstance(maps, Map) else maps
        # A single pair becomes a tuple of one pair
        maps = (maps,) if isinstance(maps[0], Map) else maps

        # Check maps are sane
        for pair in maps:
            if pair[0] is None or pair[1] is None:
                # None of this checking makes sense if one of the
                # matrix operands is a Global.
                continue
            for m in pair:
                if not isinstance(m, Map):
                    raise MapTypeError(
                        "All maps must be of type map, not type %r" % type(m))
                if len(m.values_with_halo) == 0 and m.iterset.total_size > 0:
                    raise MapValueError(
                        "Unpopulated map values when trying to build sparsity.")
            # Make sure that the "to" Set of each map in a pair is the set of
            # the corresponding DataSet set
            if not (pair[0].toset == dsets[0].set
                    and pair[1].toset == dsets[1].set):
                raise RuntimeError("Map to set must be the same as corresponding DataSet set")

            # Each pair of maps must have the same from-set (iteration set)
            if not pair[0].iterset == pair[1].iterset:
                raise RuntimeError("Iterset of both maps in a pair must be the same")

        rmaps, cmaps = zip(*maps)
        if iteration_regions is None:
            iteration_regions = tuple((ALL, ) for _ in maps)
        else:
            iteration_regions = tuple(tuple(sorted(region)) for region in iteration_regions)
        if not len(rmaps) == len(cmaps):
            raise RuntimeError("Must pass equal number of row and column maps")

        if rmaps[0] is not None and cmaps[0] is not None:
            # Each row map must have the same to-set (data set)
            if not all(m.toset == rmaps[0].toset for m in rmaps):
                raise RuntimeError("To set of all row maps must be the same")

                # Each column map must have the same to-set (data set)
            if not all(m.toset == cmaps[0].toset for m in cmaps):
                raise RuntimeError("To set of all column maps must be the same")

        # Need to return the caching object, a tuple of the processed
        # arguments and a dict of kwargs (empty in this case)
        if isinstance(dsets[0], GlobalDataSet):
            cache = None
        elif isinstance(dsets[0].set, MixedSet):
            cache = dsets[0].set[0]
        else:
            cache = dsets[0].set
        if nest is None:
            nest = configuration["matnest"]
        if block_sparse is None:
            block_sparse = configuration["block_sparsity"]

        maps = frozenset(zip(maps, iteration_regions))
        kwargs = {"name": name,
                  "nest": nest,
                  "block_sparse": block_sparse}
        return (cache,) + (tuple(dsets), maps), kwargs

    @classmethod
    def _cache_key(cls, dsets, maps, name, nest, block_sparse, *args, **kwargs):
        return (dsets, maps, nest, block_sparse)

    def __getitem__(self, idx):
        """Return :class:`Sparsity` block with row and column given by ``idx``
        or a given row of blocks."""
        try:
            i, j = idx
            return self._blocks[i][j]
        except TypeError:
            return self._blocks[idx]

    @cached_property
    def dsets(self):
        r"""A pair of :class:`DataSet`\s for the left and right function
        spaces this :class:`Sparsity` maps between."""
        return self._dsets

    @cached_property
    def maps(self):
        """A list of pairs (rmap, cmap) where each pair of
        :class:`Map` objects will later be used to assemble into this
        matrix. The iterset of each of the maps in a pair must be the
        same, while the toset of all the maps which appear first
        must be common, this will form the row :class:`Set` of the
        sparsity. Similarly, the toset of all the maps which appear
        second must be common and will form the column :class:`Set` of
        the ``Sparsity``."""
        return list(zip(self._rmaps, self._cmaps))

    @cached_property
    def cmaps(self):
        """The list of column maps this sparsity is assembled from."""
        return self._cmaps

    @cached_property
    def rmaps(self):
        """The list of row maps this sparsity is assembled from."""
        return self._rmaps

    @cached_property
    def dims(self):
        """A tuple of tuples where the ``i,j``th entry
        is a pair giving the number of rows per entry of the row
        :class:`Set` and the number of columns per entry of the column
        :class:`Set` of the ``Sparsity``.  The extents of the first
        two indices are given by the :attr:`shape` of the sparsity.
        """
        return self._dims

    @cached_property
    def shape(self):
        """Number of block rows and columns."""
        return (len(self._dsets[0] or [1]),
                len(self._dsets[1] or [1]))

    @cached_property
    def nrows(self):
        """The number of rows in the ``Sparsity``."""
        return self._nrows

    @cached_property
    def ncols(self):
        """The number of columns in the ``Sparsity``."""
        return self._ncols

    @cached_property
    def nested(self):
        r"""Whether a sparsity is monolithic (even if it has a block structure).

        To elaborate, if a sparsity maps between
        :class:`MixedDataSet`\s, it can either be nested, in which
        case it consists of as many blocks are the product of the
        length of the datasets it maps between, or monolithic.  In the
        latter case the sparsity is for the full map between the mixed
        datasets, rather than between the blocks of the non-mixed
        datasets underneath them.
        """
        return self._nested

    @cached_property
    def name(self):
        """A user-defined label."""
        return self._name

    def __iter__(self):
        r"""Iterate over all :class:`Sparsity`\s by row and then by column."""
        for row in self._blocks:
            for s in row:
                yield s

    def __str__(self):
        return "OP2 Sparsity: dsets %s, rmaps %s, cmaps %s, name %s" % \
               (self._dsets, self._rmaps, self._cmaps, self._name)

    def __repr__(self):
        return "Sparsity(%r, %r, %r)" % (self.dsets, self.maps, self.name)

    @cached_property
    def nnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        diagonal portion of the local submatrix.

        This is the same as the parameter `d_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._d_nnz

    @cached_property
    def onnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        off-diagonal portion of the local submatrix.

        This is the same as the parameter `o_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._o_nnz

    @cached_property
    def nz(self):
        return self._d_nnz.sum()

    @cached_property
    def onz(self):
        return self._o_nnz.sum()

    def __contains__(self, other):
        """Return true if other is a pair of maps in self.maps(). This
        will also return true if the elements of other have parents in
        self.maps()."""

        for maps in self.maps:
            if tuple(other) <= maps:
                return True

        return False


class Mat(DataCarrier):
    r"""OP2 matrix data. A ``Mat`` is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`.

    When a ``Mat`` is passed to :func:`pyop2.op2.par_loop`, the maps via which
    indirection occurs for the row and column space, and the access
    descriptor are passed by `calling` the ``Mat``. For instance, if a
    ``Mat`` named ``A`` is to be accessed for reading via a row :class:`Map`
    named ``R`` and a column :class:`Map` named ``C``, this is accomplished by::

     A(pyop2.READ, (R[pyop2.i[0]], C[pyop2.i[1]]))

    Notice that it is `always` necessary to index the indirection maps
    for a ``Mat``. See the :class:`Mat` documentation for more
    details.

    .. note ::

       After executing :func:`par_loop`\s that write to a ``Mat`` and
       before using it (for example to view its values), you must call
       :meth:`assemble` to finalise the writes.
    """
    @cached_property
    def pack(self):
        from pyop2.codegen.builder import MatPack
        return MatPack

    ASSEMBLED = "ASSEMBLED"
    INSERT_VALUES = "INSERT_VALUES"
    ADD_VALUES = "ADD_VALUES"

    _modes = [WRITE, INC]

    @validate_type(('sparsity', Sparsity, SparsityTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, sparsity, dtype=None, name=None):
        self._sparsity = sparsity
        self.lcomm = sparsity.lcomm
        self.rcomm = sparsity.rcomm
        self.comm = sparsity.comm
        dtype = dtype or ScalarType
        self._datatype = np.dtype(dtype)
        self._name = name or "mat_#x%x" % id(self)
        self.assembly_state = Mat.ASSEMBLED

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path, lgmaps=None, unroll_map=False):
        path_maps = as_tuple(path, Map, 2)
        if configuration["type_check"] and tuple(path_maps) not in self.sparsity:
            raise MapValueError("Path maps not in sparsity maps")
        return _make_object('Arg', data=self, map=path_maps, access=access, lgmaps=lgmaps, unroll_map=unroll_map)

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self.dims)

    def assemble(self):
        """Finalise this :class:`Mat` ready for use.

        Call this /after/ executing all the par_loops that write to
        the matrix before you want to look at it.
        """
        raise NotImplementedError("Subclass should implement this")

    def addto_values(self, rows, cols, values):
        """Add a block of values to the :class:`Mat`."""
        raise NotImplementedError(
            "Abstract Mat base class doesn't know how to set values.")

    def set_values(self, rows, cols, values):
        """Set a block of values in the :class:`Mat`."""
        raise NotImplementedError(
            "Abstract Mat base class doesn't know how to set values.")

    @cached_property
    def _argtypes_(self):
        """Ctypes argtype for this :class:`Mat`"""
        return tuple(ctypes.c_voidp for _ in self)

    @cached_property
    def dims(self):
        """A pair of integers giving the number of matrix rows and columns for
        each member of the row :class:`Set` and column :class:`Set`
        respectively. This corresponds to the ``cdim`` member of a
        :class:`DataSet`."""
        return self._sparsity._dims

    @cached_property
    def nrows(self):
        "The number of rows in the matrix (local to this process)"
        return sum(d.size * d.cdim for d in self.sparsity.dsets[0])

    @cached_property
    def nblock_rows(self):
        """The number "block" rows in the matrix (local to this process).

        This is equivalent to the number of rows in the matrix divided
        by the dimension of the row :class:`DataSet`.
        """
        assert len(self.sparsity.dsets[0]) == 1, "Block rows don't make sense for mixed Mats"
        return self.sparsity.dsets[0].size

    @cached_property
    def nblock_cols(self):
        """The number of "block" columns in the matrix (local to this process).

        This is equivalent to the number of columns in the matrix
        divided by the dimension of the column :class:`DataSet`.
        """
        assert len(self.sparsity.dsets[1]) == 1, "Block cols don't make sense for mixed Mats"
        return self.sparsity.dsets[1].size

    @cached_property
    def ncols(self):
        "The number of columns in the matrix (local to this process)"
        return sum(d.size * d.cdim for d in self.sparsity.dsets[1])

    @cached_property
    def sparsity(self):
        """:class:`Sparsity` on which the ``Mat`` is defined."""
        return self._sparsity

    @cached_property
    def _is_scalar_field(self):
        # Sparsity from Dat to MixedDat has a shape like (1, (1, 1))
        # (which you can't take the product of)
        return all(np.prod(d) == 1 for d in self.dims)

    @cached_property
    def _is_vector_field(self):
        return not self._is_scalar_field

    def change_assembly_state(self, new_state):
        """Switch the matrix assembly state."""
        if new_state == Mat.ASSEMBLED or self.assembly_state == Mat.ASSEMBLED:
            self.assembly_state = new_state
        elif new_state != self.assembly_state:
            self._flush_assembly()
            self.assembly_state = new_state
        else:
            pass

    def _flush_assembly(self):
        """Flush the in flight assembly operations (used when
        switching between inserting and adding values)."""
        pass

    @property
    def values(self):
        """A numpy array of matrix values.

        .. warning ::
            This is a dense array, so will need a lot of memory.  It's
            probably not a good idea to access this property if your
            matrix has more than around 10000 degrees of freedom.
        """
        raise NotImplementedError("Abstract base Mat does not implement values()")

    @cached_property
    def dtype(self):
        """The Python type of the data."""
        return self._datatype

    @cached_property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Mat` in bytes. This will be the correct size of the
        data payload, but does not take into account the (presumably
        small) overhead of the object and its metadata. The memory
        associated with the sparsity pattern is also not recorded.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """
        if self._sparsity._block_sparse:
            mult = np.sum(np.prod(self._sparsity.dims))
        else:
            mult = 1
        return (self._sparsity.nz + self._sparsity.onz) \
            * self.dtype.itemsize * mult

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __mul__(self, other):
        """Multiply this :class:`Mat` with the vector ``other``."""
        raise NotImplementedError("Abstract base Mat does not implement multiplication")

    def __str__(self):
        return "OP2 Mat: %s, sparsity (%s), datatype %s" \
               % (self._name, self._sparsity, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, %r, %r)" \
               % (self._sparsity, self._datatype, self._name)

# Kernel API


class Kernel(Cached):

    """OP2 kernel type.

    :param code: kernel function definition, including signature; either a
        string or an AST :class:`.Node`
    :param name: kernel function name; must match the name of the kernel
        function given in `code`
    :param opts: options dictionary for :doc:`PyOP2 IR optimisations <ir>`
        (optional, ignored if `code` is a string)
    :param include_dirs: list of additional include directories to be searched
        when compiling the kernel (optional, defaults to empty)
    :param headers: list of system headers to include when compiling the kernel
        in the form ``#include <header.h>`` (optional, defaults to empty)
    :param user_code: code snippet to be executed once at the very start of
        the generated kernel wrapper code (optional, defaults to
        empty)
    :param ldargs: A list of arguments to pass to the linker when
        compiling this Kernel.
    :param cpp: Is the kernel actually C++ rather than C?  If yes,
        then compile with the C++ compiler (kernel is wrapped in
        extern C for linkage reasons).

    Consider the case of initialising a :class:`~pyop2.Dat` with seeded random
    values in the interval 0 to 1. The corresponding :class:`~pyop2.Kernel` is
    constructed as follows: ::

      op2.Kernel("void setrand(double *x) { x[0] = (double)random()/RAND_MAX); }",
                 name="setrand",
                 headers=["#include <stdlib.h>"], user_code="srandom(10001);")

    .. note::
        When running in parallel with MPI the generated code must be the same
        on all ranks.
    """

    _cache = {}

    @classmethod
    @validate_type(('name', str, NameTypeError))
    def _cache_key(cls, code, name, opts={}, include_dirs=[], headers=[],
                   user_code="", ldargs=None, cpp=False):
        # Both code and name are relevant since there might be multiple kernels
        # extracting different functions from the same code
        # Also include the PyOP2 version, since the Kernel class might change

        if isinstance(code, Node):
            code = code.gencode()
        if isinstance(code, loopy.LoopKernel):
            from loopy.tools import LoopyKeyBuilder
            from pytools.persistent_dict import new_hash
            key_hash = new_hash()
            code.update_persistent_hash(key_hash, LoopyKeyBuilder())
            code = key_hash.hexdigest()
        hashee = (str(code) + name + str(sorted(opts.items())) + str(include_dirs)
                  + str(headers) + version + str(ldargs) + str(cpp))
        return md5(hashee.encode()).hexdigest()

    @cached_property
    def _wrapper_cache_key_(self):
        return (self._key, )

    def __init__(self, code, name, opts={}, include_dirs=[], headers=[],
                 user_code="", ldargs=None, cpp=False):
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return
        self._name = name
        self._cpp = cpp
        # Record used optimisations
        self._opts = opts
        self._include_dirs = include_dirs
        self._ldargs = ldargs if ldargs is not None else []
        self._headers = headers
        self._user_code = user_code
        assert isinstance(code, (str, Node, loopy.Program, loopy.LoopKernel))
        self._code = code
        self._initialized = True

    @property
    def name(self):
        """Kernel name, must match the kernel function name in the code."""
        return self._name

    @property
    def code(self):
        return self._code

    @cached_property
    def num_flops(self):
        if not configuration["compute_kernel_flops"]:
            return 0
        if isinstance(self.code, Node):
            v = EstimateFlops()
            return v.visit(self.code)
        elif isinstance(self.code, loopy.LoopKernel):
            op_map = loopy.get_op_map(
                self.code.copy(options=loopy.Options(ignore_boostable_into=True),
                               silenced_warnings=['insn_count_subgroups_upper_bound',
                                                  'get_x_map_guessing_subgroup_size',
                                                  'summing_if_branches_ops']),
                subgroup_size='guess')
            return op_map.filter_by(name=['add', 'sub', 'mul', 'div'], dtype=[ScalarType]).eval_and_sum({})
        else:
            return 0

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", %r)' % (self._code, self._name)

    def __eq__(self, other):
        return self.cache_key == other.cache_key


class JITModule(Cached):

    """Cached module encapsulating the generated :class:`ParLoop` stub.

    .. warning::

       Note to implementors.  This object is *cached* and therefore
       should not hold any references to objects you might want to be
       collected (such PyOP2 data objects)."""

    _cache = {}

    @classmethod
    def _cache_key(cls, kernel, iterset, *args, **kwargs):
        counter = itertools.count()
        seen = defaultdict(lambda: next(counter))
        key = ((id(dup_comm(iterset.comm)), ) + kernel._wrapper_cache_key_ + iterset._wrapper_cache_key_
               + (iterset._extruded, (iterset._extruded and iterset.constant_layers), isinstance(iterset, Subset)))

        for arg in args:
            key += arg._wrapper_cache_key_
            for map_ in arg.map_tuple:
                key += (seen[map_],)

        key += (kwargs.get("iterate", None), cls, configuration["simd_width"])

        return key


class IterationRegion(IntEnum):
    BOTTOM = 1
    TOP = 2
    INTERIOR_FACETS = 3
    ALL = 4


ON_BOTTOM = IterationRegion.BOTTOM
"""Iterate over the cells at the bottom of the column in an extruded mesh."""

ON_TOP = IterationRegion.TOP
"""Iterate over the top cells in an extruded mesh."""

ON_INTERIOR_FACETS = IterationRegion.INTERIOR_FACETS
"""Iterate over the interior facets of an extruded mesh."""

ALL = IterationRegion.ALL
"""Iterate over all cells of an extruded mesh."""


class ParLoop(object):
    """Represents the kernel, iteration space and arguments of a parallel loop
    invocation.

    .. note ::

        Users should not directly construct :class:`ParLoop` objects, but
        use :func:`pyop2.op2.par_loop` instead.

    An optional keyword argument, ``iterate``, can be used to specify
    which region of an :class:`ExtrudedSet` the parallel loop should
    iterate over.
    """

    @validate_type(('kernel', Kernel, KernelTypeError),
                   ('iterset', Set, SetTypeError))
    def __init__(self, kernel, iterset, *args, **kwargs):
        # INCs into globals need to start with zero and then sum back
        # into the input global at the end.  This has the same number
        # of reductions but means that successive par_loops
        # incrementing into a global get the "right" value in
        # parallel.
        # Don't care about MIN and MAX because they commute with the reduction
        self._reduced_globals = {}
        for i, arg in enumerate(args):
            if arg._is_global_reduction and arg.access == INC:
                glob = arg.data
                tmp = _make_object('Global', glob.dim, data=np.zeros_like(glob.data_ro), dtype=glob.dtype)
                self._reduced_globals[tmp] = glob
                args[i].data = tmp

        # Always use the current arguments, also when we hit cache
        self._actual_args = args
        self._kernel = kernel
        self._is_layered = iterset._extruded
        self._iteration_region = kwargs.get("iterate", None)
        self._pass_layer_arg = kwargs.get("pass_layer_arg", False)

        check_iterset(self.args, iterset)

        if self._pass_layer_arg:
            if not self._is_layered:
                raise ValueError("Can't request layer arg for non-extruded iteration")

        self.iterset = iterset
        self.comm = iterset.comm

        for i, arg in enumerate(self._actual_args):
            arg.position = i
            arg.indirect_position = i
        for i, arg1 in enumerate(self._actual_args):
            if arg1._is_dat and arg1._is_indirect:
                for arg2 in self._actual_args[i:]:
                    # We have to check for identity here (we really
                    # want these to be the same thing, not just look
                    # the same)
                    if arg2.data is arg1.data and arg2.map is arg1.map:
                        arg2.indirect_position = arg1.indirect_position

        self.arglist = self.prepare_arglist(iterset, *self.args)

    def prepare_arglist(self, iterset, *args):
        """Prepare the argument list for calling generated code.

        :arg iterset: The :class:`Set` iterated over.
        :arg args: A list of :class:`Args`, the argument to the :fn:`par_loop`.
        """
        return ()

    @cached_property
    def num_flops(self):
        iterset = self.iterset
        size = 1
        if iterset._extruded:
            region = self.iteration_region
            layers = np.mean(iterset.layers_array[:, 1] - iterset.layers_array[:, 0])
            if region is ON_INTERIOR_FACETS:
                size = layers - 2
            elif region not in [ON_TOP, ON_BOTTOM]:
                size = layers - 1
        return size * self._kernel.num_flops

    def log_flops(self, flops):
        pass

    @property
    @collective
    def _jitmodule(self):
        """Return the :class:`JITModule` that encapsulates the compiled par_loop code.

        Return None if the child class should deal with this in another way."""
        return None

    @cached_property
    def _parloop_event(self):
        return timed_region("ParLoopExecute")

    @collective
    def compute(self):
        """Executes the kernel over all members of the iteration space."""
        with self._parloop_event:
            orig_lgmaps = []
            for arg in self.args:
                if arg._is_mat and arg.lgmaps is not None:
                    orig_lgmaps.append(arg.data.handle.getLGMap())
                    arg.data.handle.setLGMap(*arg.lgmaps)
            self.global_to_local_begin()
            iterset = self.iterset
            arglist = self.arglist
            fun = self._jitmodule
            # Need to ensure INC globals are zero on entry to the loop
            # in case it's reused.
            for g in self._reduced_globals.keys():
                g._data[...] = 0
            self._compute(iterset.core_part, fun, *arglist)
            self.global_to_local_end()
            self._compute(iterset.owned_part, fun, *arglist)
            self.reduction_begin()
            self.local_to_global_begin()
            self.update_arg_data_state()
            for arg in reversed(self.args):
                if arg._is_mat and arg.lgmaps is not None:
                    arg.data.handle.setLGMap(*orig_lgmaps.pop())
            self.reduction_end()
            self.local_to_global_end()

    @collective
    def _compute(self, part, fun, *arglist):
        """Executes the kernel over all members of a MPI-part of the iteration space.

        :arg part: The :class:`SetPartition` to compute over
        :arg fun: The :class:`JITModule` encapsulating the compiled
             code (may be ignored by the backend).
        :arg arglist: The arguments to pass to the compiled code (may
             be ignored by the backend, depending on the exact implementation)"""
        raise RuntimeError("Must select a backend")

    @collective
    def global_to_local_begin(self):
        """Start halo exchanges."""
        for arg in self.unique_dat_args:
            arg.global_to_local_begin()

    @collective
    def global_to_local_end(self):
        """Finish halo exchanges"""
        for arg in self.unique_dat_args:
            arg.global_to_local_end()

    @collective
    def local_to_global_begin(self):
        """Start halo exchanges."""
        for arg in self.unique_dat_args:
            arg.local_to_global_begin()

    @collective
    def local_to_global_end(self):
        """Finish halo exchanges (wait on irecvs)"""
        for arg in self.unique_dat_args:
            arg.local_to_global_end()

    @cached_property
    def _reduction_event_begin(self):
        return timed_region("ParLoopRednBegin")

    @cached_property
    def _reduction_event_end(self):
        return timed_region("ParLoopRednEnd")

    @cached_property
    def _has_reduction(self):
        return len(self.global_reduction_args) > 0

    @collective
    def reduction_begin(self):
        """Start reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_begin:
            for arg in self.global_reduction_args:
                arg.reduction_begin(self.comm)

    @collective
    def reduction_end(self):
        """End reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_end:
            for arg in self.global_reduction_args:
                arg.reduction_end(self.comm)
            # Finalise global increments
            for tmp, glob in self._reduced_globals.items():
                glob._data += tmp._data

    @collective
    def update_arg_data_state(self):
        r"""Update the state of the :class:`DataCarrier`\s in the arguments to the `par_loop`.

        This marks :class:`Mat`\s that need assembly."""
        for arg in self.args:
            access = arg.access
            if access is READ:
                continue
            if arg._is_dat:
                arg.data.halo_valid = False
            if arg._is_mat:
                state = {WRITE: Mat.INSERT_VALUES,
                         INC: Mat.ADD_VALUES}[access]
                arg.data.assembly_state = state

    @cached_property
    def dat_args(self):
        return tuple(arg for arg in self.args if arg._is_dat)

    @cached_property
    def unique_dat_args(self):
        seen = {}
        unique = []
        for arg in self.dat_args:
            if arg.data not in seen:
                unique.append(arg)
                seen[arg.data] = arg
            elif arg.access != seen[arg.data].access:
                raise ValueError("Same Dat appears multiple times with different "
                                 "access descriptors")
        return tuple(unique)

    @cached_property
    def global_reduction_args(self):
        return tuple(arg for arg in self.args if arg._is_global_reduction)

    @cached_property
    def kernel(self):
        """Kernel executed by this parallel loop."""
        return self._kernel

    @cached_property
    def args(self):
        """Arguments to this parallel loop."""
        return self._actual_args

    @cached_property
    def is_layered(self):
        """Flag which triggers extrusion"""
        return self._is_layered

    @cached_property
    def iteration_region(self):
        """Specifies the part of the mesh the parallel loop will
        be iterating over. The effect is the loop only iterates over
        a certain part of an extruded mesh, for example on top cells, bottom cells or
        interior facets."""
        return self._iteration_region


def check_iterset(args, iterset):
    """Checks that the iteration set of the :class:`ParLoop` matches the
    iteration set of all its arguments. A :class:`MapValueError` is raised
    if this condition is not met."""

    if isinstance(iterset, Subset):
        _iterset = iterset.superset
    else:
        _iterset = iterset
    if configuration["type_check"]:
        if isinstance(_iterset, MixedSet):
            raise SetTypeError("Cannot iterate over MixedSets")
        for i, arg in enumerate(args):
            if arg._is_global:
                continue
            if arg._is_direct:
                if isinstance(_iterset, ExtrudedSet):
                    if arg.data.dataset.set != _iterset.parent:
                        raise MapValueError(
                            "Iterset of direct arg %s doesn't match ParLoop iterset." % i)
                elif arg.data.dataset.set != _iterset:
                    raise MapValueError(
                        "Iterset of direct arg %s doesn't match ParLoop iterset." % i)
                continue
            for j, m in enumerate(arg._map):
                if isinstance(_iterset, ExtrudedSet):
                    if m.iterset != _iterset and m.iterset not in _iterset:
                        raise MapValueError(
                            "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))
                elif m.iterset != _iterset and m.iterset not in _iterset:
                    raise MapValueError(
                        "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))


@collective
def par_loop(kernel, iterset, *args, **kwargs):
    r"""Invocation of an OP2 kernel

    :arg kernel: The :class:`Kernel` to be executed.
    :arg iterset: The iteration :class:`Set` over which the kernel should be
                  executed.
    :arg \*args: One or more :class:`base.Arg`\s constructed from a
                 :class:`Global`, :class:`Dat` or :class:`Mat` using the call
                 syntax and passing in an optionally indexed :class:`Map`
                 through which this :class:`base.Arg` is accessed and the
                 :class:`base.Access` descriptor indicating how the
                 :class:`Kernel` is going to access this data (see the example
                 below). These are the global data structures from and to
                 which the kernel will read and write.
    :kwarg iterate: Optionally specify which region of an
            :class:`ExtrudedSet` to iterate over.
            Valid values are:

              - ``ON_BOTTOM``: iterate over the bottom layer of cells.
              - ``ON_TOP`` iterate over the top layer of cells.
              - ``ALL`` iterate over all cells (the default if unspecified)
              - ``ON_INTERIOR_FACETS`` iterate over all the layers
                 except the top layer, accessing data two adjacent (in
                 the extruded direction) cells at a time.

    :kwarg pass_layer_arg: Should the wrapper pass the current layer
        into the kernel (as an ``int``). Only makes sense for
        indirect extruded iteration.

    .. warning ::
        It is the caller's responsibility that the number and type of all
        :class:`base.Arg`\s passed to the :func:`par_loop` match those expected
        by the :class:`Kernel`. No runtime check is performed to ensure this!

    :func:`par_loop` invocation is illustrated by the following example ::

      pyop2.par_loop(mass, elements,
                     mat(pyop2.INC, (elem_node[pyop2.i[0]]), elem_node[pyop2.i[1]]),
                     coords(pyop2.READ, elem_node))

    This example will execute the :class:`Kernel` ``mass`` over the
    :class:`Set` ``elements`` executing 3x3 times for each
    :class:`Set` member, assuming the :class:`Map` ``elem_node`` is of arity 3.
    The :class:`Kernel` takes four arguments, the first is a :class:`Mat` named
    ``mat``, the second is a field named ``coords``. The remaining two arguments
    indicate which local iteration space point the kernel is to execute.

    A :class:`Mat` requires a pair of :class:`Map` objects, one each
    for the row and column spaces. In this case both are the same
    ``elem_node`` map. The row :class:`Map` is indexed by the first
    index in the local iteration space, indicated by the ``0`` index
    to :data:`pyop2.i`, while the column space is indexed by
    the second local index.  The matrix is accessed to increment
    values using the ``pyop2.INC`` access descriptor.

    The ``coords`` :class:`Dat` is also accessed via the ``elem_node``
    :class:`Map`, however no indices are passed so all entries of
    ``elem_node`` for the relevant member of ``elements`` will be
    passed to the kernel as a vector.
    """
    if isinstance(kernel, types.FunctionType):
        from pyop2 import pyparloop
        return pyparloop.ParLoop(kernel, iterset, *args, **kwargs).compute()
    return _make_object('ParLoop', kernel, iterset, *args, **kwargs).compute()
