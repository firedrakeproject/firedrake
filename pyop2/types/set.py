import ctypes
import numbers

import numpy as np
import pytools

from pyop2 import (
    caching,
    datatypes as dtypes,
    exceptions as ex,
    mpi,
    utils
)


class Set:

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
    _extruded_periodic = False

    _kernel_args_ = ()
    _argtypes_ = ()

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), )

    @utils.validate_type(('size', (numbers.Integral, tuple, list, np.ndarray), ex.SizeTypeError),
                         ('name', str, ex.NameTypeError))
    def __init__(self, size, name=None, halo=None, comm=None, constrained_size=0):
        self.comm = mpi.internal_comm(comm, self)
        if isinstance(size, numbers.Integral):
            size = [size] * 3
        size = utils.as_tuple(size, numbers.Integral, 3)
        assert size[Set._CORE_SIZE] <= size[Set._OWNED_SIZE] <= \
            size[Set._GHOST_SIZE], "Set received invalid sizes: %s" % size
        self._sizes = size
        self._name = name or "set_#x%x" % id(self)
        self._halo = halo
        self._partition_size = 1024
        self._constrained_size = constrained_size

        # A cache of objects built on top of this set
        self._cache = {}

    @property
    def indices(self):
        """Returns iterator."""
        return range(self.total_size)

    @utils.cached_property
    def core_size(self):
        """Core set size.  Owned elements not touching halo elements."""
        return self._sizes[Set._CORE_SIZE]

    @utils.cached_property
    def constrained_size(self):
        return self._constrained_size

    @utils.cached_property
    def size(self):
        """Set size, owned elements."""
        return self._sizes[Set._OWNED_SIZE]

    @utils.cached_property
    def total_size(self):
        """Set size including ghost elements.
        """
        return self._sizes[Set._GHOST_SIZE]

    @utils.cached_property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return self._sizes

    @utils.cached_property
    def core_part(self):
        return SetPartition(self, 0, self.core_size)

    @utils.cached_property
    def owned_part(self):
        return SetPartition(self, self.core_size, self.size - self.core_size)

    @utils.cached_property
    def name(self):
        """User-defined label"""
        return self._name

    @utils.cached_property
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

    def __hash__(self):
        """Hash on sizes and name"""
        return hash((self._sizes, self._name))

    def __eq__(self, other):
        """Two Sets are the same if they have the same sizes and names."""
        return self._sizes == other._sizes and self._name == other._name

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
        return Subset(self, indices)

    def __contains__(self, dset):
        """Indicate whether a given DataSet is compatible with this Set."""
        from pyop2.types import DataSet
        if isinstance(dset, DataSet):
            return dset.set is self
        else:
            return False

    def __pow__(self, e):
        """Derive a :class:`DataSet` with dimension ``e``"""
        from pyop2.types import DataSet
        return DataSet(self, dim=e)

    @utils.cached_property
    def layers(self):
        """Return None (not an :class:`ExtrudedSet`)."""
        return None

    def _check_operands(self, other):
        if type(other) is Set:
            if other is not self:
                raise ValueError("Uable to perform set operations between two unrelated sets: %s and %s." % (self, other))
        elif type(other) is Subset:
            if self is not other._superset:
                raise TypeError("Superset mismatch: self (%s) != other._superset (%s)" % (self, other._superset))
        else:
            raise TypeError("Unable to perform set operations between `Set` and %s." % (type(other), ))

    def intersection(self, other):
        self._check_operands(other)
        return other

    def union(self, other):
        self._check_operands(other)
        return self

    def difference(self, other):
        self._check_operands(other)
        if other is self:
            return Subset(self, [])
        else:
            return type(other)(self, np.setdiff1d(np.asarray(range(self.total_size), dtype=dtypes.IntType), other._indices))

    def symmetric_difference(self, other):
        self._check_operands(other)
        return self.difference(other)


class GlobalSet(Set):

    _extruded = False
    _extruded_periodic = False

    """A proxy set allowing a :class:`Global` to be used in place of a
    :class:`Dat` where appropriate."""

    _kernel_args_ = ()
    _argtypes_ = ()

    def __init__(self, comm=None):
        self.comm = mpi.internal_comm(comm, self)
        self._cache = {}

    @utils.cached_property
    def core_size(self):
        return 0

    @utils.cached_property
    def size(self):
        return 1 if self.comm.rank == 0 else 0

    @utils.cached_property
    def total_size(self):
        """Total set size, including halo elements."""
        return 1 if self.comm.rank == 0 else 0

    @utils.cached_property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return (self.core_size, self.size, self.total_size)

    @utils.cached_property
    def name(self):
        """User-defined label"""
        return "GlobalSet"

    @utils.cached_property
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

    @utils.validate_type(('parent', Set, TypeError))
    def __init__(self, parent, layers, extruded_periodic=False):
        self._parent = parent
        self.comm = mpi.internal_comm(parent.comm, self)
        try:
            layers = utils.verify_reshape(layers, dtypes.IntType, (parent.total_size, 2))
            self.constant_layers = False
            if layers.min(initial=0) < 0:
                raise ex.SizeTypeError("Bottom of layers must be >= 0")
            if any(layers[:, 1] - layers[:, 0] < 1):
                raise ex.SizeTypeError("Number of layers must be >= 0")
        except ex.DataValueError:
            # Legacy, integer
            layers = np.asarray(layers, dtype=dtypes.IntType)
            if layers.shape:
                raise ex.SizeTypeError(f"Specifying layers per entity, but provided "
                                       f"{layers.shape}, needed ({parent.total_size}, 2)")
            if layers < 2:
                raise ex.SizeTypeError("Need at least two layers, not %d", layers)
            layers = np.asarray([[0, layers]], dtype=dtypes.IntType)
            self.constant_layers = True

        self._layers = layers
        self._extruded = True
        self._extruded_periodic = extruded_periodic

    @utils.cached_property
    def _kernel_args_(self):
        return (self.layers_array.ctypes.data, )

    @utils.cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return self.parent._wrapper_cache_key_ + (self.constant_layers, )

    def __getattr__(self, name):
        """Returns a :class:`Set` specific attribute."""
        value = getattr(self._parent, name)
        return value

    def __contains__(self, set):
        return set is self.parent

    def __str__(self):
        return "OP2 ExtrudedSet: %s with size %s (%s layers)" % \
            (self._name, self.size, self._layers)

    def __repr__(self):
        return "ExtrudedSet(%r, %r)" % (self._parent, self._layers)

    @utils.cached_property
    def parent(self):
        return self._parent

    @utils.cached_property
    def layers(self):
        """The layers of this extruded set."""
        if self.constant_layers:
            # Backwards compat
            return self.layers_array[0, 1]
        else:
            raise ValueError("No single layer, use layers_array attribute")

    @utils.cached_property
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
    @utils.validate_type(('superset', Set, TypeError),
                         ('indices', (list, tuple, np.ndarray), TypeError))
    def __init__(self, superset, indices):
        self.comm = mpi.internal_comm(superset.comm, self)

        # sort and remove duplicates
        indices = np.unique(indices)
        if isinstance(superset, Subset):
            # Unroll indices to point to those in the parent
            indices = superset.indices[indices]
            superset = superset.superset
        assert type(superset) is Set or type(superset) is ExtrudedSet, \
            'Subset construction failed, should not happen'

        self._superset = superset
        self._indices = utils.verify_reshape(indices, dtypes.IntType, (len(indices),))

        if len(self._indices) > 0 and (self._indices[0] < 0 or self._indices[-1] >= self._superset.total_size):
            raise ex.SubsetIndexOutOfBounds(
                'Out of bounds indices in Subset construction: [%d, %d) not [0, %d)' %
                (self._indices[0], self._indices[-1], self._superset.total_size))

        self._sizes = ((self._indices < superset.core_size).sum(),
                       (self._indices < superset.size).sum(),
                       len(self._indices))
        self._extruded = superset._extruded
        self._extruded_periodic = superset._extruded_periodic

    @utils.cached_property
    def _kernel_args_(self):
        return self._superset._kernel_args_ + (self._indices.ctypes.data, )

    @utils.cached_property
    def _argtypes_(self):
        return self._superset._argtypes_ + (ctypes.c_voidp, )

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a :class:`Set` specific attribute."""
        value = getattr(self._superset, name)
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
        return Subset(self, indices)

    @utils.cached_property
    def superset(self):
        """Returns the superset Set"""
        return self._superset

    @utils.cached_property
    def indices(self):
        """Returns the indices pointing in the superset."""
        return self._indices

    @utils.cached_property
    def owned_indices(self):
        """Return the indices that correspond to the owned entities of the
        superset.
        """
        return self.indices[self.indices < self.superset.size]

    @utils.cached_property
    def layers_array(self):
        if self._superset.constant_layers:
            return self._superset.layers_array
        else:
            return self._superset.layers_array[self.indices, ...]

    def _check_operands(self, other):
        if type(other) is Set:
            if other is not self._superset:
                raise TypeError("Superset mismatch: self._superset (%s) != other (%s)" % (self._superset, other))
        elif type(other) is Subset:
            if self._superset is not other._superset:
                raise TypeError("Unable to perform set operation between subsets of mismatching supersets (%s != %s)" % (self._superset, other._superset))
        else:
            raise TypeError("Unable to perform set operations between `Subset` and %s." % (type(other), ))

    def intersection(self, other):
        self._check_operands(other)
        if other is self._superset:
            return self
        else:
            return type(self)(self._superset, np.intersect1d(self._indices, other._indices))

    def union(self, other):
        self._check_operands(other)
        if other is self._superset:
            return other
        else:
            return type(self)(self._superset, np.union1d(self._indices, other._indices))

    def difference(self, other):
        self._check_operands(other)
        if other is self._superset:
            return Subset(other, [])
        else:
            return type(self)(self._superset, np.setdiff1d(self._indices, other._indices))

    def symmetric_difference(self, other):
        self._check_operands(other)
        if other is self._superset:
            return other.symmetric_difference(self)
        else:
            return type(self)(self._superset, np.setxor1d(self._indices, other._indices))


class SetPartition:
    def __init__(self, set, offset, size):
        self.set = set
        self.offset = offset
        self.size = size


class MixedSet(Set, caching.ObjectCached):
    r"""A container for a bag of :class:`Set`\s."""

    def __init__(self, sets):
        r""":param iterable sets: Iterable of :class:`Set`\s or :class:`ExtrudedSet`\s"""
        if self._initialized:
            return
        self._sets = sets
        assert all(s is None or isinstance(s, GlobalSet) or ((s.layers == self._sets[0].layers).all() if s.layers is not None else True) for s in sets), \
            "All components of a MixedSet must have the same number of layers."
        # TODO: do all sets need the same communicator?
        self.comm = mpi.internal_comm(
            pytools.single_valued(s.comm for s in sets if s is not None),
            self
        )
        self._initialized = True

    @utils.cached_property
    def _kernel_args_(self):
        raise NotImplementedError

    @utils.cached_property
    def _argtypes_(self):
        raise NotImplementedError

    @utils.cached_property
    def _wrapper_cache_key_(self):
        raise NotImplementedError

    @classmethod
    def _process_args(cls, sets, **kwargs):
        sets = [s for s in sets]
        try:
            sets = utils.as_tuple(sets, ExtrudedSet)
        except TypeError:
            sets = utils.as_tuple(sets, (Set, type(None)))
        cache = sets[0]
        return (cache, ) + (sets, ), kwargs

    @classmethod
    def _cache_key(cls, sets, **kwargs):
        return sets

    def __getitem__(self, idx):
        """Return :class:`Set` with index ``idx`` or a given slice of sets."""
        return self._sets[idx]

    @utils.cached_property
    def split(self):
        r"""The underlying tuple of :class:`Set`\s."""
        return self._sets

    @utils.cached_property
    def core_size(self):
        """Core set size. Owned elements not touching halo elements."""
        return sum(s.core_size for s in self._sets)

    @utils.cached_property
    def constrained_size(self):
        """Set size, owned constrained elements."""
        return sum(s.constrained_size for s in self._sets)

    @utils.cached_property
    def size(self):
        """Set size, owned elements."""
        return sum(0 if s is None else s.size for s in self._sets)

    @utils.cached_property
    def total_size(self):
        """Total set size, including halo elements."""
        return sum(s.total_size for s in self._sets)

    @utils.cached_property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return (self.core_size, self.size, self.total_size)

    @utils.cached_property
    def name(self):
        """User-defined labels."""
        return tuple(s.name for s in self._sets)

    @utils.cached_property
    def halo(self):
        r""":class:`Halo`\s associated with these :class:`Set`\s."""
        halos = tuple(s.halo for s in self._sets)
        return halos if any(halos) else None

    @utils.cached_property
    def _extruded(self):
        return isinstance(self._sets[0], ExtrudedSet)

    @utils.cached_property
    def _extruded_periodic(self):
        raise NotImplementedError("_extruded_periodic not implemented in MixedSet")

    @utils.cached_property
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
        from pyop2.types import MixedDataSet
        return MixedDataSet(self._sets, e)

    def __str__(self):
        return "OP2 MixedSet composed of Sets: %s" % (self._sets,)

    def __repr__(self):
        return "MixedSet(%r)" % (self._sets,)

    def __eq__(self, other):
        return type(self) == type(other) and self._sets == other._sets
