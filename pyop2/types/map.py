import itertools
import functools
import numbers

import numpy as np

from pyop2 import (
    caching,
    datatypes as dtypes,
    exceptions as ex,
    utils
)
from pyop2 import mpi
from pyop2.types.set import GlobalSet, MixedSet, Set


class Map:

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

    dtype = dtypes.IntType

    @utils.validate_type(('iterset', Set, ex.SetTypeError), ('toset', Set, ex.SetTypeError),
                         ('arity', numbers.Integral, ex.ArityTypeError), ('name', str, ex.NameTypeError))
    def __init__(self, iterset, toset, arity, values=None, name=None, offset=None, offset_quotient=None):
        self._iterset = iterset
        self._toset = toset
        self.comm = mpi.internal_comm(toset.comm, self)
        self._arity = arity
        self._values = utils.verify_reshape(values, dtypes.IntType,
                                            (iterset.total_size, arity), allow_none=True)
        self.shape = (iterset.total_size, arity)
        self._name = name or "map_#x%x" % id(self)
        if offset is None or len(offset) == 0:
            self._offset = None
        else:
            self._offset = utils.verify_reshape(offset, dtypes.IntType, (arity, ))
        if offset_quotient is None or len(offset_quotient) == 0:
            self._offset_quotient = None
        else:
            self._offset_quotient = utils.verify_reshape(offset_quotient, dtypes.IntType, (arity, ))
        # A cache for objects built on top of this map
        self._cache = {}

    @utils.cached_property
    def _kernel_args_(self):
        return (self._values.ctypes.data, )

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.arity, utils.tuplify(self.offset), utils.tuplify(self.offset_quotient))

    # This is necessary so that we can convert a Map to a tuple
    # (needed in as_tuple).  Because, __getitem__ no longer returns a
    # Map we have to explicitly provide an iterable interface
    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    # Here we enforce that every map stores a single, unique MapKernelArg.
    # This is required because we use object identity to determined whether
    # maps are referenced more than once in a parloop.
    @utils.cached_property
    def _global_kernel_arg(self):
        from pyop2.global_kernel import MapKernelArg

        offset = tuple(self.offset) if self.offset is not None else None
        offset_quotient = tuple(self.offset_quotient) if self.offset_quotient is not None else None
        return MapKernelArg(self.arity, offset, offset_quotient)

    @utils.cached_property
    def split(self):
        return (self,)

    @utils.cached_property
    def iterset(self):
        """:class:`Set` mapped from."""
        return self._iterset

    @utils.cached_property
    def toset(self):
        """:class:`Set` mapped to."""
        return self._toset

    @utils.cached_property
    def arity(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element."""
        return self._arity

    @utils.cached_property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return (self._arity,)

    @utils.cached_property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0, self._arity)

    @utils.cached_property
    def values(self):
        """Mapping array.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return self._values[:self.iterset.size]

    @utils.cached_property
    def values_with_halo(self):
        """Mapping array.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return self._values

    @utils.cached_property
    def name(self):
        """User-defined label"""
        return self._name

    @utils.cached_property
    def offset(self):
        """The vertical offset."""
        return self._offset

    @utils.cached_property
    def offset_quotient(self):
        """The offset quotient."""
        return self._offset_quotient

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with arity %s" \
               % (self._name, self._iterset, self._toset, self._arity)

    def __repr__(self):
        return "Map(%r, %r, %r, None, %r, %r, %r)" \
               % (self._iterset, self._toset, self._arity, self._name, self._offset, self._offset_quotient)

    def __le__(self, o):
        """self<=o if o equals self or self._parent <= o."""
        return self == o

    @utils.cached_property
    def flattened_maps(self):
        """Return all component maps.

        This is useful to flatten nested :class:`ComposedMap`s."""
        return (self, )


class PermutedMap(Map):
    """Composition of a standard :class:`Map` with a constant permutation.

    :arg map_: The map to permute.
    :arg permutation: The permutation of the map indices.

    Where normally staging to element data is performed as

    .. code-block::

       local[i] = global[map[i]]

    With a :class:`PermutedMap` we instead get

    .. code-block::

       local[i] = global[map[permutation[i]]]

    This might be useful if your local kernel wants data in a
    different order to the one that the map provides, and you don't
    want two global-sized data structures.
    """
    def __init__(self, map_, permutation):
        if not isinstance(map_, Map):
            raise TypeError("map_ must be a Map instance")
        if isinstance(map_, ComposedMap):
            raise NotImplementedError("PermutedMap of ComposedMap not implemented: simply permute before composing")
        self.map_ = map_
        self.comm = mpi.internal_comm(map_.comm, self)
        self.permutation = np.asarray(permutation, dtype=Map.dtype)
        assert (np.unique(permutation) == np.arange(map_.arity, dtype=Map.dtype)).all()

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return super()._wrapper_cache_key_ + (tuple(self.permutation),)

    # See Map._global_kernel_arg above for more information.
    @utils.cached_property
    def _global_kernel_arg(self):
        from pyop2.global_kernel import PermutedMapKernelArg

        return PermutedMapKernelArg(self.map_._global_kernel_arg, tuple(self.permutation))

    def __getattr__(self, name):
        return getattr(self.map_, name)


class ComposedMap(Map):
    """Composition of :class:`Map`s, :class:`PermutedMap`s, and/or :class:`ComposedMap`s.

    :arg maps_: The maps to compose.

    Where normally staging to element data is performed as

    .. code-block::

       local[i] = global[map[i]]

    With a :class:`ComposedMap` we instead get

    .. code-block::

       local[i] = global[maps_[0][maps_[1][maps_[2][...[i]]]]]

    This might be useful if the map you want can be represented by
    a composition of existing maps.
    """
    def __init__(self, *maps_, name=None):
        if not all(isinstance(m, Map) for m in maps_):
            raise TypeError("All maps must be Map instances")
        for tomap, frommap in zip(maps_[:-1], maps_[1:]):
            if tomap.iterset is not frommap.toset:
                raise ex.MapTypeError("tomap.iterset must match frommap.toset")
            if tomap.comm is not frommap.comm:
                raise ex.MapTypeError("All maps needs to share a communicator")
            if frommap.arity != 1:
                raise ex.MapTypeError("frommap.arity must be 1")
        self._iterset = maps_[-1].iterset
        self._toset = maps_[0].toset
        self.comm = mpi.internal_comm(self._toset.comm, self)
        self._arity = maps_[0].arity
        # Don't call super().__init__() to avoid calling verify_reshape()
        self._values = None
        self.shape = (self._iterset.total_size, self._arity)
        self._name = name or "cmap_#x%x" % id(self)
        self._offset = maps_[0]._offset
        # A cache for objects built on top of this map
        self._cache = {}
        self.maps_ = tuple(maps_)

    @utils.cached_property
    def _kernel_args_(self):
        return tuple(itertools.chain(*[m._kernel_args_ for m in self.maps_]))

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return tuple(m._wrapper_cache_key_ for m in self.maps_)

    @utils.cached_property
    def _global_kernel_arg(self):
        from pyop2.global_kernel import ComposedMapKernelArg

        return ComposedMapKernelArg(*(m._global_kernel_arg for m in self.maps_))

    @utils.cached_property
    def values(self):
        raise RuntimeError("ComposedMap does not store values directly")

    @utils.cached_property
    def values_with_halo(self):
        raise RuntimeError("ComposedMap does not store values directly")

    def __str__(self):
        return "OP2 ComposedMap of Maps: [%s]" % ",".join([str(m) for m in self.maps_])

    def __repr__(self):
        return "ComposedMap(%s)" % ",".join([repr(m) for m in self.maps_])

    def __le__(self, o):
        raise NotImplementedError("__le__ not implemented for ComposedMap")

    @utils.cached_property
    def flattened_maps(self):
        return tuple(itertools.chain(*(m.flattened_maps for m in self.maps_)))


class MixedMap(Map, caching.ObjectCached):
    r"""A container for a bag of :class:`Map`\s."""

    def __init__(self, maps):
        r""":param iterable maps: Iterable of :class:`Map`\s"""
        if self._initialized:
            return
        self._maps = maps
        # TODO: Think about different communicators on maps (c.f. MixedSet)
        # TODO: What if all maps are None?
        comms = tuple(m.comm for m in self._maps if m is not None)
        if not all(c == comms[0] for c in comms):
            raise ex.MapTypeError("All maps needs to share a communicator")
        if len(comms) == 0:
            raise ex.MapTypeError("Don't know how to make communicator")
        self.comm = mpi.internal_comm(comms[0], self)
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        maps = utils.as_tuple(args[0], type=Map, allow_none=True)
        cache = maps[0]
        return (cache, ) + (maps, ), kwargs

    @classmethod
    def _cache_key(cls, maps):
        return maps

    @utils.cached_property
    def _kernel_args_(self):
        return tuple(itertools.chain(*(m._kernel_args_ for m in self if m is not None)))

    @utils.cached_property
    def _argtypes_(self):
        return tuple(itertools.chain(*(m._argtypes_ for m in self if m is not None)))

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return tuple(m._wrapper_cache_key_ for m in self if m is not None)

    @utils.cached_property
    def split(self):
        r"""The underlying tuple of :class:`Map`\s."""
        return self._maps

    @utils.cached_property
    def iterset(self):
        """:class:`MixedSet` mapped from."""
        s, = set(m.iterset for m in self._maps)
        if len(s) == 1:
            return functools.reduce(lambda a, b: a or b, map(lambda s: s if s is None else s.iterset, self._maps))
        else:
            raise RuntimeError("Found multiple itersets.")

    @utils.cached_property
    def toset(self):
        """:class:`MixedSet` mapped to."""
        return MixedSet(tuple(GlobalSet(comm=self.comm) if m is None else
                              m.toset for m in self._maps))

    @utils.cached_property
    def arity(self):
        """Arity of the mapping: total number of toset elements mapped to per
        iterset element."""
        s, = set(m.iterset for m in self._maps)
        if len(s) == 1:
            return sum(m.arity for m in self._maps)
        else:
            raise RuntimeError("Found multiple itersets.")

    @utils.cached_property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return tuple(m.arity for m in self._maps)

    @utils.cached_property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0,) + tuple(np.cumsum(self.arities))

    @utils.cached_property
    def values(self):
        """Mapping arrays excluding data for halos.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return tuple(m.values for m in self._maps)

    @utils.cached_property
    def values_with_halo(self):
        """Mapping arrays including data for halos.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return tuple(None if m is None else
                     m.values_with_halo for m in self._maps)

    @utils.cached_property
    def name(self):
        """User-defined labels"""
        return tuple(m.name for m in self._maps)

    @utils.cached_property
    def offset(self):
        """Vertical offsets."""
        return tuple(0 if m is None else m.offset for m in self._maps)

    @utils.cached_property
    def offset_quotient(self):
        """Offsets quotient."""
        return tuple(0 if m is None else m.offset_quotient for m in self._maps)

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

    @utils.cached_property
    def flattened_maps(self):
        raise NotImplementedError("flattend_maps should not be necessary for MixedMap")
