# Copyright (c) 2026, Imperial College London and others.
# Please see the AUTHORS file in the main source directory for
# a full list of copyright holders. All rights reserved.

"""Provides common base classes for cached objects."""

import abc
import atexit
import cachetools
import collections
import contextlib
import functools
import gc
import hashlib
import os
import sys
import pickle
import weakref
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from warnings import warn  # noqa F401
from collections import defaultdict
from itertools import count
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Hashable

from petsc4py import PETSc

from pyop3 import utils
from pyop3.collections import AlwaysEmptyDict
from pyop3.config import config
from pyop3.exceptions import CacheException
from pyop3.log import debug, LOGGER
from pyop3.mpi import (
    MPI, COMM_WORLD, comm_cache_keyval, temp_internal_comm
)


_CACHE_CIDX = count()
_KNOWN_CACHES = []


# TODO: This should live in utils.py but there is a (bad) import of pyop3.cache
# that prohibits this for now.
class gc_disabled(contextlib.ContextDecorator):
    """Context manager for temporarily disabling the garbage collector.

    It may also be used as a function decorator.

    """
    def __init__(self):
        # Track GC status using a stack because recursive uses as a function
        # decorator will reuse the same object
        self._was_enabled = []

    def __enter__(self):
        self._was_enabled.append(gc.isenabled())
        gc.disable()

    def __exit__(self, *args, **kwargs):
        if self._was_enabled.pop(-1):
            gc.enable()


def _get_refcounts(lifetime_objs):
    return [sys.getrefcount(obj) for obj in lifetime_objs]


# @gc_disabled()
def _checked_get_key(cache_type, get_key, lifetime_objs=None):
    # I think that this is fine. Refcycles aren't really an issue.
    return get_key()


    if not lifetime_objs or issubclass(cache_type, weakref.WeakKeyDictionary):
        return get_key()

    # Check that we are not putting anything in the cache that would
    # create a reference cycle
    orig_refcounts = _get_refcounts(lifetime_objs)
    key = get_key()
    if _get_refcounts(lifetime_objs) != orig_refcounts:
        raise CacheException(
            "Cache key contains a reference to the object that "
            "is used to define the cache lifetime. This means "
            "that the cache will never be cleared."
        )
    return key


@gc_disabled()
def _checked_compute_value(cache_type, get_value, lifetime_objs=None):
    # I think that this is fine. Refcycles aren't really an issue.
    return get_value()

    if not lifetime_objs or issubclass(cache_type, weakref.WeakValueDictionary):
        return get_value()

    # Check that we are not putting anything in the cache that would
    # create a reference cycle
    orig_refcounts = _get_refcounts(lifetime_objs)
    value = get_value()
    if _get_refcounts(lifetime_objs) != orig_refcounts:
        raise CacheException(
            "Cache value contains a reference to the object that "
            "is used to define the cache lifetime. This means "
            "that the cache will never be cleared."
        )
    return value


def cached_on(get_obj, get_key: Callable = cachetools.keys.hashkey, *, unsafe_refcounts: bool = False):
    """
    Parameters
    ----------
    unsafe_refcounts
        Flag to disable refcount checking for cache accesses when debug checks are
        enabled. This is important to bypass cases where the wrapped function may
        inadvertently create additional references to ``obj``, for instance by
        populating extra cached properties.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            obj = get_obj(*args, **kwargs)
            if not hasattr(obj, "_pyop3_cache"):
                # Use object.__setattr__ to get around frozen dataclasses
                object.__setattr__(obj, "_pyop3_cache", collections.defaultdict(dict))
            cache = obj._pyop3_cache[func.__qualname__]

            if config.debug_checks and not unsafe_refcounts:
                key = _checked_get_key(type(cache), lambda: get_key(*args, **kwargs), [obj])
            else:
                key = get_key(*args, **kwargs)
            try:
                return cache[key]
            except KeyError:
                if config.debug_checks and not unsafe_refcounts:
                    value = _checked_compute_value(type(cache), lambda: func(*args, **kwargs), [obj])
                else:
                    value = func(*args, **kwargs)
                return cache.setdefault(key, value)
        return wrapper
    return decorator


def default_hashkey(*args, **kwargs) -> tuple[Hashable, ...]:
    args_key = tuple(utils.freeze(a) for a in args)
    kwargs_key = tuple((key, utils.freeze(value)) for key, value in kwargs.items())
    return (args_key, kwargs_key)


class CacheMixin:
    """Mixin class for objects that may be treated as a cache."""
    def __init__(self):
        self._cache = {}

    def cache_get(self, key):
        return self._cache[key]

    def cache_set(self, key, value):
        self._cache[key] = value


def cache_filter(comm=None, comm_name=None, alive=False, function=None, cache_type=None):
    """ Filter PyOP2 caches based on communicator, function or cache type.
    """
    caches = _KNOWN_CACHES
    if comm is not None:
        with temp_internal_comm(comm) as icomm:
            cache_collection = icomm.Get_attr(comm_cache_keyval)
            if cache_collection is None:
                print(f"Communicator {icomm.name} has no associated caches")
            comm_name = icomm.name
    if comm_name is not None:
        caches = filter(lambda c: c.comm_name == comm_name, caches)
    if alive:
        caches = filter(lambda c: not isinstance(c, _DeadInstrumentedCache), caches)
    if function is not None:
        if isinstance(function, str):
            caches = filter(lambda c: function in c.func_name, caches)
        else:
            caches = filter(lambda c: c.func is function, caches)
    if cache_type is not None:
        if isinstance(cache_type, str):
            caches = filter(lambda c: cache_type in c.cache_name, caches)
        else:
            caches = filter(lambda c: c.cache_name == cache_type.__class__.__qualname__, caches)
    return [*caches]


def get_comm_caches(comm: MPI.Comm) -> dict[Hashable, Mapping]:
    """Return the collection of caches that are stored on a comm.

    If a cache stash has not already been created then a new `dict` is
    created and stored.

    Parameters
    ----------
    comm :
        The communicator to get the caches from.

    Returns
    -------
    dict :
        The collection of caches.

    """
    comm_caches = comm.Get_attr(comm_cache_keyval)
    if comm_caches is None:
        comm_caches = {}
        comm.Set_attr(comm_cache_keyval, comm_caches)
    return comm_caches


class _AbstractInstrumentedCache(abc.ABC):
    def __init__(self, cidx, comm, func):
        self.cidx = cidx
        self.comm = comm
        self.comm_name = comm.name
        self.func = func
        self.func_module = func.__module__
        self.func_name = func.__qualname__
        self.known_cache_index = len(_KNOWN_CACHES)
        _KNOWN_CACHES.append(weakref.proxy(self))

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def maxsize(self) -> int:
        ...


class _InstrumentedCache(_AbstractInstrumentedCache):
    def __init__(self, cidx, comm, func, cache):
        self.cache = cache
        self.cache_name = cache.__class__.__qualname__
        try:
            self.cache_loc = cache.cachedir
        except AttributeError:
            self.cache_loc = "Memory"

        self.hit = 0
        self.miss = 0

        super().__init__(cidx, comm, func)

    def __del__(self):
        _KNOWN_CACHES[self.known_cache_index] = _DeadInstrumentedCache(self.cidx, self.cache_name, self.cache_loc, self.comm, self.func, self.hit, self.miss, self.size, self.maxsize)

    def __getitem__(self, key):
        try:
            value = self.cache[key]
        except KeyError as e:
            self.miss += 1

            if self.miss == 1000 and self.miss / (self.hit+self.miss) > 0.8:
                LOGGER.warning(
                    f"Cache '{self}' has recorded 1000 misses at a hit rate of "
                    "greater than 80%. This indicates a problem with your cache key."
                )

            raise e
        else:
            self.hit += 1
            return value

    def __setitem__(self, key, value) -> None:
        self.cache[key] = value

    def get(self, key, default=None):
        try:
            value = self[key]
        except KeyError:
            self.miss += 1
            return default
        else:
            self.hit += 1
            return value

    # TODO: singledispatch
    @property
    def size(self) -> int:
        # TODO: quite ick here
        try:
            return len(self.cache)
        except:
            return self.miss

    # TODO: singledispatch
    @property
    def maxsize(self) -> int:
        if isinstance(self.cache, cachetools.Cache):
            return self.cache.maxsize
        else:
            return -1


class _DeadInstrumentedCache(_AbstractInstrumentedCache):
    def __init__(self, cidx, cache_name, cache_loc, comm, func, nhit, nmiss, size, maxsize):
        self.cache_name = cache_name
        self.cache_loc = cache_loc
        self.hit = nhit
        self.miss = nmiss
        self._size = size
        self._maxsize = maxsize
        super().__init__(cidx, comm, func)

    @property
    def size(self) -> int:
        return self._size

    @property
    def maxsize(self) -> int:
        return self._maxsize


def print_cache_stats(*args, **kwargs):
    """Print cache statistics."""
    data = defaultdict(lambda: defaultdict(list))
    for entry in cache_filter(*args, **kwargs):
        active = not isinstance(entry, _DeadInstrumentedCache)
        data[(entry.comm_name, active)][(entry.cache_name, entry.cache_loc)].append(
            (entry.cidx, entry.func_module, entry.func_name, (entry.hit, entry.miss, entry.size, entry.maxsize))
        )

    tab = "  "
    hline = "-"*120
    col = (90, 27)
    stats_col = (6, 6, 6, 6)
    stats = ("hit", "miss", "size", "max")
    no_stats = "|".join(" "*ii for ii in stats_col)
    print(hline)
    print(f"|{'Cache':^{col[0]}}|{'Stats':^{col[1]}}|")
    subtitles = "|".join(f"{st:^{w}}" for st, w in zip(stats, stats_col))
    print("|" + " "*col[0] + f"|{subtitles:{col[1]}}|")
    print(hline)
    for ecomm, cachedict in data.items():
        active = "Active" if ecomm[1] else "Freed"
        comm_title = f"{ecomm[0]} ({active})"
        print(f"|{comm_title:{col[0]}}|{no_stats}|")
        for ecache, function_list in cachedict.items():
            cache_title = f"{tab}{ecache[0]}"
            print(f"|{cache_title:{col[0]}}|{no_stats}|")
            cache_location = f"{tab} ↳ {ecache[1]!s}"
            if len(cache_location) < col[0]:
                print(f"|{cache_location:{col[0]}}|{no_stats}|")
            else:
                print(f"|{cache_location:78}|")
            for entry in function_list:
                function_title = f"{tab*2}id={entry[0]} {'.'.join(entry[1:3])}"
                stats_row = "|".join(f"{s:{w}}" for s, w in zip(entry[3], stats_col, strict=True))
                print(f"|{function_title:{col[0]}}|{stats_row:{col[1]}}|")
        print(hline)


if config.print_cache_stats:
    atexit.register(print_cache_stats)


class _CacheMiss:
    pass


CACHE_MISS = _CacheMiss()


@functools.cache
def as_hexdigest(*args) -> str:
    """Return ``args`` as a hash string.

    Notes
    -----
    This function is relatively expensive to compute so one should avoid
    calling it wherever possible.

    """
    hash_ = hashlib.md5()
    for a in args:
        if isinstance(a, MPI.Comm):
            raise TypeError("Communicators cannot be hashed, caching will be broken!")
        hash_.update(str(a).encode())
    return hash_.hexdigest()


class DictLikeDiskAccess(MutableMapping):
    """ A Dictionary like interface for storing and retrieving objects from a disk cache.
    """
    def __init__(self, cachedir, extension=".pickle"):
        """

        :arg cachedir: The cache directory.
        :arg extension: Optional extension to use for written files.
        """
        self.cachedir = cachedir
        self.extension = extension

    def __getitem__(self, key: Hashable) -> Any:
        """Retrieve a value from the disk cache."""
        key = as_hexdigest(key)

        filepath = Path(self.cachedir, key[:2], key[2:])
        try:
            with self.open(filepath.with_suffix(self.extension), mode="rb") as fh:
                value = self.read(fh)
        except FileNotFoundError:
            raise KeyError("File not on disk, cache miss")
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Store a new value in the disk cache."""
        key = as_hexdigest(key)

        k1, k2 = key[:2], key[2:]
        basedir = Path(self.cachedir, k1)
        basedir.mkdir(parents=True, exist_ok=True)

        # Care must be taken here to ensure that the file is created safely as
        # the filesystem may be network based. `mkstemp` does so securely without
        # race conditions:
        # https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp
        # The file descriptor must also be closed after use with `os.close()`.
        fd, tempfile = mkstemp(suffix=".tmp", prefix=k2, dir=basedir, text=False)
        tempfile = Path(tempfile)
        # Open using `tempfile` (the filename) rather than the file descriptor
        # to allow redefining `self.open`
        with self.open(tempfile, mode="wb") as fh:
            self.write(fh, value)
        os.close(fd)

        # Renaming (moving) the file is guaranteed by any POSIX compliant
        # filesystem to be atomic. This may fail if somehow the destination is
        # on another filesystem, but that shouldn't happen here.
        filepath = basedir.joinpath(k2)
        tempfile.rename(filepath.with_suffix(self.extension))

    def __delitem__(self, key):
        raise NotImplementedError(f"Cannot remove items from {self.__class__.__name__}")

    def __iter__(self):
        raise NotImplementedError(f"Cannot iterate over keys in {self.__class__.__name__}")

    def __len__(self):
        raise NotImplementedError(f"Cannot query length of {self.__class__.__name__}")

    def __repr__(self):
        return f"{self.__class__.__name__}(cachedir={self.cachedir}, extension={self.extension})"

    def __eq__(self, other):
        # Instances are the same if they have the same cachedir
        return (self.cachedir == other.cachedir and self.extension == other.extension)

    def open(self, *args, **kwargs):
        return open(*args, **kwargs)

    def read(self, filehandle):
        return pickle.load(filehandle)

    def write(self, filehandle, value):
        pickle.dump(value, filehandle)


def default_get_comm(*args, **kwargs):
    """ A sensible default comm fetcher for use with `parallel_cache`.
    """
    comms = filter(
        lambda arg: isinstance(arg, MPI.Comm),
        args + tuple(kwargs.values())
    )
    try:
        comm = next(comms)
    except StopIteration:
        raise TypeError("No comms found in args or kwargs")
    return comm


def default_parallel_hashkey(*args, **kwargs) -> Hashable:
    """ A sensible default hash key for use with `parallel_cache`."""
    # We now want to actively remove any comms from args and kwargs to get
    # the same disk cache key.
    hash_args = tuple(filter(
        lambda arg: not isinstance(arg, MPI.Comm),
        args
    ))
    hash_kwargs = dict(filter(
        lambda arg: not isinstance(arg[1], MPI.Comm),
        kwargs.items()
    ))
    return default_hashkey(*hash_args, **hash_kwargs)


class DEFAULT_CACHE(dict):
    pass


# Turn on cache measurements if printing cache info is enabled
# FIXME: make a function, not global config
# if configuration["print_cache_info"]:


# TODO: One day should use the compilation comm to do the bcast
def parallel_cache(
    hashkey=default_parallel_hashkey,
    get_comm: Callable = default_get_comm,
    make_cache: Callable[[], Mapping] = lambda: DEFAULT_CACHE(),
    bcast=False,
    heavy: bool = False,
):
    """Parallel cache decorator.

    Parameters
    ----------
    hashkey :
        Callable taking ``*args`` and ``**kwargs`` and returning a hash.
    get_comm :
        Callable taking ``*args`` and ``**kwargs`` and returning the
        appropriate communicator.
    make_cache :
        Callable that will build a new cache (if one does not exist).
        This will be called every time the decorated function is called, and must return an instance
        of the same type every time it is called.
    bcast :
        If `True`, then generate the new cache value on one rank and broadcast
        to the others. If `False` then values are generated on all ranks.
        This option can only be `True` if the operation can be executed in
        serial; else it will deadlock.
    heavy :
        Do the objects stored in the cache have a large memory footprint? If
        yes then this cache is only used when a 'heavy' cache is set (see the
        `heavy_cache` context manager) and the lifetime of the objects in the
        cache are tied to the lifetime of the cache.

    """
    # Store a unique integer for each 'parallel_cache' decorator so we can
    # identify the different caches when we wrap a function in multiple of
    # them (this happens for memory and disk caches for example). This
    # identifier is different between ranks but that is fine as it is only
    # used locally.
    cache_id = next(_CACHE_CIDX)

    def decorator(func):
        @PETSc.Log.EventDecorator(f"pyop2.caching.parallel_cache.wrapper({func.__qualname__})")
        @wraps(func)
        def wrapper(*args, **kwargs):
            with temp_internal_comm(get_comm(*args, **kwargs)) as comm:
                if heavy and len(_heavy_caches) == 0:
                    caches = (AlwaysEmptyDict(),)
                    cache_type = AlwaysEmptyDict
                    value = CACHE_MISS
                else:
                    def make_instrumented_cache():
                        cache = make_cache()
                        return _InstrumentedCache(cache_id, comm, func, cache)

                    comm_caches = get_comm_caches(comm)
                    if heavy:
                        if cache_id not in comm_caches:
                            comm_caches[cache_id] = weakref.WeakKeyDictionary()

                        caches = []
                        cache_type = None
                        for lifetime_obj in _heavy_caches:
                            try:
                                cache = comm_caches[cache_id][lifetime_obj]
                            except KeyError:
                                cache = make_instrumented_cache()
                                comm_caches[cache_id][lifetime_obj] = cache

                            if cache_type is None:
                                cache_type = type(cache)
                            caches.append(cache)
                        caches = tuple(caches)
                        assert cache_type is not None
                        assert not issubclass(cache_type, DictLikeDiskAccess), "Disk caches cannot be heavy"
                    else:
                        try:
                            cache = comm_caches[cache_id]
                        except KeyError:
                            cache = make_instrumented_cache()
                            comm_caches[cache_id] = cache
                        caches = (cache,)
                        cache_type = type(cache)

                if config.debug_checks and heavy:
                    key = _checked_get_key(cache_type, lambda: hashkey(*args, **kwargs), list(_heavy_caches))
                else:
                    key = hashkey(*args, **kwargs)

                for cache in caches:
                    try:
                        value = cache[key]
                        break
                    except KeyError:
                        pass
                else:
                    value = CACHE_MISS

                if issubclass(cache_type, DictLikeDiskAccess):
                    if bcast:
                        # Since disk caches share state between ranks there are extra
                        # opportunities for mismatching hit/miss results and hence
                        # deadlocks. These include:
                        #
                        # 1. Race conditions
                        #
                        # On CI or with ensemble parallelism other processes not in this
                        # comm may write to disk, so load imbalances on the current comm
                        # may result in a hit on some ranks but not others.
                        #
                        # 2. Eager writing to disk on rank 0
                        #
                        # Since broadcasting is non-blocking for the sending rank (rank 0)
                        # it is possible for it to have written to disk before other ranks
                        # begin the cache lookup. These ranks register a cache hit.
                        #
                        # If ranks disagree on whether it was a hit or miss then some ranks
                        # will do a broadcast and others will not, ruining MPI synchronisation.
                        # To fix this we check to see if any ranks have hit cache and, if so,
                        # nominate that rank as the root of the subsequent broadcast.
                        root = comm.rank if value is not CACHE_MISS else -1
                        root = comm.allreduce(root, op=MPI.MAX)
                        if root >= 0:
                            # Found a rank with a cache hit, broadcast 'value' from it
                            value = comm.bcast(value, root=root)
                else:
                    # In-memory caches are stashed on the comm and so must always agree
                    # on their contents.
                    if (
                        config.spmd_strict
                        and not utils.is_single_valued(
                            comm.allgather(value is not CACHE_MISS)
                        )
                    ):
                        raise ValueError("Cache hit on some ranks but missed on others")

                if value is CACHE_MISS:
                    if bcast:
                        value = func(*args, **kwargs) if comm.rank == 0 else None
                        value = comm.bcast(value, root=0)
                    else:
                        if config.debug_checks and heavy:
                            value = _checked_compute_value(cache_type, lambda: func(*args, **kwargs), lifetime_objs=list(_heavy_caches))
                        else:
                            value = _checked_compute_value(cache_type, lambda: func(*args, **kwargs))

                for cache in caches:
                    cache[key] = value
                return value
        return wrapper
    return decorator


def clear_memory_cache(comm):
    """ Completely remove all PyOP2 caches on a given communicator.
    """
    with temp_internal_comm(comm) as icomm:
        if icomm.Get_attr(comm_cache_keyval) is not None:
            icomm.Set_attr(comm_cache_keyval, {})


# A small collection of default simple caches
memory_cache = parallel_cache


def serial_cache(hashkey=cachetools.keys.hashkey, cache_factory=lambda: DEFAULT_CACHE()):
    return cachetools.cached(key=hashkey, cache=cache_factory())


def disk_only_cache(*args, cachedir=config.cache_dir, **kwargs):
    return parallel_cache(*args, **kwargs, make_cache=lambda: DictLikeDiskAccess(cachedir))


def memory_and_disk_cache(*args, cachedir=config.cache_dir, **kwargs):
    def decorator(func):
        return memory_cache(*args, **kwargs)(disk_only_cache(*args, cachedir=cachedir, **kwargs)(func))
    return decorator


_heavy_caches = weakref.WeakSet()


class heavy_caches:
    """Context manager that pushes and pops lifetime objects.

    For this to be parallel safe, the contract here is that, by using this
    decorator, you are guaranteeing that all operations within the context
    manager are at most collective to the level of the the communicator of
    the lifetime objects.

    """

    def __init__(self, objs: Any) -> None:
        objs = utils.as_tuple(objs)
        self._objs = objs
        # keep track of the objects we inserted ourselves, if they were already
        # there then we don't want to remove them!
        self._added_objs = set()

    def __enter__(self) -> None:
        for obj in self._objs:
            if obj not in _heavy_caches:
                _heavy_caches.add(obj)
                self._added_objs.add(obj)

    def __exit__(self, *args) -> None:
        for obj in self._added_objs:
            _heavy_caches.remove(obj)
        self._added_objs.clear()


def with_heavy_caches(get_obj: Callable) -> Callable:
    """Function decorator that pushes and pops lifetime objects."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            obj = get_obj(*args, **kwargs)
            with heavy_caches(obj):
                return func(*args, **kwargs)
        return wrapper
    return decorator
