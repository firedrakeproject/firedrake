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

"""Provides common base classes for cached objects."""
import cachetools
import contextlib
import functools
import hashlib
import os
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

from pyop3.config import CONFIG
from pyop3.log import debug
from pyop3.mpi import (
    MPI, COMM_WORLD, comm_cache_keyval, temp_internal_comm
)
import pytools
from petsc4py import PETSc


_CACHE_CIDX = count()
_KNOWN_CACHES = []


def cached_on(obj, key=cachetools.keys.hashkey):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = obj(*args, **kwargs)
            assert isinstance(cache, CacheMixin)

            k = key(*args, **kwargs)
            try:
                return cache.cache_get(k)
            except KeyError:
                value = func(*args, **kwargs)
                cache.cache_set(k, value)
                return value
        return wrapper
    return decorator

class CacheMixin:
    """Mixin class for objects that may be treated as a cache."""
    def __init__(self):
        self._cache = {}

    def cache_get(self, key):
        return self._cache[key]

    def cache_set(self, key, value):
        self._cache[key] = value


def cache_filter(comm=None, comm_name=None, alive=True, function=None, cache_type=None):
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
        caches = filter(lambda c: c.comm != MPI.COMM_NULL, caches)
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


def get_cache_entry(comm: MPI.Comm, cache: Mapping, key: Hashable) -> Any:
    if (
        CONFIG.spmd_strict
        and not pytools.is_single_valued(comm.allgather(key))
    ):
        raise ValueError(
            f"Cache keys differ between ranks. On rank {comm.rank} got:\n{key}"
        )

    value = cache.get(key, CACHE_MISS)

    if CONFIG.debug:
        message = [f"{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: "]
        message.append(f"key={key} in cache: '{cache}' ")
        if value is CACHE_MISS:
            message.append("miss")
        else:
            message.append("hit")
        message = "".join(message)
        debug(message)

    return value


class _CacheRecord:
    """Object that records cache statistics."""
    def __init__(self, cidx, comm, func, cache):
        self.cidx = cidx
        self.comm = comm
        self.comm_name = comm.name
        self.func = func
        self.func_module = func.__module__
        self.func_name = func.__qualname__
        self.cache = weakref.ref(cache)
        fin = weakref.finalize(cache, self.finalize, cache)
        fin.atexit = False
        self.cache_name = cache.__class__.__qualname__
        try:
            self.cache_loc = cache.cachedir
        except AttributeError:
            self.cache_loc = "Memory"

    def get_stats(self, cache=None):
        if cache is None:
            cache = self.cache()
        hit = miss = size = maxsize = -1
        if cache is None:
            hit, miss, size, maxsize = self.hit, self.miss, self.size, self.maxsize
        if isinstance(cache, cachetools.Cache):
            size = cache.currsize
            maxsize = cache.maxsize
        if hasattr(cache, "instrument__"):
            hit = cache.hit
            miss = cache.miss
            if size == -1:
                try:
                    size = len(cache)
                except NotImplementedError:
                    pass
            if maxsize is None:
                try:
                    maxsize = cache.max_size
                except AttributeError:
                    pass
        return hit, miss, size, maxsize

    def finalize(self, cache):
        self.hit, self.miss, self.size, self.maxsize = self.get_stats(cache)


def print_cache_stats(*args, **kwargs):
    """ Print out the cache hit/miss/size/maxsize stats for PyOP2 caches.
    """
    data = defaultdict(lambda: defaultdict(list))
    for entry in cache_filter(*args, **kwargs):
        active = (entry.comm != MPI.COMM_NULL)
        data[(entry.comm_name, active)][(entry.cache_name, entry.cache_loc)].append(
            (entry.cidx, entry.func_module, entry.func_name, entry.get_stats())
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
                stats_row = "|".join(f"{s:{w}}" for s, w in zip(entry[3], stats_col))
                print(f"|{function_title:{col[0]}}|{stats_row:{col[1]}}|")
        print(hline)


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
    """ A sensible default hash key for use with `parallel_cache`.
    """
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
    return cachetools.keys.hashkey(*hash_args, **hash_kwargs)


def instrument(cls):
    """ Class decorator for dict-like objects for counting cache hits/misses.
    """
    @wraps(cls, updated=())
    class _wrapper(cls):
        instrument__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hit = 0
            self.miss = 0

        def get(self, key, default=None):
            value = super().get(key, default)
            if value is default:
                self.miss += 1
            else:
                self.hit += 1
            return value

        def __getitem__(self, key):
            try:
                value = super().__getitem__(key)
                self.hit += 1
            except KeyError as e:
                self.miss += 1
                raise e
            return value
    return _wrapper


class DEFAULT_CACHE(dict):
    pass


# Example of how to instrument and use different default caches:
# from functools import partial
# EXOTIC_CACHE = partial(instrument(cachetools.LRUCache), maxsize=100)

# Turn on cache measurements if printing cache info is enabled
# FIXME: make a function, not global config
# if configuration["print_cache_info"]:
#     DEFAULT_CACHE = instrument(DEFAULT_CACHE)
#     DictLikeDiskAccess = instrument(DictLikeDiskAccess)


# TODO: One day should use the compilation comm to do the bcast
def parallel_cache(
    hashkey=default_parallel_hashkey,
    get_comm: Callable = default_get_comm,
    make_cache: Callable[[], Mapping] = lambda: DEFAULT_CACHE(),
    bcast=False,
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
            # Create a PyOP2 comm associated with the key, so it is decrefed
            # when the wrapper exits

            with temp_internal_comm(get_comm(*args, **kwargs)) as comm:
                # Get the right cache from the comm
                comm_caches = get_comm_caches(comm)
                try:
                    cache = comm_caches[cache_id]
                except KeyError:
                    cache = comm_caches.setdefault(cache_id, make_cache())
                    _KNOWN_CACHES.append(_CacheRecord(cache_id, comm, func, cache))

                key = hashkey(*args, **kwargs)
                value = get_cache_entry(comm, cache, key)

                if isinstance(cache, DictLikeDiskAccess):
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
                        CONFIG.spmd_strict
                        and not pytools.is_single_valued(
                            comm.allgather(value is not CACHE_MISS)
                        )
                    ):
                        raise ValueError("Cache hit on some ranks but missed on others")

            if value is CACHE_MISS:
                if bcast:
                    value = func(*args, **kwargs) if comm.rank == 0 else None
                    value = comm.bcast(value, root=0)
                else:
                    value = func(*args, **kwargs)

            return cache.setdefault(key, value)
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


def disk_only_cache(*args, cachedir=CONFIG.cache_dir, **kwargs):
    return parallel_cache(*args, **kwargs, make_cache=lambda: DictLikeDiskAccess(cachedir))


def memory_and_disk_cache(*args, cachedir=CONFIG.cache_dir, **kwargs):
    def decorator(func):
        return memory_cache(*args, **kwargs)(disk_only_cache(*args, cachedir=cachedir, **kwargs)(func))
    return decorator


# I *think* that we are fine to not worry about comms here because we can
# be confident about collectiveness.
_active_scoped_cache = None


class active_scoped_cache:
    def __init__(self, cache):
        self._cache = cache

    def __enter__(self):
        global _active_scoped_cache

        if _active_scoped_cache is None:
            _active_scoped_cache = self._cache
            self._set_cache = True
        else:
            self._set_cache = False

    def __exit__(self, *exc):
        global _active_scoped_cache

        if self._set_cache:
            _active_scoped_cache = None


def scoped_cache(*args, **kwargs):
    """Cache decorator for 'heavy' objects.

    Unlike the other cache decorators this cache is scoped to another object
    and will be cleaned up with that object.

    If a cache scope has not been set with `active_scoped_cache` then no
    caching happens.

    """
    return cachetools.cached(cache=_active_scoped_cache, **kwargs)


# HEAVY_CACHE_COMM_KEYVAL = MPI.Comm.Create_keyval()
#
#
# def scoped_cache(*args, **kwargs):
#     """Cache decorator for 'heavy' objects.
#
#     Unlike the other cache decorators this cache is scoped to another object
#     and will be cleaned up with that object.
#
#     If a cache scope has not been set with `active_scoped_cache` then no
#     caching happens.
#
#     """
#     return memory_cache(*args, **kwargs, scoped=True)
#
#
# class active_scoped_cache:
#     """
#     """
#     def __init__(self, lifetime_obj):
#         self._lifetime_obj = lifetime_obj
#
#     def __enter__(self):
#         with temp_internal_comm(self._lifetime_obj.comm) as comm:
#             # only overwrite if no object exists yet
#             if comm.Get_attr(HEAVY_CACHE_COMM_KEYVAL) is None:
#                 comm.Set_attr(HEAVY_CACHE_COMM_KEYVAL, self._lifetime_obj)
#                 self._set = True
#             else:
#                 self._set = False
#
#     def __exit__(self, *exc):
#         if self._set:
#             with temp_internal_comm(self._lifetime_obj.comm) as comm:
#                 comm.Set_attr(HEAVY_CACHE_COMM_KEYVAL, None)
