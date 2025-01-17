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
import atexit
import cachetools
import hashlib
import os
import pickle
import weakref
from collections.abc import MutableMapping
from pathlib import Path
from warnings import warn  # noqa F401
from collections import defaultdict
from itertools import count
from functools import wraps
from tempfile import mkstemp

from pyop2.configuration import configuration
from pyop2.exceptions import CachingError, HashError  # noqa: F401
from pyop2.logger import debug
from pyop2.mpi import (
    MPI, COMM_WORLD, comm_cache_keyval, temp_internal_comm
)
from petsc4py import PETSc


# Caches created here are registered as a tuple of
#     (creation_index, comm, comm.name, function, cache)
# in _KNOWN_CACHES
_CACHE_CIDX = count()
_KNOWN_CACHES = []
# Flag for outputting information at the end of testing (do not abuse!)
_running_on_ci = bool(os.environ.get('PYOP2_CI_TESTS'))


# FIXME: (Later) Remove ObjectCached
class ObjectCached(object):
    """Base class for objects that should be cached on another object.

    Derived classes need to implement classmethods
    :meth:`_process_args` and :meth:`_cache_key` (which see for more
    details).  The object on which the cache is stored should contain
    a dict in its ``_cache`` attribute.

    .. warning::

        The derived class' :meth:`__init__` is still called if the
        object is retrieved from cache. If that is not desired,
        derived classes can set a flag indicating whether the
        constructor has already been called and immediately return
        from :meth:`__init__` if the flag is set. Otherwise the object
        will be re-initialized even if it was returned from cache!

    """

    @classmethod
    def _process_args(cls, *args, **kwargs):
        """Process the arguments to ``__init__`` into a form suitable
        for computing a cache key on.

        The first returned argument is popped off the argument list
        passed to ``__init__`` and is used as the object on which to
        cache this instance.  As such, *args* should be returned as a
        two-tuple of ``(cache_object, ) + (original_args, )``.

        *kwargs* must be a (possibly empty) dict.
        """
        raise NotImplementedError("Subclass must implement _process_args")

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """Compute a cache key from the constructor's preprocessed arguments.
        If ``None`` is returned, the object is not to be cached.

        .. note::

           The return type **must** be hashable.

        """
        raise NotImplementedError("Subclass must implement _cache_key")

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls._process_args(*args, **kwargs)
        # First argument is the object we're going to cache on
        cache_obj = args[0]
        # These are now the arguments to the subclass constructor
        args = args[1:]
        key = cls._cache_key(*args, **kwargs)

        def make_obj():
            obj = super(ObjectCached, cls).__new__(cls)
            obj._initialized = False
            # obj.__init__ will be called twice when constructing
            # something not in the cache.  The first time here, with
            # the canonicalised args, the second time directly in the
            # subclass.  But that one should hit the cache and return
            # straight away.
            obj.__init__(*args, **kwargs)
            return obj

        # Don't bother looking in caches if we're not meant to cache
        # this object.
        if key is None or cache_obj is None:
            return make_obj()

        # Does the caching object know about the caches?
        try:
            cache = cache_obj._cache
        except AttributeError:
            raise RuntimeError("Provided caching object does not have a '_cache' attribute.")

        # OK, we have a cache, let's go ahead and try and find our
        # object in it.
        try:
            return cache[key]
        except KeyError:
            obj = make_obj()
            cache[key] = obj
            return obj


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


class _CacheRecord:
    """ Object for keeping a record of Pyop2 Cache statistics.
    """
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


if _running_on_ci:
    print_cache_stats = atexit.register(print_cache_stats)


class _CacheMiss:
    pass


CACHE_MISS = _CacheMiss()


def _as_hexdigest(*args):
    hash_ = hashlib.md5()
    for a in args:
        if isinstance(a, MPI.Comm):
            raise HashError("Communicators cannot be hashed, caching will be broken!")
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

    def __getitem__(self, key):
        """Retrieve a value from the disk cache.

        :arg key: The cache key, a 2-tuple of strings.
        :returns: The cached object if found.
        """
        filepath = Path(self.cachedir, key[0][:2], key[0][2:] + key[1])
        try:
            with self.open(filepath.with_suffix(self.extension), mode="rb") as fh:
                value = self.read(fh)
        except FileNotFoundError:
            raise KeyError("File not on disk, cache miss")
        return value

    def __setitem__(self, key, value):
        """Store a new value in the disk cache.

        :arg key: The cache key, a 2-tuple of strings.
        :arg value: The new item to store in the cache.
        """
        k1, k2 = key[0][:2], key[0][2:] + key[1]
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


def default_comm_fetcher(*args, **kwargs):
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


def default_parallel_hashkey(*args, **kwargs):
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
if configuration["print_cache_info"] or _running_on_ci:
    DEFAULT_CACHE = instrument(DEFAULT_CACHE)
    DictLikeDiskAccess = instrument(DictLikeDiskAccess)


if configuration["spmd_strict"]:
    def parallel_cache(
        hashkey=default_parallel_hashkey,
        comm_fetcher=default_comm_fetcher,
        cache_factory=lambda: DEFAULT_CACHE(),
    ):
        """Parallel cache decorator (SPMD strict-enabled).
        """
        def decorator(func):
            @PETSc.Log.EventDecorator("PyOP2 Cache Wrapper")
            @wraps(func)
            def wrapper(*args, **kwargs):
                """ Extract the key and then try the memory cache before falling back
                on calling the function and populating the cache. SPMD strict ensures
                that all ranks cache hit or miss to ensure that the function evaluation
                always occurs in parallel.
                """
                k = hashkey(*args, **kwargs)
                key = _as_hexdigest(*k), func.__qualname__
                # Create a PyOP2 comm associated with the key, so it is decrefed when the wrapper exits
                with temp_internal_comm(comm_fetcher(*args, **kwargs)) as comm:
                    # Fetch the per-comm cache_collection or set it up if not present
                    # A collection is required since different types of cache can be set up on the same comm
                    cache_collection = comm.Get_attr(comm_cache_keyval)
                    if cache_collection is None:
                        cache_collection = {}
                        comm.Set_attr(comm_cache_keyval, cache_collection)
                    # If this kind of cache is already present on the
                    # cache_collection, get it, otherwise create it
                    local_cache = cache_collection.setdefault(
                        (cf := cache_factory()).__class__.__name__,
                        cf
                    )
                    local_cache = cache_collection[cf.__class__.__name__]

                    # If this is a new cache or function add it to the list of known caches
                    if (comm, comm.name, func, local_cache) not in [(c.comm, c.comm_name, c.func, c.cache()) for c in _KNOWN_CACHES]:
                        # When a comm is freed we do not hold a reference to the cache.
                        # We attach a finalizer that extracts the stats before the cache
                        # is deleted.
                        _KNOWN_CACHES.append(_CacheRecord(next(_CACHE_CIDX), comm, func, local_cache))

                    # Grab value from all ranks cache and broadcast cache hit/miss
                    value = local_cache.get(key, CACHE_MISS)
                    debug_string = f"{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: "
                    debug_string += f"key={k} in cache: {local_cache.__class__.__name__} cache "
                    if value is CACHE_MISS:
                        debug(debug_string + "miss")
                        cache_hit = False
                    else:
                        debug(debug_string + "hit")
                        cache_hit = True
                    all_present = comm.allgather(cache_hit)

                    # If not present in the cache of all ranks we force re-evaluation on all ranks
                    if not min(all_present):
                        value = CACHE_MISS

                if value is CACHE_MISS:
                    value = func(*args, **kwargs)
                return local_cache.setdefault(key, value)

            return wrapper
        return decorator
else:
    def parallel_cache(
        hashkey=default_parallel_hashkey,
        comm_fetcher=default_comm_fetcher,
        cache_factory=lambda: DEFAULT_CACHE(),
    ):
        """Parallel cache decorator.
        """
        def decorator(func):
            @PETSc.Log.EventDecorator("PyOP2 Cache Wrapper")
            @wraps(func)
            def wrapper(*args, **kwargs):
                """ Extract the key and then try the memory cache before falling back
                on calling the function and populating the cache.
                """
                k = hashkey(*args, **kwargs)
                key = _as_hexdigest(*k), func.__qualname__
                # Create a PyOP2 comm associated with the key, so it is decrefed when the wrapper exits
                with temp_internal_comm(comm_fetcher(*args, **kwargs)) as comm:
                    # Fetch the per-comm cache_collection or set it up if not present
                    # A collection is required since different types of cache can be set up on the same comm
                    cache_collection = comm.Get_attr(comm_cache_keyval)
                    if cache_collection is None:
                        cache_collection = {}
                        comm.Set_attr(comm_cache_keyval, cache_collection)
                    # If this kind of cache is already present on the
                    # cache_collection, get it, otherwise create it
                    local_cache = cache_collection.setdefault(
                        (cf := cache_factory()).__class__.__name__,
                        cf
                    )
                    local_cache = cache_collection[cf.__class__.__name__]

                    # If this is a new cache or function add it to the list of known caches
                    if (comm, comm.name, func, local_cache) not in [(c.comm, c.comm_name, c.func, c.cache()) for c in _KNOWN_CACHES]:
                        # When a comm is freed we do not hold a reference to the cache.
                        # We attach a finalizer that extracts the stats before the cache
                        # is deleted.
                        _KNOWN_CACHES.append(_CacheRecord(next(_CACHE_CIDX), comm, func, local_cache))

                    value = local_cache.get(key, CACHE_MISS)

                if value is CACHE_MISS:
                    with PETSc.Log.Event("pyop2: handle cache miss"):
                        value = func(*args, **kwargs)
                return local_cache.setdefault(key, value)

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


def serial_cache(hashkey, cache_factory=lambda: DEFAULT_CACHE()):
    return cachetools.cached(key=hashkey, cache=cache_factory())


def disk_only_cache(*args, cachedir=configuration["cache_dir"], **kwargs):
    return parallel_cache(*args, **kwargs, cache_factory=lambda: DictLikeDiskAccess(cachedir))


def memory_and_disk_cache(*args, cachedir=configuration["cache_dir"], **kwargs):
    def decorator(func):
        return memory_cache(*args, **kwargs)(disk_only_cache(*args, cachedir=cachedir, **kwargs)(func))
    return decorator
