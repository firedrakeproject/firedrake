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

import hashlib
import os
from pathlib import Path
import pickle

import cachetools

from pyop2.configuration import configuration
from pyop2.mpi import hash_comm
from pyop2.utils import cached_property


def report_cache(typ):
    """Report the size of caches of type ``typ``

    :arg typ: A class of cached object.  For example
        :class:`ObjectCached` or :class:`Cached`.
    """
    from collections import defaultdict
    from inspect import getmodule
    from gc import get_objects
    typs = defaultdict(lambda: 0)
    n = 0
    for x in get_objects():
        if isinstance(x, typ):
            typs[type(x)] += 1
            n += 1
    if n == 0:
        print("\nNo %s objects in caches" % typ.__name__)
        return
    print("\n%d %s objects in caches" % (n, typ.__name__))
    print("Object breakdown")
    print("================")
    for k, v in typs.iteritems():
        mod = getmodule(k)
        if mod is not None:
            name = "%s.%s" % (mod.__name__, k.__name__)
        else:
            name = k.__name__
        print('%s: %d' % (name, v))


class ObjectCached(object):
    """Base class for objects that should be cached on another object.

    Derived classes need to implement classmethods
    :meth:`_process_args` and :meth:`_cache_key` (which see for more
    details).  The object on which the cache is stored should contain
    a dict in its ``_cache`` attribute.

    .. warning ::

       This kind of cache sets up a circular reference.  If either of
       the objects implements ``__del__``, the Python garbage
       collector will not be able to collect this cycle, and hence
       the cache will never be evicted.

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


class Cached(object):

    """Base class providing global caching of objects. Derived classes need to
    implement classmethods :meth:`_process_args` and :meth:`_cache_key`
    and define a class attribute :attr:`_cache` of type :class:`dict`.

    .. warning::
        The derived class' :meth:`__init__` is still called if the object is
        retrieved from cache. If that is not desired, derived classes can set
        a flag indicating whether the constructor has already been called and
        immediately return from :meth:`__init__` if the flag is set. Otherwise
        the object will be re-initialized even if it was returned from cache!
    """

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls._process_args(*args, **kwargs)
        key = cls._cache_key(*args, **kwargs)

        def make_obj():
            obj = super(Cached, cls).__new__(cls)
            obj._key = key
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
        if key is None:
            return make_obj()
        try:
            return cls._cache_lookup(key)
        except (KeyError, IOError):
            obj = make_obj()
            cls._cache_store(key, obj)
            return obj

    @classmethod
    def _cache_lookup(cls, key):
        return cls._cache[key]

    @classmethod
    def _cache_store(cls, key, val):
        cls._cache[key] = val

    @classmethod
    def _process_args(cls, *args, **kwargs):
        """Pre-processes the arguments before they are being passed to
        :meth:`_cache_key` and the constructor.

        :rtype: *must* return a :class:`list` of *args* and a
            :class:`dict` of *kwargs*"""
        return args, kwargs

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """Compute the cache key given the preprocessed constructor arguments.

        :rtype: Cache key to use or ``None`` if the object is not to be cached

        .. note:: The cache key must be hashable."""
        return tuple(args) + tuple([(k, v) for k, v in kwargs.items()])

    @cached_property
    def cache_key(self):
        """Cache key."""
        return self._key


cached = cachetools.cached
"""Cache decorator for functions. See the cachetools documentation for more
information.

.. note::
    If you intend to use this decorator to cache things that are collective
    across a communicator then you must include the communicator as part of
    the cache key. Since communicators are themselves not hashable you should
    use :func:`pyop2.mpi.hash_comm`.

    You should also make sure to use unbounded caches as otherwise some ranks
    may evict results leading to deadlocks.
"""


def disk_cached(cache, cachedir=None, key=cachetools.keys.hashkey, collective=False):
    """Decorator for wrapping a function in a cache that stores values in memory and to disk.

    :arg cache: The in-memory cache, usually a :class:`dict`.
    :arg cachedir: The location of the cache directory. Defaults to ``PYOP2_CACHE_DIR``.
    :arg key: Callable returning the cache key for the function inputs. If ``collective``
        is ``True`` then this function must return a 2-tuple where the first entry is the
        communicator to be collective over and the second is the key. This is required to ensure
        that deadlocks do not occur when using different subcommunicators.
    :arg collective: If ``True`` then cache lookup is done collectively over a communicator.
    """
    if cachedir is None:
        cachedir = configuration["cache_dir"]

    def decorator(func):
        def wrapper(*args, **kwargs):
            if collective:
                comm, disk_key = key(*args, **kwargs)
                disk_key = _as_hexdigest(disk_key)
                k = hash_comm(comm), disk_key
            else:
                k = _as_hexdigest(key(*args, **kwargs))

            # first try the in-memory cache
            try:
                return cache[k]
            except KeyError:
                pass

            # then try to retrieve from disk
            if collective:
                if comm.rank == 0:
                    v = _disk_cache_get(cachedir, disk_key)
                    comm.bcast(v, root=0)
                else:
                    v = comm.bcast(None, root=0)
            else:
                v = _disk_cache_get(cachedir, k)
            if v is not None:
                return cache.setdefault(k, v)

            # if all else fails call func and populate the caches
            v = func(*args, **kwargs)
            if collective:
                if comm.rank == 0:
                    _disk_cache_set(cachedir, disk_key, v)
            else:
                _disk_cache_set(cachedir, k, v)
            return cache.setdefault(k, v)
        return wrapper
    return decorator


def _as_hexdigest(key):
    return hashlib.md5(str(key).encode()).hexdigest()


def _disk_cache_get(cachedir, key):
    """Retrieve a value from the disk cache.

    :arg cachedir: The cache directory.
    :arg key: The cache key (must be a string).
    :returns: The cached object if found, else ``None``.
    """
    filepath = Path(cachedir, key[:2], key[2:])
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _disk_cache_set(cachedir, key, value):
    """Store a new value in the disk cache.

    :arg cachedir: The cache directory.
    :arg key: The cache key (must be a string).
    :arg value: The new item to store in the cache.
    """
    k1, k2 = key[:2], key[2:]
    basedir = Path(cachedir, k1)
    basedir.mkdir(parents=True, exist_ok=True)

    tempfile = basedir.joinpath(f"{k2}_p{os.getpid()}.tmp")
    filepath = basedir.joinpath(k2)
    with open(tempfile, "wb") as f:
        pickle.dump(value, f)
    tempfile.rename(filepath)
