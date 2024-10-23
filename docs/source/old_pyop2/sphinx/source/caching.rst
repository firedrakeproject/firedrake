.. _caching:

Caching in PyOP2
================

PyOP2 makes heavy use of caches to ensure performance is not adversely
affected by too many runtime computations.  The caching in PyOP2 takes
a number of forms:

1. Disk-based caching of generated code

   Since compiling a generated code module may be an expensive
   operation, PyOP2 caches the generated code on disk such that
   subsequent runs of the same simulation will not have to pay a
   compilation cost.

2. In memory caching of generated code function pointers

   Once code has been generated and loaded into the running PyOP2
   process, we cache the resulting callable function pointer for the
   lifetime of the process, such that subsequent calls to the same
   generated code are fast.

3. In memory caching of expensive to build objects

   Some PyOP2 objects, in particular :class:`~pyop2.Sparsity` objects,
   can be expensive to construct.  Since a sparsity does not change if
   it is built again with the same arguments, we only construct the
   sparsity once for each unique set of arguments.

The caching strategies for PyOP2 follow from two axioms:

1. For PyOP2 :class:`~pyop2.Set`\s and :class:`~pyop2.Map`\s, equality
   is identity
2. Caches of generated code should depend on metadata, but not data

The first axiom implies that two :class:`~pyop2.Set`\s or
:class:`~pyop2.Map`\s compare equal if and only if they are the same
object.  The second implies that generated code must be *independent*
of the absolute size of the data the :func:`~pyop2.par_loop` that
generated it executed over.  For example, the size of the iteration
set should not be part of the key, but the arity of any maps and size
and type of every data item should be.

On consequence of these rules is that there are effectively two
separate types of cache in PyOP2, object and class caches,
distinguished by where the cache itself lives.

Class caches
------------

These are used to cache objects that depend on metadata, but not
object instances, such are generated code.  They are implemented by
the cacheable class inheriting from :class:`~.Cached`.

.. note::

   There is currently no eviction strategy for class caches, should
   they grow too large, for example by executing many different parallel
   loops, an out of memory error can occur

Object caches
-------------

These are used to cache objects that are built on top of
:class:`~pyop2.Set`\s and :class:`~pyop2.Map`\s.  They are implemented by the
cacheable class inheriting from :class:`~.ObjectCached` and the
caching instance defining a ``_cache`` attribute.

The motivation for these caches is that cache key for objects such as
sparsities relies on an identical sparsity being built if the
arguments are identical.  So that users of the API do not have to
worry too much about carrying around "temporary" objects forever such
that they will hit caches, PyOP2 builds up a hierarchy of caches of
transient objects on top of the immutable sets and maps.

So, for example, the user can build and throw away
:class:`~pyop2.DataSet`\s as normal in their code.  Internally, however,
these instances are cached on the set they are built on top of.  Thus,
in the following snippet, we have that ``ds`` and ``ds2`` are the same
object:

.. code-block:: python

   s = op2.Set(1)
   ds = op2.DataSet(s, 10)
   ds2 = op2.DataSet(s, 10)
   assert ds is ds2

The setup of these caches is such that the lifetime of objects in the
cache is tied to the lifetime of both the caching and the cached
object.  In the above example, as long as the user program holds a
reference to one of ``s``, ``ds`` or ``ds2`` all three objects will
remain live.  As soon as all references are lost, all three become
candidates for garbage collection.

.. note::

   The cache eviction strategy for these caches relies on the Python
   garbage collector, and hence on the user not holding onto
   references to some of either the cached or the caching objects for
   too long.  Should the objects on which the caches live persist, an
   out of memory error may occur.

Debugging cache leaks
---------------------

To debug potential problems with the cache, PyOP2 can be instructed to
print the size of both object and class caches at program exit.  This
can be done by setting the environment variable
``PYOP2_PRINT_CACHE_SIZE`` to 1 before running a PyOP2 program, or
passing the ``print_cache_size`` to :func:`~pyop2.init`.
