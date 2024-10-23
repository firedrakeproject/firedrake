.. _plan:

Parallel Execution Plan
=======================

For all PyOP2 backends with the exception of sequential, a parallel execution
plan is computed for each :func:`~pyop2.par_loop`. It contains information
guiding the code generator on how to partition, stage and colour the data for
efficient parallel processing.

.. _plan-partitioning:

Partitioning
------------

The iteration set is split into a number of equally sized and contiguous
mini-partitions such that the working set of each mini-partition fits into
shared memory or last level cache. This is unrelated to the partitioning
required for MPI as described in :ref:`mpi`.

.. _plan-renumbering:

Local Renumbering and Staging
-----------------------------

While a mini-partition is a contiguous chunk of the iteration set, the
indirectly accessed data it references is not necessarily contiguous. For each
mini-partition and unique :class:`~pyop2.Dat`-:class:`~pyop2.Map` pair, a
mapping from local indices within the partition to global indices is
constructed as the sorted array of unique :class:`~pyop2.Map` indices accessed
by this partition. At the same time, a global-to-local mapping is constructed
as its inverse.

Data for indirectly accessed :class:`~pyop2.Dat` arguments is staged in shared
device memory as described in :ref:`backends`. For each partition, the
local-to-global mapping indicates where data to be staged in is read from and
the global-to-local mapping gives the location in shared memory data has been
staged at. The amount of shared memory required is computed from the size of
the local-to-global mapping.

.. _plan-colouring:

Colouring
---------

A two-level colouring is used to avoid race conditions. Partitions are
coloured such that partitions of the same colour can be executed concurrently
and threads executing on a partition in parallel are coloured such that no two
threads indirectly reference the same data. Only :func:`~pyop2.par_loop`
arguments performing an indirect reduction or assembling a matrix require
colouring. Matrices are coloured per row.

For each element of a :class:`~pyop2.Set` indirectly accessed in a
:func:`~pyop2.par_loop`, a bit vector is used to record which colours
indirectly reference it. To colour each thread within a partition, the
algorithm proceeds as follows:

1. Loop over all indirectly accessed arguments and collect the colours of all
   :class:`~pyop2.Set` elements referenced by the current thread in a bit mask.
2. Choose the next available colour as the colour of the current thread.
3. Loop over all :class:`~pyop2.Set` elements indirectly accessed by the
   current thread again and set the new colour in their colour mask.

Since the bit mask is a 32-bit integer, up to 32 colours can be processed in a
single pass, which is sufficient for most applications. If not all threads can
be coloured with 32 distinct colours, the mask is reset and another pass is
made, where each newly allocated colour is offset by 32. Should another pass
be required, the offset is increased to 64 and so on until all threads are
coloured.

.. figure:: images/pyop2_colouring.svg
  :align: center

  Thread colouring within a mini-partition for a :class:`~pyop2.Dat` on
  vertices indirectly accessed in a computation over the edges. The edges are
  coloured such that no two edges touch the same vertex within the partition.

The colouring of mini-partitions is done in the same way, except that all
:class:`~pyop2.Set` elements indirectly accessed by the entire partition are
referenced, not only those accessed by a single thread.
