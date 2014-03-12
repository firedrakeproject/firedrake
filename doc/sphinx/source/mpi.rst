.. _mpi:

MPI
===

Distributed parallel computations with MPI in PyOP2 require the mesh to be
partitioned among the processors. To be able to compute over entities on their
boundaries, partitions need to access data owned by neighboring processors.
This region, called the *halo*, needs to be kept up to date and is therefore
exchanged between the processors as required.

Local Numbering
---------------

Each processor owns a partition of each :class:`~pyop2.Set`, which is again
divided into the following four sections:

* **Core**: Entities owned by this processor which can be processed without
  accessing halo data.
* **Owned**: Entities owned by this processor which access halo data when
  processed.
* **Exec halo**: Off-processor entities which are redundantly executed over
  because they touch owned entities.
* **Non-exec halo**: Off-processor entities which are not processed, but read
  when computing the exec halo.

These four sections are contiguous and local :class:`~pyop2.Set` entities
must therefore be numbered such that core entities are first, followed by
owned, exec halo and non-exec halo in that order. A good partitioning
maximises the size of the core section and minimises the halo regions. We can
therefore assume that the vast majority of local :class:`~pyop2.Set` entities
are in the core section.
