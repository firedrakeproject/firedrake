.. only:: html

   .. contents::

==========================
 Parallelism in Firedrake
==========================

Firedrake uses MPI_ for distributed memory parallelism.  This is
carried out transparently as long as your usage of Firedrake is only
through the public API.  To run your code in parallel you need you use
the MPI job launcher available on your system.  Often this program is
called ``mpiexec``.  For example, to run a simulation in a file named
``simulation.py`` on 16 processes we might use.

.. code-block:: shell

   mpiexec -n 16 python simulation.py


Printing in parallel
====================

The MPI_ execution model is that of `single program, multiple data
<https://en.wikipedia.org/wiki/SPMD>`__.  As a result, printing output
requires a little bit of care: just using :func:`~.print` will result
in every process producing output.  A sensible approach is to use
PETSc's printing facilities to handle this, as :doc:`covered in this
short demo <demos/parprint.py>`.


Expected performance improvements
=================================

Without detailed analysis, it is difficult to say precisely how much
performance improvement should be expected from running in parallel.
As a rule of thumb, it is worthwhile adding more processes as long as
the number of degrees of freedom per process is more than
around 50000.  This is explored in some depth in the :doc:`main
Firedrake paper <publications>`.  Additionally, most of the finite
element calculations performed by Firedrake are limited by the *memory
bandwidth* of the machine.  You can measure how the achieved memory
bandwidth changes depending on the number of processes used on your
machine using STREAMS_.

Parallel garbage collection
===========================

As of the PETSc v3.18 release (which Firedrake started using October
2022), there should no longer be any issue with MPI distributed PETSc
objects and Python's internal garbage collector. If you previously
disabled the Python garbage collector in your Firedrake scripts, we now
recommend you turn garbage collection back on. Randomly hanging or
deadlocking parallel code should be debugged and any suspected issues
reported by :doc:`getting in touch <contact>`.

Using MPI Communicators
=======================

By default, Firedrake parallelises across ``MPI_COMM_WORLD``.  If you
want to perform a simulation in which different subsets of processes
perform different computations (perhaps solving the same PDE for
multiple different initial conditions), this can be achieved by using
sub-communicators.  The mechanism to do so is to provide a
communicator when building the :func:`~.mesh.Mesh` you will perform the
simulation on, using the optional ``comm`` keyword argument.  All
subsequent operations using that mesh are then only collective over
the supplied communicator, rather than ``MPI_COMM_WORLD``.  For
example, to split the global communicator into two and perform two
different simulations on the two halves we would write.

.. code-block:: python3

   from firedrake import *

   comm = COMM_WORLD.Split(COMM_WORLD.rank % 2)

   if COMM_WORLD.rank % 2 == 0:
      # Even ranks create a quad mesh
      mesh = UnitSquareMesh(N, N, quadrilateral=True, comm=comm)
   else:
      # Odd ranks create a triangular mesh
      mesh = UnitSquareMesh(N, N, comm=comm)

   ...

.. note::

   If you need to create Firedrake meshes on different communicators,
   then usually the best approach is to use the :class:`~.Ensemble`,
   which manages splitting MPI communicators and communicating
   :class:`~.Function` objects between the split communicators.  More
   information on using the :class:`~.Ensemble` can be found
   :doc:`here <ensemble_parallelism>`.

To access the communicator a mesh was created on, we can use the
``mesh.comm`` property, or the function ``mesh.mpi_comm``.

.. warning::
  Do not use the internal ``mesh._comm`` attribute for communication.
  This communicator is for internal Firedrake MPI communication only.

.. _MPI: http://mpi-forum.org/
.. _STREAMS: http://www.cs.virginia.edu/stream/
