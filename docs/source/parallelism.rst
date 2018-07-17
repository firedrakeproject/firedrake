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
--------------------

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

Using MPI Communicators
=======================

By default, Firedrake parallelises across ``MPI_COMM_WORLD``.  If you
want to perform a simulation in which different subsets of processes
perform different computations (perhaps solving the same PDE for
multiple different initial conditions), this can be achieved by using
sub-communicators.  The mechanism to do so is to provide a
communicator when building the ``Mesh`` you will perform the
simulation on, using the optional ``comm`` keyword argument.  All
subsequent operations using that mesh are then only collective over
the supplied communicator, rather than ``MPI_COMM_WORLD``.  For
example, to split the global communicator into two and perform two
different simulations on the two halves we would write.

.. code-block:: python

   from firedrake import *

   comm = COMM_WORLD.Split(COMM_WORLD.rank % 2)

   if COMM_WORLD.rank % 2 == 0:
      # Even ranks create a quad mesh
      mesh = UnitSquareMesh(N, N, quadrilateral=True, comm=comm)
   else:
      # Odd ranks create a triangular mesh
      mesh = UnitSquareMesh(N, N, comm=comm)

   ...

To access the communicator a mesh was created on, we can use the
``mesh.comm`` property, or the function ``mesh.mpi_comm()``.

.. _MPI: http://mpi-forum.org/
.. _STREAMS: http://www.cs.virginia.edu/stream/
