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

Firedrake objects often contain PETSc objects, but are managed by
the Python garbage collector. It is possible that when executing in
parallel, code will deadlock as the Python garbage collector is not
collective over the MPI communicator that the PETSc objects are
collective over. If you find parallel code hanging for inexplicable
reasons, it is possible to turn off the Python garbage collector by
including these lines in your code:

.. code-block:: python

    import gc
    gc.disable()

.. warning::
    Disabling the garbage collector may cause memory leaks. It is
    possible to call the garbage collector manually using
    :func:`.gc.collect` to avoid the issue, but this may in turn
    lead to a deadlock.

The garbage collector can be turned back on with this line:

.. code-block:: python

    gc.enable()

More information can be found in
`this <https://github.com/firedrakeproject/firedrake/issues/1617>`_
issue.

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
:attr:`~.mesh.comm` property, or the function :func:`~.mesh.mpi_comm`.

Ensemble parallelism
====================

Ensemble parallelism means solving simultaneous copies of a model
with different coefficients, RHS or initial data, in situations that
require communication between the copies. Use cases include ensemble
data assimilation, uncertainty quantification, and time parallelism.

In ensemble parallelism, we split the MPI communicator into a number of
subcommunicators, each of which we refer to as an ensemble
member. Within each ensemble member, existing Firedrake functionality
allows us to specify the FE problem solves which use spatial
parallelism across the subcommunicator in the usual way. Another
set of subcommunicators then allow communication between ensemble
members.

.. figure:: images/ensemble.svg
  :align: center

  Spatial and ensemble paralellism for an ensemble with 5 members,
  each of which is executed in parallel over 5 processors.

The additional functionality required to support ensemble parallelism
is the ability to send instances of :class:`~.Function` from one
ensemble to another.  This is handled by the :class:`~.Ensemble`
class. Instantiating an ensemble requires a communicator (usually
``MPI_COMM_WORLD``) plus the number of MPI processes to be used in
each member of the ensemble (5, in the case of the example
below). Each ensemble member will have the same spatial parallelism
with the number of ensemble members given by dividing the size of the
original communicator by the number processes in each ensemble
member. The total number of processes launched by ``mpiexec`` must
therefore be equal to the product of number of ensemble members with
the number of processes to be used for each ensemble member.

.. code-block:: python

   from firedrake import *

   my_ensemble = Ensemble(COMM_WORLD, 5)

Then, the spatial sub-communicator must be passed to :func:`~.mesh.Mesh` (or via
inbuilt mesh generators in :mod:`~.utility_meshes`), so that it will then be used by function spaces
and functions derived from the mesh.

.. code-block:: python

    mesh = UnitSquareMesh(20, 20, comm=my_ensemble.comm)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

The ensemble sub-communicator is then available through the method :attr:`~.Ensemble.ensemble_comm`.

.. code-block:: python

    q = Constant(my_ensemble.ensemble_comm.rank + 1)
    u.interpolate(sin(q*pi*x)*cos(q*pi*y))

MPI communications across the spatial sub-communicator (i.e., within
an ensemble member) are handled automatically by Firedrake, whilst MPI
communications across the ensemble sub-communicator (i.e., between ensemble
members) are handled through methods of :class:`~.Ensemble`. Currently only
global reductions are supported.

.. code-block:: python

    my_ensemble.allreduce(u, usum)

Other forms of MPI communication (:meth:`~.Ensemble.send`,
:meth:`~.Ensemble.recv`, :meth:`~.Ensemble.isend`,
:meth:`~.Ensemble.irecv`) are specified but not currently implemented.

.. _MPI: http://mpi-forum.org/
.. _STREAMS: http://www.cs.virginia.edu/stream/
