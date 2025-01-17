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


Installing for parallel use
===========================

By default, Firedrake makes use of an MPICH library that is
downloaded, configured, and installed in the virtual environment as
part of the PETSc installation procedure.  If you do not intend to use
parallelism, or only use it in a limited way, this will be sufficient
for your needs.  The default MPICH installation uses ``nemesis`` as the
MPI channel, which is reasonably fast, but imposes a hard limit on the
maximum number of concurrent MPI threads equal to the number of cores
on your machine.  If you would like to be able to *oversubscribe* your
machine, and run more threads than cores, you need to change the MPICH
device at install time to ``sock``, by setting an environment variable
before you run ``firedrake-install``:

.. code-block:: shell

   export PETSC_CONFIGURE_OPTIONS="--download-mpich-device=ch3:sock"

If parallel performance is important to you (e.g., for generating
reliable timings or using a supercomputer), then you should probably
be using an MPICH library tuned for your system.  If you have a
system-wide install already available, then you can simply tell the
firedrake installer to use it, by running:

.. code-block:: shell

   python3 firedrake-install --mpiexec=mpiexec --mpicc=mpicc --mpicxx=mpicxx --mpif90=mpif90 --mpihome mpihome

where ``mpiexec``, ``mpicc``, ``mpicxx``, and ``mpif90`` are the
commands to run an MPI job and to compile C, C++, and Fortran 90 code,
respectively. ``mpihome`` is an extra variable that must point to the
root directory of the MPI installation (e.g. ``/usr`` or ``/opt/mpich``).

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

To access the communicator a mesh was created on, we can use the
``mesh.comm`` property, or the function ``mesh.mpi_comm``.

.. warning::
  Do not use the internal ``mesh._comm`` attribute for communication.
  This communicator is for internal Firedrake MPI communication only.


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

.. code-block:: python3

   from firedrake import *

   my_ensemble = Ensemble(COMM_WORLD, 5)

Then, the spatial sub-communicator must be passed to :func:`~.mesh.Mesh` (or via
inbuilt mesh generators in :mod:`~.utility_meshes`), so that it will then be used by function spaces
and functions derived from the mesh.

.. code-block:: python3

    mesh = UnitSquareMesh(20, 20, comm=my_ensemble.comm)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

The ensemble sub-communicator is then available through the attribute ``Ensemble.ensemble_comm``.

.. code-block:: python3

    q = Constant(my_ensemble.ensemble_comm.rank + 1)
    u.interpolate(sin(q*pi*x)*cos(q*pi*y))

MPI communications across the spatial sub-communicator (i.e., within
an ensemble member) are handled automatically by Firedrake, whilst MPI
communications across the ensemble sub-communicator (i.e., between ensemble
members) are handled through methods of :class:`~.Ensemble`. Currently
send/recv, reductions and broadcasts are supported, as well as their
non-blocking variants.

.. code-block:: python3

    my_ensemble.send(u, dest)
    my_ensemble.recv(u, source)

    my_ensemble.reduce(u, usum, root)
    my_ensemble.allreduce(u, usum)

    my_ensemble.bcast(u, root)

.. _MPI: http://mpi-forum.org/
.. _STREAMS: http://www.cs.virginia.edu/stream/
