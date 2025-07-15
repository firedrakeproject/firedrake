.. only:: html

   .. contents::

====================
Ensemble parallelism
====================

Ensemble parallelism means solving simultaneous copies of a model
with different coefficients, right hand sides, or initial data, in
situations that require communication between the copies. Use cases
include ensemble data assimilation, uncertainty quantification, and
time parallelism.

The Ensemble communicator
=========================

In ensemble parallelism, we split the MPI communicator into a number
of spatial subcommunicators, each of which we refer to as an
ensemble member (shown in blue in the figure below). Within each
ensemble member, existing Firedrake functionality allows us to specify
the FE problems which use spatial parallelism across the spatial
subcommunicator in the usual way. Another set of
subcommunicators - the ensemble subcommunicators - then allow
communication between ensemble members (shown in grey in the figure
below). Together, the spatial and ensemble subcommunicators form a
Cartesian product over the original global communicator.

.. figure:: images/ensemble.svg
  :align: center

  Spatial and ensemble parallelism for an ensemble with 5 members,
  each of which is executed in parallel over 5 processors.

The additional functionality required to support ensemble parallelism
is the ability to send instances of :class:`~.Function` from one
ensemble to another.  This is handled by the :class:`~.Ensemble` class.

Each ensemble member has the same spatial parallelism, so
instantiating an :class:`~.Ensemble` requires a communicator to split
(usually, but not necessarily, ``MPI_COMM_WORLD``) plus the number of
MPI processes to be used in each member of the ensemble (5, in the
case of the example below). The number of ensemble members is
implicitly calculated by dividing the size of the original
communicator by the number processes in each ensemble member. The
total number of processes launched by ``mpiexec`` must therefore be
equal to the product of the number of ensemble members with the number of
processes to be used for each ensemble member.

.. code-block:: python3

   from firedrake import *

   my_ensemble = Ensemble(COMM_WORLD, 5)

Then, the spatial sub-communicator ``Ensemble.comm`` must be passed
to :func:`~.mesh.Mesh` (or via inbuilt mesh generators in
:mod:`~.utility_meshes`), so that it will then be used by function
spaces and functions derived from the mesh.

.. code-block:: python3

    mesh = UnitSquareMesh(20, 20, comm=my_ensemble.comm)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

The ensemble sub-communicator is then available through the attribute
``Ensemble.ensemble_comm``.

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

.. warning::

   In the ``Ensemble`` communication methods, each rank sends data
   only across the ``ensemble_comm`` that it is a part of. This
   assumes not only that the total mesh is identical on each ensemble
   member, but also that the ``ensemble_comm`` connects identical
   parts of the mesh on each ensemble member. Because of this, the
   spatial partitioning of the mesh on each ``Ensemble.comm`` must be
   identical.


EnsembleFunction and EnsembleFunctionSpace
==========================================

A :class:`~.Function` is logically collective over a single spatial
communicator ``Ensemble.comm``. However, for some applications we want
to treat multiple :class:`~.Function` instances on different ensemble members
as a single collective object. For example, in time-parallel methods
we may have a :class:`~.Function` for each timestep in a timeseries, and each
timestep may live on a separate ensemble member. In this case we still
want to treat the entire timeseries as a single object which is
collective over the global communicator ``Ensemble.global_comm``.

