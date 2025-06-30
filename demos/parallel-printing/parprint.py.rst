Basic printing in parallel
==========================

.. rst-class:: emphasis

   Contributed by `Ed Bueler <mailto:elbueler@alaska.edu>`__.

This example shows how one may print various quantities in parallel.  The
`Firedrake <https://www.firedrakeproject.org>`_ public interface mostly works
as-is in parallel but several of the operations here expose the PETSc and MPI
underpinnings in order to print.

Run this example in parallel using :math:`P` processes by doing
``mpiexec -n P python3 parprint.py``.

We start with the usual import but we also import `petsc4py <https://bitbucket.org/petsc/petsc4py/>`_
so that classes ``PETSc.X`` are available.  Here ``X`` is one of the
`PETSc object types <https://petsc.org/release/manualpages/>`_,
including types like `Vec <https://petsc.org/release/manualpages/Vec/>`_::

    from firedrake import *
    from firedrake.petsc import PETSc

In serial the next line could be ``print('setting up mesh...')``  However,
in parallel that would print :math:`P` times on :math:`P` processes.  In the
following form the print happens only once (because it is done only on rank 0)::

    PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)

Next we generate a mesh.  It has an MPI communicator ``mesh.comm``, equal to
``COMM_WORLD`` by default.  By using the ``COMM_SELF`` communicator each rank
reports on the portion of the mesh it owns::

    mesh = UnitSquareMesh(3, 3)
    PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
                    % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
                    comm=COMM_SELF)

The *elements* of the mesh are owned uniquely in parallel, while the
vertices are shared via "halos" or "ghost vertices".  Note there is a nontrivial
relationship between vertices and degrees of freedom in a global PETSc Vec (below).

We use a familiar Helmholtz equation problem merely for demonstration.
First we set up a weak form just as in the
`helmholtz.py <https://www.firedrakeproject.org/demos/helmholtz.py.html>`_
demo::

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x,y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    a = (dot(grad(v), grad(u)) + v * u) * dx
    L = f * v * dx

Then solve::

    PETSc.Sys.Print('solving problem ...')
    u = Function(V)
    solve(a == L, u, options_prefix='s', solver_parameters={'ksp_type': 'cg'})

To print the solution vector in serial one could write ``print(u.dat.data)``
but then in parallel each processor would show its data separately.
So using PETSc we do a "view" of the solution vector::

    with u.dat.vec_ro as vu:
        vu.view()

Here ``vu`` is an instance of the PETSc.Vec class and ``vu.view()`` is the
equivalent of ``VecView(vu,NULL)`` using PETSc's C API.  This Vec is "global",
meaning that each degree of freedom is stored on a unique process.  The context manager
in the above usage (i.e. ``with ...``) allows Firedrake to generate a global Vec
by halo exchanges if needed.  Here we only need read-only access here so we use
``u.dat.vec_ro``; note ``u.dat.vec`` would allow read-write access.

Finally we compute and print the numerical error, relative to the exact
solution, in two norms.  The :math:`L^2` norm is computed with
``assemble`` which already includes an MPI reduction across the ``mesh.comm``
communicator::

    udiff = Function(V).interpolate(u - cos(x*pi*2)*cos(y*pi*2))
    L_2_err = sqrt(assemble(dot(udiff,udiff) * dx))

We compute the :math:`L^\infty` error a different way.  Note that
``u.dat.data.max()`` works in serial but in parallel that only
gets the max over the process-owned entries.  So again we use the ``PETSc.Vec``
approach::

    udiffabs = Function(V).interpolate(abs(udiff))
    with udiffabs.dat.vec_ro as v:
        L_inf_err = v.max()[1]
    PETSc.Sys.Print('L_2 error norm = %g, L_inf error norm = %g' \
                    % (L_2_err,L_inf_err))

.. note::

   ``max()`` on a ``PETSc.Vec`` returns an ``(index,max)`` pair, thus
   the ``[1]`` to obtain the max value.
