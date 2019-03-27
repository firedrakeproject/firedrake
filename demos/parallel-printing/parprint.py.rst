Basic printing in parallel
==========================

.. rst-class:: emphasis

   Contributed by `Ed Bueler <mailto:elbueler@alaska.edu>`__.

This example shows how one may print various quantities in parallel.  The
`Firedrake <https://www.firedrakeproject.org>`_ public interface largely works
as-is in parallel but several of the operations here expose the PETSc and MPI
underpinnings in order to print.

Run this example in parallel using :math:`P` processes by doing
``mpiexec -n P python3 parprint.py``.

We start with the usual import but we also import `petsc4py <https://bitbucket.org/petsc/petsc4py/>`_
so that classes ``PETSc.X`` are available.  Here ``X`` is one of the
`PETSc object types <https://www.mcs.anl.gov/petsc/documentation/index.html>`_,
including types like `Vec <http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/index.html>`_::

    from firedrake import *
    from firedrake.petsc import PETSc

In serial the next line could be ``print('setting up mesh...')``  However,
in parallel that would print :math:`P` times on :math:`P` processes.  In the
following form the print happens collectively across the default MPI
communicator, namely ``COMM_WORLD``::

    PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)

Next we generate a mesh.  It has an MPI communicator ``mesh.comm``,
also ``COMM_WORLD`` by default.  Here each rank reports on the portion of the
mesh it owns by using the ``COMM_SELF`` communicator::

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

And solve::

    PETSc.Sys.Print('solving problem ...')
    u = Function(V)
    solve(a == L, u, options_prefix='s', solver_parameters={'ksp_type': 'cg'})

To print the solution vector in serial one could write
``print(u.dat.data)`` but then each processor would show its data separately.
In PETSc language we do a "view" of the solution vector::

    with u.dat.vec_ro as vu:
        vu.view()

Here ``vu`` is an instance of the PETSc.Vec class and ``vu.view()`` is the
equivalent of ``VecView(vu,NULL)`` in PETSc C.  It is a "global" Vec which means
that each degree of freedom is stored on a unique process.  The context manager
in the above usage (i.e. ``with ...``) allows Firedrake to generate a global Vec
by halo exchanges if needed.  Here we only need read-only access here so we use
``u.dat.vec_ro``.  (By contrast ``u.dat.vec`` would allow read-write access.)

Finally we compute and print the numerical error, relative to the exact
solution, in two norms.  The :math:`L^2` norm is computed with
``assemble`` which already includes an MPI reduction across the ``mesh.comm``
communicator::

    uexact = Function(V)
    uexact.interpolate(cos(x*pi*2)*cos(y*pi*2))
    L_2_err = sqrt(assemble(dot(u - uexact, u - uexact) * dx))

We compute the :math:`L^\infty` error a different way.  Note that
``L_inf_err = u.dat.data.max()`` would work in serial but in parallel that only
gets the max over the process-owned entries.  So again we use the ``PETSc.Vec``
approach::

    with u.dat.vec_ro as vu:
        L_inf_err = vu.max()[1]
    PETSc.Sys.Print('L_2 error norm = %g, L_inf error norm = %g' \
                    % (L_2_err,L_inf_err))

.. note::

   ``max()`` on a ``PETSc.Vec`` returns an ``(index,max)`` pair, thus
   the ``[1]`` to obtain the max value.
