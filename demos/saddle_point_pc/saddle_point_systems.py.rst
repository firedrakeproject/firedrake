Preconditioning saddle-point systems
====================================

Introduction
------------

In this demo, we will discuss strategies for solving saddle-point
systems using the mixed formulation of the Poisson equation introduced
:doc:`previously </demos/poisson_mixed.py>` as a concrete example.
Such systems are somewhat tricky to precondition effectively, modern
approaches typically use block-factorisations.  We will encounter a
number of methods in this tutorial.  For many details and background
on solution methods for saddle point systems, :cite:`Benzi:2005` is a
nice review.  :cite:`Elman:2014` is an excellent text with a strong
focus on applications in fluid dynamics.

We start by repeating the formulation of the problem.  Starting from
the primal form of the Poisson equation, :math:`\nabla^2 u = -f`, we
introduce a vector-valued flux, :math:`\sigma = \nabla u`.  The
problem then becomes to find :math:`u` and :math:`\sigma` in some
domain :math:`\Omega` satisfying

.. math::

   \begin{aligned}
   \sigma - \nabla u &= 0 \quad &\textrm{on}\ \Omega\\
   \nabla \cdot \sigma &= -f \quad &\textrm{on}\ \Omega\\
   u &= u_0 \quad &\textrm{on}\ \Gamma_D\\
   \sigma \cdot n &= g \quad &\textrm{on}\ \Gamma_N
   \end{aligned}

for some specified function :math:`f`.  We now seek :math:`(u, \sigma)
\in V \times \Sigma` such that

.. math::

   \begin{aligned}
   \int_\Omega \sigma \cdot \tau + (\nabla \cdot \tau)\, u\,\mathrm{d}x
   &= \int_\Gamma (\tau \cdot n)\,u\,\mathrm{d}s &\quad \forall\ \tau
   \in \Sigma, \\
   \int_\Omega (\nabla \cdot \sigma)\,v\,\mathrm{d}x
   &= -\int_\Omega f\,v\,\mathrm{d}x &\quad \forall\ v \in V.
   \end{aligned}

A stable choice of discrete spaces for this problem is to pick
:math:`\Sigma_h \subset \Sigma` to be the lowest order Raviart-Thomas
space, and :math:`V_h \subset V` to be the piecewise constants,
although this is :doc:`not the only choice </demos/poisson_mixed.py>`.
For ease of exposition we choose the domain to be the unit square, and
enforce homogeneous Dirichlet conditions on all walls.  The forcing
term is chosen to be random.

Globally coupled elliptic problems, such as the Poisson problem,
require effective preconditioning to attain *mesh independent*
convergence.  By this we mean that the number of iterations of the
linear solver does not grow when the mesh is refined.  In this demo,
we will study various ways to achieve this in Firedrake.

As ever, we begin by importing the Firedrake module::

    from firedrake import *

Building the problem
--------------------

Rather than defining a mesh and function spaces straight away, since
we wish to consider the effect that mesh refinement has on the
performance of the solver, we instead define a Python function which
builds the problem we wish to solve.  This takes as arguments the size
of the mesh, the solver parameters we wish to apply, an optional
parameter specifying a "preconditioning" operator to apply, and a
final optional argument specifying whether the block system should be
assembled as a single "monolithic" matrix or a :math:`2 \times 2`
block of smaller matrices. ::

    def build_problem(mesh_size, parameters, aP=None, block_matrix=False):
        mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)

        Sigma = FunctionSpace(mesh, "RT", 1)
        V = FunctionSpace(mesh, "DG", 0)
        W = Sigma * V

Having built the function spaces, we can now proceed to defining the
problem.  We will need some trial and test functions for the spaces::

    #
        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)

along with a function to hold the forcing term, living in the
discontinuous space. ::

    #
        f = Function(V)

To initialise this function to a random value we access its :class:`~.Vector`
form and use numpy_ to set the values::

    #
        import numpy as np
        fvector = f.vector()
        fvector.set_local(np.random.uniform(size=fvector.local_size()))

Note that the homogeneous Dirichlet conditions in the primal
formulation turn into homogeneous Neumann conditions on the dual
variable and we therefore drop the surface integral terms in the
variational formulation (they are identically zero).  As a result, the
specification of the variational problem is particularly simple::

    #
        a = dot(sigma, tau)*dx + div(tau)*u*dx + div(sigma)*v*dx
        L = -f*v*dx

Now we treat the mysterious optional ``aP`` argument.  When solving a
linear system, Firedrake allows specifying that the problem should be
preconditioned with an operator different to the operator defining the
problem to be solved.  We will use this functionality in a number of
cases later.  The ``aP`` function will take one argument, the
:class:`~.FunctionSpace` defining the space, and return a bilinear
form suitable for assembling as an operator.  Obviously we only do so
if ``aP`` is provided. ::

    #
        if aP is not None:
            aP = aP(W)

Now we have all the pieces to build our linear system.  We will return a
:class:`~.LinearVariationalSolver` object from this function.  It is here that
we must specify whether we want a monolithic matrix or not, by setting the
preconditioner matrix type in the solver parameters.  ::

    #
        parameters['pmat_type'] = 'nest' if block_matrix else 'aij'

        w = Function(W)
        vpb = LinearVariationalProblem(a, L, w, aP=aP)
        solver =  LinearVariationalSolver(vpb, solver_parameters=parameters)

Finally, we return solver and solution function as a tuple. ::

    #
        return solver, w

With these preliminaries out of the way, we can now move on to
solution strategies, in particular, preconditioner options.

Preconditioner choices
----------------------

A naive approach
~~~~~~~~~~~~~~~~

To illustrate the problem, we first attempt to solve the problem on a
sequence of finer and finer meshes preconditioning the problem with
zero-fill incomplete LU factorisation.  Configuration of the solver is
carried out by providing appropriate parameters when constructing the
:class:`~.LinearVariationalSolver` object through the ``solver_parameters``
keyword argument which should be a :class:`dict` of parameters.  These
parameters are passed directly to PETSc_, and their form is described
in more detail in :doc:`/solving-interface`.  For this problem, we use
GMRES with a restart length of 100, ::

    parameters = {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 100,

solve to a relative tolerance of 1e-8, ::

    #
        "ksp_rtol": 1e-8,

and precondition with ILU(0). ::

    #
        "pc_type": "ilu",
        }

We now loop over a range of mesh sizes, assembling the system and
solving it ::

    print("Naive preconditioning")
    for n in range(8):
        solver, w = build_problem(n, parameters, block_matrix=False)
        solver.solve()

Finally, at each mesh size, we print out the number of cells in the
mesh and the number of iterations the solver took to converge ::

    #
        print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

The resulting convergence is unimpressive:

============== ================
 Mesh elements GMRES iterations
============== ================
   2                  2
   8                  12
   32                 27
   128                54
   512                111
   2048               255
   8192               717
   32768              2930
============== ================

Were this a primal Poisson problem, we would be able to use a standard
algebraic multigrid preconditioner, such as hypre_.  However, this
dual formulation is slightly more complicated.

Schur complement approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A better approach is to use a Schur complement preconditioner,
described in :ref:`mixed-preconditioning`.  The system we are trying
to solve is conceptually a :math:`2\times 2` block matrix.

.. math::

   \left(\begin{matrix} A & B \\ C & 0 \end{matrix}\right)

which admits a factorisation

.. math::

   \left(\begin{matrix} I & 0 \\ C A^{-1} & I\end{matrix}\right)
   \left(\begin{matrix}A & 0 \\ 0 & S\end{matrix}\right)
   \left(\begin{matrix} I & A^{-1} B \\ 0 & I\end{matrix}\right),

with the *Schur complement* :math:`S = -C A^{-1} B`.  The inverse of
the operator can be therefore be written as

.. math::

   P = \left(\begin{matrix} I & -A^{-1}B \\ 0 & I \end{matrix}\right)
   \left(\begin{matrix} A^{-1} & 0 \\ 0 & S^{-1}\end{matrix}\right)
   \left(\begin{matrix} I & 0 \\ -CA^{-1} & I\end{matrix}\right).

An algorithmically optimal solution
+++++++++++++++++++++++++++++++++++

If we can find a good way of approximating :math:`P` then we can use
that to precondition our original problem.  This boils down to finding
good approximations to :math:`A^{-1}` and :math:`S^{-1}`.  For our
problem, :math:`A` is just a mass matrix and so we can invert it well
with a cheap method: either a few iterations of jacobi or ILU(0) are
fine.  The troublesome term is :math:`S` which is spectrally a
Laplacian, but dense (since :math:`A^{-1}` is dense).  However, before
we worry too much about this, let us just try using a Schur complement
preconditioner.  This simple setup can be driven using only solver
options.

Note that we will exactly invert the inner blocks for :math:`A^{-1}`
and :math:`S^{-1}` using Krylov methods.  We therefore need to use
*flexible* GMRES as our outer solver, since the use of inner Krylov
methods in our preconditioner makes the application of the
preconditioner nonlinear.  This time we use the default restart length
of 30, but solve to a relative tolerance of 1e-8::

    parameters = {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-8,

this time we want a ``fieldsplit`` preconditioner. ::

    #
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",

If we use this preconditioner and invert all the blocks exactly, then
the preconditioned operator will have at most three distinct
eigenvalues :cite:`Murphy:2000` and hence GMRES should converge in at
most three iterations.  To try this, we start out by exactly
inverting :math:`A` and :math:`S` to check the convergence. ::

        "fieldsplit_0_ksp_type": "cg",
        "fieldsplit_0_pc_type": "ilu",
        "fieldsplit_0_ksp_rtol": 1e-12,
        "fieldsplit_1_ksp_type": "cg",
        "fieldsplit_1_pc_type": "none",
        "fieldsplit_1_ksp_rtol": 1e-12,
    }

Let's go ahead and run this.  Note that for this problem, we're
applying the action of blocks, so we can use a block matrix format. ::

    print("Exact full Schur complement")
    for n in range(8):
        solver, w = build_problem(n, parameters, block_matrix=True)
        solver.solve()
        print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

The resulting convergence is algorithmically good, however, the larger
problems still take a long time.

============== =================
 Mesh elements fGMRES iterations
============== =================
   2                  1
   8                  1
   32                 1
   128                1
   512                1
   2048               1
   8192               1
   32768              1
============== =================

We can improve things by building a matrix used to precondition the
inversion of the Schur complement.  Note how we're currently not using
any preconditioning, and so the inner solver struggles (this can be
observed by additionally running with the parameter
``"fieldsplit_1_ksp_converged_reason": True``.

As we increase the number of mesh elements, the solver inverting
:math:`S` takes more and more iterations, which means that we take
longer and longer to solve the problem as the mesh is refined.

============== ==================
 Mesh elements CG iterations on S
============== ==================
   2                  2
   8                  7
   32                 32
   128                73
   512                149
   2048               289
   8192               553
   32768              1143
============== ==================


Approximating the Schur complement
++++++++++++++++++++++++++++++++++

Fortunately, PETSc gives us some options to try here.  For our problem
a diagonal "mass-lumping" of the velocity mass matrix gives a good
approximation to :math:`A^{-1}`.  Under these circumstances :math:`S_p
= -C \mathrm{diag}(A)^{-1} B` is spectrally close to :math:`S`, but
sparse, and can be used to precondition the solver inverting
:math:`S`.  To do this, we need some additional parameters.  First we
repeat those that remain unchanged ::

    parameters = {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_ksp_type": "cg",
        "fieldsplit_0_pc_type": "ilu",
        "fieldsplit_0_ksp_rtol": 1e-12,
        "fieldsplit_1_ksp_type": "cg",
        "fieldsplit_1_ksp_rtol": 1e-12,

Now we tell PETSc to construct :math:`S_p` using the diagonal of
:math:`A`, and to precondition the resulting linear system using
algebraic multigrid from the hypre suite. ::

        "pc_fieldsplit_schur_precondition": "selfp",
        "fieldsplit_1_pc_type": "hypre"
    }

.. note::

   For this set of options to work, you will have needed to build
   PETSc_ with support for hypre_ (for example, by specifying
   ``--download-hypre`` when configuring).

Let's see what happens. ::

    print("Schur complement with S_p")
    for n in range(8):
        solver, w = build_problem(n, parameters, block_matrix=True)
        solver.solve()
        print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

This is much better, the problem takes much less time to solve and
when observing the iteration counts for inverting :math:`S` we can see
why.

============== ==================
 Mesh elements CG iterations on S
============== ==================
   2                  2
   8                  8
   32                 17
   128                18
   512                19
   2048               19
   8192               19
   32768              19
============== ==================

We can now think about backing off the accuracy of the inner solves.
Effectively computing a worse approximation to :math:`P` that we hope
is faster, despite taking more GMRES iterations.  Additionally we can
try dropping some terms in the factorisation of :math:`P`, by adjusting
``pc_fieldsplit_schur_fact_type`` from ``full`` to one of ``upper``,
``lower``, or ``diag`` we make the preconditioner slightly worse, but
gain because we require fewer applications of :math:`A^{-1}`.  For our
problem where computing :math:`A^{-1}` is cheap, this is not a great
problem, however for many fluids problems :math:`A^{-1}` is expensive
and it pays to experiment.

For example, we might wish to try a full factorisation, but
approximate :math:`A^{-1}` by a single application of ILU(0) and
:math:`S^{-1}` by a single multigrid V-cycle on :math:`S_p`.  To do
this, we use the following set of parameters. ::

    parameters = {
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "preonly",
        "pc_fieldsplit_schur_precondition": "selfp",
        "fieldsplit_1_pc_type": "hypre"
    }

Note how we can switch back to GMRES here, our inner solves are linear
and so we no longer need a flexible Krylov method. ::

    print("Schur complement with S_p and inexact inner inverses")
    for n in range(8):
        solver, w = build_problem(n, parameters, block_matrix=True)
        solver.solve()
        print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

This results in the following GMRES iteration counts

============== ==================
 Mesh elements  GMRES iterations
============== ==================
   2                  2
   8                  9
   32                 11
   128                13
   512                13
   2048               12
   8192               12
   32768              12
============== ==================

and the solves take only a few seconds.

Providing the Schur complement approximation
++++++++++++++++++++++++++++++++++++++++++++

Instead of asking PETSc to build an approximation to :math:`S` which
we then use to solve the problem, we can provide one ourselves.
Recall that :math:`S` is spectrally a Laplacian only in a
discontinuous space.  A natural choice is therefore to use an interior
penalty DG formulation for the Laplacian term on the block of the scalar
variable. We can provide it as an :class:`~.AuxiliaryOperatorPC` via a python preconditioner. ::

    class DGLaplacian(AuxiliaryOperatorPC):
        def form(self, pc, u, v):
            W = u.function_space()
            n = FacetNormal(W.mesh())
            alpha = Constant(4.0)
            gamma = Constant(8.0)
            h = CellSize(W.mesh())
            h_avg = (h('+') + h('-'))/2
            a_dg = -(inner(grad(u), grad(v))*dx \
                - inner(jump(u, n), avg(grad(v)))*dS \
                - inner(avg(grad(u)), jump(v, n), )*dS \
                + alpha/h_avg * inner(jump(u, n), jump(v, n))*dS \
                - inner(u*n, grad(v))*ds \
                - inner(grad(u), v*n)*ds \
                + (gamma/h)*inner(u, v)*ds)
            bcs = None
            return (a_dg, bcs)

    parameters = {
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": __name__+ ".DGLaplacian",
        "fieldsplit_1_aux_pc_type": "hypre"
    }

    print("DG approximation for S_p")
    for n in range(8):
        solver, w = build_problem(n, parameters, aP=None, block_matrix=False)
        solver.solve()
        print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

This actually results in slightly worse convergence than the diagonal
approximation we used above.

============== ==================
 Mesh elements  GMRES iterations
============== ==================
    2                 2
    8                 9
    32                12
    128               13
    512               14
    2048              13
    8192              13
    32768             13
============== ==================

Block diagonal preconditioners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternate approach to using a Schur complement is to use a
block-diagonal preconditioner.  To do this, we note that the
mesh-dependent ill conditioning of linear operators comes from working
in the wrong norm.  To convert into working in the correct norm, we
can precondition our problem using the *Riesz map* for the spaces.
For details on the mathematics behind this approach see for example
:cite:`Kirby:2010`.

We are working in a space :math:`W \subset H(\text{div}) \times L^2`,
and as such, the appropriate Riesz map is just :math:`H(\text{div})`
inner product in :math:`\Sigma` and the :math:`L^2` inner product in
:math:`V`.  As was the case for the DG Laplacian, we do this by
providing a function that constructs this operator to our
``build_problem`` function. ::

    def riesz(W):
        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)

        return (dot(sigma, tau) + div(sigma)*div(tau) + u*v)*dx

Now we set up the solver parameters.  We will still use a
``fieldsplit`` preconditioner, but this time it will be additive,
rather than a Schur complement. ::

    parameters = {
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",

Now we choose how to invert the two blocks.  The second block is easy,
it is just a mass matrix in a discontinuous space and is therefore
inverted exactly using a single application of zero-fill ILU. ::

    #
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "ilu",

The :math:`H(\text{div})` inner product is the tricky part. For a
first attempt, we will invert it with a direct solver.  This is a reasonable
option up to a few tens of thousands of degrees of freedom. ::

    #
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
    }

.. note::

   For larger problems, you will probably need to use a sparse direct
   solver such as MUMPS_, which may be selected by additionally
   specifying ``"fieldsplit_0_pc_factor_mat_solver_type": "mumps"``.

   To use MUMPS_ you will need to have configured PETSc_ appropriately
   (using at the very least ``--download-mumps``).

Let's see what the iteration count looks like now. ::

    print("Riesz-map preconditioner")
    for n in range(8):
        solver, w = build_problem(n, parameters, aP=riesz, block_matrix=True)
        solver.solve()
        print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

============== ==================
 Mesh elements  GMRES iterations
============== ==================
   2                  3
   8                  5
   32                 5
   128                5
   512                5
   2048               5
   8192               5
   32768              5
============== ==================


Firedrake provides some facility to solve the :math:`H(\mathrm{div})`
Riesz map in a scalable way. In particular either by employing a
geometric multigrid method with overlapping Schwarz smoothers (using
:class:`.PatchPC`), or using the algebraic approach of
:cite:`Hiptmair:2007` provided by `Hypre's
<https://hypre.readthedocs.io/en/latest/>`__ "auxiliary space"
preconditioners ``AMS`` and ``ADS``. See the separate manual page on
:doc:`../preconditioning`.

A runnable python script version of this demo is available :demo:`here
<saddle_point_systems.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames

.. _PETSc: https://petsc.org/
.. _hypre: https://hypre.readthedocs.io/en/latest/
.. _numpy: https://www.numpy.org
.. _MUMPS: https://mumps-solver.org/index.php
