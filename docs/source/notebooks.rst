:orphan: true

Introductory Jupyter notebooks
==============================

These notebooks provide an introduction to usage of Firedrake, and are
designed to familiarise you with manipulating Firedrake objects to
solve finite element problems.  The rendered notebooks below are executed
as part of the documentation build; each page also links to a version you
can run yourself on Google Colab.

Running the notebooks locally
-----------------------------

To run the notebooks, you will need to `install jupyter
<https://jupyter.org/install.html>`__ *inside* your activated
Firedrake virtualenv.

These notebooks are maintained in the Firedrake repository as
`jupytext <https://jupytext.readthedocs.io>`__ ``py:percent`` scripts, so all
the material is available in your Firedrake installation source directory.  If
you cloned Firedrake in ``Documents/firedrake``, then the notebooks are in the
directory ``Documents/firedrake/docs/notebooks``.  Jupyter (with the jupytext
extension installed) can open the ``.py`` files directly as notebooks, or you
can convert one to ``.ipynb`` with, for example::

    jupytext --to ipynb 01-spd-helmholtz.py

Running the notebooks on Google Colab
-------------------------------------

Thanks to the excellent `FEM on
Colab <https://fem-on-colab.github.io/index.html>`__ by `Francesco
Ballarin <https://www.francescoballarin.it>`__, you can run the notebooks on
Google Colab through your web browser, without installing Firedrake.  Each
rendered notebook below carries a link to its Colab version.

The notebooks
-------------

:doc:`A first example <notebooks/01-spd-helmholtz>`
    In this notebook, we solve the symmetric positive definite "Helmholtz"
    equation, and learn about meshes and function spaces.

:doc:`Incorporating strong boundary conditions <notebooks/02-poisson>`
    Next, we modify the problem slightly and solve the Poisson equation.  We
    introduce strong (Dirichlet) boundary conditions and how to use them.

:doc:`A vector-valued problem <notebooks/03-elasticity>`
    Moving on from scalar problems, we look at our first vector-valued problem,
    namely the equations of linear elasticity.  In this notebook, we learn
    about some of UFL's support for tensor algebra, and start looking at
    configuring linear solvers.

:doc:`A time-dependent, nonlinear, problem <notebooks/04-burgers>`
    This notebook looks at a simple nonlinear problem, the viscous Burgers'
    equation, and also treats simple timestepping schemes.  We learn about
    formulating nonlinear, as opposed to linear problems, and also a little bit
    about how to write efficient Firedrake code.

:doc:`A mixed formulation of the Poisson equation <notebooks/05-mixed-poisson>`
    Here we look at our first mixed finite element problem, a dual formulation
    of the Poisson equation.  This equation also appears in the context of flow
    in porous media, as Darcy flow.  We introduce mixed function spaces and how
    to work with them.  Equations with multiple variables are typically more
    challenging to precondition, and so we discuss some of the preconditioning
    strategies for such block systems, and how to control them using PETSc
    solver options.

:doc:`PDE-constrained optimisation <notebooks/06-pde-constrained-optimisation>`
    Now that we've learnt how to solve some PDEs, we might want to consider
    optimisation subject to PDE constraints.  This notebook introduces the use
    of `dolfin-adjoint <http://www.dolfin-adjoint.org/>`__ to solve PDE
    constrained optimisation problems.  We solve the Stokes equations and
    minimise energy loss due to heat, controlling inflow/outflow in a pipe.

:doc:`Geometric multigrid <notebooks/07-geometric-multigrid>`
    This notebook looks a little bit at the support Firedrake has for geometric
    multigrid, and how you can configure complex multilevel solvers purely
    using PETSc options.

:doc:`Solver composition <notebooks/08-composable-solvers>`
    We next dive a little deeper into the advanced ways in which Firedrake and
    PETSc enable solvers and preconditioners to be composed in arbitrarily
    complex ways to create an optimal solution strategy for a particular
    problem.

:doc:`Hybridisation <notebooks/09-hybridisation>`
    Building on the theme of composable solvers, we now explore Firedrake's
    capabilities in the area of static condensation and hybridisation.

:doc:`Sum factorisation <notebooks/10-sum-factorisation>`
    In this notebook, we take a look under the hood at the sorts of performance
    optimisation that Firedrake's compilers can generate.  In this case, we
    focus on sum factorisation for tensor product elements.

:doc:`Solving adjoint problems <notebooks/11-extract-adjoint-solutions>`
    In some cases, it can be useful to extract adjoint solution data which is
    written to tape by ``dolfin-adjoint``.  This notebook shows how to solve
    adjoint equations using firedrake-adjoint.

:doc:`Running on HPC <notebooks/12-HPC_demo>`
    When it comes to running Firedrake on a high performance computer there are
    a range of different techniques to get the best performance from your code.
    This HPC demonstration notebook builds up a multigrid solver for an
    elliptic problem specifically designed for solving very large problems
    using Firedrake on HPC.

.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/01-spd-helmholtz
   notebooks/02-poisson
   notebooks/03-elasticity
   notebooks/04-burgers
   notebooks/05-mixed-poisson
   notebooks/06-pde-constrained-optimisation
   notebooks/07-geometric-multigrid
   notebooks/08-composable-solvers
   notebooks/09-hybridisation
   notebooks/10-sum-factorisation
   notebooks/11-extract-adjoint-solutions
   notebooks/12-HPC_demo
